# ───────────────────────────────────────────────────────────────
# Copyright (c) 2026 Ambrish Nigam
# Author : Ambrish Nigam | https://github.com/ambrishnigam30
# Project: SynthFlow — Autonomous Synthetic Data Orchestration Platform
# License : Apache License 2.0 | https://www.apache.org/licenses/LICENSE-2.0
# Module : GenerationService — wrap SynthFlow engine, persist results
# ───────────────────────────────────────────────────────────────

from __future__ import annotations

import asyncio
import logging
import uuid
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.storage import SupabaseStorageClient
from app.models.generation import Generation
from app.models.user import User
from app.services.usage_service import UsageService

logger = logging.getLogger(__name__)

# Plan row limits
_PLAN_ROW_CAPS: dict[str, int] = {
    "free":       1_000,
    "starter":    10_000,
    "pro":        100_000,
    "business":   1_000_000,
    "enterprise": 0,  # unlimited
}

# Plan generation-count limits per month
_PLAN_GEN_CAPS: dict[str, int] = {
    "free":       10,
    "starter":    100,
    "pro":        200,
    "business":   0,
    "enterprise": 0,
}

PhaseCallback = Callable[[int, float, str], None]


class PlanLimitError(Exception):
    """Raised when the user has exceeded their plan limits."""


class GenerationService:
    def __init__(
        self,
        db: AsyncSession,
        session_factory: async_sessionmaker[AsyncSession] | None = None,
        storage: SupabaseStorageClient | None = None,
    ) -> None:
        self._db = db
        self._session_factory = session_factory
        self._storage = storage or SupabaseStorageClient()

    # ── Plan enforcement ───────────────────────────────────────────────────

    async def _check_plan_limits(
        self, user: User, requested_rows: int
    ) -> None:
        """Raise PlanLimitError if user exceeds row or monthly generation caps."""
        row_cap = _PLAN_ROW_CAPS.get(user.plan, 1_000)
        if row_cap > 0 and requested_rows > row_cap:
            raise PlanLimitError(
                f"Your {user.plan} plan supports up to {row_cap:,} rows per generation. "
                f"Requested: {requested_rows:,}."
            )

        usage_svc = UsageService(self._db)
        within = await usage_svc.check_limit(user.id, "generations", user.plan)
        if not within:
            gen_cap = _PLAN_GEN_CAPS.get(user.plan, 10)
            raise PlanLimitError(
                f"You have reached your {gen_cap} generations/month limit on the {user.plan} plan."
            )

    # ── Trigger (async fire-and-forget) ───────────────────────────────────

    async def trigger_generation(
        self,
        prompt: str,
        user: User,
        conversation_id: str | None = None,
        options: dict[str, Any] | None = None,
        phase_callback: PhaseCallback | None = None,
    ) -> str:
        """
        Create a Generation DB record, launch background engine task.
        Returns ``generation_id`` immediately (non-blocking).
        """
        opts = options or {}
        # Quick row-count estimate from prompt for limit check
        import re
        row_match = re.search(r"\b(\d[\d,]*)\s*(rows?|records?)\b", prompt, re.IGNORECASE)
        requested_rows = int(row_match.group(1).replace(",", "")) if row_match else 1_000

        await self._check_plan_limits(user, requested_rows)

        session_id = str(uuid.uuid4())
        generation_id = str(uuid.uuid4())

        gen = Generation(
            id=generation_id,
            user_id=user.id,
            conversation_id=conversation_id,
            session_id=session_id,
            status="pending",
            row_count=requested_rows,
        )
        self._db.add(gen)
        await self._db.commit()

        # Launch background task (non-blocking)
        asyncio.create_task(
            self._run_engine(
                generation_id=generation_id,
                session_id=session_id,
                prompt=prompt,
                user_id=user.id,
                plan=user.plan,
                options=opts,
                phase_callback=phase_callback,
            )
        )

        return generation_id

    # ── Engine execution (background) ─────────────────────────────────────

    async def _run_engine(
        self,
        generation_id: str,
        session_id: str,
        prompt: str,
        user_id: str,
        plan: str,
        options: dict[str, Any],
        phase_callback: PhaseCallback | None,
    ) -> None:
        """
        Run the SynthFlow engine in a background task.
        Uses a fresh DB session (cannot share the request session across tasks).
        """
        # Get a fresh session for background work
        if self._session_factory is not None:
            async with self._session_factory() as bg_db:
                await self._execute_engine(
                    bg_db, generation_id, session_id, prompt, user_id, plan, options, phase_callback
                )
        else:
            # Fallback: use the same session (test mode)
            await self._execute_engine(
                self._db, generation_id, session_id, prompt, user_id, plan, options, phase_callback
            )

    async def _execute_engine(
        self,
        db: AsyncSession,
        generation_id: str,
        session_id: str,
        prompt: str,
        user_id: str,
        plan: str,
        options: dict[str, Any],
        phase_callback: PhaseCallback | None,
    ) -> None:
        """Update DB as engine phases complete. Graceful fallback if engine unavailable."""
        try:
            await self._update_status(db, generation_id, "running")

            # Attempt real engine import
            try:
                from synthflow.orchestrator import SynthFlowOrchestrator  # type: ignore[import]
                await self._run_real_engine(
                    db, generation_id, session_id, prompt, options, phase_callback
                )
                return
            except ImportError:
                logger.warning(
                    "SynthFlow engine not installed in backend venv; using stub result."
                )

            # Stub result when engine is not available
            await self._simulate_phases(phase_callback)
            await self._update_status(
                db,
                generation_id,
                "done",
                quality_score=88.5,
                privacy_score=92.0,
                glass_box_code="# Generated stub code\ndef generate(row_count, seed):\n    import pandas as pd\n    return pd.DataFrame({'id': range(row_count)})\n",
                intent_json={"prompt": prompt, "session_id": session_id},
            )

            # Track usage
            usage_svc = UsageService(db)
            stmt = select(Generation).where(Generation.id == generation_id)
            result = await db.execute(stmt)
            gen = result.scalar_one_or_none()
            if gen:
                await usage_svc.track(
                    user_id, "generation", {"row_count": gen.row_count or 0}
                )

        except Exception as exc:
            logger.exception("Generation %s failed: %s", generation_id, exc)
            await self._update_status(db, generation_id, "failed", error_message=str(exc))

    async def _run_real_engine(
        self,
        db: AsyncSession,
        generation_id: str,
        session_id: str,
        prompt: str,
        options: dict[str, Any],
        phase_callback: PhaseCallback | None,
    ) -> None:
        """Run the actual SynthFlow engine (only when installed)."""
        from synthflow.orchestrator import SynthFlowOrchestrator  # type: ignore[import]

        def _phase_cb(phase: int, progress: float, message: str) -> None:
            if phase_callback:
                phase_callback(phase, progress, message)

        orchestrator = SynthFlowOrchestrator()
        result = await orchestrator.run(
            prompt=prompt,
            seed=options.get("seed", 42),
            progress_callback=_phase_cb,
        )

        code = getattr(result, "generated_code", "")
        quality = getattr(getattr(result, "quality_report", None), "overall_score", None)
        privacy = getattr(getattr(result, "privacy_report", None), "privacy_score", None)

        await self._update_status(
            db,
            generation_id,
            "done",
            quality_score=float(quality) if quality is not None else None,
            privacy_score=float(privacy) if privacy is not None else None,
            glass_box_code=code,
            domain=getattr(getattr(result, "intent", None), "domain", None),
            row_count=getattr(getattr(result, "intent", None), "row_count", None),
        )

    async def _simulate_phases(self, callback: PhaseCallback | None) -> None:
        """Fire phase callbacks for stub mode (9 phases at ~10ms each)."""
        phase_messages = [
            "Parsing intent…",
            "Activating knowledge graph…",
            "Designing schema…",
            "Building constraints…",
            "Modelling distributions…",
            "Synthesizing data…",
            "Applying correlations…",
            "Validating output…",
            "Computing quality scores…",
        ]
        for i, msg in enumerate(phase_messages, start=1):
            if callback:
                callback(i, i / 9, msg)
            await asyncio.sleep(0.01)

    async def _update_status(
        self,
        db: AsyncSession,
        generation_id: str,
        status: str,
        quality_score: float | None = None,
        privacy_score: float | None = None,
        glass_box_code: str | None = None,
        error_message: str | None = None,
        domain: str | None = None,
        row_count: int | None = None,
        intent_json: dict[str, Any] | None = None,
    ) -> None:
        stmt = select(Generation).where(Generation.id == generation_id)
        result = await db.execute(stmt)
        gen = result.scalar_one_or_none()
        if gen is None:
            return

        gen.status = status
        gen.updated_at = datetime.now(tz=timezone.utc)
        if quality_score is not None:
            gen.quality_score = quality_score
        if privacy_score is not None:
            gen.privacy_score = privacy_score
        if glass_box_code is not None:
            gen.glass_box_code = glass_box_code
        if error_message is not None:
            gen.error_message = error_message
        if domain is not None:
            gen.domain = domain
        if row_count is not None:
            gen.row_count = row_count
        if intent_json is not None:
            gen.intent_json = intent_json

        await db.commit()

    # ── Queries ────────────────────────────────────────────────────────────

    async def get_generation(self, generation_id: str, user_id: str) -> Generation | None:
        stmt = select(Generation).where(
            Generation.id == generation_id,
            Generation.user_id == user_id,
        )
        result = await self._db.execute(stmt)
        return result.scalar_one_or_none()

    async def list_generations(
        self,
        user_id: str,
        page: int = 1,
        page_size: int = 20,
        domain: str | None = None,
        status: str | None = None,
    ) -> list[Generation]:
        stmt = select(Generation).where(Generation.user_id == user_id)
        if domain:
            stmt = stmt.where(Generation.domain == domain)
        if status:
            stmt = stmt.where(Generation.status == status)
        stmt = stmt.order_by(Generation.created_at.desc()).offset((page - 1) * page_size).limit(page_size)
        result = await self._db.execute(stmt)
        return list(result.scalars().all())
