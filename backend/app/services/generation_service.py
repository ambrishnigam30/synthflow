# ───────────────────────────────────────────────────────────────
# Copyright (c) 2026 Ambrish Nigam
# Author : Ambrish Nigam | https://github.com/ambrishnigam30
# Project: SynthFlow — Autonomous Synthetic Data Orchestration Platform
# License : Apache License 2.0 | https://www.apache.org/licenses/LICENSE-2.0
# Module : GenerationService — wrap SynthFlow engine, persist results
# ───────────────────────────────────────────────────────────────

from __future__ import annotations

import asyncio
import json
import logging
import math
import re
import uuid
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.security import decrypt_api_key
from app.core.storage import SupabaseStorageClient
from app.models.generation import Generation
from app.models.llm_config import LLMConfig
from app.models.user import User
from app.services.usage_service import UsageService

logger = logging.getLogger(__name__)


def _make_json_safe(obj: Any) -> Any:
    """Convert pandas/numpy types to JSON-serializable Python types."""
    if obj is None:
        return None
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    if isinstance(obj, (pd.Timestamp, datetime)):
        return obj.isoformat()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    try:
        if pd.isna(obj):
            return None
    except (TypeError, ValueError):
        pass
    return obj


def _sanitize_for_json(data: Any) -> Any:
    """Final sweep: replace any remaining NaN/Infinity with None for valid JSON."""
    text = json.dumps(data, default=str)
    text = text.replace(": NaN", ": null").replace(":NaN", ":null")
    text = text.replace(": Infinity", ": null").replace(":Infinity", ":null")
    text = text.replace(": -Infinity", ": null").replace(":-Infinity", ":null")
    return json.loads(text)

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

# Callback types
PhaseCallback = Callable[[int, float, str], None]
DoneCallback = Callable[[str, dict[str, Any]], None]
ErrorCallback = Callable[[str, str, int, float], None]  # (generation_id, error_message, phase, progress)


class PlanLimitError(Exception):
    """Raised when the user has exceeded their plan limits."""


def _parse_row_count(prompt: str, default: int = 100) -> int:
    """
    Extract a row count from a natural-language prompt.

    Handles patterns like:
      "Generate 50 Indian healthcare patient records"
      "5,000 rows of banking data"
      "create 100 records"
    Returns ``default`` if no number is found near a row/record keyword.
    """
    # Find a number that appears before "rows" or "records" with up to 8 words in between
    match = re.search(
        r"\b(\d[\d,]*)\b(?:\s+\w+){0,8}\s+(?:rows?|records?)\b",
        prompt,
        re.IGNORECASE,
    )
    if match:
        return int(match.group(1).replace(",", ""))
    # Fallback: any standalone large number in the prompt (likely a row count)
    numbers = re.findall(r"\b(\d{2,}[\d,]*)\b", prompt)
    if numbers:
        return int(numbers[0].replace(",", ""))
    return default


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
        done_callback: DoneCallback | None = None,
        error_callback: ErrorCallback | None = None,
    ) -> str:
        """
        Create a Generation DB record, launch background engine task.
        Returns ``generation_id`` immediately (non-blocking).
        """
        opts = options or {}
        requested_rows = _parse_row_count(prompt)

        await self._check_plan_limits(user, requested_rows)

        generation_id = str(uuid.uuid4())
        session_id = str(uuid.uuid4())

        gen = Generation(
            id=generation_id,
            user_id=user.id,
            conversation_id=conversation_id,
            session_id=session_id,
            status="pending",
            row_count=requested_rows,
            intent_json={"prompt": prompt},
        )
        self._db.add(gen)
        await self._db.commit()

        # Fetch the user's active LLM config (provider + decrypted key)
        llm_provider: str | None = None
        llm_api_key: str | None = None
        llm_model: str | None = None
        try:
            cfg_stmt = (
                select(LLMConfig)
                .where(LLMConfig.user_id == user.id, LLMConfig.is_active == True)  # noqa: E712
                .order_by(LLMConfig.is_default.desc())
                .limit(1)
            )
            cfg_result = await self._db.execute(cfg_stmt)
            llm_cfg = cfg_result.scalar_one_or_none()
            if llm_cfg is not None:
                llm_provider = llm_cfg.provider
                llm_model = llm_cfg.model_name
                try:
                    llm_api_key = decrypt_api_key(llm_cfg.encrypted_api_key)
                except Exception as dec_exc:
                    logger.warning("Failed to decrypt LLM API key for user %s: %s", user.id, dec_exc)
        except Exception as cfg_exc:
            logger.warning("Could not load LLM config for user %s: %s", user.id, cfg_exc)

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
                done_callback=done_callback,
                error_callback=error_callback,
                llm_provider=llm_provider,
                llm_api_key=llm_api_key,
                llm_model=llm_model,
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
        done_callback: DoneCallback | None = None,
        error_callback: ErrorCallback | None = None,
        llm_provider: str | None = None,
        llm_api_key: str | None = None,
        llm_model: str | None = None,
    ) -> None:
        """Run the SynthFlow engine in a background task with a fresh DB session."""
        if self._session_factory is not None:
            async with self._session_factory() as bg_db:
                await self._execute_engine(
                    bg_db, generation_id, session_id, prompt, user_id, plan, options,
                    phase_callback, done_callback, error_callback,
                    llm_provider, llm_api_key, llm_model,
                )
        else:
            await self._execute_engine(
                self._db, generation_id, session_id, prompt, user_id, plan, options,
                phase_callback, done_callback, error_callback,
                llm_provider, llm_api_key, llm_model,
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
        done_callback: DoneCallback | None = None,
        error_callback: ErrorCallback | None = None,
        llm_provider: str | None = None,
        llm_api_key: str | None = None,
        llm_model: str | None = None,
    ) -> None:
        """Drive the real SynthFlow engine. Fails loudly if the engine can't run."""
        # Track the last phase/progress so error_callback can report the failed phase.
        # Defined outside the try block so the except clause can always reference them.
        last_phase: list[int] = [0]
        last_progress: list[float] = [0.0]

        try:
            await self._update_status(db, generation_id, "running")

            # Guard: no LLM key → fail early with a helpful message
            if not llm_provider or not llm_api_key:
                msg = (
                    "No LLM API key configured. "
                    "Please go to Settings → Providers and add your API key."
                )
                await self._update_status(db, generation_id, "failed", error_message=msg)
                if error_callback:
                    error_callback(generation_id, msg, 0, 0.0)
                return

            orig_phase_cb = phase_callback

            def tracking_phase_cb(phase: int, progress: float, message: str) -> None:
                last_phase[0] = phase
                last_progress[0] = progress
                if orig_phase_cb:
                    orig_phase_cb(phase, progress, message)

            await self._run_real_engine(
                db, generation_id, session_id, prompt, options,
                tracking_phase_cb, done_callback,
                llm_provider, llm_api_key, llm_model,
            )

            # Track usage on success
            usage_svc = UsageService(db)
            stmt = select(Generation).where(Generation.id == generation_id)
            result = await db.execute(stmt)
            gen = result.scalar_one_or_none()
            if gen:
                await usage_svc.track(user_id, "generation", {"row_count": gen.row_count or 0})

        except Exception as exc:
            logger.exception("Generation %s failed: %s", generation_id, exc)
            error_msg = str(exc)
            try:
                await self._update_status(db, generation_id, "failed", error_message=error_msg)
            except Exception as db_exc:
                logger.error("Could not update failed status for %s: %s", generation_id, db_exc)
            if error_callback:
                error_callback(generation_id, error_msg, last_phase[0], last_progress[0])

    async def _run_real_engine(
        self,
        db: AsyncSession,
        generation_id: str,
        session_id: str,
        prompt: str,
        options: dict[str, Any],
        phase_callback: PhaseCallback | None,
        done_callback: DoneCallback | None = None,
        llm_provider: str | None = None,
        llm_api_key: str | None = None,
        llm_model: str | None = None,
    ) -> None:
        """Run the SynthFlow engine pipeline and update the DB with the result."""
        from synthflow.core import SynthFlowContainer
        from synthflow.llm_client import LLMClient, LLMConfigError
        from synthflow.orchestrator import OrchestrationError, SynthFlowOrchestrator

        # Look up the persisted row_count from DB (already stored by trigger_generation)
        stmt = select(Generation).where(Generation.id == generation_id)
        result = await db.execute(stmt)
        gen = result.scalar_one_or_none()
        row_count: int | None = gen.row_count if gen else None
        seed: int = options.get("seed", 42)

        # Async progress_callback that bridges phase_callback (sync fire-and-forget)
        # to the engine's expected Callable[[int, float, str], Awaitable[None]]
        async def _engine_progress(phase: int, fraction: float, message: str) -> None:
            if phase_callback:
                try:
                    phase_callback(phase, fraction, message)
                except Exception as cb_exc:
                    logger.debug("Phase callback error (ignored): %s", cb_exc)

        # Wire up the engine with the user's LLM key
        llm_client = LLMClient(
            provider=llm_provider,  # type: ignore[arg-type]
            api_key=llm_api_key,    # type: ignore[arg-type]
            model=llm_model or None,
        )
        try:
            container = SynthFlowContainer(llm_client=llm_client)
            orchestrator = SynthFlowOrchestrator(container)

            logger.info(
                "Generation %s starting: provider=%s model=%s rows=%s prompt=%r",
                generation_id, llm_provider, llm_model, row_count, prompt[:80],
            )

            engine_result = await orchestrator.generate(
                prompt=prompt,
                row_count=row_count,
                seed=seed,
                scenario_text=options.get("scenario"),
                enable_dirty_data=options.get("enable_dirty_data", True),
                enable_sdv=options.get("enable_sdv", False),
                progress_callback=_engine_progress,
            )
        except LLMConfigError as exc:
            raise RuntimeError(f"LLM configuration error: {exc}") from exc
        except OrchestrationError as exc:
            raise RuntimeError(f"Engine pipeline failed: {exc}") from exc
        finally:
            # Always close the httpx client
            await llm_client._http.aclose()

        # ── Extract results from GenerationResult ────────────────────────
        df = engine_result.dataframe
        quality_report = engine_result.quality_report
        privacy_report = engine_result.privacy_report
        intent = engine_result.intent
        generated_code = engine_result.generated_code

        quality_score: float | None = None
        if quality_report is not None:
            quality_score = float(getattr(quality_report, "overall_score", 0.0) or 0.0)

        privacy_score: float | None = None
        if privacy_report is not None:
            privacy_score = float(getattr(privacy_report, "privacy_score", 0.0) or 0.0)

        domain: str | None = getattr(intent, "domain", None) if intent else None
        final_row_count: int = len(df) if df is not None else 0

        preview_rows: list[dict[str, Any]] = []
        col_count: int = 0
        if df is not None and hasattr(df, "head"):
            try:
                _preview_size = min(100, len(df))
                preview_df = df.head(_preview_size).copy()
                for col in preview_df.columns:
                    preview_df[col] = preview_df[col].apply(_make_json_safe)
                raw_rows = preview_df.to_dict(orient="records")
                # Final sweep: catch any NaN/Infinity that slipped through
                preview_rows = _sanitize_for_json(raw_rows)
                col_count = len(df.columns)
            except Exception as df_exc:
                logger.warning("Could not extract preview rows: %s", df_exc)

        schema_dict: dict[str, Any] = {}
        if engine_result.schema is not None:
            try:
                schema_dict = engine_result.schema.model_dump()
            except Exception:
                pass

        intent_dict: dict[str, Any] = {"prompt": prompt}
        if intent is not None:
            try:
                intent_dict = intent.model_dump()
                intent_dict["prompt"] = prompt
            except Exception:
                pass

        # Save DataFrame as CSV for fast, reliable downloads (avoids re-executing glass_box_code)
        csv_storage_path: str | None = None
        if df is not None and len(df) > 0:
            try:
                import os
                _csv_dir = "/tmp/synthflow_data"
                os.makedirs(_csv_dir, exist_ok=True)
                _csv_path = f"{_csv_dir}/{generation_id}.csv"
                df.to_csv(_csv_path, index=False)
                csv_storage_path = _csv_path
                logger.info("Generation %s: saved CSV to %s", generation_id, _csv_path)
            except Exception as _csv_exc:
                logger.warning("Generation %s: could not save CSV: %s", generation_id, _csv_exc)

        # Persist final result to DB
        await self._update_status(
            db,
            generation_id,
            "done",
            quality_score=quality_score,
            privacy_score=privacy_score,
            glass_box_code=generated_code,
            domain=domain,
            row_count=final_row_count,
            schema_json=schema_dict,
            intent_json=intent_dict,
            storage_path=csv_storage_path,
        )

        logger.info(
            "Generation %s complete: %d rows, %d cols, quality=%.1f",
            generation_id, final_row_count, col_count, quality_score or 0.0,
        )

        # Notify the WebSocket handler that generation is done
        if done_callback:
            done_callback(generation_id, {
                "quality_score": quality_score or 0.0,
                "preview_rows": preview_rows,
                "row_count": final_row_count,
                "col_count": col_count,
                "domain": domain or "",
                "schema": schema_dict,
                "download_urls": {
                    "csv": f"/api/generate/{generation_id}/download?fmt=csv",
                    "excel": f"/api/generate/{generation_id}/download?fmt=xlsx",
                    "json": f"/api/generate/{generation_id}/download?fmt=json",
                    "parquet": f"/api/generate/{generation_id}/download?fmt=parquet",
                },
            })

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
        schema_json: dict[str, Any] | None = None,
        intent_json: dict[str, Any] | None = None,
        storage_path: str | None = None,
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
        if schema_json is not None:
            gen.schema_json = schema_json
        if intent_json is not None:
            gen.intent_json = intent_json
        if storage_path is not None:
            gen.storage_path = storage_path

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
