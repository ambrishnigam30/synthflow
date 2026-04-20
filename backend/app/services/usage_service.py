# ───────────────────────────────────────────────────────────────
# Copyright (c) 2026 Ambrish Nigam
# Author : Ambrish Nigam | https://github.com/ambrishnigam30
# Project: SynthFlow — Autonomous Synthetic Data Orchestration Platform
# License : Apache License 2.0 | https://www.apache.org/licenses/LICENSE-2.0
# Module : UsageService — track and enforce per-user plan limits
# ───────────────────────────────────────────────────────────────

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.billing import UsageMonthly, UsageRecord

logger = logging.getLogger(__name__)

# Plan caps: (max_generations/month, max_rows/generation, max_uploads/month)
# 0 = unlimited
_PLAN_CAPS: dict[str, dict[str, int]] = {
    "free":       {"generations": 10,   "rows": 1_000,     "uploads": 1},
    "starter":    {"generations": 100,  "rows": 10_000,    "uploads": 5},
    "pro":        {"generations": 200,  "rows": 100_000,   "uploads": 10},
    "business":   {"generations": 0,    "rows": 1_000_000, "uploads": 50},
    "enterprise": {"generations": 0,    "rows": 0,         "uploads": 0},
}


def _year_month() -> str:
    """Return current year-month string, e.g. '2026-04'."""
    return datetime.now(tz=timezone.utc).strftime("%Y-%m")


class UsageService:
    def __init__(self, db: AsyncSession) -> None:
        self._db = db

    async def track(
        self,
        user_id: str,
        action: str,
        metadata: dict[str, Any] | None = None,
        quantity: int = 1,
    ) -> None:
        """Record a usage event and increment monthly counters."""
        record = UsageRecord(
            id=str(uuid.uuid4()),
            user_id=user_id,
            action=action,
            quantity=quantity,
            metadata_json=metadata or {},
        )
        self._db.add(record)

        ym = _year_month()
        stmt = select(UsageMonthly).where(
            UsageMonthly.user_id == user_id,
            UsageMonthly.year_month == ym,
        )
        result = await self._db.execute(stmt)
        monthly = result.scalar_one_or_none()

        if monthly is None:
            monthly = UsageMonthly(
                id=str(uuid.uuid4()),
                user_id=user_id,
                year_month=ym,
            )
            self._db.add(monthly)

        if action == "generation":
            monthly.generations_count = (monthly.generations_count or 0) + quantity
            rows = (metadata or {}).get("row_count", 0)
            monthly.rows_generated = (monthly.rows_generated or 0) + int(rows)
        elif action == "dataset_upload":
            monthly.datasets_uploaded = (monthly.datasets_uploaded or 0) + quantity
        elif action == "api_call":
            monthly.api_calls = (monthly.api_calls or 0) + quantity

        monthly.updated_at = datetime.now(tz=timezone.utc)
        await self._db.commit()

    async def get_current_usage(self, user_id: str) -> dict[str, Any]:
        """Return usage summary for the current billing month."""
        ym = _year_month()
        stmt = select(UsageMonthly).where(
            UsageMonthly.user_id == user_id,
            UsageMonthly.year_month == ym,
        )
        result = await self._db.execute(stmt)
        monthly = result.scalar_one_or_none()

        if monthly is None:
            return {
                "year_month": ym,
                "generations_count": 0,
                "rows_generated": 0,
                "datasets_uploaded": 0,
                "api_calls": 0,
            }
        return {
            "year_month": monthly.year_month,
            "generations_count": monthly.generations_count or 0,
            "rows_generated": monthly.rows_generated or 0,
            "datasets_uploaded": monthly.datasets_uploaded or 0,
            "api_calls": monthly.api_calls or 0,
        }

    async def check_limit(self, user_id: str, action: str, plan: str, value: int = 1) -> bool:
        """
        Return True if the user is within their plan limit for *action*.
        False means they have exceeded it.
        """
        caps = _PLAN_CAPS.get(plan, _PLAN_CAPS["free"])
        cap = caps.get(action, 0)
        if cap == 0:
            return True  # unlimited

        usage = await self.get_current_usage(user_id)

        if action == "generations":
            return (usage["generations_count"] + value) <= cap
        if action == "rows":
            return value <= cap
        if action == "uploads":
            return (usage["datasets_uploaded"] + value) <= cap
        return True

    async def get_monthly_usage(self, user_id: str, months: int = 6) -> list[dict[str, Any]]:
        """Return usage history for the last *months* billing periods."""
        stmt = (
            select(UsageMonthly)
            .where(UsageMonthly.user_id == user_id)
            .order_by(UsageMonthly.year_month.desc())
            .limit(months)
        )
        result = await self._db.execute(stmt)
        rows = result.scalars().all()
        return [
            {
                "year_month": r.year_month,
                "generations_count": r.generations_count or 0,
                "rows_generated": r.rows_generated or 0,
                "datasets_uploaded": r.datasets_uploaded or 0,
                "api_calls": r.api_calls or 0,
            }
            for r in rows
        ]
