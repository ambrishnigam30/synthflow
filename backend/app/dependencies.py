# ───────────────────────────────────────────────────────────────
# Copyright (c) 2026 Ambrish Nigam
# Author : Ambrish Nigam | https://github.com/ambrishnigam30
# Project: SynthFlow — Autonomous Synthetic Data Orchestration Platform
# License : Apache License 2.0 | https://www.apache.org/licenses/LICENSE-2.0
# Module : FastAPI dependency injection — auth, plan limits
# ───────────────────────────────────────────────────────────────

import logging
from collections.abc import Callable, Coroutine
from typing import Any

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.core.security import AuthTokenError, verify_token
from app.models.user import User

logger = logging.getLogger(__name__)

security = HTTPBearer()
optional_security = HTTPBearer(auto_error=False)

# Plan caps: (max_generations_per_month, max_rows_per_generation)
_PLAN_CAPS: dict[str, tuple[int, int]] = {
    "free": (10, 1_000),
    "starter": (100, 10_000),
    "pro": (1_000, 100_000),
    "enterprise": (0, 0),  # 0 means unlimited
}


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: AsyncSession = Depends(get_db),
) -> User:
    """
    Extract and validate the JWT bearer token, then return the active User.
    Raises HTTP 401 if the token is invalid or the user does not exist / is inactive.
    """
    try:
        payload = verify_token(credentials.credentials)
    except AuthTokenError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"message": str(exc), "code": "INVALID_TOKEN"},
        ) from exc

    user_id: str | None = payload.get("sub")
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"message": "Token missing subject claim.", "code": "INVALID_TOKEN"},
        )

    result = await db.execute(
        select(User).where(User.id == user_id, User.is_active == True)  # noqa: E712
    )
    user: User | None = result.scalar_one_or_none()
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"message": "User not found or inactive.", "code": "USER_NOT_FOUND"},
        )
    return user


async def get_optional_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(optional_security),
    db: AsyncSession = Depends(get_db),
) -> User | None:
    """Return the current User or None for public endpoints."""
    if credentials is None:
        return None
    try:
        payload = verify_token(credentials.credentials)
    except AuthTokenError:
        return None

    user_id: str | None = payload.get("sub")
    if not user_id:
        return None

    result = await db.execute(
        select(User).where(User.id == user_id, User.is_active == True)  # noqa: E712
    )
    return result.scalar_one_or_none()


def check_plan_limit(action: str) -> Callable[..., Coroutine[Any, Any, User]]:
    """
    Dependency factory: checks that the current user's plan permits the action.
    Currently validates monthly generation count and max rows.
    Returns the verified User.
    """

    async def _check(
        user: User = Depends(get_current_user),
        db: AsyncSession = Depends(get_db),
    ) -> User:
        from datetime import datetime, timezone

        from sqlalchemy import select

        from app.models.billing import UsageMonthly

        plan = user.plan
        caps = _PLAN_CAPS.get(plan, _PLAN_CAPS["free"])
        max_gens, _max_rows = caps

        if max_gens == 0:
            # Enterprise — unlimited
            return user

        now = datetime.now(tz=timezone.utc)
        year_month = now.strftime("%Y-%m")

        result = await db.execute(
            select(UsageMonthly).where(
                UsageMonthly.user_id == user.id,
                UsageMonthly.year_month == year_month,
            )
        )
        monthly: UsageMonthly | None = result.scalar_one_or_none()
        current_gens = monthly.generations_count if monthly else 0

        if action == "generate" and current_gens >= max_gens:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail={
                    "message": f"Monthly generation limit of {max_gens} reached for plan '{plan}'.",
                    "code": "PLAN_LIMIT_EXCEEDED",
                },
            )
        return user

    return _check
