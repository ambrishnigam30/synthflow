# ───────────────────────────────────────────────────────────────
# Copyright (c) 2026 Ambrish Nigam
# Author : Ambrish Nigam | https://github.com/ambrishnigam30
# Project: SynthFlow — Autonomous Synthetic Data Orchestration Platform
# License : Apache License 2.0 | https://www.apache.org/licenses/LICENSE-2.0
# Module : Public Developer API — /api/v1/* with X-API-Key authentication
# ───────────────────────────────────────────────────────────────

from __future__ import annotations

import hashlib
import logging

from fastapi import APIRouter, Depends, Header, HTTPException, Query, status
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.models.generation import Generation
from app.models.user import User
from app.models.webhook import APIKey
from app.schemas.generation import GenerateRequest
from app.services.generation_service import GenerationService, PlanLimitError

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1", tags=["public-api"])

# Plans that allow API access
_API_PLANS = {"pro", "business", "enterprise"}

# Rate limits (requests per hour per plan) — enforced via response headers only;
# actual distributed limiting requires Redis (out of scope here).
_RATE_LIMITS: dict[str, int] = {
    "pro": 100,
    "business": 1000,
    "enterprise": 0,  # unlimited
}


def _ok(data: object) -> dict:
    return {"data": data, "error": None}


# ── API Key dependency ─────────────────────────────────────────────────────────

async def get_api_key_user(
    x_api_key: str = Header(..., alias="X-API-Key"),
    db: AsyncSession = Depends(get_db),
) -> User:
    """
    Resolve an X-API-Key header to an active User.
    Keys are stored as SHA-256 hashes; we never store the plaintext.
    """
    key_hash = hashlib.sha256(x_api_key.encode()).hexdigest()

    stmt = select(APIKey).where(
        APIKey.key_hash == key_hash,
        APIKey.is_active.is_(True),
    )
    result = await db.execute(stmt)
    api_key: APIKey | None = result.scalar_one_or_none()

    if api_key is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"message": "Invalid or revoked API key.", "code": "INVALID_API_KEY"},
        )

    # Update last-used timestamp (fire-and-forget; don't block on failure)
    from datetime import datetime, timezone
    api_key.last_used_at = datetime.now(tz=timezone.utc)
    try:
        await db.commit()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not update last_used_at for API key %s: %s", api_key.id, exc)
        await db.rollback()

    # Fetch owning user
    user_stmt = select(User).where(User.id == api_key.user_id, User.is_active.is_(True))
    user_result = await db.execute(user_stmt)
    user: User | None = user_result.scalar_one_or_none()

    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"message": "User not found or inactive.", "code": "USER_NOT_FOUND"},
        )

    if user.plan not in _API_PLANS:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={
                "message": "API access requires Pro plan or above.",
                "code": "PLAN_REQUIRED",
            },
        )

    return user


# ── Endpoints ──────────────────────────────────────────────────────────────────

class V1GenerateRequest(BaseModel):
    prompt: str
    conversation_id: str | None = None
    options: dict | None = None


@router.post("/generate", status_code=status.HTTP_202_ACCEPTED)
async def v1_trigger_generation(
    body: V1GenerateRequest,
    user: User = Depends(get_api_key_user),
    db: AsyncSession = Depends(get_db),
) -> dict:
    """POST /api/v1/generate — trigger a generation via API key auth."""
    svc = GenerationService(db)
    try:
        generation_id = await svc.trigger_generation(
            prompt=body.prompt,
            user=user,
            conversation_id=body.conversation_id,
            options=body.options,
        )
    except PlanLimitError as exc:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail={"message": str(exc), "code": "PLAN_LIMIT_EXCEEDED"},
        ) from exc

    return _ok(
        {
            "generation_id": generation_id,
            "status": "pending",
            "poll_url": f"/api/v1/generate/{generation_id}",
        }
    )


@router.get("/generate/{generation_id}")
async def v1_get_generation(
    generation_id: str,
    user: User = Depends(get_api_key_user),
    db: AsyncSession = Depends(get_db),
) -> dict:
    """GET /api/v1/generate/:id — poll generation status."""
    svc = GenerationService(db)
    gen = await svc.get_generation(generation_id, user.id)
    if gen is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"message": "Generation not found.", "code": "NOT_FOUND"},
        )

    return _ok(
        {
            "generation_id": gen.id,
            "status": gen.status,
            "domain": gen.domain,
            "row_count": gen.row_count,
            "quality_score": gen.quality_score,
            "created_at": gen.created_at.isoformat() if gen.created_at else None,
            "updated_at": gen.updated_at.isoformat() if gen.updated_at else None,
        }
    )


@router.get("/generate/{generation_id}/download")
async def v1_download_generation(
    generation_id: str,
    fmt: str = Query("csv", pattern="^(csv|parquet|json|xlsx)$"),
    user: User = Depends(get_api_key_user),
    db: AsyncSession = Depends(get_db),
) -> RedirectResponse:
    """GET /api/v1/generate/:id/download — download completed generation."""
    svc = GenerationService(db)
    gen = await svc.get_generation(generation_id, user.id)
    if gen is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"message": "Generation not found.", "code": "NOT_FOUND"},
        )
    if gen.status != "done":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"message": "Generation not complete yet.", "code": "NOT_READY"},
        )

    signed_url = gen.storage_path or f"/static/generations/{generation_id}.{fmt}"
    return RedirectResponse(url=signed_url, status_code=302)
