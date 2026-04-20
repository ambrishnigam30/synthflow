# ───────────────────────────────────────────────────────────────
# Copyright (c) 2026 Ambrish Nigam
# Author : Ambrish Nigam | https://github.com/ambrishnigam30
# Project: SynthFlow — Autonomous Synthetic Data Orchestration Platform
# License : Apache License 2.0 | https://www.apache.org/licenses/LICENSE-2.0
# Module : API Keys REST API — create, list, revoke
# ───────────────────────────────────────────────────────────────

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.core.security import generate_api_key
from app.dependencies import get_current_user
from app.models.user import User
from app.models.webhook import APIKey

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/keys", tags=["api-keys"])

_PLAN_ALLOWED = {"pro", "business", "enterprise"}


def _ok(data: object) -> dict:
    return {"data": data, "error": None}


def _require_pro(user: User) -> None:
    if user.plan not in _PLAN_ALLOWED:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={
                "message": "API key access requires Pro plan or above.",
                "code": "PLAN_REQUIRED",
            },
        )


class CreateKeyRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    scopes: list[str] = Field(default_factory=lambda: ["generate:read", "generate:write"])


def _key_to_dict(key: APIKey, full_key: str | None = None) -> dict[str, Any]:
    return {
        "id": key.id,
        "name": key.name,
        "key_prefix": key.key_prefix,
        "full_key": full_key,  # only on creation; None otherwise
        "scopes": key.scopes,
        "is_active": key.is_active,
        "last_used_at": key.last_used_at.isoformat() if key.last_used_at else None,
        "created_at": key.created_at.isoformat(),
    }


# ── Endpoints ──────────────────────────────────────────────────────────────────

@router.post("", status_code=status.HTTP_201_CREATED)
async def create_api_key(
    body: CreateKeyRequest,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> dict:
    """POST /api/keys — generate a new API key (full key returned ONCE)."""
    _require_pro(user)

    full_key, key_hash, key_prefix = generate_api_key()

    api_key = APIKey(
        id=str(uuid.uuid4()),
        user_id=user.id,
        name=body.name,
        key_prefix=key_prefix,
        key_hash=key_hash,
        scopes=body.scopes,
        is_active=True,
    )
    db.add(api_key)
    await db.commit()
    await db.refresh(api_key)

    return _ok(_key_to_dict(api_key, full_key=full_key))


@router.get("")
async def list_api_keys(
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> dict:
    """GET /api/keys — list API keys; only prefix is returned, never full key."""
    _require_pro(user)

    stmt = select(APIKey).where(
        APIKey.user_id == user.id, APIKey.is_active.is_(True)
    ).order_by(APIKey.created_at.desc())
    result = await db.execute(stmt)
    keys = result.scalars().all()

    return _ok({"keys": [_key_to_dict(k) for k in keys]})


@router.delete("/{key_id}", status_code=status.HTTP_204_NO_CONTENT)
async def revoke_api_key(
    key_id: str,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> None:
    """DELETE /api/keys/:id — revoke an API key (sets is_active=False)."""
    _require_pro(user)

    stmt = select(APIKey).where(APIKey.id == key_id, APIKey.user_id == user.id)
    result = await db.execute(stmt)
    key = result.scalar_one_or_none()
    if key is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"message": "API key not found.", "code": "NOT_FOUND"},
        )

    key.is_active = False
    await db.commit()
