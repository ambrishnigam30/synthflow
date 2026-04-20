# ───────────────────────────────────────────────────────────────
# Copyright (c) 2026 Ambrish Nigam
# Author : Ambrish Nigam | https://github.com/ambrishnigam30
# Project: SynthFlow — Autonomous Synthetic Data Orchestration Platform
# License : Apache License 2.0 | https://www.apache.org/licenses/LICENSE-2.0
# Module : Webhooks REST API — CRUD, deliveries, test (business plan+)
# ───────────────────────────────────────────────────────────────

from __future__ import annotations

import logging
import secrets
import uuid
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.dependencies import get_current_user
from app.models.user import User
from app.models.webhook import Webhook, WebhookDelivery
from app.services.webhook_service import WebhookService

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/webhooks", tags=["webhooks"])

_PLAN_ALLOWED = {"business", "enterprise"}


def _ok(data: object) -> dict:
    return {"data": data, "error": None}


def _require_business(user: User) -> None:
    if user.plan not in _PLAN_ALLOWED:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={
                "message": "Webhooks require Business plan or above.",
                "code": "PLAN_REQUIRED",
            },
        )


class CreateWebhookRequest(BaseModel):
    url: str
    events: list[str] = ["*"]
    is_active: bool = True


class UpdateWebhookRequest(BaseModel):
    url: str | None = None
    events: list[str] | None = None
    is_active: bool | None = None


def _webhook_dict(wh: Webhook) -> dict[str, Any]:
    return {
        "id": wh.id,
        "url": wh.url,
        "events": wh.events,
        "is_active": wh.is_active,
        "created_at": wh.created_at.isoformat(),
        "updated_at": wh.updated_at.isoformat(),
    }


def _delivery_dict(d: WebhookDelivery) -> dict[str, Any]:
    return {
        "id": d.id,
        "webhook_id": d.webhook_id,
        "event_type": d.event_type,
        "payload": d.payload,
        "status_code": d.status_code,
        "response_body": d.response_body,
        "success": d.success,
        "created_at": d.created_at.isoformat(),
    }


# ── Endpoints ──────────────────────────────────────────────────────────────────

@router.post("", status_code=status.HTTP_201_CREATED)
async def create_webhook(
    body: CreateWebhookRequest,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> dict:
    """POST /api/webhooks — register a new webhook endpoint."""
    _require_business(user)

    webhook = Webhook(
        id=str(uuid.uuid4()),
        user_id=user.id,
        url=body.url,
        events=body.events,
        is_active=body.is_active,
        secret=secrets.token_hex(32),
    )
    db.add(webhook)
    await db.commit()
    await db.refresh(webhook)

    data = _webhook_dict(webhook)
    data["secret"] = webhook.secret  # returned only on creation
    return _ok(data)


@router.get("")
async def list_webhooks(
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> dict:
    """GET /api/webhooks — list all webhooks for the current user."""
    _require_business(user)

    stmt = select(Webhook).where(Webhook.user_id == user.id).order_by(Webhook.created_at.desc())
    result = await db.execute(stmt)
    webhooks = result.scalars().all()
    return _ok({"webhooks": [_webhook_dict(wh) for wh in webhooks]})


@router.patch("/{webhook_id}")
async def update_webhook(
    webhook_id: str,
    body: UpdateWebhookRequest,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> dict:
    """PATCH /api/webhooks/:id — update a webhook."""
    _require_business(user)

    stmt = select(Webhook).where(Webhook.id == webhook_id, Webhook.user_id == user.id)
    result = await db.execute(stmt)
    webhook = result.scalar_one_or_none()
    if webhook is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"message": "Webhook not found.", "code": "NOT_FOUND"},
        )

    if body.url is not None:
        webhook.url = body.url
    if body.events is not None:
        webhook.events = body.events
    if body.is_active is not None:
        webhook.is_active = body.is_active

    await db.commit()
    await db.refresh(webhook)
    return _ok(_webhook_dict(webhook))


@router.delete("/{webhook_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_webhook(
    webhook_id: str,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> None:
    """DELETE /api/webhooks/:id — delete a webhook."""
    _require_business(user)

    stmt = select(Webhook).where(Webhook.id == webhook_id, Webhook.user_id == user.id)
    result = await db.execute(stmt)
    webhook = result.scalar_one_or_none()
    if webhook is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"message": "Webhook not found.", "code": "NOT_FOUND"},
        )

    await db.delete(webhook)
    await db.commit()


@router.get("/{webhook_id}/deliveries")
async def list_deliveries(
    webhook_id: str,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> dict:
    """GET /api/webhooks/:id/deliveries — list recent delivery attempts."""
    _require_business(user)

    wh_stmt = select(Webhook).where(Webhook.id == webhook_id, Webhook.user_id == user.id)
    wh_result = await db.execute(wh_stmt)
    if wh_result.scalar_one_or_none() is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"message": "Webhook not found.", "code": "NOT_FOUND"},
        )

    stmt = (
        select(WebhookDelivery)
        .where(WebhookDelivery.webhook_id == webhook_id)
        .order_by(WebhookDelivery.created_at.desc())
        .limit(50)
    )
    result = await db.execute(stmt)
    deliveries = result.scalars().all()
    return _ok({"deliveries": [_delivery_dict(d) for d in deliveries]})


@router.post("/{webhook_id}/test", status_code=status.HTTP_202_ACCEPTED)
async def test_webhook(
    webhook_id: str,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> dict:
    """POST /api/webhooks/:id/test — fire a test event to the webhook URL."""
    _require_business(user)

    wh_stmt = select(Webhook).where(Webhook.id == webhook_id, Webhook.user_id == user.id)
    wh_result = await db.execute(wh_stmt)
    webhook = wh_result.scalar_one_or_none()
    if webhook is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"message": "Webhook not found.", "code": "NOT_FOUND"},
        )

    svc = WebhookService(db)
    results = await svc.deliver(
        user_id=user.id,
        event_type="webhook.test",
        payload={"message": "This is a test event from SynthFlow.", "webhook_id": webhook_id},
    )
    return _ok({"dispatched": True, "results": results})
