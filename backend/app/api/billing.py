# ───────────────────────────────────────────────────────────────
# Copyright (c) 2026 Ambrish Nigam
# Author : Ambrish Nigam | https://github.com/ambrishnigam30
# Project: SynthFlow — Autonomous Synthetic Data Orchestration Platform
# License : Apache License 2.0 | https://www.apache.org/licenses/LICENSE-2.0
# Module : Billing REST API — plans, subscriptions, usage, webhooks
# ───────────────────────────────────────────────────────────────

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, Header, HTTPException, Request, status
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.dependencies import get_current_user
from app.models.user import User
from app.services.billing_service import BillingError, BillingService, WebhookVerificationError
from app.services.usage_service import UsageService

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/billing", tags=["billing"])


def _ok(data: object) -> dict:
    return {"data": data, "error": None}


# ── Request schemas ────────────────────────────────────────────────────────────

class SubscribeRequest(BaseModel):
    plan: str = Field(..., pattern="^(free|pro|business|enterprise)$")
    payment_provider: str = Field("manual", pattern="^(manual|stripe|razorpay)$")
    external_subscription_id: str | None = None


# ── Endpoints ──────────────────────────────────────────────────────────────────

@router.get("/plans")
async def list_plans() -> dict:
    """GET /api/billing/plans — public, no auth required."""
    return _ok({"plans": BillingService.get_plans()})


@router.post("/subscribe", status_code=status.HTTP_201_CREATED)
async def subscribe(
    body: SubscribeRequest,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> dict:
    """POST /api/billing/subscribe — create or upgrade subscription."""
    svc = BillingService(db)
    try:
        sub = await svc.create_subscription(
            user=user,
            plan=body.plan,
            payment_provider=body.payment_provider,
            external_subscription_id=body.external_subscription_id,
        )
    except BillingError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"message": str(exc), "code": "BILLING_ERROR"},
        ) from exc

    return _ok(
        {
            "subscription_id": sub.id,
            "plan": sub.plan,
            "status": sub.status,
            "current_period_start": sub.current_period_start.isoformat()
            if sub.current_period_start
            else None,
            "current_period_end": sub.current_period_end.isoformat()
            if sub.current_period_end
            else None,
        }
    )


@router.post("/cancel")
async def cancel_subscription(
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> dict:
    """POST /api/billing/cancel — cancel subscription at period end."""
    svc = BillingService(db)
    try:
        sub = await svc.cancel_subscription(user)
    except BillingError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"message": str(exc), "code": "BILLING_ERROR"},
        ) from exc

    return _ok(
        {
            "subscription_id": sub.id,
            "plan": sub.plan,
            "status": sub.status,
        }
    )


@router.get("/usage")
async def get_usage(
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> dict:
    """GET /api/billing/usage — current period usage summary."""
    usage_svc = UsageService(db)
    usage = await usage_svc.get_current_usage(user.id)
    billing_svc = BillingService(db)
    sub = await billing_svc.get_subscription(user.id)

    # Get limits for current plan
    from app.services.billing_service import PLANS

    plan_def = next((p for p in PLANS if p["id"] == user.plan), PLANS[0])
    features = plan_def["features"]

    return _ok(
        {
            "plan": user.plan,
            "subscription_status": sub.status if sub else "none",
            "usage": usage,
            "limits": {
                "generations_per_month": features["generations_per_month"],
                "max_rows_per_generation": features["max_rows_per_generation"],
            },
        }
    )


@router.get("/invoices")
async def get_invoices(
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> dict:
    """GET /api/billing/invoices — list past invoices."""
    svc = BillingService(db)
    invoices = await svc.get_invoices(user.id)
    return _ok({"invoices": invoices})


@router.post("/webhook/stripe", include_in_schema=False)
async def stripe_webhook(
    request: Request,
    db: AsyncSession = Depends(get_db),
    stripe_signature: str = Header("", alias="stripe-signature"),
) -> dict:
    """POST /api/billing/webhook/stripe — Stripe webhook receiver (no auth)."""
    body = await request.body()
    svc = BillingService(db)
    try:
        result = await svc.handle_stripe_webhook(body, stripe_signature)
    except WebhookVerificationError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"message": str(exc), "code": "WEBHOOK_VERIFICATION_FAILED"},
        ) from exc
    return result


@router.post("/webhook/razorpay", include_in_schema=False)
async def razorpay_webhook(
    request: Request,
    db: AsyncSession = Depends(get_db),
    x_razorpay_signature: str = Header("", alias="x-razorpay-signature"),
) -> dict:
    """POST /api/billing/webhook/razorpay — Razorpay webhook receiver (no auth)."""
    body = await request.body()
    svc = BillingService(db)
    try:
        result = await svc.handle_razorpay_webhook(body, x_razorpay_signature)
    except WebhookVerificationError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"message": str(exc), "code": "WEBHOOK_VERIFICATION_FAILED"},
        ) from exc
    return result
