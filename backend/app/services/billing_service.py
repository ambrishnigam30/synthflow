# ───────────────────────────────────────────────────────────────
# Copyright (c) 2026 Ambrish Nigam
# Author : Ambrish Nigam | https://github.com/ambrishnigam30
# Project: SynthFlow — Autonomous Synthetic Data Orchestration Platform
# License : Apache License 2.0 | https://www.apache.org/licenses/LICENSE-2.0
# Module : BillingService — plan management, subscriptions, payment webhooks
# ───────────────────────────────────────────────────────────────

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.models.billing import Subscription
from app.models.user import User

logger = logging.getLogger(__name__)

# ── Plan catalogue ─────────────────────────────────────────────────────────────

PLANS: list[dict[str, Any]] = [
    {
        "id": "free",
        "name": "Free",
        "price_usd": 0,
        "price_inr": 0,
        "billing_period": "month",
        "features": {
            "generations_per_month": 10,
            "max_rows_per_generation": 1_000,
            "datasets_upload": 1,
            "api_access": False,
            "team_members": 0,
            "webhooks": False,
            "priority_support": False,
        },
    },
    {
        "id": "pro",
        "name": "Pro",
        "price_usd": 19,
        "price_inr": 499,
        "billing_period": "month",
        "features": {
            "generations_per_month": 200,
            "max_rows_per_generation": 100_000,
            "datasets_upload": 10,
            "api_access": True,
            "team_members": 0,
            "webhooks": False,
            "priority_support": False,
        },
    },
    {
        "id": "business",
        "name": "Business",
        "price_usd": 49,
        "price_inr": 1_999,
        "billing_period": "month",
        "features": {
            "generations_per_month": 0,  # unlimited
            "max_rows_per_generation": 1_000_000,
            "datasets_upload": 50,
            "api_access": True,
            "team_members": 10,
            "webhooks": True,
            "priority_support": True,
        },
    },
    {
        "id": "enterprise",
        "name": "Enterprise",
        "price_usd": None,  # custom
        "price_inr": None,
        "billing_period": "custom",
        "features": {
            "generations_per_month": 0,
            "max_rows_per_generation": 0,
            "datasets_upload": 0,
            "api_access": True,
            "team_members": 0,
            "webhooks": True,
            "priority_support": True,
        },
    },
]

_PLAN_IDS = {p["id"] for p in PLANS}


class BillingError(Exception):
    """Raised when a billing operation fails."""


class WebhookVerificationError(Exception):
    """Raised when payment webhook signature verification fails."""


class BillingService:
    def __init__(self, db: AsyncSession) -> None:
        self._db = db

    # ── Plans ──────────────────────────────────────────────────────────────────

    @staticmethod
    def get_plans() -> list[dict[str, Any]]:
        """Return all available plan definitions."""
        return PLANS

    # ── Subscription management ────────────────────────────────────────────────

    async def get_subscription(self, user_id: str) -> Subscription | None:
        """Return the user's current subscription, or None."""
        stmt = select(Subscription).where(Subscription.user_id == user_id)
        result = await self._db.execute(stmt)
        return result.scalar_one_or_none()

    async def create_subscription(
        self,
        user: User,
        plan: str,
        payment_provider: str = "manual",
        external_subscription_id: str | None = None,
    ) -> Subscription:
        """
        Create or replace the user's subscription.
        In test / manual mode (no payment provider configured), activates immediately.
        """
        if plan not in _PLAN_IDS:
            raise BillingError(f"Unknown plan: {plan!r}. Valid plans: {sorted(_PLAN_IDS)}")

        # Cancel existing subscription first
        existing = await self.get_subscription(user.id)
        if existing is not None:
            existing.status = "cancelled"
            existing.updated_at = datetime.now(tz=timezone.utc)

        now = datetime.now(tz=timezone.utc)
        period_end = now + timedelta(days=30)

        sub = Subscription(
            id=str(uuid.uuid4()),
            user_id=user.id,
            plan=plan,
            status="active",
            current_period_start=now,
            current_period_end=period_end,
        )
        if payment_provider == "stripe" and external_subscription_id:
            sub.stripe_subscription_id = external_subscription_id
        elif payment_provider == "razorpay" and external_subscription_id:
            sub.razorpay_subscription_id = external_subscription_id

        self._db.add(sub)

        # Update user plan
        user.plan = plan
        await self._db.commit()
        await self._db.refresh(sub)
        return sub

    async def cancel_subscription(self, user: User) -> Subscription:
        """
        Cancel the user's subscription at period end.
        Sets status = 'cancel_at_period_end'; plan stays active until period expires.
        """
        sub = await self.get_subscription(user.id)
        if sub is None:
            raise BillingError("No active subscription found.")
        if sub.status in {"cancelled", "cancel_at_period_end"}:
            raise BillingError("Subscription is already cancelled.")

        sub.status = "cancel_at_period_end"
        sub.updated_at = datetime.now(tz=timezone.utc)
        await self._db.commit()
        await self._db.refresh(sub)
        return sub

    async def get_invoices(self, user_id: str) -> list[dict[str, Any]]:
        """
        Return a list of past subscription periods as invoice proxies.
        In production this would fetch from Stripe/Razorpay.
        """
        sub = await self.get_subscription(user_id)
        if sub is None:
            return []

        invoices: list[dict[str, Any]] = []
        if sub.status in {"active", "cancel_at_period_end", "cancelled"}:
            invoices.append(
                {
                    "id": sub.id,
                    "plan": sub.plan,
                    "status": sub.status,
                    "period_start": sub.current_period_start.isoformat()
                    if sub.current_period_start
                    else None,
                    "period_end": sub.current_period_end.isoformat()
                    if sub.current_period_end
                    else None,
                    "amount_usd": next(
                        (p["price_usd"] for p in PLANS if p["id"] == sub.plan), 0
                    ),
                    "payment_provider": (
                        "stripe"
                        if sub.stripe_subscription_id
                        else "razorpay"
                        if sub.razorpay_subscription_id
                        else "manual"
                    ),
                }
            )
        return invoices

    # ── Razorpay webhook ───────────────────────────────────────────────────────

    async def handle_razorpay_webhook(
        self, body: bytes, signature_header: str
    ) -> dict[str, Any]:
        """
        Verify Razorpay webhook signature and process payment events.
        Signature format: HMAC-SHA256 of body using razorpay_key_secret.
        """
        if not settings.razorpay_key_secret:
            logger.warning("Razorpay not configured — webhook ignored.")
            return {"status": "ignored", "reason": "not_configured"}

        expected = hmac.new(
            settings.razorpay_key_secret.encode(),
            body,
            hashlib.sha256,
        ).hexdigest()
        if not hmac.compare_digest(expected, signature_header):
            raise WebhookVerificationError("Razorpay signature mismatch.")

        try:
            event: dict[str, Any] = json.loads(body)
        except json.JSONDecodeError as exc:
            raise WebhookVerificationError(f"Invalid JSON payload: {exc}") from exc

        return await self._process_razorpay_event(event)

    async def _process_razorpay_event(self, event: dict[str, Any]) -> dict[str, Any]:
        """Route Razorpay events to the appropriate handler."""
        event_type: str = event.get("event", "")
        payload: dict[str, Any] = event.get("payload", {})

        if event_type == "payment.captured":
            # Extract subscription ID from payment notes
            notes: dict = payload.get("payment", {}).get("entity", {}).get("notes", {})
            subscription_id: str | None = notes.get("subscription_id")
            if subscription_id:
                await self._activate_by_external_id(subscription_id, "razorpay")

        elif event_type in {"subscription.cancelled", "subscription.expired"}:
            subscription_id = (
                payload.get("subscription", {}).get("entity", {}).get("id")
            )
            if subscription_id:
                await self._cancel_by_external_id(subscription_id, "razorpay")

        return {"status": "processed", "event": event_type}

    # ── Stripe webhook ─────────────────────────────────────────────────────────

    async def handle_stripe_webhook(
        self, body: bytes, stripe_signature: str
    ) -> dict[str, Any]:
        """
        Verify Stripe webhook signature and process payment events.
        Stripe format: t=<timestamp>,v1=<hmac_sha256>
        """
        if not settings.stripe_webhook_secret:
            logger.warning("Stripe not configured — webhook ignored.")
            return {"status": "ignored", "reason": "not_configured"}

        # Parse Stripe-Signature header
        sig_parts: dict[str, str] = {}
        for part in stripe_signature.split(","):
            if "=" in part:
                k, v = part.split("=", 1)
                sig_parts[k.strip()] = v.strip()

        timestamp = sig_parts.get("t", "")
        v1_sig = sig_parts.get("v1", "")
        if not timestamp or not v1_sig:
            raise WebhookVerificationError("Malformed Stripe-Signature header.")

        signed_payload = f"{timestamp}.".encode() + body
        expected = hmac.new(
            settings.stripe_webhook_secret.encode(),
            signed_payload,
            hashlib.sha256,
        ).hexdigest()
        if not hmac.compare_digest(expected, v1_sig):
            raise WebhookVerificationError("Stripe signature mismatch.")

        try:
            event: dict[str, Any] = json.loads(body)
        except json.JSONDecodeError as exc:
            raise WebhookVerificationError(f"Invalid JSON payload: {exc}") from exc

        return await self._process_stripe_event(event)

    async def _process_stripe_event(self, event: dict[str, Any]) -> dict[str, Any]:
        """Route Stripe events to the appropriate handler."""
        event_type: str = event.get("type", "")
        data_object: dict[str, Any] = event.get("data", {}).get("object", {})

        if event_type == "invoice.payment_succeeded":
            subscription_id: str | None = data_object.get("subscription")
            if subscription_id:
                await self._activate_by_external_id(subscription_id, "stripe")

        elif event_type in {
            "customer.subscription.deleted",
            "customer.subscription.paused",
        }:
            subscription_id = data_object.get("id")
            if subscription_id:
                await self._cancel_by_external_id(subscription_id, "stripe")

        elif event_type == "customer.subscription.updated":
            subscription_id = data_object.get("id")
            status: str = data_object.get("status", "")
            if subscription_id and status == "active":
                await self._activate_by_external_id(subscription_id, "stripe")

        return {"status": "processed", "event": event_type}

    # ── Internal helpers ───────────────────────────────────────────────────────

    async def _activate_by_external_id(
        self, external_id: str, provider: str
    ) -> None:
        """Activate a subscription matched by provider-specific external ID."""
        if provider == "stripe":
            stmt = select(Subscription).where(
                Subscription.stripe_subscription_id == external_id
            )
        else:
            stmt = select(Subscription).where(
                Subscription.razorpay_subscription_id == external_id
            )
        result = await self._db.execute(stmt)
        sub = result.scalar_one_or_none()
        if sub is None:
            logger.warning("Subscription not found for external_id=%s", external_id)
            return

        sub.status = "active"
        sub.updated_at = datetime.now(tz=timezone.utc)

        # Sync user plan
        user_result = await self._db.execute(
            select(User).where(User.id == sub.user_id)
        )
        user = user_result.scalar_one_or_none()
        if user:
            user.plan = sub.plan
        await self._db.commit()

    async def _cancel_by_external_id(
        self, external_id: str, provider: str
    ) -> None:
        """Cancel a subscription matched by provider-specific external ID."""
        if provider == "stripe":
            stmt = select(Subscription).where(
                Subscription.stripe_subscription_id == external_id
            )
        else:
            stmt = select(Subscription).where(
                Subscription.razorpay_subscription_id == external_id
            )
        result = await self._db.execute(stmt)
        sub = result.scalar_one_or_none()
        if sub is None:
            return

        sub.status = "cancelled"
        sub.updated_at = datetime.now(tz=timezone.utc)

        user_result = await self._db.execute(
            select(User).where(User.id == sub.user_id)
        )
        user = user_result.scalar_one_or_none()
        if user:
            user.plan = "free"
        await self._db.commit()
