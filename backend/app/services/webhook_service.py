# ───────────────────────────────────────────────────────────────
# Copyright (c) 2026 Ambrish Nigam
# Author : Ambrish Nigam | https://github.com/ambrishnigam30
# Project: SynthFlow — Autonomous Synthetic Data Orchestration Platform
# License : Apache License 2.0 | https://www.apache.org/licenses/LICENSE-2.0
# Module : WebhookService — HMAC-signed delivery with retry and logging
# ───────────────────────────────────────────────────────────────

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
import uuid
from typing import Any

import httpx
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.webhook import Webhook, WebhookDelivery

logger = logging.getLogger(__name__)

# Retry delays: 2 s, 4 s, 8 s (3 attempts total)
_RETRY_DELAYS: list[float] = [2.0, 4.0, 8.0]
_TIMEOUT_SECONDS = 10.0


def _sign_payload(secret: str, payload_bytes: bytes) -> str:
    """Return an HMAC-SHA256 signature for the given payload bytes."""
    return hmac.new(secret.encode(), payload_bytes, hashlib.sha256).hexdigest()


class WebhookService:
    def __init__(self, db: AsyncSession) -> None:
        self._db = db

    async def deliver(
        self,
        user_id: str,
        event_type: str,
        payload: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """
        Find all active webhooks for *user_id* that subscribe to *event_type*,
        then attempt delivery to each with up to 3 retries.
        Returns a list of delivery-result dicts.
        """
        stmt = select(Webhook).where(
            Webhook.user_id == user_id,
            Webhook.is_active.is_(True),
        )
        result = await self._db.execute(stmt)
        webhooks = result.scalars().all()

        results: list[dict[str, Any]] = []
        for webhook in webhooks:
            # Only deliver if webhook subscribes to this event type
            subscribed: list[str] = webhook.events or []
            if event_type not in subscribed and "*" not in subscribed:
                continue
            delivery_result = await self._deliver_one(webhook, event_type, payload)
            results.append(delivery_result)
        return results

    async def _deliver_one(
        self,
        webhook: Webhook,
        event_type: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        """Attempt delivery to a single webhook URL with retries."""
        body: dict[str, Any] = {
            "id": str(uuid.uuid4()),
            "type": event_type,
            "data": payload,
        }
        payload_bytes = json.dumps(body, default=str).encode()
        signature = _sign_payload(webhook.secret, payload_bytes)
        headers = {
            "Content-Type": "application/json",
            "X-SynthFlow-Signature": f"sha256={signature}",
            "X-SynthFlow-Event": event_type,
        }

        last_status: int | None = None
        last_body: str = ""
        success = False

        for attempt, delay in enumerate(_RETRY_DELAYS, start=1):
            try:
                async with httpx.AsyncClient(timeout=_TIMEOUT_SECONDS) as client:
                    resp = await client.post(
                        webhook.url,
                        content=payload_bytes,
                        headers=headers,
                    )
                last_status = resp.status_code
                last_body = resp.text[:2000]
                if 200 <= resp.status_code < 300:
                    success = True
                    break
                logger.warning(
                    "Webhook %s attempt %d/%d returned HTTP %d",
                    webhook.id, attempt, len(_RETRY_DELAYS), resp.status_code,
                )
            except (httpx.RequestError, httpx.TimeoutException) as exc:
                logger.warning(
                    "Webhook %s attempt %d/%d failed: %s",
                    webhook.id, attempt, len(_RETRY_DELAYS), exc,
                )
                last_body = str(exc)

            if attempt < len(_RETRY_DELAYS):
                await asyncio.sleep(delay)

        # Persist delivery record
        delivery = WebhookDelivery(
            id=str(uuid.uuid4()),
            webhook_id=webhook.id,
            event_type=event_type,
            payload=body,
            status_code=last_status,
            response_body=last_body,
            success=success,
        )
        self._db.add(delivery)
        try:
            await self._db.commit()
        except Exception as exc:
            logger.warning("Could not persist webhook delivery: %s", exc)

        return {
            "webhook_id": webhook.id,
            "url": webhook.url,
            "event_type": event_type,
            "success": success,
            "status_code": last_status,
        }
