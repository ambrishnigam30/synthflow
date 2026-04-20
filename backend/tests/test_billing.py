# ───────────────────────────────────────────────────────────────
# Copyright (c) 2026 Ambrish Nigam
# Author : Ambrish Nigam | https://github.com/ambrishnigam30
# Project: SynthFlow — Autonomous Synthetic Data Orchestration Platform
# License : Apache License 2.0 | https://www.apache.org/licenses/LICENSE-2.0
# Module : B-005 Billing integration tests
# ───────────────────────────────────────────────────────────────

import pytest
from httpx import AsyncClient

pytestmark = pytest.mark.anyio

_SIGNUP = {"email": "billing@synthflow.ai", "password": "Secure1234!", "full_name": "Billing User"}


async def _auth(client: AsyncClient) -> str:
    """Register a user and return bearer token."""
    resp = await client.post("/api/auth/signup", json=_SIGNUP)
    return resp.json()["data"]["access_token"]


async def _headers(client: AsyncClient) -> dict:
    token = await _auth(client)
    return {"Authorization": f"Bearer {token}"}


# ── B-005-01: List plans ──────────────────────────────────────────────────────

async def test_b005_01_list_plans(client: AsyncClient) -> None:
    """B-005-01: GET /api/billing/plans → 4 plans, no auth required."""
    resp = await client.get("/api/billing/plans")
    assert resp.status_code == 200
    body = resp.json()
    assert body["error"] is None
    plans = body["data"]["plans"]
    assert len(plans) == 4
    ids = {p["id"] for p in plans}
    assert ids == {"free", "pro", "business", "enterprise"}


# ── B-005-02: Subscribe ───────────────────────────────────────────────────────

async def test_b005_02_subscribe(client: AsyncClient) -> None:
    """B-005-02: POST /api/billing/subscribe → subscription created."""
    hdrs = await _headers(client)
    resp = await client.post(
        "/api/billing/subscribe",
        json={"plan": "pro", "payment_provider": "manual"},
        headers=hdrs,
    )
    assert resp.status_code == 201
    body = resp.json()
    assert body["error"] is None
    data = body["data"]
    assert data["plan"] == "pro"
    assert data["status"] == "active"
    assert "subscription_id" in data
    assert data["current_period_end"] is not None


# ── B-005-03: Cancel subscription ─────────────────────────────────────────────

async def test_b005_03_cancel_subscription(client: AsyncClient) -> None:
    """B-005-03: POST /api/billing/cancel → status = cancel_at_period_end."""
    hdrs = await _headers(client)
    # Subscribe first
    await client.post(
        "/api/billing/subscribe",
        json={"plan": "pro", "payment_provider": "manual"},
        headers=hdrs,
    )
    resp = await client.post("/api/billing/cancel", headers=hdrs)
    assert resp.status_code == 200
    body = resp.json()
    assert body["error"] is None
    assert body["data"]["status"] == "cancel_at_period_end"


# ── B-005-04: Usage tracking ──────────────────────────────────────────────────

async def test_b005_04_usage_endpoint_returns_data(client: AsyncClient) -> None:
    """B-005-04: GET /api/billing/usage → returns plan + usage object."""
    hdrs = await _headers(client)
    resp = await client.get("/api/billing/usage", headers=hdrs)
    assert resp.status_code == 200
    body = resp.json()
    assert body["error"] is None
    data = body["data"]
    assert "plan" in data
    assert "usage" in data
    assert "limits" in data
    assert data["plan"] == "free"  # default plan


# ── B-005-05: Razorpay webhook (not configured → ignored) ─────────────────────

async def test_b005_05_razorpay_webhook_not_configured(client: AsyncClient) -> None:
    """B-005-05: POST /api/billing/webhook/razorpay → ignored when not configured."""
    import json

    payload = json.dumps(
        {
            "event": "payment.captured",
            "payload": {"payment": {"entity": {"notes": {"subscription_id": "rp_xxx"}}}},
        }
    ).encode()

    resp = await client.post(
        "/api/billing/webhook/razorpay",
        content=payload,
        headers={"Content-Type": "application/json", "x-razorpay-signature": "dummy"},
    )
    # When razorpay is not configured, returns 200 with ignored status
    assert resp.status_code == 200
    body = resp.json()
    assert body.get("status") == "ignored"
    assert body.get("reason") == "not_configured"
