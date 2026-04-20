# ───────────────────────────────────────────────────────────────
# Copyright (c) 2026 Ambrish Nigam
# Author : Ambrish Nigam | https://github.com/ambrishnigam30
# Project: SynthFlow — Autonomous Synthetic Data Orchestration Platform
# License : Apache License 2.0 | https://www.apache.org/licenses/LICENSE-2.0
# Module : B-006 API Keys integration tests
# ───────────────────────────────────────────────────────────────

import pytest
from httpx import AsyncClient

pytestmark = pytest.mark.anyio

_SIGNUP = {"email": "apikeys@synthflow.ai", "password": "Secure1234!", "full_name": "Key User"}


async def _auth_headers(client: AsyncClient) -> dict:
    """Register a user, upgrade to pro, and return JWT auth headers."""
    resp = await client.post("/api/auth/signup", json=_SIGNUP)
    token = resp.json()["data"]["access_token"]
    hdrs = {"Authorization": f"Bearer {token}"}
    # Upgrade to pro so API key creation is allowed
    await client.post(
        "/api/billing/subscribe",
        json={"plan": "pro", "payment_provider": "manual"},
        headers=hdrs,
    )
    # Re-fetch token after plan change (plan is embedded in DB, not JWT)
    login_resp = await client.post(
        "/api/auth/login",
        json={"email": _SIGNUP["email"], "password": _SIGNUP["password"]},
    )
    new_token = login_resp.json()["data"]["access_token"]
    return {"Authorization": f"Bearer {new_token}"}


# ── B-006-01: Create API key ──────────────────────────────────────────────────

async def test_b006_01_create_api_key(client: AsyncClient) -> None:
    """B-006-01: POST /api/keys → full key returned once, prefixed sf_live_."""
    hdrs = await _auth_headers(client)
    resp = await client.post(
        "/api/keys",
        json={"name": "My Test Key"},
        headers=hdrs,
    )
    assert resp.status_code == 201
    body = resp.json()
    assert body["error"] is None
    data = body["data"]
    assert "full_key" in data
    assert data["full_key"].startswith("sf_live_")
    assert "key_prefix" in data
    assert data["key_prefix"] == data["full_key"][:12]
    assert data["is_active"] is True


# ── B-006-02: List shows prefix only ─────────────────────────────────────────

async def test_b006_02_list_shows_prefix_only(client: AsyncClient) -> None:
    """B-006-02: GET /api/keys → only prefix visible, full_key is None."""
    hdrs = await _auth_headers(client)
    # Create a key first
    await client.post("/api/keys", json={"name": "List Test Key"}, headers=hdrs)

    resp = await client.get("/api/keys", headers=hdrs)
    assert resp.status_code == 200
    body = resp.json()
    assert body["error"] is None
    keys = body["data"]["keys"]
    assert len(keys) >= 1
    for k in keys:
        assert k["full_key"] is None  # never exposed in list
        assert k["key_prefix"] is not None
        assert len(k["key_prefix"]) <= 12


# ── B-006-03: Revoke key ──────────────────────────────────────────────────────

async def test_b006_03_revoke_key(client: AsyncClient) -> None:
    """B-006-03: DELETE /api/keys/:id → key is_active becomes False."""
    hdrs = await _auth_headers(client)
    # Create a key
    create_resp = await client.post("/api/keys", json={"name": "Revoke Test"}, headers=hdrs)
    key_id = create_resp.json()["data"]["id"]

    # Revoke it
    del_resp = await client.delete(f"/api/keys/{key_id}", headers=hdrs)
    assert del_resp.status_code == 204

    # It should no longer appear in the list (list only returns active keys)
    list_resp = await client.get("/api/keys", headers=hdrs)
    key_ids = [k["id"] for k in list_resp.json()["data"]["keys"]]
    assert key_id not in key_ids


# ── B-006-04: API auth with key ───────────────────────────────────────────────

async def test_b006_04_api_auth_with_key(client: AsyncClient) -> None:
    """B-006-04: X-API-Key header authenticates to /api/v1/generate."""
    hdrs = await _auth_headers(client)
    # Create an API key
    create_resp = await client.post("/api/keys", json={"name": "Auth Test"}, headers=hdrs)
    full_key = create_resp.json()["data"]["full_key"]

    # Use it to call the public API
    resp = await client.post(
        "/api/v1/generate",
        json={"prompt": "Generate 10 healthcare records for testing"},
        headers={"X-API-Key": full_key},
    )
    # Should be 202 (accepted) — generation is async
    assert resp.status_code == 202
    body = resp.json()
    assert body["error"] is None
    assert "generation_id" in body["data"]
    assert body["data"]["status"] == "pending"


# ── B-006-05: Revoked key rejected ───────────────────────────────────────────

async def test_b006_05_revoked_key_rejected(client: AsyncClient) -> None:
    """B-006-05: Revoked X-API-Key returns 401."""
    hdrs = await _auth_headers(client)
    # Create and immediately revoke a key
    create_resp = await client.post("/api/keys", json={"name": "Revoke Auth Test"}, headers=hdrs)
    key_data = create_resp.json()["data"]
    full_key = key_data["full_key"]
    key_id = key_data["id"]

    await client.delete(f"/api/keys/{key_id}", headers=hdrs)

    # Revoked key should be rejected
    resp = await client.post(
        "/api/v1/generate",
        json={"prompt": "Generate 10 records"},
        headers={"X-API-Key": full_key},
    )
    assert resp.status_code == 401
