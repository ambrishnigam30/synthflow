# ───────────────────────────────────────────────────────────────
# Copyright (c) 2026 Ambrish Nigam
# Author : Ambrish Nigam | https://github.com/ambrishnigam30
# Project: SynthFlow — Autonomous Synthetic Data Orchestration Platform
# License : Apache License 2.0 | https://www.apache.org/licenses/LICENSE-2.0
# Module : B-001 Auth integration tests
# ───────────────────────────────────────────────────────────────

import pytest
from httpx import AsyncClient

pytestmark = pytest.mark.anyio

# ── Helpers ──────────────────────────────────────────────────────────────────

SIGNUP_PAYLOAD = {"email": "test@synthflow.ai", "password": "Secure1234!", "full_name": "Test User"}
LOGIN_PAYLOAD = {"email": "test@synthflow.ai", "password": "Secure1234!"}


async def _signup(client: AsyncClient) -> dict:
    """Helper: register a user and return the response JSON."""
    resp = await client.post("/api/auth/signup", json=SIGNUP_PAYLOAD)
    return resp.json()


# ── Tests ─────────────────────────────────────────────────────────────────────


async def test_b001_01_signup_valid(client: AsyncClient) -> None:
    """B-001-01: POST /api/auth/signup with valid data → 200 + tokens + user."""
    resp = await client.post("/api/auth/signup", json=SIGNUP_PAYLOAD)
    assert resp.status_code == 200
    body = resp.json()
    assert body["error"] is None
    data = body["data"]
    assert "access_token" in data
    assert "refresh_token" in data
    assert data["token_type"] == "bearer"
    user = data["user"]
    assert user["email"] == SIGNUP_PAYLOAD["email"]
    assert user["full_name"] == SIGNUP_PAYLOAD["full_name"]
    assert user["plan"] == "free"


async def test_b001_02_signup_duplicate_email(client: AsyncClient) -> None:
    """B-001-02: Signing up with the same email twice → 409."""
    await _signup(client)
    resp = await client.post("/api/auth/signup", json=SIGNUP_PAYLOAD)
    assert resp.status_code == 409
    body = resp.json()
    # FastAPI wraps HTTPException detail — check for error content
    assert body.get("detail") is not None or (
        body.get("data") is None and body.get("error") is not None
    )


async def test_b001_03_login_valid(client: AsyncClient) -> None:
    """B-001-03: POST /api/auth/login with correct credentials → 200 + access_token."""
    await _signup(client)
    resp = await client.post("/api/auth/login", json=LOGIN_PAYLOAD)
    assert resp.status_code == 200
    body = resp.json()
    assert body["error"] is None
    data = body["data"]
    assert "access_token" in data
    assert "refresh_token" in data


async def test_b001_04_login_wrong_password(client: AsyncClient) -> None:
    """B-001-04: Login with wrong password → 401."""
    await _signup(client)
    resp = await client.post(
        "/api/auth/login",
        json={"email": SIGNUP_PAYLOAD["email"], "password": "wrongpassword"},
    )
    assert resp.status_code == 401


async def test_b001_05_get_me_valid_token(client: AsyncClient) -> None:
    """B-001-05: GET /api/auth/me with valid token → 200 + user profile."""
    body = await _signup(client)
    token = body["data"]["access_token"]
    resp = await client.get("/api/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert resp.status_code == 200
    me_body = resp.json()
    assert me_body["error"] is None
    user = me_body["data"]
    assert user["email"] == SIGNUP_PAYLOAD["email"]


async def test_b001_06_get_me_expired_token(client: AsyncClient) -> None:
    """B-001-06: GET /api/auth/me with an expired / invalid token → 401."""
    fake_token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJmYWtlIiwiZXhwIjoxfQ.fake"
    resp = await client.get("/api/auth/me", headers={"Authorization": f"Bearer {fake_token}"})
    assert resp.status_code == 401


async def test_b001_07_refresh_token(client: AsyncClient) -> None:
    """B-001-07: POST /api/auth/refresh with valid refresh token → new access_token."""
    body = await _signup(client)
    refresh_token = body["data"]["refresh_token"]
    resp = await client.post(
        "/api/auth/refresh", json={"refresh_token": refresh_token}
    )
    assert resp.status_code == 200
    refresh_body = resp.json()
    assert refresh_body["error"] is None
    assert "access_token" in refresh_body["data"]


async def test_b001_08_get_me_no_token(client: AsyncClient) -> None:
    """B-001-08: GET /api/auth/me without Authorization header → 401/403."""
    resp = await client.get("/api/auth/me")
    assert resp.status_code in {401, 403}
