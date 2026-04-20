# ───────────────────────────────────────────────────────────────
# Copyright (c) 2026 Ambrish Nigam
# Author : Ambrish Nigam | https://github.com/ambrishnigam30
# Project: SynthFlow — Autonomous Synthetic Data Orchestration Platform
# License : Apache License 2.0 | https://www.apache.org/licenses/LICENSE-2.0
# Module : B-003 Generation integration tests
# ───────────────────────────────────────────────────────────────

import uuid
from datetime import datetime, timezone

import pytest
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.billing import UsageMonthly
from app.models.generation import Generation

pytestmark = pytest.mark.anyio

# ── Helpers ───────────────────────────────────────────────────────────────────

_USER = {"email": "genuser@synthflow.ai", "password": "Pass1234!", "full_name": "Gen User"}
_PROMPT = "Generate 100 Indian healthcare records for Maharashtra"


async def _signup_get_token(client: AsyncClient) -> tuple[str, str]:
    """Sign up and return (access_token, user_id)."""
    resp = await client.post("/api/auth/signup", json=_USER)
    assert resp.status_code == 200, resp.text
    data = resp.json()["data"]
    return data["access_token"], data["user"]["id"]


def _auth(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


async def _trigger(client: AsyncClient, token: str, prompt: str = _PROMPT) -> str:
    """Trigger a generation and return the generation_id."""
    resp = await client.post(
        "/api/generate",
        json={"prompt": prompt},
        headers=_auth(token),
    )
    assert resp.status_code == 202, resp.text
    return resp.json()["data"]["generation_id"]



# ── Tests ──────────────────────────────────────────────────────────────────────


async def test_b003_01_trigger_generation(client: AsyncClient) -> None:
    """B-003-01: POST /api/generate → 202 + generation_id + status=pending."""
    token, _ = await _signup_get_token(client)
    resp = await client.post(
        "/api/generate",
        json={"prompt": _PROMPT},
        headers=_auth(token),
    )
    assert resp.status_code == 202
    body = resp.json()
    assert body["error"] is None
    data = body["data"]
    assert "generation_id" in data
    assert data["status"] == "pending"
    # generation_id must be a non-empty string
    assert isinstance(data["generation_id"], str)
    assert len(data["generation_id"]) > 0


async def test_b003_02_get_generation_status(
    client: AsyncClient, db_session: AsyncSession
) -> None:
    """B-003-02: GET /api/generate/{id} returns status object."""
    token, _ = await _signup_get_token(client)
    gen_id = await _trigger(client, token)

    resp = await client.get(f"/api/generate/{gen_id}", headers=_auth(token))
    assert resp.status_code == 200
    body = resp.json()
    assert body["error"] is None
    data = body["data"]
    assert data["generation_id"] == gen_id
    assert data["status"] in {"pending", "running", "done", "failed"}
    assert "created_at" in data
    assert "updated_at" in data


async def test_b003_03_download_completed_generation(
    client: AsyncClient, db_session: AsyncSession
) -> None:
    """B-003-03: Download completed generation → 302 redirect."""
    token, user_id = await _signup_get_token(client)

    # Create a completed generation directly in DB (no background task — avoids race condition)
    gen_id = str(uuid.uuid4())
    gen = Generation(
        id=gen_id,
        user_id=user_id,
        session_id=str(uuid.uuid4()),
        status="done",
        row_count=100,
        quality_score=88.5,
        privacy_score=92.0,
    )
    db_session.add(gen)
    await db_session.commit()

    resp = await client.get(
        f"/api/generate/{gen_id}/download",
        headers=_auth(token),
        follow_redirects=False,
    )
    assert resp.status_code == 302
    assert resp.headers.get("location"), "Redirect must have a Location header"


async def test_b003_04_get_glass_box_code(
    client: AsyncClient, db_session: AsyncSession
) -> None:
    """B-003-04: GET /api/generate/{id}/code returns Python source."""
    token, user_id = await _signup_get_token(client)

    # Create a completed generation with glass_box_code directly in DB
    gen_id = str(uuid.uuid4())
    glass_code = (
        "# Glass Box code\n"
        "def generate(row_count, seed):\n"
        "    import pandas as pd\n"
        "    return pd.DataFrame({'id': range(row_count)})\n"
    )
    gen = Generation(
        id=gen_id,
        user_id=user_id,
        session_id=str(uuid.uuid4()),
        status="done",
        row_count=100,
        glass_box_code=glass_code,
    )
    db_session.add(gen)
    await db_session.commit()

    resp = await client.get(f"/api/generate/{gen_id}/code", headers=_auth(token))
    assert resp.status_code == 200
    body = resp.json()
    assert body["error"] is None
    code = body["data"]["code"]
    assert isinstance(code, str)
    assert "def generate" in code
    assert body["data"]["generation_id"] == gen_id


async def test_b003_05_plan_limit_enforced(
    client: AsyncClient, db_session: AsyncSession
) -> None:
    """B-003-05: Exceeding free plan's 10 generations/month → 429."""
    token, user_id = await _signup_get_token(client)

    # Insert a monthly usage record at the free plan limit (10 generations)
    ym = datetime.now(tz=timezone.utc).strftime("%Y-%m")
    monthly = UsageMonthly(
        id=str(uuid.uuid4()),
        user_id=user_id,
        year_month=ym,
        generations_count=10,  # free plan cap
        rows_generated=0,
        datasets_uploaded=0,
        api_calls=0,
        updated_at=datetime.now(tz=timezone.utc),
    )
    db_session.add(monthly)
    await db_session.commit()

    resp = await client.post(
        "/api/generate",
        json={"prompt": _PROMPT},
        headers=_auth(token),
    )
    assert resp.status_code == 429
    body = resp.json()
    # FastAPI wraps HTTPException detail
    detail = body.get("detail") or {}
    if isinstance(detail, dict):
        assert detail.get("code") == "PLAN_LIMIT_EXCEEDED"
