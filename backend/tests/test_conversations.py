# ───────────────────────────────────────────────────────────────
# Copyright (c) 2026 Ambrish Nigam
# Author : Ambrish Nigam | https://github.com/ambrishnigam30
# Project: SynthFlow — Autonomous Synthetic Data Orchestration Platform
# License : Apache License 2.0 | https://www.apache.org/licenses/LICENSE-2.0
# Module : B-002 Conversations integration tests
# ───────────────────────────────────────────────────────────────

import uuid

import pytest
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.conversation import Message

pytestmark = pytest.mark.anyio

# ── Helpers ───────────────────────────────────────────────────────────────────

_USER_A = {"email": "conva@synthflow.ai", "password": "Pass1234!", "full_name": "User A"}
_USER_B = {"email": "convb@synthflow.ai", "password": "Pass1234!", "full_name": "User B"}


async def _signup_get_token(client: AsyncClient, payload: dict) -> tuple[str, str]:
    """Sign up a user and return (access_token, user_id)."""
    resp = await client.post("/api/auth/signup", json=payload)
    assert resp.status_code == 200, resp.text
    data = resp.json()["data"]
    return data["access_token"], data["user"]["id"]


def _auth(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


# ── Tests ──────────────────────────────────────────────────────────────────────


async def test_b002_01_create_conversation(client: AsyncClient) -> None:
    """B-002-01: POST /api/conversations with valid data → 201 + conversation."""
    token, _ = await _signup_get_token(client, _USER_A)
    resp = await client.post(
        "/api/conversations",
        json={"title": "Test conversation", "conversation_type": "generate"},
        headers=_auth(token),
    )
    assert resp.status_code == 201
    body = resp.json()
    assert body["error"] is None
    data = body["data"]
    assert "id" in data
    assert data["title"] == "Test conversation"
    assert "user_id" in data
    assert data["messages"] == []


async def test_b002_02_list_conversations(client: AsyncClient) -> None:
    """B-002-02: GET /api/conversations for authenticated user → array."""
    token, _ = await _signup_get_token(client, _USER_A)

    # Create two conversations
    for title in ("First", "Second"):
        await client.post(
            "/api/conversations",
            json={"title": title},
            headers=_auth(token),
        )

    resp = await client.get("/api/conversations", headers=_auth(token))
    assert resp.status_code == 200
    body = resp.json()
    assert body["error"] is None
    convs = body["data"]["conversations"]
    assert isinstance(convs, list)
    assert len(convs) == 2


async def test_b002_03_get_conversation_with_messages(
    client: AsyncClient, db_session: AsyncSession
) -> None:
    """B-002-03: GET /api/conversations/{id} returns conversation + messages."""
    token, _ = await _signup_get_token(client, _USER_A)

    # Create conversation
    resp = await client.post(
        "/api/conversations",
        json={"title": "With messages"},
        headers=_auth(token),
    )
    conv_id: str = resp.json()["data"]["id"]

    # Insert a message directly into DB
    msg = Message(
        id=str(uuid.uuid4()),
        conversation_id=conv_id,
        role="user",
        content="Hello, generate data",
    )
    db_session.add(msg)
    await db_session.commit()

    resp = await client.get(f"/api/conversations/{conv_id}", headers=_auth(token))
    assert resp.status_code == 200
    body = resp.json()
    assert body["error"] is None
    data = body["data"]
    assert data["id"] == conv_id
    messages = data["messages"]
    assert len(messages) == 1
    assert messages[0]["role"] == "user"
    assert messages[0]["content"] == "Hello, generate data"


async def test_b002_04_delete_conversation(client: AsyncClient) -> None:
    """B-002-04: DELETE /api/conversations/{id} → 204."""
    token, _ = await _signup_get_token(client, _USER_A)

    resp = await client.post(
        "/api/conversations",
        json={"title": "To delete"},
        headers=_auth(token),
    )
    conv_id: str = resp.json()["data"]["id"]

    resp = await client.delete(f"/api/conversations/{conv_id}", headers=_auth(token))
    assert resp.status_code == 204

    # Confirm it's gone
    resp = await client.get(f"/api/conversations/{conv_id}", headers=_auth(token))
    assert resp.status_code == 404


async def test_b002_05_other_user_cannot_access(client: AsyncClient) -> None:
    """B-002-05: User B cannot access User A's conversation → 404."""
    token_a, _ = await _signup_get_token(client, _USER_A)
    token_b, _ = await _signup_get_token(client, _USER_B)

    # A creates a conversation
    resp = await client.post(
        "/api/conversations",
        json={"title": "Private conversation"},
        headers=_auth(token_a),
    )
    conv_id: str = resp.json()["data"]["id"]

    # B tries to access it
    resp = await client.get(f"/api/conversations/{conv_id}", headers=_auth(token_b))
    assert resp.status_code == 404
