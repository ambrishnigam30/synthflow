# ───────────────────────────────────────────────────────────────
# Copyright (c) 2026 Ambrish Nigam
# Author : Ambrish Nigam | https://github.com/ambrishnigam30
# Project: SynthFlow — Autonomous Synthetic Data Orchestration Platform
# License : Apache License 2.0 | https://www.apache.org/licenses/LICENSE-2.0
# Module : WebSocket chat endpoint — real-time generation progress
# ───────────────────────────────────────────────────────────────

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, WebSocket, WebSocketDisconnect, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import AsyncSessionLocal, get_db
from app.core.security import AuthTokenError, verify_token
from app.models.conversation import Conversation, Message
from app.models.user import User
from app.services.chat_service import ChatService
from app.services.generation_service import GenerationService, PlanLimitError

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/ws", tags=["websocket"])


async def _auth_ws(websocket: WebSocket, db: AsyncSession) -> User | None:
    """Authenticate WebSocket connection via JWT query param."""
    token = websocket.query_params.get("token")
    if not token:
        return None
    try:
        payload = verify_token(token)
        user_id: str = payload.get("sub", "")
        stmt = select(User).where(User.id == user_id, User.is_active.is_(True))
        result = await db.execute(stmt)
        return result.scalar_one_or_none()
    except AuthTokenError:
        return None


@router.websocket("/chat/{conversation_id}")
async def websocket_chat(
    websocket: WebSocket,
    conversation_id: str,
    db: AsyncSession = Depends(get_db),
) -> None:
    """
    WebSocket endpoint for real-time chat + generation progress.

    Auth: JWT passed as `?token=...` query parameter.

    Receives: {"type": "message", "content": "...", "options": {...}}
    Sends:
      {"type": "text_chunk", "content": "..."}
      {"type": "generation_start", "generation_id": "..."}
      {"type": "phase_update", "phase": N, "progress": 0.0-1.0, "message": "..."}
      {"type": "generation_done", "generation_id": "...", "quality_score": float}
      {"type": "error", "message": "..."}
    """
    user = await _auth_ws(websocket, db)
    if user is None:
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return

    # Accept the connection first so we can send error messages if needed
    await websocket.accept()

    # Verify or auto-create the conversation.
    # The frontend may use locally generated UUIDs that don't exist in the DB yet.
    stmt = select(Conversation).where(
        Conversation.id == conversation_id,
        Conversation.user_id == user.id,
    )
    result = await db.execute(stmt)
    conv = result.scalar_one_or_none()
    if conv is None:
        conv = Conversation(
            id=conversation_id,
            user_id=user.id,
            title="New conversation",
        )
        db.add(conv)
        try:
            await db.commit()
            await db.refresh(conv)
        except Exception:
            await db.rollback()
            await websocket.close(code=status.WS_1011_INTERNAL_ERROR)
            return

    # Notify client that connection is ready
    await websocket.send_json({"type": "connected", "conversation_id": conversation_id})

    chat_svc = ChatService(db)
    gen_svc = GenerationService(db, session_factory=AsyncSessionLocal)

    try:
        while True:
            raw = await websocket.receive_text()
            try:
                msg_data = json.loads(raw)
            except json.JSONDecodeError:
                await websocket.send_json({"type": "error", "message": "Invalid JSON."})
                continue

            if msg_data.get("type") != "message":
                continue

            content: str = msg_data.get("content", "").strip()
            if not content:
                continue

            options: dict = msg_data.get("options", {})

            # Save user message
            await chat_svc.save_message(conversation_id, "user", content)

            # Classify intent
            intent, context = await chat_svc.route_message(conversation_id, content, user)

            if intent == "generation":
                # Trigger generation
                generation_id: str | None = None
                try:
                    def phase_cb(phase: int, progress: float, message: str) -> None:
                        import asyncio
                        asyncio.create_task(
                            websocket.send_json({
                                "type": "phase_update",
                                "phase": phase,
                                "progress": round(progress, 3),
                                "message": message,
                            })
                        )

                    def done_cb(gen_id: str, result: dict) -> None:
                        import asyncio
                        payload = {
                            "type": "generation_done",
                            "generation_id": gen_id,
                            "quality_score": result.get("quality_score", 0.0),
                            "preview_rows": result.get("preview_rows", []),
                            "row_count": result.get("row_count", 0),
                            "col_count": result.get("col_count", 0),
                            "domain": result.get("domain", ""),
                            "schema": result.get("schema", {}),
                            "download_urls": result.get(
                                "download_urls",
                                {"csv": "", "excel": "", "json": "", "parquet": ""},
                            ),
                        }
                        # Use send_text with a manually serialized string so that
                        # any NaN/Infinity values (invalid JSON) are replaced with null.
                        safe_json = (
                            json.dumps(payload, default=str)
                            .replace(": NaN", ": null").replace(":NaN", ":null")
                            .replace(": Infinity", ": null").replace(":Infinity", ":null")
                            .replace(": -Infinity", ": null").replace(":-Infinity", ":null")
                        )
                        asyncio.create_task(websocket.send_text(safe_json))

                    def error_cb(gen_id: str, error_msg: str, phase: int, progress: float) -> None:
                        import asyncio

                        async def _send_failure() -> None:
                            try:
                                if phase > 0:
                                    await websocket.send_json({
                                        "type": "phase_update",
                                        "phase": phase,
                                        "progress": round(progress, 3),
                                        "message": f"Failed: {error_msg}",
                                        "status": "failed",
                                    })
                                await websocket.send_json({
                                    "type": "error",
                                    "message": error_msg,
                                })
                            except Exception:
                                pass

                        asyncio.create_task(_send_failure())

                    generation_id = await gen_svc.trigger_generation(
                        prompt=content,
                        user=user,
                        conversation_id=conversation_id,
                        options=options,
                        phase_callback=phase_cb,
                        done_callback=done_cb,
                        error_callback=error_cb,
                    )
                    await websocket.send_json({
                        "type": "generation_start",
                        "generation_id": generation_id,
                    })
                except PlanLimitError as exc:
                    await websocket.send_json({"type": "error", "message": str(exc)})
                    continue

                # Save assistant ack
                await chat_svc.save_message(
                    conversation_id,
                    "assistant",
                    f"Starting generation (ID: {generation_id})…",
                )

            else:
                # Conversational reply
                reply_parts: list[str] = []
                async for chunk in chat_svc.conversational_reply(
                    content, context["history"], user_id=user.id
                ):
                    await websocket.send_json({"type": "text_chunk", "content": chunk})
                    reply_parts.append(chunk)

                full_reply = "".join(reply_parts)
                await chat_svc.save_message(conversation_id, "assistant", full_reply)

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected: conversation=%s user=%s", conversation_id, user.id)
    except Exception as exc:
        logger.exception("WebSocket error: %s", exc)
        try:
            await websocket.send_json({"type": "error", "message": "Internal server error."})
        except Exception:
            pass
