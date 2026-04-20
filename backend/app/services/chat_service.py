# ───────────────────────────────────────────────────────────────
# Copyright (c) 2026 Ambrish Nigam
# Author : Ambrish Nigam | https://github.com/ambrishnigam30
# Project: SynthFlow — Autonomous Synthetic Data Orchestration Platform
# License : Apache License 2.0 | https://www.apache.org/licenses/LICENSE-2.0
# Module : ChatService — intent classification and message routing
# ───────────────────────────────────────────────────────────────

from __future__ import annotations

import logging
import re
import uuid
from collections.abc import AsyncGenerator
from typing import Any, Literal

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.conversation import Message
from app.models.user import User

logger = logging.getLogger(__name__)

_GENERATION_KEYWORDS = re.compile(
    r"\b(generate|create|produce|build|make|synthesize|give me|generate me|"
    r"i need|can you make|please create)\b.*\b(data|dataset|records|rows|sample|csv)\b",
    re.IGNORECASE,
)
_MODIFICATION_KEYWORDS = re.compile(
    r"\b(add|remove|delete|change|modify|update|increase|decrease|make it|"
    r"adjust|set|filter|include|exclude)\b",
    re.IGNORECASE,
)

IntentType = Literal["generation", "modification", "conversation"]


class ChatService:
    def __init__(self, db: AsyncSession) -> None:
        self._db = db

    # ── Intent classification ──────────────────────────────────────────────

    async def classify_intent(self, message: str) -> IntentType:
        """
        Classify a chat message into one of three intent types:
        - ``"generation"``:   user wants to generate / create a new dataset.
        - ``"modification"``: user wants to modify the last generated dataset.
        - ``"conversation"``: general question or discussion.
        """
        if _GENERATION_KEYWORDS.search(message):
            return "generation"

        # Row-count pattern heuristic ("500 rows", "5k records")
        if re.search(r"\b\d[\d,]*\s*(rows?|records?|samples?)\b", message, re.IGNORECASE):
            return "generation"

        if _MODIFICATION_KEYWORDS.search(message):
            return "modification"

        return "conversation"

    # ── Message history ────────────────────────────────────────────────────

    async def get_recent_messages(
        self, conversation_id: str, limit: int = 10
    ) -> list[dict[str, Any]]:
        """Return the last *limit* messages for LLM context."""
        stmt = (
            select(Message)
            .where(Message.conversation_id == conversation_id)
            .order_by(Message.created_at.desc())
            .limit(limit)
        )
        result = await self._db.execute(stmt)
        messages = list(reversed(result.scalars().all()))
        return [{"role": m.role, "content": m.content} for m in messages]

    async def save_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        token_count: int | None = None,
    ) -> Message:
        """Persist a message to the database."""
        msg = Message(
            id=str(uuid.uuid4()),
            conversation_id=conversation_id,
            role=role,
            content=content,
            token_count=token_count,
        )
        self._db.add(msg)
        await self._db.commit()
        await self._db.refresh(msg)
        return msg

    # ── Conversational response (no generation) ────────────────────────────

    async def conversational_reply(
        self,
        message: str,
        history: list[dict[str, Any]],
    ) -> AsyncGenerator[str, None]:
        """
        Stream a conversational reply for non-generation messages.
        Uses a simple template when no LLM is configured; real LLM used in production.
        """
        # Placeholder: echo a canned response chunk-by-chunk for testing
        # In production this calls the configured LLM provider.
        reply = (
            f"I understand you're asking about: {message[:80]}. "
            "I can help you generate synthetic data or answer questions about data science. "
            "To generate a dataset, try: 'Generate 1000 healthcare records for India'."
        )
        for word in reply.split():
            yield word + " "

    # ── Route message ──────────────────────────────────────────────────────

    async def route_message(
        self,
        conversation_id: str,
        message: str,
        user: User,
    ) -> tuple[IntentType, dict[str, Any]]:
        """
        Classify the message and return (intent, routing_context).
        routing_context carries what the generation_service or response handler needs.
        """
        intent = await self.classify_intent(message)
        history = await self.get_recent_messages(conversation_id)

        context: dict[str, Any] = {
            "message": message,
            "history": history,
            "conversation_id": conversation_id,
            "user_id": user.id,
        }

        if intent == "modification":
            # Find last generation in this conversation to get context
            context["is_modification"] = True

        return intent, context
