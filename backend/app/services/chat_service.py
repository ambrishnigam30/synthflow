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

from app.core.security import decrypt_api_key
from app.models.conversation import Message
from app.models.llm_config import LLMConfig
from app.models.user import User

logger = logging.getLogger(__name__)

_MODIFICATION_KEYWORDS = re.compile(
    r"\b(add|remove|delete|change|modify|update|increase|decrease|make it|"
    r"adjust|set|filter|include|exclude)\b",
    re.IGNORECASE,
)

IntentType = Literal["generation", "modification", "conversation"]


def is_generation_request(message: str) -> bool:
    """Detect if a user message is requesting data generation."""
    msg = message.lower().strip()

    has_number = bool(re.search(r"\b\d+\b", msg))

    action_words = [
        "generate", "create", "make", "build", "produce", "give me",
        "i need", "list of", "prepare", "simulate", "synthesize",
        "get me", "fabricate", "show me", "provide",
    ]
    has_action = any(word in msg for word in action_words)

    data_nouns = [
        "record", "row", "entr", "dataset", "data", "sample",
        "transaction", "patient", "student", "employee", "order",
        "customer", "account", "invoice", "report", "log", "event",
        "user", "product", "item", "ticket", "claim", "policy",
        "payment", "loan", "deposit", "person", "people", "member",
        "school", "hospital", "bank", "company", "store", "shop",
    ]
    has_data_noun = any(noun in msg for noun in data_nouns)

    return (
        (has_number and has_data_noun)
        or (has_action and has_data_noun)
        or (has_action and has_number)
    )


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
        if is_generation_request(message):
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

    # ── LLM config lookup ──────────────────────────────────────────────────

    async def _get_llm_config(self, user_id: str) -> tuple[str | None, str | None, str | None]:
        """Return (provider, decrypted_api_key, model) for the user's active LLM config."""
        try:
            stmt = (
                select(LLMConfig)
                .where(LLMConfig.user_id == user_id, LLMConfig.is_active == True)  # noqa: E712
                .order_by(LLMConfig.is_default.desc())
                .limit(1)
            )
            result = await self._db.execute(stmt)
            cfg = result.scalar_one_or_none()
            if cfg is None:
                return None, None, None
            api_key = decrypt_api_key(cfg.encrypted_api_key)
            return cfg.provider, api_key, cfg.model_name
        except Exception as exc:
            logger.warning("Could not load LLM config: %s", exc)
            return None, None, None

    # ── Conversational response (no generation) ────────────────────────────

    async def conversational_reply(
        self,
        message: str,
        history: list[dict[str, Any]],
        user_id: str | None = None,
    ) -> AsyncGenerator[str, None]:
        """
        Stream a conversational reply for non-generation messages.
        Uses the user's configured LLM provider when available;
        falls back to a canned response for testing.
        """
        # Try to use the real LLM engine if installed and configured
        if user_id:
            provider, api_key, model = await self._get_llm_config(user_id)
            if provider and api_key:
                try:
                    from synthflow.llm_client import LLMClient  # type: ignore[import]
                    client = LLMClient(
                        provider=provider,
                        api_key=api_key,
                        model=model,
                    )
                    system_prompt = (
                        "You are the SynthFlow AI assistant — an expert in synthetic data generation. "
                        "Help users understand synthetic data, data quality, privacy, and statistics. "
                        "When users ask for data generation, remind them to use the generate command."
                    )
                    # Build messages list including history
                    messages = [{"role": m["role"], "content": m["content"]} for m in history[-8:]]
                    messages.append({"role": "user", "content": message})
                    response = await client.complete(
                        prompt=message,
                        system=system_prompt,
                        messages=messages,
                    )
                    # Stream word-by-word
                    for word in response.split():
                        yield word + " "
                    return
                except ImportError:
                    logger.debug("SynthFlow LLM client not available; using fallback reply.")
                except Exception as llm_exc:
                    logger.warning("LLM conversational reply failed: %s", llm_exc)

        # Fallback: canned template response
        reply = (
            "I can help you generate synthetic data! Try describing what you need, like: "
            "'50 student records from Indian schools' or '100 banking transactions from Mumbai'. "
            "Include the number of rows and the type of data you want."
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
            # pass through so chat handler can call conversational_reply(user_id=...)
        }

        if intent == "modification":
            # Find last generation in this conversation to get context
            context["is_modification"] = True

        return intent, context
