# ───────────────────────────────────────────────────────────────
# Copyright (c) 2026 Ambrish Nigam
# Author : Ambrish Nigam | https://github.com/ambrishnigam30
# Project: SynthFlow — Autonomous Synthetic Data Orchestration Platform
# License : Apache License 2.0 | https://www.apache.org/licenses/LICENSE-2.0
# Module : Pydantic schemas for chat / conversations
# ───────────────────────────────────────────────────────────────

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class SendMessageRequest(BaseModel):
    content: str = Field(..., min_length=1, max_length=32_000)
    options: dict[str, Any] = Field(default_factory=dict)


class ChatMessage(BaseModel):
    id: str
    conversation_id: str
    role: str  # "user" | "assistant" | "system"
    content: str
    token_count: int | None = None
    created_at: datetime

    model_config = {"from_attributes": True}


class ConversationBase(BaseModel):
    id: str
    user_id: str
    title: str | None = None
    model_provider: str | None = None
    total_tokens: int = 0
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class ConversationResponse(ConversationBase):
    messages: list[ChatMessage] = Field(default_factory=list)


class ConversationListItem(ConversationBase):
    last_message_at: datetime | None = None
    message_count: int = 0


class CreateConversationRequest(BaseModel):
    title: str | None = Field(None, max_length=500)
    conversation_type: str = Field("generate", pattern="^(generate|explore)$")


class UpdateConversationRequest(BaseModel):
    title: str | None = Field(None, max_length=500)
    archived: bool | None = None
