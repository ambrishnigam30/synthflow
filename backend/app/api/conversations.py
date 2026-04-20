# ───────────────────────────────────────────────────────────────
# Copyright (c) 2026 Ambrish Nigam
# Author : Ambrish Nigam | https://github.com/ambrishnigam30
# Project: SynthFlow — Autonomous Synthetic Data Orchestration Platform
# License : Apache License 2.0 | https://www.apache.org/licenses/LICENSE-2.0
# Module : Conversations REST API
# ───────────────────────────────────────────────────────────────

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.dependencies import get_current_user
from app.models.conversation import Conversation, Message
from app.models.user import User
from app.schemas.chat import (
    ChatMessage,
    ConversationListItem,
    ConversationResponse,
    CreateConversationRequest,
    UpdateConversationRequest,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/conversations", tags=["conversations"])


def _ok(data: object) -> dict:
    return {"data": data, "error": None}


@router.post("", status_code=status.HTTP_201_CREATED)
async def create_conversation(
    body: CreateConversationRequest,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> dict:
    conv = Conversation(
        id=str(uuid.uuid4()),
        user_id=user.id,
        title=body.title,
    )
    db.add(conv)
    await db.commit()
    await db.refresh(conv)

    return _ok(
        ConversationResponse(
            id=conv.id,
            user_id=conv.user_id,
            title=conv.title,
            model_provider=conv.model_provider,
            total_tokens=conv.total_tokens,
            created_at=conv.created_at,
            updated_at=conv.updated_at,
            messages=[],
        ).model_dump()
    )


@router.get("")
async def list_conversations(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> dict:
    stmt = (
        select(Conversation)
        .where(Conversation.user_id == user.id)
        .order_by(Conversation.updated_at.desc())
        .offset((page - 1) * page_size)
        .limit(page_size)
    )
    result = await db.execute(stmt)
    convs = result.scalars().all()

    items = [
        ConversationListItem(
            id=c.id,
            user_id=c.user_id,
            title=c.title,
            model_provider=c.model_provider,
            total_tokens=c.total_tokens,
            created_at=c.created_at,
            updated_at=c.updated_at,
        ).model_dump()
        for c in convs
    ]
    return _ok({"conversations": items, "page": page, "page_size": page_size})


@router.get("/{conversation_id}")
async def get_conversation(
    conversation_id: str,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> dict:
    conv = await _get_owned(conversation_id, user.id, db)

    msgs_stmt = (
        select(Message)
        .where(Message.conversation_id == conversation_id)
        .order_by(Message.created_at.asc())
    )
    msgs_result = await db.execute(msgs_stmt)
    messages = msgs_result.scalars().all()

    msg_out = [
        ChatMessage(
            id=m.id,
            conversation_id=m.conversation_id,
            role=m.role,
            content=m.content,
            token_count=m.token_count,
            created_at=m.created_at,
        ).model_dump()
        for m in messages
    ]

    return _ok(
        ConversationResponse(
            id=conv.id,
            user_id=conv.user_id,
            title=conv.title,
            model_provider=conv.model_provider,
            total_tokens=conv.total_tokens,
            created_at=conv.created_at,
            updated_at=conv.updated_at,
            messages=msg_out,  # type: ignore[arg-type]
        ).model_dump()
    )


@router.patch("/{conversation_id}")
async def update_conversation(
    conversation_id: str,
    body: UpdateConversationRequest,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> dict:
    conv = await _get_owned(conversation_id, user.id, db)

    if body.title is not None:
        conv.title = body.title
    conv.updated_at = datetime.now(tz=timezone.utc)
    await db.commit()
    await db.refresh(conv)

    return _ok(
        ConversationListItem(
            id=conv.id,
            user_id=conv.user_id,
            title=conv.title,
            model_provider=conv.model_provider,
            total_tokens=conv.total_tokens,
            created_at=conv.created_at,
            updated_at=conv.updated_at,
        ).model_dump()
    )


@router.delete("/{conversation_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_conversation(
    conversation_id: str,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> None:
    conv = await _get_owned(conversation_id, user.id, db)
    await db.delete(conv)
    await db.commit()


async def _get_owned(conversation_id: str, user_id: str, db: AsyncSession) -> Conversation:
    stmt = select(Conversation).where(
        Conversation.id == conversation_id,
        Conversation.user_id == user_id,
    )
    result = await db.execute(stmt)
    conv = result.scalar_one_or_none()
    if conv is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"message": "Conversation not found.", "code": "NOT_FOUND"},
        )
    return conv
