# ───────────────────────────────────────────────────────────────
# Copyright (c) 2026 Ambrish Nigam
# Author : Ambrish Nigam | https://github.com/ambrishnigam30
# Project: SynthFlow — Autonomous Synthetic Data Orchestration Platform
# License : Apache License 2.0 | https://www.apache.org/licenses/LICENSE-2.0
# Module : Generation and HealEvent SQLAlchemy models
# ───────────────────────────────────────────────────────────────

import uuid

import sqlalchemy as sa
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base
from app.models.base import TimestampMixin


class Generation(Base, TimestampMixin):
    __tablename__ = "generations"

    id: Mapped[str] = mapped_column(
        sa.String(36),
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
    )
    user_id: Mapped[str] = mapped_column(
        sa.String(36),
        sa.ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    conversation_id: Mapped[str | None] = mapped_column(
        sa.String(36),
        sa.ForeignKey("conversations.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    session_id: Mapped[str] = mapped_column(
        sa.String(255), unique=True, nullable=False
    )
    domain: Mapped[str | None] = mapped_column(sa.String(255), nullable=True)
    sub_domain: Mapped[str | None] = mapped_column(sa.String(255), nullable=True)
    row_count: Mapped[int | None] = mapped_column(sa.Integer, nullable=True)
    status: Mapped[str] = mapped_column(
        sa.String(50), nullable=False, server_default="pending"
    )
    quality_score: Mapped[float | None] = mapped_column(sa.Float, nullable=True)
    privacy_score: Mapped[float | None] = mapped_column(sa.Float, nullable=True)
    storage_path: Mapped[str | None] = mapped_column(sa.Text, nullable=True)
    glass_box_code: Mapped[str | None] = mapped_column(sa.Text, nullable=True)
    schema_json: Mapped[dict | None] = mapped_column(sa.JSON, nullable=True)
    intent_json: Mapped[dict | None] = mapped_column(sa.JSON, nullable=True)
    error_message: Mapped[str | None] = mapped_column(sa.Text, nullable=True)


class HealEvent(Base):
    __tablename__ = "heal_events"

    id: Mapped[str] = mapped_column(
        sa.String(36),
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
    )
    generation_id: Mapped[str] = mapped_column(
        sa.String(36),
        sa.ForeignKey("generations.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    attempt: Mapped[int] = mapped_column(sa.Integer, nullable=False)
    error_text: Mapped[str] = mapped_column(sa.Text, nullable=False)
    healed_code: Mapped[str] = mapped_column(sa.Text, nullable=False)
    success: Mapped[bool] = mapped_column(sa.Boolean, nullable=False)
    created_at: Mapped[sa.DateTime] = mapped_column(
        sa.DateTime(timezone=True),
        server_default=sa.func.now(),
        nullable=False,
    )
