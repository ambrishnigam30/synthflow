# ───────────────────────────────────────────────────────────────
# Copyright (c) 2026 Ambrish Nigam
# Author : Ambrish Nigam | https://github.com/ambrishnigam30
# Project: SynthFlow — Autonomous Synthetic Data Orchestration Platform
# License : Apache License 2.0 | https://www.apache.org/licenses/LICENSE-2.0
# Module : APIKey, Webhook, and WebhookDelivery SQLAlchemy models
# ───────────────────────────────────────────────────────────────

import uuid
from datetime import datetime

import sqlalchemy as sa
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base
from app.models.base import TimestampMixin


class APIKey(Base):
    __tablename__ = "api_keys"

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
    name: Mapped[str] = mapped_column(sa.String(255), nullable=False)
    key_prefix: Mapped[str] = mapped_column(sa.String(12), nullable=False)
    key_hash: Mapped[str] = mapped_column(sa.String(255), nullable=False)
    scopes: Mapped[list] = mapped_column(sa.JSON, nullable=False, server_default="[]")
    is_active: Mapped[bool] = mapped_column(
        sa.Boolean, nullable=False, server_default=sa.true()
    )
    last_used_at: Mapped[datetime | None] = mapped_column(
        sa.DateTime(timezone=True), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        sa.DateTime(timezone=True),
        server_default=sa.func.now(),
        nullable=False,
    )


class Webhook(Base, TimestampMixin):
    __tablename__ = "webhooks"

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
    url: Mapped[str] = mapped_column(sa.Text, nullable=False)
    events: Mapped[list] = mapped_column(sa.JSON, nullable=False, server_default="[]")
    is_active: Mapped[bool] = mapped_column(
        sa.Boolean, nullable=False, server_default=sa.true()
    )
    secret: Mapped[str] = mapped_column(sa.String(255), nullable=False)


class WebhookDelivery(Base):
    __tablename__ = "webhook_deliveries"

    id: Mapped[str] = mapped_column(
        sa.String(36),
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
    )
    webhook_id: Mapped[str] = mapped_column(
        sa.String(36),
        sa.ForeignKey("webhooks.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    event_type: Mapped[str] = mapped_column(sa.String(100), nullable=False)
    payload: Mapped[dict] = mapped_column(sa.JSON, nullable=False)
    status_code: Mapped[int | None] = mapped_column(sa.Integer, nullable=True)
    response_body: Mapped[str | None] = mapped_column(sa.Text, nullable=True)
    success: Mapped[bool] = mapped_column(sa.Boolean, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        sa.DateTime(timezone=True),
        server_default=sa.func.now(),
        nullable=False,
    )
