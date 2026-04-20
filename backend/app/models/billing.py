# ───────────────────────────────────────────────────────────────
# Copyright (c) 2026 Ambrish Nigam
# Author : Ambrish Nigam | https://github.com/ambrishnigam30
# Project: SynthFlow — Autonomous Synthetic Data Orchestration Platform
# License : Apache License 2.0 | https://www.apache.org/licenses/LICENSE-2.0
# Module : Subscription, UsageRecord, and UsageMonthly SQLAlchemy models
# ───────────────────────────────────────────────────────────────

import uuid
from datetime import datetime

import sqlalchemy as sa
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base
from app.models.base import TimestampMixin


class Subscription(Base, TimestampMixin):
    __tablename__ = "subscriptions"

    id: Mapped[str] = mapped_column(
        sa.String(36),
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
    )
    user_id: Mapped[str] = mapped_column(
        sa.String(36),
        sa.ForeignKey("users.id", ondelete="CASCADE"),
        unique=True,
        nullable=False,
    )
    plan: Mapped[str] = mapped_column(sa.String(50), nullable=False)
    status: Mapped[str] = mapped_column(
        sa.String(50), nullable=False, server_default="active"
    )
    stripe_subscription_id: Mapped[str | None] = mapped_column(
        sa.String(255), nullable=True
    )
    razorpay_subscription_id: Mapped[str | None] = mapped_column(
        sa.String(255), nullable=True
    )
    current_period_start: Mapped[datetime | None] = mapped_column(
        sa.DateTime(timezone=True), nullable=True
    )
    current_period_end: Mapped[datetime | None] = mapped_column(
        sa.DateTime(timezone=True), nullable=True
    )


class UsageRecord(Base):
    __tablename__ = "usage_records"

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
    action: Mapped[str] = mapped_column(sa.String(100), nullable=False)
    quantity: Mapped[int] = mapped_column(sa.Integer, nullable=False, server_default="1")
    metadata_json: Mapped[dict | None] = mapped_column(sa.JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        sa.DateTime(timezone=True),
        server_default=sa.func.now(),
        nullable=False,
    )


class UsageMonthly(Base):
    __tablename__ = "usage_monthly"

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
    year_month: Mapped[str] = mapped_column(sa.String(7), nullable=False)
    rows_generated: Mapped[int] = mapped_column(
        sa.BigInteger, nullable=False, server_default="0"
    )
    generations_count: Mapped[int] = mapped_column(
        sa.Integer, nullable=False, server_default="0"
    )
    datasets_uploaded: Mapped[int] = mapped_column(
        sa.Integer, nullable=False, server_default="0"
    )
    api_calls: Mapped[int] = mapped_column(
        sa.Integer, nullable=False, server_default="0"
    )
    updated_at: Mapped[datetime] = mapped_column(
        sa.DateTime(timezone=True),
        server_default=sa.func.now(),
        onupdate=sa.func.now(),
        nullable=False,
    )

    __table_args__ = (
        sa.UniqueConstraint("user_id", "year_month", name="uq_usage_monthly_user_month"),
    )
