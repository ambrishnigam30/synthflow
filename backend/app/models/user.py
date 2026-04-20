# ───────────────────────────────────────────────────────────────
# Copyright (c) 2026 Ambrish Nigam
# Author : Ambrish Nigam | https://github.com/ambrishnigam30
# Project: SynthFlow — Autonomous Synthetic Data Orchestration Platform
# License : Apache License 2.0 | https://www.apache.org/licenses/LICENSE-2.0
# Module : User SQLAlchemy model
# ───────────────────────────────────────────────────────────────

import uuid

import sqlalchemy as sa
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base
from app.models.base import TimestampMixin


class User(Base, TimestampMixin):
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(
        sa.String(36),
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
    )
    email: Mapped[str] = mapped_column(
        sa.String(255),
        unique=True,
        nullable=False,
        index=True,
    )
    full_name: Mapped[str | None] = mapped_column(sa.String(255), nullable=True)
    avatar_url: Mapped[str | None] = mapped_column(sa.Text, nullable=True)
    hashed_password: Mapped[str | None] = mapped_column(sa.Text, nullable=True)
    auth_provider: Mapped[str] = mapped_column(
        sa.String(50), nullable=False, server_default="email"
    )
    plan: Mapped[str] = mapped_column(
        sa.String(50), nullable=False, server_default="free"
    )
    stripe_customer_id: Mapped[str | None] = mapped_column(
        sa.String(255), nullable=True
    )
    razorpay_customer_id: Mapped[str | None] = mapped_column(
        sa.String(255), nullable=True
    )
    onboarding_completed: Mapped[bool] = mapped_column(
        sa.Boolean, nullable=False, server_default=sa.false()
    )
    is_active: Mapped[bool] = mapped_column(
        sa.Boolean, nullable=False, server_default=sa.true()
    )
