# ───────────────────────────────────────────────────────────────
# Copyright (c) 2026 Ambrish Nigam
# Author : Ambrish Nigam | https://github.com/ambrishnigam30
# Project: SynthFlow — Autonomous Synthetic Data Orchestration Platform
# License : Apache License 2.0 | https://www.apache.org/licenses/LICENSE-2.0
# Module : LLMConfig SQLAlchemy model for per-user LLM provider credentials
# ───────────────────────────────────────────────────────────────

import uuid

import sqlalchemy as sa
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base
from app.models.base import TimestampMixin


class LLMConfig(Base, TimestampMixin):
    __tablename__ = "llm_configs"

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
    provider: Mapped[str] = mapped_column(sa.String(100), nullable=False)
    encrypted_api_key: Mapped[str] = mapped_column(sa.Text, nullable=False)
    model_name: Mapped[str | None] = mapped_column(sa.String(255), nullable=True)
    is_active: Mapped[bool] = mapped_column(
        sa.Boolean, nullable=False, server_default=sa.true()
    )

    __table_args__ = (
        sa.UniqueConstraint("user_id", "provider", name="uq_llm_config_user_provider"),
    )
