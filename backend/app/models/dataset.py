# ───────────────────────────────────────────────────────────────
# Copyright (c) 2026 Ambrish Nigam
# Author : Ambrish Nigam | https://github.com/ambrishnigam30
# Project: SynthFlow — Autonomous Synthetic Data Orchestration Platform
# License : Apache License 2.0 | https://www.apache.org/licenses/LICENSE-2.0
# Module : UploadedDataset SQLAlchemy model
# ───────────────────────────────────────────────────────────────

import uuid

import sqlalchemy as sa
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base
from app.models.base import TimestampMixin


class UploadedDataset(Base, TimestampMixin):
    __tablename__ = "uploaded_datasets"

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
    description: Mapped[str | None] = mapped_column(sa.Text, nullable=True)
    file_path: Mapped[str] = mapped_column(sa.Text, nullable=False)
    file_size: Mapped[int] = mapped_column(sa.BigInteger, nullable=False)
    row_count: Mapped[int | None] = mapped_column(sa.Integer, nullable=True)
    column_count: Mapped[int | None] = mapped_column(sa.Integer, nullable=True)
    schema_summary: Mapped[dict | None] = mapped_column(sa.JSON, nullable=True)
