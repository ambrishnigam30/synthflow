# ───────────────────────────────────────────────────────────────
# Copyright (c) 2026 Ambrish Nigam
# Author : Ambrish Nigam | https://github.com/ambrishnigam30
# Project: SynthFlow — Autonomous Synthetic Data Orchestration Platform
# License : Apache License 2.0 | https://www.apache.org/licenses/LICENSE-2.0
# Module : Team and TeamMember SQLAlchemy models
# ───────────────────────────────────────────────────────────────

import uuid
from datetime import datetime

import sqlalchemy as sa
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base
from app.models.base import TimestampMixin


class Team(Base, TimestampMixin):
    __tablename__ = "teams"

    id: Mapped[str] = mapped_column(
        sa.String(36),
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
    )
    name: Mapped[str] = mapped_column(sa.String(255), nullable=False)
    owner_id: Mapped[str] = mapped_column(
        sa.String(36),
        sa.ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    plan: Mapped[str] = mapped_column(
        sa.String(50), nullable=False, server_default="team"
    )


class TeamMember(Base):
    __tablename__ = "team_members"

    id: Mapped[str] = mapped_column(
        sa.String(36),
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
    )
    team_id: Mapped[str] = mapped_column(
        sa.String(36),
        sa.ForeignKey("teams.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    user_id: Mapped[str] = mapped_column(
        sa.String(36),
        sa.ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    role: Mapped[str] = mapped_column(
        sa.String(50), nullable=False, server_default="member"
    )
    created_at: Mapped[datetime] = mapped_column(
        sa.DateTime(timezone=True),
        server_default=sa.func.now(),
        nullable=False,
    )

    __table_args__ = (
        sa.UniqueConstraint("team_id", "user_id", name="uq_team_member"),
    )
