# ───────────────────────────────────────────────────────────────
# Copyright (c) 2026 Ambrish Nigam
# Author : Ambrish Nigam | https://github.com/ambrishnigam30
# Project: SynthFlow — Autonomous Synthetic Data Orchestration Platform
# License : Apache License 2.0 | https://www.apache.org/licenses/LICENSE-2.0
# Module : Teams REST API — CRUD, member management (business plan+)
# ───────────────────────────────────────────────────────────────

from __future__ import annotations

import logging
import uuid
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, EmailStr, Field
from sqlalchemy import or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.dependencies import get_current_user
from app.models.team import Team, TeamMember
from app.models.user import User

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/teams", tags=["teams"])

_PLAN_ALLOWED = {"business", "enterprise"}


def _ok(data: object) -> dict:
    return {"data": data, "error": None}


def _require_business(user: User) -> None:
    if user.plan not in _PLAN_ALLOWED:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={
                "message": "Teams require Business plan or above.",
                "code": "PLAN_REQUIRED",
            },
        )


class CreateTeamRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    slug: str | None = Field(None, max_length=100)


class UpdateTeamRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)


class InviteMemberRequest(BaseModel):
    email: EmailStr
    role: str = Field("member", pattern="^(admin|member)$")


def _team_dict(team: Team, role: str | None = None) -> dict[str, Any]:
    return {
        "id": team.id,
        "name": team.name,
        "owner_id": team.owner_id,
        "plan": team.plan,
        "role": role,
        "created_at": team.created_at.isoformat(),
        "updated_at": team.updated_at.isoformat(),
    }


async def _get_team_with_role(
    team_id: str, user_id: str, db: AsyncSession, require_owner: bool = False
) -> tuple[Team, str]:
    """
    Return (team, role) for the given user.
    Raises 404 if team not found or user not a member.
    Raises 403 if require_owner=True and user is not owner.
    """
    stmt = select(Team).where(Team.id == team_id)
    result = await db.execute(stmt)
    team = result.scalar_one_or_none()
    if team is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"message": "Team not found.", "code": "NOT_FOUND"},
        )

    if team.owner_id == user_id:
        role = "owner"
    else:
        mem_stmt = select(TeamMember).where(
            TeamMember.team_id == team_id, TeamMember.user_id == user_id
        )
        mem_result = await db.execute(mem_stmt)
        member = mem_result.scalar_one_or_none()
        if member is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={"message": "Team not found.", "code": "NOT_FOUND"},
            )
        role = member.role

    if require_owner and role != "owner":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={"message": "Only the team owner can perform this action.", "code": "FORBIDDEN"},
        )
    return team, role


# ── Endpoints ──────────────────────────────────────────────────────────────────

@router.post("", status_code=status.HTTP_201_CREATED)
async def create_team(
    body: CreateTeamRequest,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> dict:
    """POST /api/teams — create a new team (business plan+)."""
    _require_business(user)

    team = Team(
        id=str(uuid.uuid4()),
        name=body.name,
        owner_id=user.id,
    )
    db.add(team)

    # Add owner as a member with role "owner"
    member = TeamMember(
        id=str(uuid.uuid4()),
        team_id=team.id,
        user_id=user.id,
        role="owner",
    )
    db.add(member)
    await db.commit()
    await db.refresh(team)

    return _ok(_team_dict(team, role="owner"))


@router.get("")
async def list_teams(
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> dict:
    """GET /api/teams — list teams where the user is owner or member."""
    _require_business(user)

    # Teams where user is owner
    owned_stmt = select(Team).where(Team.owner_id == user.id)
    owned_result = await db.execute(owned_stmt)
    owned = list(owned_result.scalars().all())

    # Teams where user is a member (not owner)
    mem_stmt = select(TeamMember).where(
        TeamMember.user_id == user.id, TeamMember.role != "owner"
    )
    mem_result = await db.execute(mem_stmt)
    memberships = mem_result.scalars().all()

    member_team_ids = [m.team_id for m in memberships]
    member_teams: list[Team] = []
    if member_team_ids:
        mt_stmt = select(Team).where(Team.id.in_(member_team_ids))
        mt_result = await db.execute(mt_stmt)
        member_teams = list(mt_result.scalars().all())

    teams_out = [_team_dict(t, role="owner") for t in owned]
    for t in member_teams:
        role = next((m.role for m in memberships if m.team_id == t.id), "member")
        teams_out.append(_team_dict(t, role=role))

    return _ok({"teams": teams_out})


@router.patch("/{team_id}")
async def update_team(
    team_id: str,
    body: UpdateTeamRequest,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> dict:
    """PATCH /api/teams/:id — update team name (owner or admin)."""
    _require_business(user)
    team, role = await _get_team_with_role(team_id, user.id, db)

    if role not in {"owner", "admin"}:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={"message": "Only owners and admins can update the team.", "code": "FORBIDDEN"},
        )

    team.name = body.name
    await db.commit()
    await db.refresh(team)
    return _ok(_team_dict(team, role=role))


@router.delete("/{team_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_team(
    team_id: str,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> None:
    """DELETE /api/teams/:id — delete team (owner only)."""
    _require_business(user)
    team, _ = await _get_team_with_role(team_id, user.id, db, require_owner=True)

    await db.delete(team)
    await db.commit()


@router.post("/{team_id}/invite", status_code=status.HTTP_201_CREATED)
async def invite_member(
    team_id: str,
    body: InviteMemberRequest,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> dict:
    """POST /api/teams/:id/invite — invite a user by email (owner or admin)."""
    _require_business(user)
    team, role = await _get_team_with_role(team_id, user.id, db)

    if role not in {"owner", "admin"}:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={"message": "Only owners and admins can invite members.", "code": "FORBIDDEN"},
        )

    # Look up the invitee by email
    invitee_stmt = select(User).where(User.email == body.email, User.is_active.is_(True))
    invitee_result = await db.execute(invitee_stmt)
    invitee = invitee_result.scalar_one_or_none()

    if invitee is None:
        # In production this would send an email invitation; for now return gracefully
        return _ok(
            {
                "status": "invitation_queued",
                "email": body.email,
                "message": "User does not have a SynthFlow account. An invitation email will be sent.",
            }
        )

    # Check if already a member
    existing_stmt = select(TeamMember).where(
        TeamMember.team_id == team_id, TeamMember.user_id == invitee.id
    )
    existing_result = await db.execute(existing_stmt)
    existing = existing_result.scalar_one_or_none()
    if existing is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={"message": "User is already a team member.", "code": "ALREADY_MEMBER"},
        )

    member = TeamMember(
        id=str(uuid.uuid4()),
        team_id=team_id,
        user_id=invitee.id,
        role=body.role,
    )
    db.add(member)
    await db.commit()

    return _ok(
        {
            "status": "added",
            "user_id": invitee.id,
            "email": body.email,
            "role": body.role,
            "team_id": team_id,
        }
    )


@router.delete("/{team_id}/members/{member_user_id}", status_code=status.HTTP_204_NO_CONTENT)
async def remove_member(
    team_id: str,
    member_user_id: str,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> None:
    """DELETE /api/teams/:id/members/:user_id — remove a member (owner or admin)."""
    _require_business(user)
    team, role = await _get_team_with_role(team_id, user.id, db)

    if role not in {"owner", "admin"}:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={"message": "Only owners and admins can remove members.", "code": "FORBIDDEN"},
        )

    # Cannot remove the owner
    if member_user_id == team.owner_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"message": "Cannot remove the team owner.", "code": "CANNOT_REMOVE_OWNER"},
        )

    mem_stmt = select(TeamMember).where(
        TeamMember.team_id == team_id, TeamMember.user_id == member_user_id
    )
    mem_result = await db.execute(mem_stmt)
    member = mem_result.scalar_one_or_none()
    if member is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"message": "Member not found.", "code": "NOT_FOUND"},
        )

    await db.delete(member)
    await db.commit()
