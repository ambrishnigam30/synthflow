# ───────────────────────────────────────────────────────────────
# Copyright (c) 2026 Ambrish Nigam
# Author : Ambrish Nigam | https://github.com/ambrishnigam30
# Project: SynthFlow — Autonomous Synthetic Data Orchestration Platform
# License : Apache License 2.0 | https://www.apache.org/licenses/LICENSE-2.0
# Module : Auth API router — signup, login, OAuth, refresh, profile
# ───────────────────────────────────────────────────────────────

import logging
import uuid
from typing import Any

import httpx
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.core.database import get_db
from app.core.security import (
    AuthTokenError,
    create_access_token,
    create_refresh_token,
    hash_password,
    verify_password,
    verify_token,
)
from app.dependencies import get_current_user
from app.models.user import User
from app.schemas.auth import (
    ChangePasswordRequest,
    GoogleAuthRequest,
    LoginRequest,
    RefreshRequest,
    SignupRequest,
    TokenResponse,
    UpdateProfileRequest,
    UserProfile,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/auth", tags=["auth"])


def _ok(data: Any) -> dict[str, Any]:
    return {"data": data, "error": None}


def _err(message: str, code: str) -> dict[str, Any]:
    return {"data": None, "error": {"message": message, "code": code}}


def _build_token_response(user: User) -> dict[str, Any]:
    profile = UserProfile.model_validate(user)
    return TokenResponse(
        access_token=create_access_token(user.id),
        refresh_token=create_refresh_token(user.id),
        token_type="bearer",
        user=profile,
    ).model_dump()


@router.post("/signup")
async def signup(
    body: SignupRequest,
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    """Register a new user with email and password."""
    new_user = User(
        id=str(uuid.uuid4()),
        email=body.email.lower(),
        full_name=body.full_name,
        hashed_password=hash_password(body.password),
        auth_provider="email",
        plan="free",
    )
    db.add(new_user)
    try:
        await db.commit()
        await db.refresh(new_user)
    except IntegrityError:
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=_err("An account with this email already exists.", "EMAIL_TAKEN"),
        )
    return _ok(_build_token_response(new_user))


@router.post("/login")
async def login(
    body: LoginRequest,
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    """Authenticate with email and password, return JWT tokens."""
    result = await db.execute(
        select(User).where(User.email == body.email.lower(), User.is_active == True)  # noqa: E712
    )
    user: User | None = result.scalar_one_or_none()

    if user is None or not user.hashed_password or not verify_password(body.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=_err("Invalid email or password.", "INVALID_CREDENTIALS"),
        )
    return _ok(_build_token_response(user))


@router.post("/google")
async def google_auth(
    body: GoogleAuthRequest,
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    """
    Authenticate via Google ID token.
    Validates the token against Supabase, then upserts the user.
    """
    if not settings.supabase_url or not settings.supabase_key:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail=_err("Google OAuth is not configured.", "OAUTH_NOT_CONFIGURED"),
        )

    # Validate Google ID token via Supabase
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{settings.supabase_url}/auth/v1/token?grant_type=id_token",
                json={"provider": "google", "id_token": body.id_token},
                headers={"apikey": settings.supabase_key, "Content-Type": "application/json"},
            )
            resp.raise_for_status()
            supabase_data: dict[str, Any] = resp.json()
    except httpx.HTTPStatusError as exc:
        logger.warning("Supabase Google auth failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=_err("Google token validation failed.", "GOOGLE_AUTH_FAILED"),
        ) from exc

    sb_user: dict[str, Any] = supabase_data.get("user", {})
    email: str | None = sb_user.get("email")
    if not email:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=_err("Could not extract email from Google token.", "GOOGLE_AUTH_FAILED"),
        )

    result = await db.execute(select(User).where(User.email == email.lower()))
    user: User | None = result.scalar_one_or_none()

    if user is None:
        user = User(
            id=str(uuid.uuid4()),
            email=email.lower(),
            full_name=sb_user.get("user_metadata", {}).get("full_name"),
            avatar_url=sb_user.get("user_metadata", {}).get("avatar_url"),
            auth_provider="google",
            plan="free",
        )
        db.add(user)
        try:
            await db.commit()
            await db.refresh(user)
        except IntegrityError:
            await db.rollback()
            result2 = await db.execute(select(User).where(User.email == email.lower()))
            user = result2.scalar_one()

    return _ok(_build_token_response(user))


@router.post("/refresh")
async def refresh_token(
    body: RefreshRequest,
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    """Exchange a valid refresh token for a new access token."""
    try:
        payload = verify_token(body.refresh_token)
    except AuthTokenError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=_err(str(exc), "INVALID_TOKEN"),
        ) from exc

    if payload.get("type") != "refresh":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=_err("Not a refresh token.", "INVALID_TOKEN"),
        )

    user_id: str | None = payload.get("sub")
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=_err("Token missing subject.", "INVALID_TOKEN"),
        )

    result = await db.execute(
        select(User).where(User.id == user_id, User.is_active == True)  # noqa: E712
    )
    user: User | None = result.scalar_one_or_none()
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=_err("User not found.", "USER_NOT_FOUND"),
        )

    return _ok({"access_token": create_access_token(user.id), "token_type": "bearer"})


@router.get("/me")
async def get_me(
    current_user: User = Depends(get_current_user),
) -> dict[str, Any]:
    """Return the current user's profile."""
    return _ok(UserProfile.model_validate(current_user).model_dump())


@router.patch("/me")
async def update_me(
    body: UpdateProfileRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    """Update the current user's profile fields."""
    if body.full_name is not None:
        current_user.full_name = body.full_name
    if body.avatar_url is not None:
        current_user.avatar_url = body.avatar_url

    db.add(current_user)
    await db.commit()
    await db.refresh(current_user)
    return _ok(UserProfile.model_validate(current_user).model_dump())


@router.post("/change-password")
async def change_password(
    body: ChangePasswordRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    """Verify current password and update to a new hashed password."""
    if not current_user.hashed_password:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=_err(
                "Password change is not available for accounts signed in via Google.",
                "OAUTH_ACCOUNT",
            ),
        )
    if not verify_password(body.current_password, current_user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=_err("Current password is incorrect.", "WRONG_PASSWORD"),
        )
    current_user.hashed_password = hash_password(body.new_password)
    db.add(current_user)
    await db.commit()
    return _ok({"message": "Password updated successfully."})
