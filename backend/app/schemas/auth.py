# ───────────────────────────────────────────────────────────────
# Copyright (c) 2026 Ambrish Nigam
# Author : Ambrish Nigam | https://github.com/ambrishnigam30
# Project: SynthFlow — Autonomous Synthetic Data Orchestration Platform
# License : Apache License 2.0 | https://www.apache.org/licenses/LICENSE-2.0
# Module : Pydantic schemas for auth endpoints
# ───────────────────────────────────────────────────────────────

from pydantic import BaseModel, EmailStr, Field


class SignupRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8)
    full_name: str | None = None


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class GoogleAuthRequest(BaseModel):
    id_token: str


class RefreshRequest(BaseModel):
    refresh_token: str


class UserProfile(BaseModel):
    id: str
    email: str
    full_name: str | None
    avatar_url: str | None
    plan: str
    auth_provider: str
    onboarding_completed: bool

    model_config = {"from_attributes": True}


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    user: UserProfile


class UpdateProfileRequest(BaseModel):
    full_name: str | None = None
    avatar_url: str | None = None
