# ───────────────────────────────────────────────────────────────
# Copyright (c) 2026 Ambrish Nigam
# Author : Ambrish Nigam | https://github.com/ambrishnigam30
# Project: SynthFlow — Autonomous Synthetic Data Orchestration Platform
# License : Apache License 2.0 | https://www.apache.org/licenses/LICENSE-2.0
# Module : Pydantic schemas for generation requests / responses
# ───────────────────────────────────────────────────────────────

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=5, max_length=8_000)
    conversation_id: str | None = None
    options: dict[str, Any] = Field(default_factory=dict)


class PhaseUpdate(BaseModel):
    phase: int = Field(..., ge=1, le=9)
    progress: float = Field(..., ge=0.0, le=1.0)
    message: str = ""


class GenerationResponse(BaseModel):
    generation_id: str
    status: str  # "pending" | "running" | "done" | "failed"
    domain: str | None = None
    sub_domain: str | None = None
    row_count: int | None = None
    quality_score: float | None = None
    privacy_score: float | None = None
    glass_box_code: str | None = None
    generation_schema: dict[str, Any] | None = None
    intent_json: dict[str, Any] | None = None
    error_message: str | None = None
    download_url: str | None = None
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True, "populate_by_name": True}


class GenerationStatus(BaseModel):
    generation_id: str
    status: str
    current_phase: int | None = None
    progress: float = 0.0
    message: str = ""
    quality_score: float | None = None
    error_message: str | None = None


class GenerationListItem(BaseModel):
    id: str
    session_id: str
    conversation_id: str | None = None
    prompt: str | None = None
    domain: str | None = None
    sub_domain: str | None = None
    row_count: int | None = None
    status: str
    quality_score: float | None = None
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}
