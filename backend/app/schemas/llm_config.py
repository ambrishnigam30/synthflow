# ───────────────────────────────────────────────────────────────
# Copyright (c) 2026 Ambrish Nigam
# Author : Ambrish Nigam | https://github.com/ambrishnigam30
# Project: SynthFlow — Autonomous Synthetic Data Orchestration Platform
# License : Apache License 2.0 | https://www.apache.org/licenses/LICENSE-2.0
# Module : Pydantic schemas for LLM config endpoints
# ───────────────────────────────────────────────────────────────

from pydantic import BaseModel


class LLMConfigCreateRequest(BaseModel):
    provider: str
    api_key: str
    model_name: str | None = None
    is_default: bool = False


class LLMConfigResponse(BaseModel):
    id: str
    provider: str
    model_name: str | None
    masked_key: str
    is_active: bool
    is_default: bool = False

    model_config = {"from_attributes": True}


class LLMTestResponse(BaseModel):
    success: bool
    message: str
