# ───────────────────────────────────────────────────────────────
# Copyright (c) 2026 Ambrish Nigam
# Author : Ambrish Nigam | https://github.com/ambrishnigam30
# Project: SynthFlow — Autonomous Synthetic Data Orchestration Platform
# License : Apache License 2.0 | https://www.apache.org/licenses/LICENSE-2.0
# Module : Pydantic schemas for dataset upload and querying
# ───────────────────────────────────────────────────────────────

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class ColumnSummary(BaseModel):
    name: str
    dtype: str
    null_count: int = 0
    unique_count: int = 0
    sample_values: list[Any] = Field(default_factory=list)


class DatasetInfo(BaseModel):
    id: str
    name: str
    description: str | None = None
    file_size: int
    row_count: int | None = None
    column_count: int | None = None
    schema_summary: dict[str, Any] | None = None
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class DatasetUploadResponse(BaseModel):
    dataset: DatasetInfo
    preview_rows: list[dict[str, Any]] = Field(default_factory=list)
    columns: list[ColumnSummary] = Field(default_factory=list)


class DatasetQueryRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=2_000)
    max_rows: int = Field(100, ge=1, le=10_000)


class QueryResult(BaseModel):
    question: str
    operation: str = ""
    data: list[dict[str, Any]] = Field(default_factory=list)
    row_count: int = 0
    columns: list[str] = Field(default_factory=list)
    summary: str = ""


class DatasetQueryResponse(BaseModel):
    result: QueryResult
    execution_time_ms: int = 0
