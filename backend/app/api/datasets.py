# ───────────────────────────────────────────────────────────────
# Copyright (c) 2026 Ambrish Nigam
# Author : Ambrish Nigam | https://github.com/ambrishnigam30
# Project: SynthFlow — Autonomous Synthetic Data Orchestration Platform
# License : Apache License 2.0 | https://www.apache.org/licenses/LICENSE-2.0
# Module : Datasets REST API — upload, list, query, delete
# ───────────────────────────────────────────────────────────────

from __future__ import annotations

import logging
import time

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.dependencies import get_current_user
from app.models.user import User
from app.schemas.dataset import DatasetInfo, DatasetQueryRequest, DatasetQueryResponse, DatasetUploadResponse
from app.services.dataset_service import (
    DatasetFileTooLargeError,
    DatasetNotFoundError,
    DatasetService,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/datasets", tags=["datasets"])

_MAX_UPLOAD_BYTES = 50 * 1024 * 1024  # 50 MB


def _ok(data: object) -> dict:
    return {"data": data, "error": None}


@router.post("/upload", status_code=status.HTTP_201_CREATED)
async def upload_dataset(
    file: UploadFile = File(...),
    description: str | None = Form(None),
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> dict:
    file_bytes = await file.read()

    if len(file_bytes) > _MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail={"message": "File exceeds the 50 MB upload limit.", "code": "FILE_TOO_LARGE"},
        )

    svc = DatasetService(db)
    try:
        info, preview_rows, columns = await svc.upload(
            file_bytes=file_bytes,
            filename=file.filename or "upload.csv",
            user=user,
            description=description,
        )
    except DatasetFileTooLargeError as exc:
        raise HTTPException(
            status_code=413,
            detail={"message": str(exc), "code": "FILE_TOO_LARGE"},
        ) from exc

    return _ok(
        DatasetUploadResponse(
            dataset=info,
            preview_rows=preview_rows,
            columns=columns,
        ).model_dump()
    )


@router.get("")
async def list_datasets(
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> dict:
    svc = DatasetService(db)
    datasets = await svc.list_datasets(user.id)
    items = [DatasetInfo.model_validate(d).model_dump() for d in datasets]
    return _ok({"datasets": items, "count": len(items)})


@router.get("/{dataset_id}")
async def get_dataset(
    dataset_id: str,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> dict:
    svc = DatasetService(db)
    try:
        dataset = await svc.get_dataset_info(dataset_id, user.id)
    except DatasetNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"message": str(exc), "code": "NOT_FOUND"},
        ) from exc
    return _ok(DatasetInfo.model_validate(dataset).model_dump())


@router.delete("/{dataset_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_dataset(
    dataset_id: str,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> None:
    svc = DatasetService(db)
    try:
        await svc.delete(dataset_id, user)
    except DatasetNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"message": str(exc), "code": "NOT_FOUND"},
        ) from exc


@router.post("/{dataset_id}/query")
async def query_dataset(
    dataset_id: str,
    body: DatasetQueryRequest,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> dict:
    svc = DatasetService(db)
    t0 = time.monotonic()
    try:
        result = await svc.query(
            dataset_id=dataset_id,
            question=body.question,
            user=user,
            max_rows=body.max_rows,
        )
    except DatasetNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"message": str(exc), "code": "NOT_FOUND"},
        ) from exc

    elapsed_ms = int((time.monotonic() - t0) * 1000)
    return _ok(
        DatasetQueryResponse(
            result=result,
            execution_time_ms=elapsed_ms,
        ).model_dump()
    )
