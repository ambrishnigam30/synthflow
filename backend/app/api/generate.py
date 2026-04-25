# ───────────────────────────────────────────────────────────────
# Copyright (c) 2026 Ambrish Nigam
# Author : Ambrish Nigam | https://github.com/ambrishnigam30
# Project: SynthFlow — Autonomous Synthetic Data Orchestration Platform
# License : Apache License 2.0 | https://www.apache.org/licenses/LICENSE-2.0
# Module : Generation REST API — trigger, status, download, Glass Box code
# ───────────────────────────────────────────────────────────────

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import RedirectResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.dependencies import get_current_user
from app.models.user import User
from app.schemas.generation import GenerateRequest, GenerationListItem, GenerationResponse
from app.services.generation_service import GenerationService, PlanLimitError

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/generate", tags=["generation"])


def _ok(data: object) -> dict:
    return {"data": data, "error": None}


@router.get("", summary="List generations")
async def list_generations(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    limit: int | None = Query(None, ge=1, le=500),
    domain: str | None = Query(None),
    generation_status: str | None = Query(None, alias="status"),
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> dict:
    svc = GenerationService(db)
    effective_page_size = limit if limit is not None else page_size
    gens = await svc.list_generations(
        user_id=user.id,
        page=page,
        page_size=effective_page_size,
        domain=domain,
        status=generation_status,
    )
    items = [
        GenerationListItem(
            id=g.id,
            session_id=g.session_id,
            conversation_id=g.conversation_id,
            prompt=(g.intent_json or {}).get("prompt") if g.intent_json else None,
            domain=g.domain,
            sub_domain=g.sub_domain,
            row_count=g.row_count,
            status=g.status,
            quality_score=g.quality_score,
            created_at=g.created_at,
            updated_at=g.updated_at,
        ).model_dump()
        for g in gens
    ]
    return _ok({"items": items, "total": len(items), "page": page, "page_size": effective_page_size})


@router.post("", status_code=status.HTTP_202_ACCEPTED)
async def trigger_generation(
    body: GenerateRequest,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> dict:
    svc = GenerationService(db)
    try:
        generation_id = await svc.trigger_generation(
            prompt=body.prompt,
            user=user,
            conversation_id=body.conversation_id,
            options=body.options,
        )
    except PlanLimitError as exc:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail={"message": str(exc), "code": "PLAN_LIMIT_EXCEEDED"},
        ) from exc

    return _ok({"generation_id": generation_id, "status": "pending"})


@router.get("/{generation_id}")
async def get_generation_status(
    generation_id: str,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> dict:
    svc = GenerationService(db)
    gen = await svc.get_generation(generation_id, user.id)
    if gen is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"message": "Generation not found.", "code": "NOT_FOUND"},
        )

    return _ok(
        GenerationResponse(
            generation_id=gen.id,
            status=gen.status,
            domain=gen.domain,
            sub_domain=gen.sub_domain,
            row_count=gen.row_count,
            quality_score=gen.quality_score,
            privacy_score=gen.privacy_score,
            glass_box_code=gen.glass_box_code,
            generation_schema=gen.schema_json,
            intent_json=gen.intent_json,
            error_message=gen.error_message,
            created_at=gen.created_at,
            updated_at=gen.updated_at,
        ).model_dump()
    )


@router.get("/{generation_id}/download")
async def download_generation(
    generation_id: str,
    fmt: str = Query("csv", pattern="^(csv|parquet|json|xlsx)$"),
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> RedirectResponse:
    svc = GenerationService(db)
    gen = await svc.get_generation(generation_id, user.id)
    if gen is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"message": "Generation not found.", "code": "NOT_FOUND"},
        )
    if gen.status != "done":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"message": "Generation not complete yet.", "code": "NOT_READY"},
        )
    # Redirect to Supabase signed URL (placeholder for test env)
    signed_url = gen.storage_path or f"/static/generations/{generation_id}.{fmt}"
    return RedirectResponse(url=signed_url, status_code=302)


@router.get("/{generation_id}/code")
async def get_glass_box_code(
    generation_id: str,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> dict:
    svc = GenerationService(db)
    gen = await svc.get_generation(generation_id, user.id)
    if gen is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"message": "Generation not found.", "code": "NOT_FOUND"},
        )
    if not gen.glass_box_code:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"message": "Glass Box code not yet available.", "code": "NOT_READY"},
        )
    return _ok({"code": gen.glass_box_code, "generation_id": generation_id})


@router.get("/{generation_id}/report")
async def get_quality_report(
    generation_id: str,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> dict:
    svc = GenerationService(db)
    gen = await svc.get_generation(generation_id, user.id)
    if gen is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"message": "Generation not found.", "code": "NOT_FOUND"},
        )
    return _ok(
        {
            "generation_id": generation_id,
            "quality_score": gen.quality_score,
            "privacy_score": gen.privacy_score,
            "status": gen.status,
            "schema": gen.schema_json,  # raw DB field, not schema response field
        }
    )
