# ───────────────────────────────────────────────────────────────
# Copyright (c) 2026 Ambrish Nigam
# Author : Ambrish Nigam | https://github.com/ambrishnigam30
# Project: SynthFlow — Autonomous Synthetic Data Orchestration Platform
# License : Apache License 2.0 | https://www.apache.org/licenses/LICENSE-2.0
# Module : LLM config API router — manage per-user LLM provider credentials
# ───────────────────────────────────────────────────────────────

import logging
import uuid
from typing import Any

import httpx
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.core.security import decrypt_api_key, encrypt_api_key
from app.dependencies import get_current_user
from app.models.llm_config import LLMConfig
from app.models.user import User
from app.schemas.llm_config import LLMConfigCreateRequest, LLMConfigResponse, LLMTestResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/llm-config", tags=["llm-config"])


def _ok(data: Any) -> dict[str, Any]:
    return {"data": data, "error": None}


def _err(message: str, code: str) -> dict[str, Any]:
    return {"data": None, "error": {"message": message, "code": code}}


def _mask_key(encrypted_key: str) -> str:
    """Return a masked representation showing the last 4 chars of the real key."""
    try:
        plain = decrypt_api_key(encrypted_key)
        suffix = plain[-4:] if len(plain) >= 4 else plain
        return "••••••••••••" + suffix
    except Exception:
        return "••••••••••••"


@router.post("/")
async def upsert_llm_config(
    body: LLMConfigCreateRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    """Save or update a provider API key (encrypted). Upserts by (user_id, provider)."""
    encrypted = encrypt_api_key(body.api_key)

    result = await db.execute(
        select(LLMConfig).where(
            LLMConfig.user_id == current_user.id,
            LLMConfig.provider == body.provider,
        )
    )
    existing: LLMConfig | None = result.scalar_one_or_none()

    # If setting as default, unset all other providers first
    if body.is_default:
        all_result = await db.execute(
            select(LLMConfig).where(
                LLMConfig.user_id == current_user.id,
                LLMConfig.is_active == True,  # noqa: E712
            )
        )
        for other in all_result.scalars().all():
            other.is_default = False

    if existing is not None:
        existing.encrypted_api_key = encrypted
        existing.model_name = body.model_name
        existing.is_active = True
        existing.is_default = body.is_default
        config = existing
    else:
        config = LLMConfig(
            id=str(uuid.uuid4()),
            user_id=current_user.id,
            provider=body.provider,
            encrypted_api_key=encrypted,
            model_name=body.model_name,
            is_active=True,
            is_default=body.is_default,
        )
        db.add(config)

    try:
        await db.commit()
        await db.refresh(config)
    except IntegrityError as exc:
        await db.rollback()
        logger.warning("LLM config upsert integrity error: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=_err("Could not save LLM config.", "DB_ERROR"),
        ) from exc

    response = LLMConfigResponse(
        id=config.id,
        provider=config.provider,
        model_name=config.model_name,
        masked_key=_mask_key(config.encrypted_api_key),
        is_active=config.is_active,
        is_default=config.is_default,
    )
    return _ok(response.model_dump())


@router.get("/")
async def list_llm_configs(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    """List the current user's LLM configurations with API keys masked."""
    result = await db.execute(
        select(LLMConfig).where(
            LLMConfig.user_id == current_user.id,
            LLMConfig.is_active == True,  # noqa: E712
        )
    )
    configs = result.scalars().all()
    items = [
        LLMConfigResponse(
            id=c.id,
            provider=c.provider,
            model_name=c.model_name,
            masked_key=_mask_key(c.encrypted_api_key),
            is_active=c.is_active,
            is_default=c.is_default,
        ).model_dump()
        for c in configs
    ]
    return _ok(items)


@router.delete("/{provider}")
async def delete_llm_config(
    provider: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    """Soft-delete (deactivate) an LLM config for the given provider."""
    result = await db.execute(
        select(LLMConfig).where(
            LLMConfig.user_id == current_user.id,
            LLMConfig.provider == provider,
            LLMConfig.is_active == True,  # noqa: E712
        )
    )
    config: LLMConfig | None = result.scalar_one_or_none()
    if config is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=_err(f"No active config found for provider '{provider}'.", "NOT_FOUND"),
        )
    config.is_active = False
    await db.commit()
    return _ok({"deleted": True, "provider": provider})


@router.post("/{provider}/test")
async def test_llm_config(
    provider: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    """Test the stored API key for the given provider by making a lightweight call."""
    result = await db.execute(
        select(LLMConfig).where(
            LLMConfig.user_id == current_user.id,
            LLMConfig.provider == provider,
            LLMConfig.is_active == True,  # noqa: E712
        )
    )
    config: LLMConfig | None = result.scalar_one_or_none()
    if config is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=_err(f"No active config found for provider '{provider}'.", "NOT_FOUND"),
        )

    api_key = decrypt_api_key(config.encrypted_api_key)
    success, message = await _test_provider_key(provider, api_key)
    response = LLMTestResponse(success=success, message=message)
    return _ok(response.model_dump())


@router.patch("/{provider}/default")
async def set_default_provider(
    provider: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    """Set the given provider as the default without requiring the API key again."""
    all_result = await db.execute(
        select(LLMConfig).where(
            LLMConfig.user_id == current_user.id,
            LLMConfig.is_active == True,  # noqa: E712
        )
    )
    configs = all_result.scalars().all()
    found = False
    for c in configs:
        c.is_default = c.provider == provider
        if c.provider == provider:
            found = True

    if not found:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=_err(f"No active config found for provider '{provider}'.", "NOT_FOUND"),
        )

    await db.commit()
    return _ok({"default_provider": provider})


async def _test_provider_key(provider: str, api_key: str) -> tuple[bool, str]:
    """
    Attempt a minimal API call to the provider to validate the key.
    Returns (success, message).
    """
    provider_lower = provider.lower()

    try:
        if provider_lower == "openai":
            async with httpx.AsyncClient() as client:
                resp = await client.get(
                    "https://api.openai.com/v1/models",
                    headers={"Authorization": f"Bearer {api_key}"},
                    timeout=10.0,
                )
                if resp.status_code == 200:
                    return True, "OpenAI API key is valid."
                return False, f"OpenAI returned HTTP {resp.status_code}."

        if provider_lower == "anthropic":
            async with httpx.AsyncClient() as client:
                resp = await client.get(
                    "https://api.anthropic.com/v1/models",
                    headers={"x-api-key": api_key, "anthropic-version": "2023-06-01"},
                    timeout=10.0,
                )
                if resp.status_code == 200:
                    return True, "Anthropic API key is valid."
                return False, f"Anthropic returned HTTP {resp.status_code}."

        if provider_lower in {"gemini", "google"}:
            async with httpx.AsyncClient() as client:
                resp = await client.get(
                    f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}",
                    timeout=10.0,
                )
                if resp.status_code == 200:
                    return True, "Gemini API key is valid."
                return False, f"Google returned HTTP {resp.status_code}."

        if provider_lower == "groq":
            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": "llama-3.3-70b-versatile",
                        "messages": [{"role": "user", "content": "Hi"}],
                        "max_tokens": 5,
                    },
                    timeout=15.0,
                )
                if resp.status_code == 200:
                    return True, "Groq API key is valid."
                return False, f"Groq returned HTTP {resp.status_code}."

        return False, f"Provider '{provider}' is not supported for testing."

    except httpx.RequestError as exc:
        import traceback
        traceback.print_exc()
        logger.error(
            "LLM key test request error for provider %r: %s: %s",
            provider, type(exc).__name__, exc,
        )
        return False, f"{type(exc).__name__}: {exc}"
    except Exception as exc:
        import traceback
        traceback.print_exc()
        logger.error(
            "LLM key test unexpected error for provider %r: %s: %s",
            provider, type(exc).__name__, exc,
        )
        return False, f"{type(exc).__name__}: {exc}"
