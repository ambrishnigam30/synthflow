# ───────────────────────────────────────────────────────────────
# Copyright (c) 2026 Ambrish Nigam
# Author : Ambrish Nigam | https://github.com/ambrishnigam30
# Project: SynthFlow — Autonomous Synthetic Data Orchestration Platform
# License : Apache License 2.0 | https://www.apache.org/licenses/LICENSE-2.0
# Module : FastAPI application entry point
# ───────────────────────────────────────────────────────────────

import logging
from contextlib import asynccontextmanager
from collections.abc import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(application: FastAPI) -> AsyncGenerator[None, None]:
    logger.info("SynthFlow backend starting up (version %s).", settings.app_version)
    yield
    logger.info("SynthFlow backend shutting down.")


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ──────────────────────────────────────────────────────────────────
from app.api.auth import router as auth_router
from app.api.llm_config import router as llm_config_router
from app.api.conversations import router as conversations_router
from app.api.generate import router as generate_router
from app.api.chat import router as chat_router
from app.api.datasets import router as datasets_router
from app.api.billing import router as billing_router
from app.api.teams import router as teams_router
from app.api.api_keys import router as api_keys_router
from app.api.webhooks import router as webhooks_router
from app.api.public_api import router as public_api_router

app.include_router(auth_router)
app.include_router(llm_config_router)
app.include_router(conversations_router)
app.include_router(generate_router)
app.include_router(chat_router)
app.include_router(datasets_router)
app.include_router(billing_router)
app.include_router(teams_router)
app.include_router(api_keys_router)
app.include_router(webhooks_router)
app.include_router(public_api_router)


# ── Health ───────────────────────────────────────────────────────────────────
@app.get("/health")
async def health_check() -> dict:
    return {"data": {"status": "ok", "version": settings.app_version}, "error": None}
