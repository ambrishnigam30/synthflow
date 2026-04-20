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

app.include_router(auth_router)
app.include_router(llm_config_router)


# ── Health ───────────────────────────────────────────────────────────────────
@app.get("/health")
async def health_check() -> dict:
    return {"data": {"status": "ok", "version": settings.app_version}, "error": None}
