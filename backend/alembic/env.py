# ───────────────────────────────────────────────────────────────
# Copyright (c) 2026 Ambrish Nigam
# Author : Ambrish Nigam | https://github.com/ambrishnigam30
# Project: SynthFlow — Autonomous Synthetic Data Orchestration Platform
# License : Apache License 2.0 | https://www.apache.org/licenses/LICENSE-2.0
# Module : Alembic environment — async SQLAlchemy + auto-import of all models
# ───────────────────────────────────────────────────────────────

from __future__ import annotations

import asyncio
import os
from logging.config import fileConfig

from alembic import context
from sqlalchemy import pool
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import async_engine_from_config

# ── Alembic Config ────────────────────────────────────────────────────────────
config = context.config

# Interpret the config file for Python logging.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# ── DATABASE_URL override ─────────────────────────────────────────────────────
# Always read DATABASE_URL from the environment (or .env) — never hardcode.
database_url = os.environ.get("DATABASE_URL", "")
if not database_url:
    # Attempt to load from backend .env
    try:
        from app.config import settings  # type: ignore[import]
        database_url = settings.database_url
    except Exception:
        pass

if database_url:
    config.set_main_option("sqlalchemy.url", database_url)

# ── Import ALL models so Alembic sees their metadata ─────────────────────────
# This guarantees autogenerate picks up every table.
from app.core.database import Base  # noqa: E402  # type: ignore[import]
import app.models.user          # noqa: F401, E402
import app.models.conversation  # noqa: F401, E402
import app.models.generation    # noqa: F401, E402
import app.models.dataset       # noqa: F401, E402
import app.models.billing       # noqa: F401, E402
import app.models.webhook       # noqa: F401, E402
import app.models.llm_config    # noqa: F401, E402
import app.models.team          # noqa: F401, E402

target_metadata = Base.metadata

# ── Offline migrations ────────────────────────────────────────────────────────

def run_migrations_offline() -> None:
    """
    Run migrations without an active database connection.
    Emits SQL to stdout so it can be reviewed or piped.
    """
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
        compare_server_default=True,
    )
    with context.begin_transaction():
        context.run_migrations()


# ── Online migrations (async) ─────────────────────────────────────────────────

def do_run_migrations(connection: Connection) -> None:
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        compare_type=True,
        compare_server_default=True,
    )
    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    """Create an async engine and run migrations through a sync connection."""
    connectable = async_engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)
    await connectable.dispose()


def run_migrations_online() -> None:
    asyncio.run(run_async_migrations())


# ── Entry point ───────────────────────────────────────────────────────────────

if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
