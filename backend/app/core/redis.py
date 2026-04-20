# ───────────────────────────────────────────────────────────────
# Copyright (c) 2026 Ambrish Nigam
# Author : Ambrish Nigam | https://github.com/ambrishnigam30
# Project: SynthFlow — Autonomous Synthetic Data Orchestration Platform
# License : Apache License 2.0 | https://www.apache.org/licenses/LICENSE-2.0
# Module : Async Redis client with graceful fallback
# ───────────────────────────────────────────────────────────────

import logging
from typing import TYPE_CHECKING

import redis.asyncio as aioredis
from redis.exceptions import RedisError

from app.config import settings

logger = logging.getLogger(__name__)


class RedisClient:
    """Async Redis wrapper with graceful fallback when Redis is unavailable."""

    def __init__(self) -> None:
        self._client: aioredis.Redis | None = None
        self._available: bool = True

    async def _get_client(self) -> aioredis.Redis | None:
        if not self._available:
            return None
        if self._client is None:
            try:
                self._client = aioredis.from_url(
                    settings.redis_url,
                    encoding="utf-8",
                    decode_responses=True,
                )
                # Ping to confirm connection
                await self._client.ping()
            except (RedisError, OSError, ConnectionRefusedError) as exc:
                logger.warning("Redis unavailable, falling back to no-op mode: %s", exc)
                self._client = None
                self._available = False
        return self._client

    async def get(self, key: str) -> str | None:
        """Return the value for key, or None if not found / Redis unavailable."""
        client = await self._get_client()
        if client is None:
            return None
        try:
            return await client.get(key)
        except RedisError as exc:
            logger.warning("Redis get error for key %r: %s", key, exc)
            return None

    async def set(self, key: str, value: str, expire: int | None = None) -> None:
        """Set key=value with optional TTL in seconds. No-op if Redis unavailable."""
        client = await self._get_client()
        if client is None:
            return
        try:
            await client.set(key, value, ex=expire)
        except RedisError as exc:
            logger.warning("Redis set error for key %r: %s", key, exc)

    async def delete(self, key: str) -> None:
        """Delete a key. No-op if Redis unavailable."""
        client = await self._get_client()
        if client is None:
            return
        try:
            await client.delete(key)
        except RedisError as exc:
            logger.warning("Redis delete error for key %r: %s", key, exc)

    async def incr(self, key: str) -> int:
        """Atomically increment key. Returns 0 if Redis unavailable."""
        client = await self._get_client()
        if client is None:
            return 0
        try:
            result: int = await client.incr(key)
            return result
        except RedisError as exc:
            logger.warning("Redis incr error for key %r: %s", key, exc)
            return 0

    async def rate_limit_check(
        self,
        user_id: str,
        action: str,
        limit: int,
        window_seconds: int,
    ) -> bool:
        """
        Sliding-window rate limit check using INCR + EXPIRE.

        Returns True if the request is within the limit, False if exceeded.
        Falls back to True (allow) when Redis is unavailable.
        """
        client = await self._get_client()
        if client is None:
            return True

        key = f"rate:{action}:{user_id}"
        try:
            current = await client.incr(key)
            if current == 1:
                # First hit in this window — set expiry
                await client.expire(key, window_seconds)
            return current <= limit
        except RedisError as exc:
            logger.warning("Redis rate_limit_check error for %r/%r: %s", user_id, action, exc)
            return True


redis_client = RedisClient()
