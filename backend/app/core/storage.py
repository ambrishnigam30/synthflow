# ───────────────────────────────────────────────────────────────
# Copyright (c) 2026 Ambrish Nigam
# Author : Ambrish Nigam | https://github.com/ambrishnigam30
# Project: SynthFlow — Autonomous Synthetic Data Orchestration Platform
# License : Apache License 2.0 | https://www.apache.org/licenses/LICENSE-2.0
# Module : Supabase Storage client with graceful fallback
# ───────────────────────────────────────────────────────────────

import logging

import httpx

from app.config import settings

logger = logging.getLogger(__name__)


class SupabaseStorageClient:
    """Supabase Storage wrapper with graceful fallback when not configured."""

    @property
    def _configured(self) -> bool:
        return bool(settings.supabase_url and settings.supabase_key)

    def _auth_headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {settings.supabase_key}",
            "apikey": settings.supabase_key,
        }

    async def upload_file(
        self,
        bucket: str,
        path: str,
        file_bytes: bytes,
        content_type: str = "application/octet-stream",
    ) -> str:
        """
        Upload bytes to Supabase Storage.
        Returns the storage path on success.
        """
        if not self._configured:
            logger.warning("Supabase not configured — upload_file is a no-op, returning placeholder path.")
            return f"placeholder://{bucket}/{path}"

        url = f"{settings.supabase_url}/storage/v1/object/{bucket}/{path}"
        headers = {
            **self._auth_headers(),
            "Content-Type": content_type,
        }
        async with httpx.AsyncClient() as client:
            response = await client.post(url, content=file_bytes, headers=headers)
            response.raise_for_status()
        return f"{bucket}/{path}"

    async def download_file(self, bucket: str, path: str) -> bytes:
        """Download a file from Supabase Storage and return its bytes."""
        if not self._configured:
            logger.warning("Supabase not configured — download_file is a no-op.")
            return b""

        url = f"{settings.supabase_url}/storage/v1/object/{bucket}/{path}"
        async with httpx.AsyncClient() as client:
            response = await client.get(url, headers=self._auth_headers())
            response.raise_for_status()
            return response.content

    async def delete_file(self, bucket: str, path: str) -> None:
        """Delete a file from Supabase Storage."""
        if not self._configured:
            logger.warning("Supabase not configured — delete_file is a no-op.")
            return

        url = f"{settings.supabase_url}/storage/v1/object/{bucket}/{path}"
        async with httpx.AsyncClient() as client:
            response = await client.delete(url, headers=self._auth_headers())
            response.raise_for_status()

    async def get_signed_url(
        self,
        bucket: str,
        path: str,
        expires_in: int = 3600,
    ) -> str:
        """Return a signed download URL valid for expires_in seconds."""
        if not self._configured:
            logger.warning("Supabase not configured — get_signed_url returning placeholder.")
            return f"placeholder://{bucket}/{path}?expires_in={expires_in}"

        url = f"{settings.supabase_url}/storage/v1/object/sign/{bucket}/{path}"
        async with httpx.AsyncClient() as client:
            response = await client.post(
                url,
                json={"expiresIn": expires_in},
                headers=self._auth_headers(),
            )
            response.raise_for_status()
            data: dict = response.json()
        signed_url: str = data.get("signedURL", "")
        if not signed_url.startswith("http"):
            signed_url = f"{settings.supabase_url}{signed_url}"
        return signed_url


storage_client = SupabaseStorageClient()
