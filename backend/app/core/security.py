# ───────────────────────────────────────────────────────────────
# Copyright (c) 2026 Ambrish Nigam
# Author : Ambrish Nigam | https://github.com/ambrishnigam30
# Project: SynthFlow — Autonomous Synthetic Data Orchestration Platform
# License : Apache License 2.0 | https://www.apache.org/licenses/LICENSE-2.0
# Module : Security utilities — JWT, bcrypt, AES-256-GCM, API key generation
# ───────────────────────────────────────────────────────────────

import base64
import hashlib
import os
import secrets
import string
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from jose import JWTError, jwt
from passlib.context import CryptContext

from app.config import settings

_pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

_NONCE_SIZE = 12  # bytes for AES-GCM nonce


class AuthTokenError(Exception):
    """Raised when a JWT is invalid, expired, or malformed."""


def hash_password(password: str) -> str:
    """Hash a plain-text password using bcrypt."""
    return _pwd_context.hash(password)


def verify_password(plain: str, hashed: str) -> bool:
    """Verify a plain-text password against a bcrypt hash."""
    return _pwd_context.verify(plain, hashed)


def _build_token(payload: dict[str, Any], expires_delta: timedelta) -> str:
    expire = datetime.now(tz=timezone.utc) + expires_delta
    data = {**payload, "exp": expire, "iat": datetime.now(tz=timezone.utc)}
    return jwt.encode(data, settings.jwt_secret, algorithm=settings.jwt_algorithm)


def create_access_token(user_id: str) -> str:
    """Create a JWT access token valid for 15 minutes."""
    return _build_token(
        {"sub": user_id, "type": "access"},
        timedelta(minutes=settings.access_token_expire_minutes),
    )


def create_refresh_token(user_id: str) -> str:
    """Create a JWT refresh token valid for 7 days."""
    return _build_token(
        {"sub": user_id, "type": "refresh"},
        timedelta(days=settings.refresh_token_expire_days),
    )


def verify_token(token: str) -> dict[str, Any]:
    """
    Decode and validate a JWT.
    Raises AuthTokenError if invalid or expired.
    """
    try:
        payload: dict[str, Any] = jwt.decode(
            token, settings.jwt_secret, algorithms=[settings.jwt_algorithm]
        )
        return payload
    except JWTError as exc:
        raise AuthTokenError(f"Invalid or expired token: {exc}") from exc


def _derive_aes_key() -> bytes:
    """Derive a 32-byte AES key from the encryption_key setting via SHA-256."""
    return hashlib.sha256(settings.encryption_key.encode()).digest()


def encrypt_api_key(key: str) -> str:
    """
    Encrypt an API key with AES-256-GCM.
    Returns base64-encoded nonce + ciphertext.
    """
    aes_key = _derive_aes_key()
    aesgcm = AESGCM(aes_key)
    nonce = os.urandom(_NONCE_SIZE)
    ciphertext = aesgcm.encrypt(nonce, key.encode(), None)
    combined = nonce + ciphertext
    return base64.b64encode(combined).decode()


def decrypt_api_key(encrypted: str) -> str:
    """
    Decrypt an AES-256-GCM encrypted API key.
    Expects base64-encoded nonce + ciphertext.
    """
    aes_key = _derive_aes_key()
    aesgcm = AESGCM(aes_key)
    combined = base64.b64decode(encrypted.encode())
    nonce = combined[:_NONCE_SIZE]
    ciphertext = combined[_NONCE_SIZE:]
    plaintext = aesgcm.decrypt(nonce, ciphertext, None)
    return plaintext.decode()


def generate_api_key() -> tuple[str, str, str]:
    """
    Generate a new API key.

    Returns:
        (full_key, key_hash, key_prefix)
        - full_key  : "sf_live_" + 40 random alphanumeric chars
        - key_hash  : SHA-256 hex digest of full_key
        - key_prefix: first 12 characters of full_key
    """
    alphabet = string.ascii_letters + string.digits
    random_part = "".join(secrets.choice(alphabet) for _ in range(40))
    full_key = f"sf_live_{random_part}"
    key_hash = hashlib.sha256(full_key.encode()).hexdigest()
    key_prefix = full_key[:12]
    return full_key, key_hash, key_prefix
