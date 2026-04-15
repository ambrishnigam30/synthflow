# ───────────────────────────────────────────────────────────────
# Copyright (c) 2026 Ambrish Nigam
# Author : Ambrish Nigam | https://github.com/ambrishnigam30
# Project: SynthFlow — Autonomous Synthetic Data Orchestration Platform
# License : Apache License 2.0 | https://www.apache.org/licenses/LICENSE-2.0
# Module : Unified async LLM client (Gemini / OpenAI / Groq) + MockLLMClient
# ───────────────────────────────────────────────────────────────

from __future__ import annotations

import hashlib
import json
from typing import Any, Optional

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from synthflow.utils.helpers import LLMParseError, safe_json_loads


# ── Exceptions ─────────────────────────────────────────────────────────────

class LLMConfigError(Exception):
    """Raised when LLM provider configuration is invalid or missing."""


# ── Provider registry ──────────────────────────────────────────────────────

MODEL_REGISTRY: dict[str, dict[str, Any]] = {
    "gemini": {
        "default_model": "gemini-1.5-pro",
        "models": ["gemini-1.5-pro", "gemini-1.5-flash", "gemini-2.0-flash-exp"],
        "context_window": 1_000_000,
        "base_url": "https://generativelanguage.googleapis.com/v1beta",
    },
    "openai": {
        "default_model": "gpt-4o",
        "models": ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo"],
        "context_window": 128_000,
        "base_url": "https://api.openai.com/v1",
    },
    "groq": {
        "default_model": "llama-3.3-70b-versatile",
        "models": ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "mixtral-8x7b-32768"],
        "context_window": 32_768,
        "base_url": "https://api.groq.com/openai/v1",
    },
}

# Exceptions that should trigger a retry (network + rate-limit errors)
_RETRYABLE = (
    httpx.ConnectError,
    httpx.ReadTimeout,
    httpx.WriteTimeout,
    httpx.RemoteProtocolError,
    httpx.HTTPStatusError,
)


# ── Mock client (deterministic, no API calls) ──────────────────────────────

class MockLLMClient:
    """
    Deterministic mock LLM client for unit tests.

    Returns the same JSON for the same prompt (keyed on MD5 of prompt).
    No network calls are ever made.
    """

    def __init__(self) -> None:
        self.provider = "mock"
        self.model = "mock-model"
        self.call_count: int = 0
        self._overrides: dict[str, str] = {}

    def set_response(self, prompt_contains: str, response: str) -> None:
        """Register a specific response for prompts containing *prompt_contains*."""
        self._overrides[prompt_contains] = response

    async def complete(
        self,
        prompt: str,
        system_prompt: str = "",
        json_mode: bool = False,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> str:
        self.call_count += 1
        for key, resp in self._overrides.items():
            if key in prompt:
                return resp
        # Deterministic default: hash the prompt to a stable response
        h = hashlib.md5(prompt.encode()).hexdigest()[:8]
        return json.dumps(
            {
                "mock": True,
                "prompt_hash": h,
                "domain": "healthcare",
                "row_count": 100,
                "columns": ["patient_id", "name", "age"],
            }
        )

    def parse_json_response(self, text: str) -> dict[str, Any]:
        return safe_json_loads(text)  # type: ignore[return-value]


# ── Real async client ──────────────────────────────────────────────────────

class LLMClient:
    """
    Unified async LLM client supporting Gemini, OpenAI, and Groq.

    Usage::

        async with LLMClient(provider="openai", api_key="sk-...") as client:
            text = await client.complete("Generate a schema for …")
            data = client.parse_json_response(text)
    """

    def __init__(
        self,
        provider: str,
        api_key: str,
        model: Optional[str] = None,
    ) -> None:
        if provider not in MODEL_REGISTRY:
            raise LLMConfigError(
                f"Unknown provider '{provider}'. "
                f"Supported providers: {sorted(MODEL_REGISTRY)}"
            )
        self.provider = provider
        self.api_key = api_key
        self._registry: dict[str, Any] = MODEL_REGISTRY[provider]
        self.model: str = model or self._registry["default_model"]
        self._http = httpx.AsyncClient(timeout=httpx.Timeout(60.0))

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=8),
        retry=retry_if_exception_type(_RETRYABLE),
        reraise=True,
    )
    async def complete(
        self,
        prompt: str,
        system_prompt: str = "",
        json_mode: bool = False,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> str:
        """
        Send *prompt* to the configured provider and return the completion text.

        Retries up to 3 times with exponential back-off (2 s, 4 s, 8 s).
        """
        if self.provider in ("openai", "groq"):
            return await self._complete_openai_compat(
                prompt, system_prompt, json_mode, temperature, max_tokens
            )
        if self.provider == "gemini":
            return await self._complete_gemini(
                prompt, system_prompt, json_mode, temperature, max_tokens
            )
        raise LLMConfigError(f"Provider '{self.provider}' has no implementation")

    async def _complete_openai_compat(
        self,
        prompt: str,
        system_prompt: str,
        json_mode: bool,
        temperature: float,
        max_tokens: int,
    ) -> str:
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        resp = await self._http.post(
            f"{self._registry['base_url']}/chat/completions",
            json=payload,
            headers={"Authorization": f"Bearer {self.api_key}"},
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]  # type: ignore[no-any-return]

    async def _complete_gemini(
        self,
        prompt: str,
        system_prompt: str,
        json_mode: bool,
        temperature: float,
        max_tokens: int,
    ) -> str:
        contents: list[dict[str, Any]] = []
        if system_prompt:
            contents.append({"role": "user", "parts": [{"text": system_prompt}]})
            contents.append({"role": "model", "parts": [{"text": "Understood."}]})
        contents.append({"role": "user", "parts": [{"text": prompt}]})

        payload: dict[str, Any] = {
            "contents": contents,
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens,
            },
        }
        if json_mode:
            payload["generationConfig"]["responseMimeType"] = "application/json"

        resp = await self._http.post(
            f"{self._registry['base_url']}/models/{self.model}:generateContent"
            f"?key={self.api_key}",
            json=payload,
        )
        resp.raise_for_status()
        return resp.json()["candidates"][0]["content"]["parts"][0]["text"]  # type: ignore[no-any-return]

    def parse_json_response(self, text: str) -> dict[str, Any]:
        """Parse JSON from LLM text using the 6-strategy fallback chain."""
        result = safe_json_loads(text)
        if not isinstance(result, dict):
            raise LLMParseError(f"Expected JSON object, got {type(result).__name__}")
        return result

    async def __aenter__(self) -> "LLMClient":
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self._http.aclose()
