# ───────────────────────────────────────────────────────────────
# Copyright (c) 2026 Ambrish Nigam
# Author : Ambrish Nigam | https://github.com/ambrishnigam30
# Project: SynthFlow — Autonomous Synthetic Data Orchestration Platform
# License : Apache License 2.0 | https://www.apache.org/licenses/LICENSE-2.0
# Module : Tests E-003-01 through E-003-08 — LLM client tests
# ───────────────────────────────────────────────────────────────

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from synthflow.llm_client import LLMClient, LLMConfigError, MockLLMClient, MODEL_REGISTRY
from synthflow.utils.helpers import LLMParseError


# ── Helpers ────────────────────────────────────────────────────────────────

def _openai_response(content: str) -> MagicMock:
    """Build a fake httpx response mimicking OpenAI's JSON structure."""
    mock = MagicMock()
    mock.json.return_value = {"choices": [{"message": {"content": content}}]}
    mock.raise_for_status = MagicMock()
    return mock


# ── E-003-01: Mock returns deterministic JSON ──────────────────────────────

@pytest.mark.asyncio
async def test_mock_llm_returns_deterministic_response() -> None:
    """MockLLMClient returns the same response for the same prompt."""
    client = MockLLMClient()
    prompt = "Generate Indian healthcare data"
    r1 = await client.complete(prompt)
    r2 = await client.complete(prompt)
    assert r1 == r2


@pytest.mark.asyncio
async def test_mock_llm_increments_call_count() -> None:
    """call_count increments with each complete() call."""
    client = MockLLMClient()
    assert client.call_count == 0
    await client.complete("p1")
    await client.complete("p2")
    assert client.call_count == 2


@pytest.mark.asyncio
async def test_mock_llm_returns_valid_json() -> None:
    """Default MockLLMClient response is valid JSON."""
    import json
    client = MockLLMClient()
    resp = await client.complete("any prompt")
    parsed = json.loads(resp)
    assert isinstance(parsed, dict)


@pytest.mark.asyncio
async def test_mock_llm_override_response() -> None:
    """set_response() makes the mock return a specific response for matching prompts."""
    client = MockLLMClient()
    client.set_response("healthcare", '{"domain": "healthcare"}')
    resp = await client.complete("Generate healthcare data")
    assert resp == '{"domain": "healthcare"}'


# ── E-003-02: LLMClient retries 3 times on failure ────────────────────────

@pytest.mark.asyncio
async def test_llm_client_retries_3_times_on_failure(mocker: Any) -> None:
    """LLMClient retries exactly 3 times before raising on repeated failure."""
    mocker.patch("asyncio.sleep", return_value=None)

    client = LLMClient(provider="openai", api_key="test-key")
    mock_post = mocker.patch.object(
        client._http,
        "post",
        new_callable=AsyncMock,
        side_effect=httpx.ConnectError("Connection refused"),
    )

    with pytest.raises(httpx.ConnectError):
        await client.complete("test prompt")

    assert mock_post.call_count == 3


# ── E-003-03: Parse JSON from markdown fences ─────────────────────────────

@pytest.mark.asyncio
async def test_llm_client_parses_json_from_markdown_fences(mocker: Any) -> None:
    """Response wrapped in ```json ... ``` is parsed correctly."""
    mocker.patch("asyncio.sleep", return_value=None)
    client = LLMClient(provider="openai", api_key="test-key")
    content = '```json\n{"domain": "finance", "rows": 1000}\n```'
    mock_post = mocker.patch.object(
        client._http, "post", new_callable=AsyncMock,
        return_value=_openai_response(content),
    )

    raw = await client.complete("prompt")
    parsed = client.parse_json_response(raw)

    assert parsed["domain"] == "finance"
    assert parsed["rows"] == 1000
    mock_post.assert_called_once()


# ── E-003-04: Parse JSON with preamble ────────────────────────────────────

@pytest.mark.asyncio
async def test_llm_client_parses_json_with_preamble(mocker: Any) -> None:
    """'Here is the output: {...}' is parsed correctly."""
    mocker.patch("asyncio.sleep", return_value=None)
    client = LLMClient(provider="openai", api_key="test-key")
    content = 'Here is the structured output: {"domain": "retail", "rows": 250} Thanks!'
    mocker.patch.object(
        client._http, "post", new_callable=AsyncMock,
        return_value=_openai_response(content),
    )

    raw = await client.complete("prompt")
    parsed = client.parse_json_response(raw)

    assert parsed["domain"] == "retail"
    assert parsed["rows"] == 250


# ── E-003-05: Temperature parameter forwarded ─────────────────────────────

@pytest.mark.asyncio
async def test_temperature_parameter_passed(mocker: Any) -> None:
    """temperature= is forwarded in the JSON payload."""
    mocker.patch("asyncio.sleep", return_value=None)
    client = LLMClient(provider="openai", api_key="test-key")
    mock_post = mocker.patch.object(
        client._http, "post", new_callable=AsyncMock,
        return_value=_openai_response('{"ok": true}'),
    )

    await client.complete("p", temperature=0.25)

    payload = mock_post.call_args.kwargs["json"]
    assert payload["temperature"] == 0.25


# ── E-003-06: Max tokens parameter forwarded ──────────────────────────────

@pytest.mark.asyncio
async def test_max_tokens_parameter_passed(mocker: Any) -> None:
    """max_tokens= is forwarded in the JSON payload."""
    mocker.patch("asyncio.sleep", return_value=None)
    client = LLMClient(provider="openai", api_key="test-key")
    mock_post = mocker.patch.object(
        client._http, "post", new_callable=AsyncMock,
        return_value=_openai_response('{"ok": true}'),
    )

    await client.complete("p", max_tokens=2048)

    payload = mock_post.call_args.kwargs["json"]
    assert payload["max_tokens"] == 2048


# ── E-003-07: Registry has all 3 providers ────────────────────────────────

def test_provider_registry_has_all_three() -> None:
    """MODEL_REGISTRY contains gemini, openai, and groq."""
    assert "gemini" in MODEL_REGISTRY
    assert "openai" in MODEL_REGISTRY
    assert "groq" in MODEL_REGISTRY


def test_provider_registry_has_required_keys() -> None:
    """Each provider entry has default_model, models, context_window, base_url."""
    for provider, entry in MODEL_REGISTRY.items():
        assert "default_model" in entry, f"{provider} missing default_model"
        assert "models" in entry, f"{provider} missing models"
        assert "context_window" in entry, f"{provider} missing context_window"
        assert "base_url" in entry, f"{provider} missing base_url"
        assert entry["default_model"] in entry["models"]


# ── E-003-08: Invalid provider raises LLMConfigError ─────────────────────

def test_invalid_provider_raises_config_error() -> None:
    """LLMClient(provider='invalid') raises LLMConfigError."""
    with pytest.raises(LLMConfigError, match="invalid"):
        LLMClient(provider="invalid", api_key="key")


def test_parse_json_response_raises_on_garbage() -> None:
    """parse_json_response raises LLMParseError on un-parseable text."""
    client = LLMClient(provider="openai", api_key="test-key")
    with pytest.raises(LLMParseError):
        client.parse_json_response("this is definitely not JSON at all")


def test_groq_client_uses_correct_base_url() -> None:
    """Groq client is configured with the Groq API base URL."""
    client = LLMClient(provider="groq", api_key="gsk_test")
    assert "groq.com" in client._registry["base_url"]
