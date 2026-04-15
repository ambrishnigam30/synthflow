# ───────────────────────────────────────────────────────────────
# Copyright (c) 2026 Ambrish Nigam
# Author : Ambrish Nigam | https://github.com/ambrishnigam30
# Project: SynthFlow — Autonomous Synthetic Data Orchestration Platform
# License : Apache License 2.0 | https://www.apache.org/licenses/LICENSE-2.0
# Module : Tests E-005-01 through E-005-08 — CognitiveIntentEngine tests
# ───────────────────────────────────────────────────────────────

from __future__ import annotations

import json

import pytest

from synthflow.engines.intent_engine import (
    CognitiveIntentEngine,
    DOMAIN_KEYWORD_MAP,
    _compute_seed,
)
from synthflow.engines.memory_store import MemoryContextStore
from synthflow.llm_client import MockLLMClient
from synthflow.models.schemas import IntentObject


# ── Helpers ────────────────────────────────────────────────────────────────

def _engine_with_garbage_llm() -> CognitiveIntentEngine:
    """Engine whose LLM always returns garbage → triggers keyword fallback."""
    client = MockLLMClient()
    client.set_response("", "this is not json at all!!!")
    return CognitiveIntentEngine(client)


def _engine_with_good_llm(domain: str = "healthcare", row_count: int = 500) -> CognitiveIntentEngine:
    """Engine whose LLM returns a well-formed IntentObject JSON."""
    client = MockLLMClient()
    resp = json.dumps({
        "domain": domain,
        "sub_domain": None,
        "row_count": row_count,
        "country": "India",
        "state_province": "Maharashtra",
        "city": None,
        "economic_tier": None,
        "scenario": None,
        "output_format": "csv",
    })
    # Override for any prompt (use empty string key to match everything)
    client._overrides[""] = resp  # type: ignore[attr-defined]
    return CognitiveIntentEngine(client)


# ── E-005-01: Domain extraction ────────────────────────────────────────────

@pytest.mark.asyncio
async def test_intent_engine_extracts_domain_from_healthcare_prompt() -> None:
    """'Generate Indian healthcare records' → domain='healthcare' via fallback."""
    engine = _engine_with_garbage_llm()
    intent = await engine.parse("Generate Indian healthcare patient records")
    assert intent.domain == "healthcare"


@pytest.mark.asyncio
async def test_intent_engine_extracts_banking_domain() -> None:
    """'Bank loan transaction data' → domain='banking'."""
    engine = _engine_with_garbage_llm()
    intent = await engine.parse("Generate bank loan transaction data for customers")
    assert intent.domain == "banking"


@pytest.mark.asyncio
async def test_intent_engine_extracts_hr_domain() -> None:
    """'employee payroll salary records' → domain='hr'."""
    engine = _engine_with_garbage_llm()
    intent = await engine.parse("Generate employee payroll and salary data")
    assert intent.domain == "hr"


# ── E-005-02: Row count extraction ────────────────────────────────────────

@pytest.mark.asyncio
async def test_intent_engine_extracts_row_count_5000() -> None:
    """'Generate 5000 records' → row_count=5000."""
    engine = _engine_with_garbage_llm()
    intent = await engine.parse("Generate 5000 healthcare records for Indian patients")
    assert intent.row_count == 5000


@pytest.mark.asyncio
async def test_intent_engine_extracts_row_count_with_comma() -> None:
    """'10,000 records' → row_count=10000."""
    engine = _engine_with_garbage_llm()
    intent = await engine.parse("Create 10,000 rows of employee data")
    assert intent.row_count == 10000


@pytest.mark.asyncio
async def test_intent_engine_extracts_row_count_generate_verb() -> None:
    """'Generate 250 samples' → row_count=250."""
    engine = _engine_with_garbage_llm()
    intent = await engine.parse("Generate 250 samples of customer purchase data")
    assert intent.row_count == 250


# ── E-005-03: Region extraction ───────────────────────────────────────────

@pytest.mark.asyncio
async def test_intent_engine_extracts_india_region() -> None:
    """'Indian healthcare data' → region.country='India'."""
    engine = _engine_with_garbage_llm()
    intent = await engine.parse("Generate Indian healthcare patient data")
    assert intent.region is not None
    assert intent.region.country == "India"


@pytest.mark.asyncio
async def test_intent_engine_extracts_us_region() -> None:
    """'American retail data' → region.country='United States'."""
    engine = _engine_with_garbage_llm()
    intent = await engine.parse("Generate American retail customer purchase data")
    assert intent.region is not None
    assert intent.region.country == "United States"


@pytest.mark.asyncio
async def test_intent_engine_extracts_maharashtra_state() -> None:
    """'Maharashtra patient data' → region.state_province='Maharashtra', country='India'."""
    engine = _engine_with_garbage_llm()
    intent = await engine.parse("Generate Maharashtra hospital patient records")
    assert intent.region is not None
    assert intent.region.country == "India"
    assert intent.region.state_province == "Maharashtra"


# ── E-005-04: Fallback on LLM failure ─────────────────────────────────────

@pytest.mark.asyncio
async def test_intent_engine_falls_back_on_llm_garbage() -> None:
    """LLM returns garbage → keyword classifier produces valid IntentObject."""
    engine = _engine_with_garbage_llm()
    intent = await engine.parse("Generate insurance policy claim records for India")
    assert isinstance(intent, IntentObject)
    assert intent.domain in DOMAIN_KEYWORD_MAP
    assert intent.row_count > 0


@pytest.mark.asyncio
async def test_intent_engine_falls_back_produces_valid_intent() -> None:
    """Fallback always returns a valid IntentObject with seed and row_count."""
    engine = _engine_with_garbage_llm()
    intent = await engine.parse("Create some data")
    assert isinstance(intent, IntentObject)
    assert isinstance(intent.seed, int)
    assert intent.row_count >= 1


# ── E-005-05: Deterministic seed ──────────────────────────────────────────

@pytest.mark.asyncio
async def test_intent_engine_deterministic_seed_same_prompt() -> None:
    """Same prompt always produces same seed."""
    engine = _engine_with_garbage_llm()
    prompt = "Generate 500 Indian healthcare patient records"
    r1 = await engine.parse(prompt)
    r2 = await engine.parse(prompt)
    assert r1.seed == r2.seed


def test_compute_seed_deterministic() -> None:
    """_compute_seed is a pure function — same input → same output."""
    assert _compute_seed("hello") == _compute_seed("hello")
    assert _compute_seed("hello") != _compute_seed("world")


def test_compute_seed_within_int31() -> None:
    """Seed must be a valid positive 31-bit integer."""
    seed = _compute_seed("Generate 5000 Indian healthcare records")
    assert 0 <= seed < 2**31


# ── E-005-06: Cache hit ────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_intent_engine_cache_hit_returns_same_object() -> None:
    """Second call with same prompt uses cache, call_count does not increment."""
    client = MockLLMClient()
    store = MemoryContextStore(":memory:")
    engine = CognitiveIntentEngine(client, memory_store=store)

    prompt = "Generate 100 Indian healthcare records"
    r1 = await engine.parse(prompt)
    calls_after_first = client.call_count

    r2 = await engine.parse(prompt)
    # LLM should not be called again
    assert client.call_count == calls_after_first
    assert r1.domain == r2.domain
    assert r1.seed == r2.seed


# ── E-005-07: Default row count ────────────────────────────────────────────

@pytest.mark.asyncio
async def test_intent_engine_default_row_count_1000() -> None:
    """'Generate healthcare data' (no number) → row_count=1000."""
    engine = _engine_with_garbage_llm()
    intent = await engine.parse("Generate healthcare data for Indian hospitals")
    assert intent.row_count == 1000


# ── E-005-08: Multi-word domain ───────────────────────────────────────────

@pytest.mark.asyncio
async def test_intent_engine_real_estate_domain() -> None:
    """'real estate property listings' → domain='real_estate'."""
    engine = _engine_with_garbage_llm()
    intent = await engine.parse("Generate real estate property listing data")
    assert intent.domain == "real_estate"


@pytest.mark.asyncio
async def test_intent_engine_llm_domain_used_when_valid() -> None:
    """When LLM returns a valid domain JSON, it takes precedence over keywords."""
    engine = _engine_with_good_llm(domain="banking", row_count=200)
    intent = await engine.parse("Generate some data please")
    assert intent.domain == "banking"
    assert intent.row_count == 200
