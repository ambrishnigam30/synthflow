# ───────────────────────────────────────────────────────────────
# Copyright (c) 2026 Ambrish Nigam
# Author : Ambrish Nigam | https://github.com/ambrishnigam30
# Project: SynthFlow — Autonomous Synthetic Data Orchestration Platform
# License : Apache License 2.0 | https://www.apache.org/licenses/LICENSE-2.0
# Module : Tests for MemoryContextStore
# ───────────────────────────────────────────────────────────────

from __future__ import annotations

import uuid

import pytest

from synthflow.models.schemas import (
    GenerationResult,
    IntentObject,
    RegionInfo,
)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_result(
    session_id: str,
    domain: str = "healthcare",
    row_count: int = 100,
    seed: int = 42,
    quality_score: float = 85.0,
) -> GenerationResult:
    """Build a minimal GenerationResult for store tests."""
    from synthflow.models.schemas import QualityReport

    intent = IntentObject(
        domain=domain,
        region=RegionInfo(country="India"),
        row_count=row_count,
        seed=seed,
    )
    qr = QualityReport(overall_score=quality_score)
    return GenerationResult(
        session_id=session_id,
        intent=intent,
        row_count=row_count,
        seed=seed,
        quality_report=qr,
    )


# ── Tests ──────────────────────────────────────────────────────────────────────

def test_memory_store_save_and_get_session() -> None:
    """save_session then get_session returns the same session_id."""
    from synthflow.engines.memory_store import MemoryContextStore

    store = MemoryContextStore(db_path=":memory:")
    session_id = str(uuid.uuid4())
    result = _make_result(session_id=session_id)

    store.save_session(result=result, prompt="test healthcare data")

    record = store.get_session(session_id)
    assert record is not None
    assert record["session_id"] == session_id


def test_memory_store_get_session_unknown_returns_none() -> None:
    """get_session with unknown ID returns None."""
    from synthflow.engines.memory_store import MemoryContextStore

    store = MemoryContextStore(db_path=":memory:")
    result = store.get_session("does-not-exist")
    assert result is None


def test_memory_store_list_sessions() -> None:
    """After saving 3 sessions, list_sessions returns 3 records."""
    from synthflow.engines.memory_store import MemoryContextStore

    store = MemoryContextStore(db_path=":memory:")

    for i in range(3):
        sid = str(uuid.uuid4())
        result = _make_result(session_id=sid, quality_score=80.0 + i)
        store.save_session(result=result, prompt=f"prompt {i}")

    sessions = store.list_sessions(limit=10)
    assert len(sessions) == 3


def test_memory_store_list_sessions_respects_limit() -> None:
    """list_sessions(limit=2) returns at most 2 records even if 5 exist."""
    from synthflow.engines.memory_store import MemoryContextStore

    store = MemoryContextStore(db_path=":memory:")

    for _ in range(5):
        sid = str(uuid.uuid4())
        store.save_session(result=_make_result(session_id=sid), prompt="x")

    sessions = store.list_sessions(limit=2)
    assert len(sessions) <= 2


def test_memory_store_cache_and_get_intent() -> None:
    """cache_intent then get_cached_intent returns the same domain."""
    from synthflow.engines.memory_store import MemoryContextStore

    store = MemoryContextStore(db_path=":memory:")

    prompt_hash = "abc123test"
    intent = IntentObject(
        domain="retail",
        region=RegionInfo(country="US"),
        row_count=500,
        seed=99,
    )

    store.cache_intent(prompt_hash, intent)
    cached = store.get_cached_intent(prompt_hash)

    assert cached is not None
    assert cached.domain == "retail"


def test_memory_store_get_cached_intent_miss_returns_none() -> None:
    """get_cached_intent with unknown hash returns None."""
    from synthflow.engines.memory_store import MemoryContextStore

    store = MemoryContextStore(db_path=":memory:")
    result = store.get_cached_intent("no_such_hash")
    assert result is None


def test_memory_store_delete_session() -> None:
    """delete_session removes the record; get_session returns None afterwards."""
    from synthflow.engines.memory_store import MemoryContextStore

    store = MemoryContextStore(db_path=":memory:")
    session_id = str(uuid.uuid4())

    store.save_session(
        result=_make_result(session_id=session_id),
        prompt="test",
    )

    # Confirm it exists first
    assert store.get_session(session_id) is not None

    store.delete_session(session_id)
    result = store.get_session(session_id)
    assert result is None


def test_memory_store_session_record_fields() -> None:
    """Saved session record has expected keys and correct domain."""
    from synthflow.engines.memory_store import MemoryContextStore

    store = MemoryContextStore(db_path=":memory:")
    session_id = str(uuid.uuid4())

    result = _make_result(session_id=session_id, domain="finance", row_count=200)
    store.save_session(result=result, prompt="financial data")

    record = store.get_session(session_id)
    assert record is not None
    assert record["domain"] == "finance"
    assert record["row_count"] == 200
    assert record["session_id"] == session_id


def test_memory_store_cache_knowledge_and_retrieve() -> None:
    """cache_knowledge then get_cached_knowledge returns the correct domain."""
    from synthflow.engines.memory_store import MemoryContextStore
    from synthflow.models.schemas import CausalKnowledgeBundle

    store = MemoryContextStore(db_path=":memory:")

    bundle = CausalKnowledgeBundle(
        domain="education",
        region=RegionInfo(country="India"),
    )
    bundle_key = "edu_india_v1"

    store.cache_knowledge(bundle_key, bundle)
    retrieved = store.get_cached_knowledge(bundle_key)

    assert retrieved is not None
    assert retrieved.domain == "education"
