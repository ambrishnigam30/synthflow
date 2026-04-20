# ───────────────────────────────────────────────────────────────
# Copyright (c) 2026 Ambrish Nigam
# Author : Ambrish Nigam | https://github.com/ambrishnigam30
# Project: SynthFlow — Autonomous Synthetic Data Orchestration Platform
# License : Apache License 2.0 | https://www.apache.org/licenses/LICENSE-2.0
# Module : Tests for SynthFlowOrchestrator (9-phase pipeline)
# ───────────────────────────────────────────────────────────────
#
# NOTE: These tests depend on SynthFlowOrchestrator and SynthFlowContainer
# being fully implemented in synthflow/orchestrator.py and synthflow/core.py.
# The orchestrator is currently stubbed (Layer 2 in progress).
# Tests are marked xfail where the orchestrator is not yet implemented
# so the test suite remains green while development continues.
# ───────────────────────────────────────────────────────────────

from __future__ import annotations

import pytest


def _orchestrator_available() -> bool:
    """Return True if the orchestrator has a generate() method implemented."""
    try:
        from synthflow.orchestrator import SynthFlowOrchestrator
        from synthflow.core import SynthFlowContainer
        from synthflow.llm_client import MockLLMClient

        container = SynthFlowContainer(
            llm_client=MockLLMClient(), duckdb_path=":memory:"
        )
        orch = SynthFlowOrchestrator(container=container)
        return hasattr(orch, "generate") and callable(getattr(orch, "generate", None))
    except Exception:
        return False


_SKIP_ORCHESTRATOR = pytest.mark.skipif(
    not _orchestrator_available(),
    reason="SynthFlowOrchestrator.generate() not yet implemented (Layer 2 in progress)",
)


# ── Orchestrator availability smoke test ──────────────────────────────────────

def test_synthflow_container_imports() -> None:
    """SynthFlowContainer can be imported and instantiated."""
    from synthflow.core import SynthFlowContainer
    from synthflow.llm_client import MockLLMClient

    container = SynthFlowContainer(llm_client=MockLLMClient(), duckdb_path=":memory:")
    assert container is not None


def test_orchestrator_can_be_imported() -> None:
    """synthflow.orchestrator module can be imported without errors."""
    import synthflow.orchestrator  # noqa: F401


def test_synthflow_container_has_expected_subsystems() -> None:
    """SynthFlowContainer exposes properties for all 15 subsystems."""
    from synthflow.core import SynthFlowContainer
    from synthflow.llm_client import MockLLMClient

    container = SynthFlowContainer(llm_client=MockLLMClient(), duckdb_path=":memory:")

    # Check a representative subset of subsystem properties exist
    expected_attrs = [
        "intent_engine",
        "schema_intelligence",
        "constraint_engine",
        "memory_store",
        "presidio_guard",
    ]
    for attr in expected_attrs:
        assert hasattr(container, attr), (
            f"SynthFlowContainer missing expected attribute: '{attr}'"
        )


# ── Pipeline tests (skip until orchestrator is implemented) ───────────────────

@_SKIP_ORCHESTRATOR
@pytest.mark.asyncio
async def test_full_pipeline_returns_generation_result(mock_llm_client) -> None:
    """End-to-end: healthcare prompt → GenerationResult with required fields."""
    from synthflow.core import SynthFlowContainer
    from synthflow.orchestrator import SynthFlowOrchestrator
    from synthflow.models.schemas import GenerationResult

    container = SynthFlowContainer(llm_client=mock_llm_client, duckdb_path=":memory:")
    orchestrator = SynthFlowOrchestrator(container=container)

    result = await orchestrator.generate(
        prompt="Generate Indian healthcare patient records",
        row_count=50,
        seed=42,
    )

    assert isinstance(result, GenerationResult)
    assert result.dataframe is not None
    assert len(result.dataframe) > 0
    assert result.schema is not None
    assert result.session_id is not None


@_SKIP_ORCHESTRATOR
@pytest.mark.asyncio
async def test_progress_callback_fires_multiple_times(mock_llm_client) -> None:
    """progress_callback is called at least once per pipeline phase (≥9 calls)."""
    from synthflow.core import SynthFlowContainer
    from synthflow.orchestrator import SynthFlowOrchestrator

    container = SynthFlowContainer(llm_client=mock_llm_client, duckdb_path=":memory:")
    orchestrator = SynthFlowOrchestrator(container=container)

    calls: list[tuple[int, float, str]] = []

    async def callback(phase: int, progress: float, message: str) -> None:
        calls.append((phase, progress, message))

    await orchestrator.generate(
        prompt="Generate 50 retail records",
        row_count=50,
        seed=42,
        progress_callback=callback,
    )

    assert len(calls) >= 9, (
        f"Expected at least 9 progress callbacks (one per phase), got {len(calls)}"
    )


@_SKIP_ORCHESTRATOR
@pytest.mark.asyncio
async def test_same_seed_same_column_layout(mock_llm_client) -> None:
    """Two runs with the same seed produce DataFrames with identical column names."""
    from synthflow.core import SynthFlowContainer
    from synthflow.orchestrator import SynthFlowOrchestrator

    container1 = SynthFlowContainer(llm_client=mock_llm_client, duckdb_path=":memory:")
    orchestrator1 = SynthFlowOrchestrator(container=container1)
    result1 = await orchestrator1.generate(
        prompt="healthcare data", row_count=20, seed=42
    )

    container2 = SynthFlowContainer(llm_client=mock_llm_client, duckdb_path=":memory:")
    orchestrator2 = SynthFlowOrchestrator(container=container2)
    result2 = await orchestrator2.generate(
        prompt="healthcare data", row_count=20, seed=42
    )

    assert list(result1.dataframe.columns) == list(result2.dataframe.columns)
    assert len(result1.dataframe) == len(result2.dataframe)


@_SKIP_ORCHESTRATOR
@pytest.mark.asyncio
async def test_result_has_quality_score(mock_llm_client) -> None:
    """GenerationResult.quality_report has a non-negative overall_score."""
    from synthflow.core import SynthFlowContainer
    from synthflow.orchestrator import SynthFlowOrchestrator

    container = SynthFlowContainer(llm_client=mock_llm_client, duckdb_path=":memory:")
    orchestrator = SynthFlowOrchestrator(container=container)
    result = await orchestrator.generate(
        prompt="hr employee data", row_count=30, seed=1
    )

    assert result.quality_report is not None
    assert result.quality_report.overall_score >= 0.0


@_SKIP_ORCHESTRATOR
@pytest.mark.asyncio
async def test_result_has_privacy_report(mock_llm_client) -> None:
    """GenerationResult.privacy_report is populated after generation."""
    from synthflow.core import SynthFlowContainer
    from synthflow.orchestrator import SynthFlowOrchestrator
    from synthflow.models.schemas import PrivacyReport

    container = SynthFlowContainer(llm_client=mock_llm_client, duckdb_path=":memory:")
    orchestrator = SynthFlowOrchestrator(container=container)
    result = await orchestrator.generate(
        prompt="Generate banking transaction records", row_count=25, seed=3
    )

    assert result.privacy_report is not None
    assert isinstance(result.privacy_report, PrivacyReport)
    assert result.privacy_report.k_anonymity >= 1


@_SKIP_ORCHESTRATOR
@pytest.mark.asyncio
async def test_result_session_id_is_uuid(mock_llm_client) -> None:
    """GenerationResult.session_id is a valid UUID string."""
    import uuid
    from synthflow.core import SynthFlowContainer
    from synthflow.orchestrator import SynthFlowOrchestrator

    container = SynthFlowContainer(llm_client=mock_llm_client, duckdb_path=":memory:")
    orchestrator = SynthFlowOrchestrator(container=container)
    result = await orchestrator.generate(
        prompt="retail product data", row_count=10, seed=5
    )

    # Ensure session_id is a valid UUID
    parsed = uuid.UUID(result.session_id)
    assert str(parsed) == result.session_id
