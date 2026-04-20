# ───────────────────────────────────────────────────────────────
# Copyright (c) 2026 Ambrish Nigam
# Author : Ambrish Nigam | https://github.com/ambrishnigam30
# Project: SynthFlow — Autonomous Synthetic Data Orchestration Platform
# License : Apache License 2.0 | https://www.apache.org/licenses/LICENSE-2.0
# Module : Tests for CorrelationDriftEngine
# ───────────────────────────────────────────────────────────────

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from synthflow.engines.correlation_engine import CorrelationDriftEngine
from synthflow.models.schemas import CrossColumnCorrelation


# ── compute_actual_correlations ────────────────────────────────────────────


def test_correlation_engine_compute_returns_dict() -> None:
    """compute_actual_correlations returns a dict keyed by column-pair tuples."""
    engine = CorrelationDriftEngine()
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"x": rng.normal(size=50), "y": rng.normal(size=50)})
    result = engine.compute_actual_correlations(df)
    assert isinstance(result, dict)
    assert ("x", "y") in result or ("y", "x") in result


def test_correlation_engine_compute_high_correlation() -> None:
    """Strongly correlated columns produce |rho| close to 1."""
    engine = CorrelationDriftEngine()
    rng = np.random.default_rng(42)
    x = rng.normal(size=200)
    df = pd.DataFrame({"a": x, "b": x + rng.normal(scale=0.05, size=200)})
    result = engine.compute_actual_correlations(df)
    rho = result.get(("a", "b"), result.get(("b", "a"), 0.0))
    assert abs(rho) > 0.95


def test_correlation_engine_compute_low_correlation() -> None:
    """Independent columns produce |rho| close to 0."""
    engine = CorrelationDriftEngine()
    rng = np.random.default_rng(7)
    df = pd.DataFrame({
        "p": rng.normal(size=300),
        "q": rng.normal(size=300),
    })
    result = engine.compute_actual_correlations(df)
    rho = result.get(("p", "q"), result.get(("q", "p"), None))
    assert rho is not None
    assert abs(rho) < 0.20


def test_correlation_engine_compute_non_numeric_ignored() -> None:
    """Non-numeric columns are skipped; result still valid for numeric pairs."""
    engine = CorrelationDriftEngine()
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "name": ["Alice", "Bob", "Carol"] * 30,
        "age": rng.integers(20, 60, size=90),
        "salary": rng.lognormal(10, 0.5, size=90),
    })
    result = engine.compute_actual_correlations(df)
    assert isinstance(result, dict)
    # name column should not appear in results
    for pair in result.keys():
        assert "name" not in pair


# ── detect_drift ───────────────────────────────────────────────────────────


def test_correlation_engine_detects_drift() -> None:
    """Pair with actual_rho≈0.1 and target_rho=0.6 triggers a DriftEvent."""
    engine = CorrelationDriftEngine()
    rng = np.random.default_rng(42)
    n = 200
    x = rng.normal(size=n)
    y = x * 0.1 + rng.normal(size=n) * 0.99
    df = pd.DataFrame({"age": x, "bill_amount": y})

    actual = engine.compute_actual_correlations(df)
    targets = [CrossColumnCorrelation(col_a="age", col_b="bill_amount", strength=0.6)]
    drift_events = engine.detect_drift(actual, targets, threshold=0.15)
    assert len(drift_events) > 0


def test_correlation_engine_no_drift_when_close() -> None:
    """Actual rho close to target rho produces no drift events."""
    engine = CorrelationDriftEngine()
    rng = np.random.default_rng(42)
    n = 200
    x = rng.normal(size=n)
    # Produce actual rho ≈ 0.55; target = 0.5 → delta < 0.15
    y = x * 0.6 + rng.normal(size=n) * 0.8
    df = pd.DataFrame({"a": x, "b": y})
    actual = engine.compute_actual_correlations(df)
    targets = [CrossColumnCorrelation(col_a="a", col_b="b", strength=0.5)]
    drift_events = engine.detect_drift(actual, targets, threshold=0.15)
    assert len(drift_events) == 0


def test_correlation_engine_missing_column_skipped() -> None:
    """Target pair referencing absent column is silently skipped."""
    engine = CorrelationDriftEngine()
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"x": rng.normal(size=50)})
    actual = engine.compute_actual_correlations(df)
    targets = [CrossColumnCorrelation(col_a="x", col_b="nonexistent", strength=0.8)]
    drift_events = engine.detect_drift(actual, targets, threshold=0.15)
    # Should not raise; column pair absent from actual → no event
    assert isinstance(drift_events, list)


def test_correlation_engine_drift_event_fields() -> None:
    """DriftEvent exposes col_a, col_b, actual_rho, target_rho, delta."""
    engine = CorrelationDriftEngine()
    rng = np.random.default_rng(42)
    n = 200
    x = rng.normal(size=n)
    y = rng.normal(size=n)  # uncorrelated
    df = pd.DataFrame({"x": x, "y": y})
    actual = engine.compute_actual_correlations(df)
    targets = [CrossColumnCorrelation(col_a="x", col_b="y", strength=0.9)]
    events = engine.detect_drift(actual, targets, threshold=0.15)
    assert len(events) > 0
    ev = events[0]
    assert ev.col_a == "x"
    assert ev.col_b == "y"
    assert 0.0 <= ev.delta <= 2.0
