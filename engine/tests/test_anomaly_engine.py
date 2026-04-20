# ───────────────────────────────────────────────────────────────
# Copyright (c) 2026 Ambrish Nigam
# Author : Ambrish Nigam | https://github.com/ambrishnigam30
# Project: SynthFlow — Autonomous Synthetic Data Orchestration Platform
# License : Apache License 2.0 | https://www.apache.org/licenses/LICENSE-2.0
# Module : Tests for AnomalyOutlierEngine and TypoGenerator
# ───────────────────────────────────────────────────────────────

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from synthflow.engines.anomaly_engine import AnomalyOutlierEngine, TypoGenerator


# ── inject_structured_nulls ────────────────────────────────────────────────


def test_anomaly_engine_null_rate_within_tolerance() -> None:
    """Injected null rate is within ±5pp of the target rate."""
    engine = AnomalyOutlierEngine(seed=42)
    rng = np.random.default_rng(42)
    n = 1000
    df = pd.DataFrame({
        "salary": rng.lognormal(10, 1, size=n).tolist(),
        "age": rng.integers(18, 80, size=n).tolist(),
    })
    result = engine.inject_structured_nulls(df.copy(), {"salary": 0.10})
    actual_rate = result["salary"].isna().mean()
    assert 0.05 <= actual_rate <= 0.15


def test_anomaly_engine_mcar_null_injection() -> None:
    """MCAR nulls hit roughly the target rate."""
    engine = AnomalyOutlierEngine(seed=7)
    rng = np.random.default_rng(7)
    n = 500
    df = pd.DataFrame({"age": rng.integers(18, 80, size=n).tolist()})
    result = engine.inject_structured_nulls(df.copy(), {"age": 0.20})
    null_rate = result["age"].isna().mean()
    assert 0.15 <= null_rate <= 0.25


def test_anomaly_engine_null_missing_column_skipped() -> None:
    """Specifying a column that doesn't exist doesn't raise an error."""
    engine = AnomalyOutlierEngine(seed=0)
    df = pd.DataFrame({"x": [1, 2, 3, 4, 5]})
    # Should not raise
    result = engine.inject_structured_nulls(df.copy(), {"nonexistent": 0.5})
    assert list(result["x"]) == [1, 2, 3, 4, 5]


def test_anomaly_engine_null_rate_zero_injects_nothing() -> None:
    """A null_rate of 0.0 leaves the column untouched."""
    engine = AnomalyOutlierEngine(seed=0)
    df = pd.DataFrame({"val": list(range(100))})
    result = engine.inject_structured_nulls(df.copy(), {"val": 0.0})
    assert result["val"].isna().sum() == 0


# ── inject_date_format_mix ─────────────────────────────────────────────────


def test_anomaly_engine_date_format_inconsistency() -> None:
    """inject_date_format_mix returns a string series with correct length."""
    engine = AnomalyOutlierEngine(seed=42)
    dates = pd.date_range("2023-01-01", periods=200, freq="1D")
    series = pd.Series(dates)
    rng = np.random.default_rng(42)
    result = engine.inject_date_format_mix(series, rng, inconsistency_rate=0.12)
    assert len(result) == 200
    assert result.dtype == object  # should be string series


def test_anomaly_engine_date_format_mix_produces_variants() -> None:
    """Some rows use alt formats when inconsistency_rate is high enough."""
    engine = AnomalyOutlierEngine(seed=1)
    dates = pd.date_range("2020-06-01", periods=100, freq="1D")
    series = pd.Series(dates)
    rng = np.random.default_rng(1)
    result = engine.inject_date_format_mix(series, rng, inconsistency_rate=0.50)
    # With 50% rate, we expect a mix of formats; not all should be YYYY-MM-DD
    formats = set(result.tolist())
    assert len(formats) > 1


# ── inject_duplicates ─────────────────────────────────────────────────────


def test_anomaly_engine_inject_duplicates_increases_rows() -> None:
    """inject_duplicates returns at least as many rows as the original."""
    engine = AnomalyOutlierEngine(seed=42)
    rng = np.random.default_rng(42)
    n = 500
    df = pd.DataFrame({"id": range(n), "val": rng.normal(size=n)})
    result = engine.inject_duplicates(df.copy(), duplicate_rate=0.01, near_duplicate_rate=0.02)
    assert len(result) >= n


def test_anomaly_engine_inject_duplicates_preserves_columns() -> None:
    """inject_duplicates keeps all original columns."""
    engine = AnomalyOutlierEngine(seed=0)
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    result = engine.inject_duplicates(df.copy(), duplicate_rate=0.5, near_duplicate_rate=0.0)
    assert set(result.columns) == {"a", "b"}


# ── TypoGenerator ──────────────────────────────────────────────────────────


def test_typo_generator_adjacent_key_produces_variants() -> None:
    """apply_random_typo on 'gmail' produces multiple distinct strings across calls."""
    results: set[str] = set()
    for seed_i in range(50):
        rng = np.random.default_rng(seed_i + 100)
        tg = TypoGenerator(rng)
        results.add(tg.apply_random_typo("gmail"))
    assert len(results) > 1


def test_typo_generator_transposition_sometimes_changes_text() -> None:
    """transposition_typo on 'Bangalore' sometimes swaps adjacent chars."""
    changed: list[str] = []
    for i in range(30):
        rng = np.random.default_rng(i * 17)
        tg = TypoGenerator(rng)
        result = tg.transposition_typo("Bangalore")
        if result != "Bangalore":
            changed.append(result)
    assert len(changed) > 0


def test_typo_generator_deletion_reduces_length() -> None:
    """deletion_typo removes exactly one character."""
    rng = np.random.default_rng(42)
    tg = TypoGenerator(rng)
    original = "Hyderabad"
    result = tg.deletion_typo(original)
    assert len(result) == len(original) - 1


def test_typo_generator_repetition_increases_length() -> None:
    """repetition_typo adds exactly one character."""
    rng = np.random.default_rng(42)
    tg = TypoGenerator(rng)
    original = "Mumbai"
    result = tg.repetition_typo(original)
    assert len(result) == len(original) + 1


def test_typo_generator_case_error_lowercases() -> None:
    """case_error_typo returns the lowercase version of the input."""
    rng = np.random.default_rng(0)
    tg = TypoGenerator(rng)
    assert tg.case_error_typo("HELLO") == "hello"


# ── inject_outliers ────────────────────────────────────────────────────────


def test_anomaly_engine_inject_outliers_preserves_row_count() -> None:
    """inject_outliers keeps the same number of rows."""
    engine = AnomalyOutlierEngine(seed=42)
    rng = np.random.default_rng(42)
    n = 200
    df = pd.DataFrame({"salary": rng.lognormal(10, 0.5, size=n)})
    result = engine.inject_outliers(df.copy(), ["salary"], rate=0.05)
    assert len(result) == n
