# ───────────────────────────────────────────────────────────────
# Copyright (c) 2026 Ambrish Nigam
# Author : Ambrish Nigam | https://github.com/ambrishnigam30
# Project: SynthFlow — Autonomous Synthetic Data Orchestration Platform
# License : Apache License 2.0 | https://www.apache.org/licenses/LICENSE-2.0
# Module : Tests for PatternRhythmLibrary
# ───────────────────────────────────────────────────────────────

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from synthflow.engines.pattern_library import PatternRhythmLibrary
from synthflow.models.schemas import TemporalPatterns


# ── apply_temporal_patterns ────────────────────────────────────────────────


def test_pattern_library_applies_temporal_without_error(sample_knowledge_bundle) -> None:
    """apply_temporal_patterns runs without exception and preserves row count."""
    library = PatternRhythmLibrary()
    rng = np.random.default_rng(42)
    n = 100
    df = pd.DataFrame({
        "created_at": pd.date_range("2023-01-01", periods=n, freq="3h"),
        "revenue": rng.lognormal(10, 1, size=n),
    })
    result = library.apply_temporal_patterns(
        df, sample_knowledge_bundle.temporal_patterns, timestamp_columns=["created_at"]
    )
    assert result is not None
    assert len(result) == n


def test_pattern_library_empty_ts_columns_returns_unchanged() -> None:
    """Empty timestamp_columns list → DataFrame returned unchanged."""
    library = PatternRhythmLibrary()
    df = pd.DataFrame({"revenue": [100, 200, 300]})
    patterns = TemporalPatterns()
    result = library.apply_temporal_patterns(df, patterns, timestamp_columns=[])
    assert len(result) == 3
    assert list(result["revenue"]) == [100, 200, 300]


def test_pattern_library_missing_ts_column_ignored() -> None:
    """A timestamp_column that doesn't exist in df is skipped gracefully."""
    library = PatternRhythmLibrary()
    df = pd.DataFrame({"revenue": [1.0, 2.0, 3.0]})
    patterns = TemporalPatterns(day_of_week_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 0.3])
    result = library.apply_temporal_patterns(df, patterns, timestamp_columns=["nonexistent_col"])
    assert len(result) == 3


def test_pattern_library_dow_weights_preserve_row_count() -> None:
    """DOW resampling keeps the same number of rows."""
    library = PatternRhythmLibrary()
    n = 50
    df = pd.DataFrame({
        "event_at": pd.date_range("2023-06-01", periods=n, freq="1D"),
        "amount": np.random.default_rng(1).lognormal(5, 1, size=n),
    })
    patterns = TemporalPatterns(day_of_week_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 0.3])
    result = library.apply_temporal_patterns(df, patterns, timestamp_columns=["event_at"])
    assert len(result) == n


def test_pattern_library_result_has_same_columns() -> None:
    """apply_temporal_patterns does not drop or add any columns."""
    library = PatternRhythmLibrary()
    n = 20
    df = pd.DataFrame({
        "ts": pd.date_range("2023-01-01", periods=n, freq="1D"),
        "val": range(n),
    })
    patterns = TemporalPatterns()
    result = library.apply_temporal_patterns(df, patterns, timestamp_columns=["ts"])
    assert set(result.columns) == set(df.columns)


# ── apply_autocorrelation ──────────────────────────────────────────────────


def test_pattern_library_apply_autocorrelation_preserves_rows() -> None:
    """apply_autocorrelation returns DataFrame with same number of rows."""
    library = PatternRhythmLibrary()
    rng = np.random.default_rng(42)
    n = 100
    df = pd.DataFrame({
        "created_at": pd.date_range("2023-01-01", periods=n, freq="1h"),
        "revenue": rng.normal(1000, 100, size=n),
    })
    result = library.apply_autocorrelation(df, rho=0.8, ts_columns=["revenue"], sort_by="created_at")
    assert result is not None
    assert len(result) == n
    assert "revenue" in result.columns


def test_pattern_library_autocorrelation_zero_rho_returns_unchanged() -> None:
    """rho=0 is a no-op — DataFrame returned as-is."""
    library = PatternRhythmLibrary()
    df = pd.DataFrame({
        "ts": pd.date_range("2023-01-01", periods=10, freq="1D"),
        "val": list(range(10)),
    })
    result = library.apply_autocorrelation(df, rho=0.0, ts_columns=["ts"])
    assert len(result) == 10


def test_pattern_library_autocorrelation_preserves_columns() -> None:
    """apply_autocorrelation preserves all column names."""
    library = PatternRhythmLibrary()
    rng = np.random.default_rng(5)
    n = 30
    df = pd.DataFrame({
        "date": pd.date_range("2023-01-01", periods=n, freq="1D"),
        "sales": rng.normal(500, 50, size=n),
        "cost": rng.normal(300, 30, size=n),
    })
    result = library.apply_autocorrelation(df, rho=0.5, ts_columns=["sales", "cost"], sort_by="date")
    assert set(result.columns) == set(df.columns)
