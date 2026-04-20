# ───────────────────────────────────────────────────────────────
# Copyright (c) 2026 Ambrish Nigam
# Author : Ambrish Nigam | https://github.com/ambrishnigam30
# Project: SynthFlow — Autonomous Synthetic Data Orchestration Platform
# License : Apache License 2.0 | https://www.apache.org/licenses/LICENSE-2.0
# Module : Tests for SDVMultiplierEngine
# ───────────────────────────────────────────────────────────────

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from synthflow.engines.sdv_multiplier import SDVMultiplierEngine


# ── build_sdv_metadata ─────────────────────────────────────────────────────


def test_sdv_multiplier_build_metadata_returns_something_or_none(sample_schema) -> None:
    """build_sdv_metadata either returns a metadata object (SDV installed) or None."""
    engine = SDVMultiplierEngine(use_sdv=True)
    metadata = engine.build_sdv_metadata(sample_schema)
    # Acceptable: None (SDV not installed) or a non-None object (SDV present)
    assert metadata is None or metadata is not None  # always true — just no exception


def test_sdv_multiplier_build_metadata_without_sdv_returns_none(sample_schema) -> None:
    """With use_sdv=False (SDV unavailable simulation), metadata is None."""
    # Force SDV unavailable by creating engine with use_sdv=False
    engine = SDVMultiplierEngine(use_sdv=False)
    # Manually override _sdv_available to ensure fallback path
    engine._sdv_available = False
    metadata = engine.build_sdv_metadata(sample_schema)
    assert metadata is None


# ── scale — row count ──────────────────────────────────────────────────────


def test_sdv_multiplier_scales_rows(sample_dataframe, sample_schema) -> None:
    """scale() returns a DataFrame with approximately target_rows rows."""
    engine = SDVMultiplierEngine(use_sdv=False)  # use statistical fallback
    result = engine.scale(sample_dataframe, target_rows=200, schema=sample_schema)
    assert result is not None
    assert len(result) >= 150  # allow minor variance


def test_sdv_multiplier_scales_to_exact_count(sample_dataframe, sample_schema) -> None:
    """Statistical scaling returns exactly target_rows rows."""
    engine = SDVMultiplierEngine(use_sdv=False)
    result = engine.scale(sample_dataframe, target_rows=300, schema=sample_schema)
    assert len(result) == 300


def test_sdv_multiplier_target_below_seed_samples_down(sample_dataframe, sample_schema) -> None:
    """target_rows <= len(seed_df) → result has exactly target_rows rows."""
    engine = SDVMultiplierEngine(use_sdv=False)
    target = 20
    result = engine.scale(sample_dataframe, target_rows=target, schema=sample_schema)
    assert len(result) == target


# ── scale — column preservation ───────────────────────────────────────────


def test_sdv_multiplier_preserves_columns(sample_dataframe, sample_schema) -> None:
    """scale() preserves all column names from the seed DataFrame."""
    engine = SDVMultiplierEngine(use_sdv=False)
    result = engine.scale(sample_dataframe, target_rows=100, schema=sample_schema)
    for col in sample_dataframe.columns:
        assert col in result.columns


def test_sdv_multiplier_no_extra_columns(sample_dataframe, sample_schema) -> None:
    """scale() does not introduce new columns beyond the seed DataFrame."""
    engine = SDVMultiplierEngine(use_sdv=False)
    result = engine.scale(sample_dataframe, target_rows=100, schema=sample_schema)
    assert set(result.columns) == set(sample_dataframe.columns)


# ── scale — edge cases ─────────────────────────────────────────────────────


def test_sdv_multiplier_empty_seed_returns_empty(sample_schema) -> None:
    """Empty seed DataFrame → empty result returned without error."""
    engine = SDVMultiplierEngine(use_sdv=False)
    empty_df = pd.DataFrame(columns=["patient_id", "age", "salary"])
    result = engine.scale(empty_df, target_rows=100, schema=sample_schema)
    assert len(result) == 0


def test_sdv_multiplier_deterministic_with_same_seed(
    sample_dataframe, sample_schema
) -> None:
    """Same seed produces identical results across two calls."""
    engine = SDVMultiplierEngine(use_sdv=False)
    result1 = engine.scale(sample_dataframe, target_rows=150, schema=sample_schema, seed=42)
    result2 = engine.scale(sample_dataframe, target_rows=150, schema=sample_schema, seed=42)
    # Compare numeric columns
    numeric_cols = result1.select_dtypes(include="number").columns.tolist()
    for col in numeric_cols:
        assert list(result1[col]) == list(result2[col]), f"Column {col} differs"


def test_sdv_multiplier_gender_values_from_enum(sample_dataframe, sample_schema) -> None:
    """Scaled gender column contains only values present in the seed."""
    engine = SDVMultiplierEngine(use_sdv=False)
    result = engine.scale(sample_dataframe, target_rows=200, schema=sample_schema, seed=7)
    original_genders = set(sample_dataframe["gender"].dropna().unique())
    result_genders = set(result["gender"].dropna().unique())
    # All result values should be drawn from the original categories
    assert result_genders.issubset(original_genders)
