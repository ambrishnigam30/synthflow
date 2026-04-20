# ───────────────────────────────────────────────────────────────
# Copyright (c) 2026 Ambrish Nigam
# Author : Ambrish Nigam | https://github.com/ambrishnigam30
# Project: SynthFlow — Autonomous Synthetic Data Orchestration Platform
# License : Apache License 2.0 | https://www.apache.org/licenses/LICENSE-2.0
# Module : Tests for DeterministicRealismEngine
# ───────────────────────────────────────────────────────────────

from __future__ import annotations

import numpy as np
import pytest

from synthflow.engines.realism_engine import DeterministicRealismEngine


# ── get_cell_rng ───────────────────────────────────────────────────────────


def test_realism_engine_same_seed_same_value() -> None:
    """get_cell_rng(42, 'age', 0) always produces same random stream."""
    engine = DeterministicRealismEngine()
    rng1 = engine.get_cell_rng(42, "age", 0)
    rng2 = engine.get_cell_rng(42, "age", 0)
    assert rng1.integers(1000) == rng2.integers(1000)


def test_realism_engine_different_rows_different_values() -> None:
    """Row 0 and row 1 of same column produce different values."""
    engine = DeterministicRealismEngine()
    rng0 = engine.get_cell_rng(42, "salary", 0)
    rng1 = engine.get_cell_rng(42, "salary", 1)
    assert rng0.integers(1_000_000) != rng1.integers(1_000_000)


def test_realism_engine_different_columns_different_values() -> None:
    """Same (seed, row) but different column names produce different values."""
    engine = DeterministicRealismEngine()
    rng_a = engine.get_cell_rng(42, "age", 5)
    rng_b = engine.get_cell_rng(42, "name", 5)
    assert rng_a.integers(1_000_000) != rng_b.integers(1_000_000)


def test_realism_engine_different_seeds_different_values() -> None:
    """Different global seeds produce different streams for the same (column, row)."""
    engine = DeterministicRealismEngine()
    rng_a = engine.get_cell_rng(42, "age", 0)
    rng_b = engine.get_cell_rng(99, "age", 0)
    assert rng_a.integers(1_000_000) != rng_b.integers(1_000_000)


# ── sample_column ──────────────────────────────────────────────────────────


def test_realism_engine_sample_enum_column(sample_schema) -> None:
    """Enum column samples only from allowed values."""
    engine = DeterministicRealismEngine()
    gender_col = next(
        c for t in sample_schema.tables for c in t.columns if c.name == "gender"
    )
    values = engine.sample_column(gender_col, 20, 42, {})
    assert len(values) == 20
    assert all(v in ["Male", "Female", "Other"] for v in values)


def test_realism_engine_sample_enum_column_correct_count(sample_schema) -> None:
    """sample_column returns exactly n_rows values."""
    engine = DeterministicRealismEngine()
    gender_col = next(
        c for t in sample_schema.tables for c in t.columns if c.name == "gender"
    )
    values = engine.sample_column(gender_col, 50, 1, {})
    assert len(values) == 50


def test_realism_engine_sample_integer_column(sample_schema) -> None:
    """Integer column (age) produces numeric values."""
    engine = DeterministicRealismEngine()
    age_col = next(
        c for t in sample_schema.tables for c in t.columns if c.name == "age"
    )
    values = engine.sample_column(age_col, 10, 42, {})
    assert len(values) == 10
    assert all(isinstance(v, (int, float)) for v in values)


# ── format_value ───────────────────────────────────────────────────────────


def test_realism_engine_format_value_indian() -> None:
    """Indian number format inserts commas into large salary values."""
    engine = DeterministicRealismEngine()
    result = engine.format_value(1_234_567.0, "salary", "indian")
    assert "," in str(result)


def test_realism_engine_format_value_western() -> None:
    """Western number format also inserts commas for large values."""
    engine = DeterministicRealismEngine()
    result = engine.format_value(1_234_567.0, "salary", "western")
    assert "," in str(result)


def test_realism_engine_format_value_non_numeric_passthrough() -> None:
    """Non-numeric values pass through format_value unchanged."""
    engine = DeterministicRealismEngine()
    result = engine.format_value("Mumbai", "city", "indian")
    assert result == "Mumbai"
