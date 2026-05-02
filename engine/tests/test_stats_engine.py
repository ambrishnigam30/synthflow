# ───────────────────────────────────────────────────────────────
# Copyright (c) 2026 Ambrish Nigam
# Author : Ambrish Nigam | https://github.com/ambrishnigam30
# Project: SynthFlow — Autonomous Synthetic Data Orchestration Platform
# License : Apache License 2.0 | https://www.apache.org/licenses/LICENSE-2.0
# Module : Tests for StatisticalModelingCore
# ───────────────────────────────────────────────────────────────

from __future__ import annotations

import pytest

from synthflow.engines.stats_engine import StatisticalModelingCore


# ── validate_distribution ──────────────────────────────────────────────────


def test_stats_engine_salary_must_be_lognormal(sample_schema, sample_knowledge_bundle) -> None:
    """semantic_type='salary' with distribution='normal' gets corrected to 'lognormal'."""
    engine = StatisticalModelingCore()
    result = engine.validate_distribution("salary", "normal")
    assert result == "lognormal"


def test_stats_engine_age_must_be_truncated_normal() -> None:
    engine = StatisticalModelingCore()
    assert engine.validate_distribution("age", "pareto") == "truncated_normal"


def test_stats_engine_count_must_be_poisson() -> None:
    engine = StatisticalModelingCore()
    assert engine.validate_distribution("count", "normal") == "poisson"


def test_stats_engine_probability_must_be_beta() -> None:
    engine = StatisticalModelingCore()
    assert engine.validate_distribution("probability", "normal") == "beta"


def test_stats_engine_unknown_type_returns_input() -> None:
    """Unknown semantic type — distribution accepted unchanged."""
    engine = StatisticalModelingCore()
    result = engine.validate_distribution("unknown_type", "normal")
    assert result == "normal"


def test_stats_engine_income_must_be_lognormal() -> None:
    engine = StatisticalModelingCore()
    assert engine.validate_distribution("income", "normal") == "lognormal"


def test_stats_engine_duration_must_be_exponential() -> None:
    engine = StatisticalModelingCore()
    assert engine.validate_distribution("duration", "normal") == "exponential"


# ── model() ───────────────────────────────────────────────────────────────


def test_stats_engine_model_returns_distribution_map() -> None:
    """model() on a mixed schema returns a DistributionMap."""
    from synthflow.models.schemas import (
        CausalKnowledgeBundle,
        ColumnDefinition,
        RegionInfo,
        SchemaDefinition,
        SchemaTable,
    )

    engine = StatisticalModelingCore()
    cols = [
        ColumnDefinition(name="id", data_type="string", is_primary_key=True, unique=True),
        ColumnDefinition(name="age", data_type="integer", semantic_type="age"),
        ColumnDefinition(name="bill_amount", data_type="float", semantic_type="salary"),
    ]
    schema = SchemaDefinition(tables=[SchemaTable(name="t", columns=cols)])
    bundle = CausalKnowledgeBundle(domain="test", region=RegionInfo(country="India"))
    dist_map = engine.model(schema, bundle)
    assert dist_map is not None
    assert hasattr(dist_map, "column_distributions")


def test_stats_engine_model_salary_column_is_lognormal() -> None:
    """bill_amount with semantic_type='salary' is mapped to lognormal."""
    from synthflow.models.schemas import (
        CausalKnowledgeBundle,
        ColumnDefinition,
        RegionInfo,
        SchemaDefinition,
        SchemaTable,
    )

    engine = StatisticalModelingCore()
    cols = [
        ColumnDefinition(name="id", data_type="string", is_primary_key=True, unique=True),
        ColumnDefinition(name="bill_amount", data_type="float", semantic_type="salary"),
    ]
    schema = SchemaDefinition(tables=[SchemaTable(name="t", columns=cols)])
    bundle = CausalKnowledgeBundle(domain="test", region=RegionInfo(country="India"))
    dist_map = engine.model(schema, bundle)
    bill_spec = dist_map.column_distributions.get("bill_amount")
    assert bill_spec is not None
    assert bill_spec.distribution_type == "lognormal"


def test_stats_engine_model_age_column_is_truncated_normal() -> None:
    """age column with semantic_type='age' is mapped to truncated_normal."""
    from synthflow.models.schemas import (
        CausalKnowledgeBundle,
        ColumnDefinition,
        RegionInfo,
        SchemaDefinition,
        SchemaTable,
    )

    engine = StatisticalModelingCore()
    cols = [
        ColumnDefinition(name="id", data_type="string", is_primary_key=True, unique=True),
        ColumnDefinition(name="age", data_type="integer", semantic_type="age"),
    ]
    schema = SchemaDefinition(tables=[SchemaTable(name="t", columns=cols)])
    bundle = CausalKnowledgeBundle(domain="test", region=RegionInfo(country="India"))
    dist_map = engine.model(schema, bundle)
    age_spec = dist_map.column_distributions.get("age")
    assert age_spec is not None
    assert age_spec.distribution_type == "truncated_normal"


def test_stats_engine_model_covers_numeric_columns() -> None:
    """DistributionMap covers all columns."""
    from synthflow.models.schemas import (
        CausalKnowledgeBundle,
        ColumnDefinition,
        RegionInfo,
        SchemaDefinition,
        SchemaTable,
    )

    engine = StatisticalModelingCore()
    cols = [
        ColumnDefinition(name="pk", data_type="string", is_primary_key=True, unique=True),
        ColumnDefinition(name="age", data_type="integer", semantic_type="age"),
        ColumnDefinition(name="salary", data_type="float", semantic_type="salary"),
        ColumnDefinition(name="tenure", data_type="float", semantic_type="duration"),
    ]
    schema = SchemaDefinition(tables=[SchemaTable(name="t", columns=cols)])
    bundle = CausalKnowledgeBundle(domain="test", region=RegionInfo(country="India"))
    dist_map = engine.model(schema, bundle)
    expected_cols = {"pk", "age", "salary", "tenure"}
    assert expected_cols == set(dist_map.column_distributions.keys())
