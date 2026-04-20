# ───────────────────────────────────────────────────────────────
# Copyright (c) 2026 Ambrish Nigam
# Author : Ambrish Nigam | https://github.com/ambrishnigam30
# Project: SynthFlow — Autonomous Synthetic Data Orchestration Platform
# License : Apache License 2.0 | https://www.apache.org/licenses/LICENSE-2.0
# Module : Tests for ValidationHygieneEngine
# ───────────────────────────────────────────────────────────────

from __future__ import annotations

import uuid

import numpy as np
import pandas as pd
import pytest

from synthflow.engines.validation_engine import ValidationHygieneEngine
from synthflow.models.schemas import (
    CausalKnowledgeBundle,
    ColumnDefinition,
    ConstraintSet,
    RegionInfo,
    SchemaDefinition,
    SchemaTable,
)


# ── audit — structural checks ──────────────────────────────────────────────


def test_validation_engine_returns_exactly_10_checks(
    sample_dataframe, sample_schema, sample_knowledge_bundle
) -> None:
    """audit() returns a ValidationReport with exactly 10 checks."""
    engine = ValidationHygieneEngine()
    report = engine.audit(sample_dataframe, sample_schema, ConstraintSet(), sample_knowledge_bundle)
    assert len(report.checks) == 10


def test_validation_engine_overall_score_range(
    sample_dataframe, sample_schema, sample_knowledge_bundle
) -> None:
    """Overall score is between 0 and 100 inclusive."""
    engine = ValidationHygieneEngine()
    report = engine.audit(sample_dataframe, sample_schema, ConstraintSet(), sample_knowledge_bundle)
    assert 0.0 <= report.overall_score <= 100.0


def test_validation_engine_check_names_are_10_expected(
    sample_dataframe, sample_schema, sample_knowledge_bundle
) -> None:
    """All 10 canonical check names are present in the report."""
    engine = ValidationHygieneEngine()
    report = engine.audit(sample_dataframe, sample_schema, ConstraintSet(), sample_knowledge_bundle)
    expected_names = {
        "schema_completeness", "type_conformance", "null_policy", "range_validity",
        "enum_validity", "uniqueness", "temporal_ordering", "causal_physics",
        "statistical_sanity", "correlation_consistency",
    }
    actual_names = {c.check_name for c in report.checks}
    assert actual_names == expected_names


# ── schema_completeness ────────────────────────────────────────────────────


def test_validation_engine_schema_completeness_pass(
    sample_dataframe, sample_schema, sample_knowledge_bundle
) -> None:
    """All schema columns present → schema_completeness score is 1.0."""
    engine = ValidationHygieneEngine()
    report = engine.audit(sample_dataframe, sample_schema, ConstraintSet(), sample_knowledge_bundle)
    check = next(c for c in report.checks if c.check_name == "schema_completeness")
    assert check.score == 1.0


def test_validation_engine_missing_column_reduces_score(
    sample_schema, sample_knowledge_bundle
) -> None:
    """Missing columns → schema_completeness score < 1.0."""
    engine = ValidationHygieneEngine()
    df = pd.DataFrame({"patient_id": ["x"], "first_name": ["a"]})
    report = engine.audit(df, sample_schema, ConstraintSet(), sample_knowledge_bundle)
    check = next(c for c in report.checks if c.check_name == "schema_completeness")
    assert check.score < 1.0


# ── uniqueness ─────────────────────────────────────────────────────────────


def test_validation_engine_unique_violation() -> None:
    """Duplicate PK values → uniqueness check has score < 1.0."""
    engine = ValidationHygieneEngine()
    col = ColumnDefinition(name="id", data_type="string", is_primary_key=True, unique=True)
    schema = SchemaDefinition(tables=[SchemaTable(name="t", columns=[col])])
    df = pd.DataFrame({"id": ["a", "a", "b"]})  # one duplicate
    bundle = CausalKnowledgeBundle(domain="test", region=RegionInfo(country="US"))
    report = engine.audit(df, schema, ConstraintSet(), bundle)
    check = next(c for c in report.checks if c.check_name == "uniqueness")
    assert check.score < 1.0


def test_validation_engine_uniqueness_passes_with_unique_pk(
    sample_dataframe, sample_schema, sample_knowledge_bundle
) -> None:
    """sample_dataframe has unique patient_ids → uniqueness check passes."""
    engine = ValidationHygieneEngine()
    report = engine.audit(sample_dataframe, sample_schema, ConstraintSet(), sample_knowledge_bundle)
    check = next(c for c in report.checks if c.check_name == "uniqueness")
    assert check.passed is True


# ── null_policy ────────────────────────────────────────────────────────────


def test_validation_engine_null_policy_violation() -> None:
    """Null in a nullable=False column → null_policy score < 1.0."""
    engine = ValidationHygieneEngine()
    pk_col = ColumnDefinition(name="id", data_type="string", is_primary_key=True)
    col = ColumnDefinition(name="name", data_type="string", nullable=False)
    schema = SchemaDefinition(tables=[SchemaTable(name="t", columns=[pk_col, col])])
    df = pd.DataFrame({"id": ["1", "2", "3"], "name": ["Alice", None, "Carol"]})
    bundle = CausalKnowledgeBundle(domain="test", region=RegionInfo(country="US"))
    report = engine.audit(df, schema, ConstraintSet(), bundle)
    check = next(c for c in report.checks if c.check_name == "null_policy")
    assert check.score < 1.0


# ── enum_validity ──────────────────────────────────────────────────────────


def test_validation_engine_enum_validity_passes(
    sample_dataframe, sample_schema, sample_knowledge_bundle
) -> None:
    """gender column values are all in enum → enum_validity passes."""
    engine = ValidationHygieneEngine()
    report = engine.audit(sample_dataframe, sample_schema, ConstraintSet(), sample_knowledge_bundle)
    check = next(c for c in report.checks if c.check_name == "enum_validity")
    assert check.score == 1.0


def test_validation_engine_enum_validity_violation() -> None:
    """Invalid enum value → enum_validity score < 1.0."""
    engine = ValidationHygieneEngine()
    pk_col = ColumnDefinition(name="id", data_type="string", is_primary_key=True)
    col = ColumnDefinition(
        name="status", data_type="string", enum_values=["active", "inactive"]
    )
    schema = SchemaDefinition(tables=[SchemaTable(name="t", columns=[pk_col, col])])
    df = pd.DataFrame({"id": ["1", "2", "3"], "status": ["active", "unknown_value", "inactive"]})
    bundle = CausalKnowledgeBundle(domain="test", region=RegionInfo(country="US"))
    report = engine.audit(df, schema, ConstraintSet(), bundle)
    check = next(c for c in report.checks if c.check_name == "enum_validity")
    assert check.score < 1.0


# ── row / column count ─────────────────────────────────────────────────────


def test_validation_engine_report_row_count(
    sample_dataframe, sample_schema, sample_knowledge_bundle
) -> None:
    """ValidationReport.row_count matches the DataFrame length."""
    engine = ValidationHygieneEngine()
    report = engine.audit(sample_dataframe, sample_schema, ConstraintSet(), sample_knowledge_bundle)
    assert report.row_count == len(sample_dataframe)
