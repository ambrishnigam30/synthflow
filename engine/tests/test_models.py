# ───────────────────────────────────────────────────────────────
# Copyright (c) 2026 Ambrish Nigam
# Author : Ambrish Nigam | https://github.com/ambrishnigam30
# Project: SynthFlow — Autonomous Synthetic Data Orchestration Platform
# License : Apache License 2.0 | https://www.apache.org/licenses/LICENSE-2.0
# Module : Tests E-001-01 through E-001-10 — Pydantic model tests
# ───────────────────────────────────────────────────────────────

from __future__ import annotations

import pandas as pd
import pytest
from pydantic import ValidationError

from synthflow.models.schemas import (
    BusinessContext,
    CausalDagRule,
    CausalKnowledgeBundle,
    CausalOrSpurious,
    ColumnDefinition,
    ColumnKnowledge,
    ConditionalParameterEntry,
    ConditionalParameterTable,
    ConstraintRule,
    ConstraintSet,
    ConstraintType,
    CorrelationDirection,
    CrossColumnCorrelation,
    DirtyDataProfile,
    DistributionMap,
    DistributionSpec,
    DriftEvent,
    EconomicTier,
    GenerationResult,
    GeographicalConstraints,
    HealEvent,
    ImpliedTimeRange,
    IntentObject,
    LocaleInfo,
    NameCulturalPatterns,
    NullMechanism,
    PrivacyReport,
    PrivacySensitivity,
    QualityReport,
    RegionInfo,
    RelationshipType,
    ScenarioParams,
    SchemaDefinition,
    SchemaTable,
    Severity,
    SessionRecord,
    SpecialEvent,
    TableRelationship,
    TableRelationshipType,
    TemporalPatterns,
    ValidationCheck,
    ValidationReport,
    ViolationAction,
)


# ── Helpers ────────────────────────────────────────────────────────────────

def _make_minimal_schema() -> SchemaDefinition:
    """Minimal valid SchemaDefinition with one PK column."""
    col = ColumnDefinition(name="id", is_primary_key=True)
    table = SchemaTable(name="t", columns=[col])
    return SchemaDefinition(tables=[table])


# ── E-001-01: all 33 models instantiate ───────────────────────────────────

def test_all_33_models_can_be_instantiated() -> None:
    """Every model can be created with minimal valid data — no exceptions."""
    schema = _make_minimal_schema()

    instances = [
        RegionInfo(country="India"),
        LocaleInfo(),
        ImpliedTimeRange(),
        IntentObject(domain="healthcare", row_count=100),
        ColumnKnowledge(column_name="age"),
        CausalDagRule(
            parent_column="a", child_column="b",
            lambda_str='lambda row, rng: row["a"] + 1',
        ),
        GeographicalConstraints(),
        SpecialEvent(name="Diwali", month=10),
        TemporalPatterns(),
        CrossColumnCorrelation(col_a="age", col_b="salary"),
        DirtyDataProfile(),
        BusinessContext(),
        NameCulturalPatterns(),
        CausalKnowledgeBundle(domain="healthcare"),
        ColumnDefinition(name="col1"),
        TableRelationship(
            parent_table="a", parent_column="id",
            child_table="b", child_column="a_id",
        ),
        SchemaTable(
            name="patients",
            columns=[ColumnDefinition(name="id", is_primary_key=True)],
        ),
        schema,
        ConstraintRule(
            name="age_range", constraint_type=ConstraintType.RANGE, columns=["age"]
        ),
        ConstraintSet(),
        DistributionSpec(distribution_type="normal"),
        DistributionMap(),
        ScenarioParams(scenario_name="recession"),
        ValidationCheck(check_name="null_check", score=1.0),
        ValidationReport(),
        PrivacyReport(),
        QualityReport(),
        GenerationResult(),
        SessionRecord(),
        HealEvent(session_id="abc", attempt=1, error_message="err"),
        DriftEvent(col_a="x", col_b="y", actual_rho=0.1, target_rho=0.6, delta=0.5),
        ConditionalParameterEntry(condition='row["age"] > 60', value=1.2),
        ConditionalParameterTable(column_name="discount"),
    ]
    assert len(instances) == 33


# ── E-001-02: row_count > 10M rejected ────────────────────────────────────

def test_intent_object_rejects_row_count_above_10m() -> None:
    """IntentObject raises ValueError if row_count > 10_000_000."""
    with pytest.raises(ValidationError, match="10,000,000"):
        IntentObject(domain="healthcare", row_count=20_000_000)


# ── E-001-03: domain normalised to lowercase ──────────────────────────────

def test_intent_object_normalizes_domain_to_lowercase() -> None:
    """domain='Healthcare' becomes 'healthcare'."""
    obj = IntentObject(domain="Healthcare", row_count=100)
    assert obj.domain == "healthcare"


def test_intent_object_normalizes_domain_strips_whitespace() -> None:
    """Leading/trailing whitespace is stripped from domain."""
    obj = IntentObject(domain="  Finance  ", row_count=10)
    assert obj.domain == "finance"


# ── E-001-04: extra fields allowed ────────────────────────────────────────

def test_intent_object_allows_extra_fields() -> None:
    """IntentObject(extra_field='x') does not raise."""
    obj = IntentObject(domain="healthcare", row_count=50, custom_metadata="test")
    assert obj.custom_metadata == "test"  # type: ignore[attr-defined]


# ── E-001-05: temporal weights normalised ─────────────────────────────────

def test_knowledge_bundle_normalizes_temporal_weights() -> None:
    """day_of_week_weights [2,2,2,2,2,2,2] normalised to [~0.142 × 7]."""
    tp = TemporalPatterns(day_of_week_weights=[2, 2, 2, 2, 2, 2, 2])
    assert len(tp.day_of_week_weights) == 7
    for w in tp.day_of_week_weights:
        assert abs(w - 1 / 7) < 1e-9
    assert abs(sum(tp.day_of_week_weights) - 1.0) < 1e-9


def test_knowledge_bundle_normalizes_unequal_weights() -> None:
    """Unequal weights are normalised so they sum to 1.0."""
    tp = TemporalPatterns(day_of_week_weights=[1, 2, 3, 4, 5, 6, 7])
    assert abs(sum(tp.day_of_week_weights) - 1.0) < 1e-9


# ── E-001-06: schema without PK rejected ──────────────────────────────────

def test_schema_definition_rejects_table_without_primary_key() -> None:
    """SchemaDefinition raises if any table has no is_primary_key=True column."""
    col_no_pk = ColumnDefinition(name="name")  # is_primary_key defaults False
    table = SchemaTable(name="no_pk_table", columns=[col_no_pk])
    with pytest.raises(ValidationError, match="primary key"):
        SchemaDefinition(tables=[table])


# ── E-001-07: RegionInfo round-trip serialisation ─────────────────────────

def test_region_info_serialization_roundtrip() -> None:
    """RegionInfo → dict → RegionInfo produces an identical object."""
    original = RegionInfo(
        country="India",
        state_province="Karnataka",
        city="Bengaluru",
        subregion="South",
        continent="Asia",
    )
    restored = RegionInfo.from_dict(original.to_dict())
    assert original == restored


# ── E-001-08: ColumnDefinition defaults ───────────────────────────────────

def test_column_definition_defaults() -> None:
    """Minimal ColumnDefinition has nullable=True and null_rate=0.0."""
    col = ColumnDefinition(name="test_col")
    assert col.nullable is True
    assert col.null_rate == 0.0
    assert col.is_primary_key is False
    assert col.is_foreign_key is False


# ── E-001-09: GenerationResult validates dataframe type ───────────────────

def test_generation_result_rejects_non_dataframe() -> None:
    """Passing a non-DataFrame for dataframe raises ValidationError."""
    with pytest.raises(ValidationError):
        GenerationResult(dataframe="not_a_dataframe")

    with pytest.raises(ValidationError):
        GenerationResult(dataframe={"key": "value"})


def test_generation_result_accepts_dataframe() -> None:
    """GenerationResult accepts a real pd.DataFrame."""
    df = pd.DataFrame({"a": [1, 2, 3]})
    result = GenerationResult(dataframe=df)
    assert result.dataframe is df


def test_generation_result_accepts_none() -> None:
    """GenerationResult dataframe=None is valid."""
    result = GenerationResult(dataframe=None)
    assert result.dataframe is None


# ── E-001-10: enum values ─────────────────────────────────────────────────

def test_null_mechanism_mcar_value() -> None:
    """NullMechanism.MCAR == 'MCAR'."""
    assert NullMechanism.MCAR == "MCAR"
    assert NullMechanism.MAR == "MAR"
    assert NullMechanism.MNAR == "MNAR"


def test_all_enum_string_values() -> None:
    """All str-enum members compare equal to their string value."""
    assert EconomicTier.HIGH == "high"
    assert ViolationAction.CLIP == "clip"
    assert ConstraintType.RANGE == "range"
    assert Severity.CRITICAL == "critical"
    assert PrivacySensitivity.RESTRICTED == "restricted"
    assert TableRelationshipType.ONE_TO_MANY == "one_to_many"
    assert CorrelationDirection.POSITIVE == "positive"
    assert CausalOrSpurious.CAUSAL == "causal"
    assert RelationshipType.MANY_TO_MANY == "many_to_many"
