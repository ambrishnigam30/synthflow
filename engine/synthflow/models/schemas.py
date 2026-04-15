# ───────────────────────────────────────────────────────────────
# Copyright (c) 2026 Ambrish Nigam
# Author : Ambrish Nigam | https://github.com/ambrishnigam30
# Project: SynthFlow — Autonomous Synthetic Data Orchestration Platform
# License : Apache License 2.0 | https://www.apache.org/licenses/LICENSE-2.0
# Module : Pydantic v2 data models — all 33 domain models
# ───────────────────────────────────────────────────────────────

from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Optional

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


# ══════════════════════════════════════════════════════════════════════
# Enums
# ══════════════════════════════════════════════════════════════════════

class EconomicTier(str, Enum):
    LOW = "low"
    LOWER_MIDDLE = "lower_middle"
    UPPER_MIDDLE = "upper_middle"
    HIGH = "high"


class NullMechanism(str, Enum):
    MCAR = "MCAR"
    MAR = "MAR"
    MNAR = "MNAR"


class RelationshipType(str, Enum):
    ONE_TO_ONE = "one_to_one"
    ONE_TO_MANY = "one_to_many"
    MANY_TO_MANY = "many_to_many"


class ViolationAction(str, Enum):
    CLIP = "clip"
    REJECT = "reject"
    FLAG = "flag"
    IMPUTE = "impute"


class ConstraintType(str, Enum):
    RANGE = "range"
    ENUM = "enum"
    UNIQUE = "unique"
    TEMPORAL = "temporal"
    REFERENTIAL = "referential"
    EXPRESSION = "expression"
    CAUSAL = "causal"


class Severity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class PrivacySensitivity(str, Enum):
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"


class TableRelationshipType(str, Enum):
    ONE_TO_ONE = "one_to_one"
    ONE_TO_MANY = "one_to_many"
    MANY_TO_MANY = "many_to_many"


class CorrelationDirection(str, Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NONE = "none"


class CausalOrSpurious(str, Enum):
    CAUSAL = "causal"
    SPURIOUS = "spurious"
    UNKNOWN = "unknown"


# ══════════════════════════════════════════════════════════════════════
# Base
# ══════════════════════════════════════════════════════════════════════

class SynthFlowBase(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SynthFlowBase":
        return cls.model_validate(data)


# ══════════════════════════════════════════════════════════════════════
# Models 1–13: primitives and knowledge building blocks
# ══════════════════════════════════════════════════════════════════════

# 1
class RegionInfo(SynthFlowBase):
    country: str = Field(description="Country name, e.g. 'India'")
    state_province: Optional[str] = Field(default=None, description="State or province")
    city: Optional[str] = Field(default=None, description="City name")
    subregion: Optional[str] = Field(default=None, description="Sub-region or district")
    continent: Optional[str] = Field(default=None, description="Continent")


# 2
class LocaleInfo(SynthFlowBase):
    faker_locale: str = Field(default="en_IN", description="Faker locale (never pa_IN; use hi_IN for Punjab)")
    currency: str = Field(default="INR", description="ISO 4217 currency code")
    number_format: str = Field(default="indian", description="'western' or 'indian'")
    date_format: str = Field(default="%Y-%m-%d", description="strftime date format")
    timezone: str = Field(default="Asia/Kolkata", description="Timezone name")


# 3
class ImpliedTimeRange(SynthFlowBase):
    start_year: int = Field(default=2020, description="Start year for temporal data")
    end_year: int = Field(default=2024, description="End year for temporal data")
    reference_date: Optional[str] = Field(default=None, description="ISO date string reference point")


# 4  — extra="allow" so LLM can add arbitrary fields
class IntentObject(SynthFlowBase):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    domain: str = Field(description="Data domain, e.g. 'healthcare'")
    sub_domain: Optional[str] = Field(default=None, description="Sub-domain, e.g. 'cardiology'")
    region: Optional[RegionInfo] = Field(default=None, description="Geographic region")
    row_count: int = Field(default=1000, ge=1, description="Number of rows to generate")
    seed: Optional[int] = Field(default=None, description="Random seed for reproducibility")
    implied_columns: list[str] = Field(default_factory=list, description="Columns implied by prompt")
    economic_tier: Optional[EconomicTier] = Field(default=None)
    time_range: Optional[ImpliedTimeRange] = Field(default=None)
    locale: Optional[LocaleInfo] = Field(default=None)
    scenario: Optional[str] = Field(default=None, description="Scenario modifier")
    output_format: str = Field(default="csv", description="Output format: csv, json, parquet, excel")

    @field_validator("row_count")
    @classmethod
    def cap_row_count(cls, v: int) -> int:
        if v > 10_000_000:
            raise ValueError(f"row_count {v:,} exceeds maximum of 10,000,000")
        return v

    @field_validator("domain")
    @classmethod
    def lowercase_domain(cls, v: str) -> str:
        return v.lower().strip()


# 5
class ColumnKnowledge(SynthFlowBase):
    column_name: str = Field(description="Column name")
    description: str = Field(default="", description="Human-readable description")
    distribution_hint: Optional[str] = Field(default=None, description="Distribution type hint")
    value_examples: list[Any] = Field(default_factory=list, description="Example values")
    semantic_type: Optional[str] = Field(default=None, description="Semantic type, e.g. 'age'")
    min_value: Optional[float] = Field(default=None)
    max_value: Optional[float] = Field(default=None)
    unit: Optional[str] = Field(default=None, description="Unit of measurement")


# 6
class CausalDagRule(SynthFlowBase):
    parent_column: str = Field(description="Parent column name")
    child_column: str = Field(description="Child column derived from parent")
    lambda_str: str = Field(description="Python lambda: 'lambda row, rng: ...'")
    description: str = Field(default="", description="Human-readable rule description")


# 7
class GeographicalConstraints(SynthFlowBase):
    valid_cities: list[str] = Field(default_factory=list)
    valid_states: list[str] = Field(default_factory=list)
    coordinate_bounds: Optional[dict[str, float]] = Field(
        default=None, description="lat_min, lat_max, lon_min, lon_max"
    )
    country_code: Optional[str] = Field(default=None, description="ISO 3166-1 alpha-2")


# 8
class SpecialEvent(SynthFlowBase):
    name: str = Field(description="Event name, e.g. 'Diwali'")
    month: int = Field(ge=1, le=12)
    day: Optional[int] = Field(default=None, ge=1, le=31)
    day_multiplier: float = Field(default=1.0, description="Multiplier for affected columns")
    affected_columns: list[str] = Field(default_factory=list)
    duration_days: int = Field(default=1, ge=1)


# 9
class TemporalPatterns(SynthFlowBase):
    day_of_week_weights: list[float] = Field(
        default_factory=lambda: [1 / 7] * 7,
        description="7 weights Mon–Sun, auto-normalised to sum=1.0",
    )
    hour_of_day_weights: list[float] = Field(
        default_factory=lambda: [1 / 24] * 24,
        description="24 weights hour 0–23, auto-normalised to sum=1.0",
    )
    monthly_seasonality: dict[str, float] = Field(
        default_factory=dict, description="Month str (1–12) → multiplier"
    )
    special_events: list[SpecialEvent] = Field(default_factory=list)
    has_autocorrelation: bool = Field(default=False)
    autocorrelation_rho: float = Field(default=0.0, ge=-1.0, le=1.0)

    @field_validator("day_of_week_weights")
    @classmethod
    def normalise_dow(cls, v: list[float]) -> list[float]:
        total = sum(v)
        if total <= 0:
            raise ValueError("day_of_week_weights must not all be zero")
        return [w / total for w in v]

    @field_validator("hour_of_day_weights")
    @classmethod
    def normalise_hod(cls, v: list[float]) -> list[float]:
        total = sum(v)
        if total <= 0:
            raise ValueError("hour_of_day_weights must not all be zero")
        return [w / total for w in v]


# 10
class CrossColumnCorrelation(SynthFlowBase):
    col_a: str = Field(description="First column")
    col_b: str = Field(description="Second column")
    direction: CorrelationDirection = Field(default=CorrelationDirection.POSITIVE)
    strength: float = Field(default=0.5, ge=-1.0, le=1.0)
    causal_or_spurious: CausalOrSpurious = Field(default=CausalOrSpurious.UNKNOWN)


# 11
class DirtyDataProfile(SynthFlowBase):
    null_rate: float = Field(default=0.05, ge=0.0, le=1.0)
    null_mechanism: NullMechanism = Field(default=NullMechanism.MCAR)
    typo_rate: float = Field(default=0.02, ge=0.0, le=1.0)
    outlier_rate: float = Field(default=0.01, ge=0.0, le=1.0)
    duplicate_rate: float = Field(default=0.005, ge=0.0, le=1.0)
    near_duplicate_rate: float = Field(default=0.005, ge=0.0, le=1.0)
    date_format_inconsistency_rate: float = Field(default=0.1, ge=0.0, le=1.0)


# 12
class BusinessContext(SynthFlowBase):
    organization_type: str = Field(default="", description="e.g. 'hospital', 'bank'")
    size_tier: str = Field(default="medium")
    regulatory_environment: list[str] = Field(default_factory=list)
    notes: str = Field(default="")
    industry_code: Optional[str] = Field(default=None)


# 13
class NameCulturalPatterns(SynthFlowBase):
    first_name_pool_hint: str = Field(default="")
    last_name_pool_hint: str = Field(default="")
    salutation_pattern: str = Field(default="Mr/Ms")
    name_order: str = Field(default="first_last")
    middle_name_probability: float = Field(default=0.3, ge=0.0, le=1.0)


# ══════════════════════════════════════════════════════════════════════
# Model 14: CausalKnowledgeBundle — extra="allow"
# ══════════════════════════════════════════════════════════════════════

# 14
class CausalKnowledgeBundle(SynthFlowBase):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    domain: str = Field(description="Domain this bundle covers")
    sub_domain: Optional[str] = Field(default=None)
    region: Optional[RegionInfo] = Field(default=None)
    temporal_patterns: Optional[TemporalPatterns] = Field(default=None)
    column_knowledge: list[ColumnKnowledge] = Field(default_factory=list)
    dag_rules: list[CausalDagRule] = Field(default_factory=list)
    correlations: list[CrossColumnCorrelation] = Field(default_factory=list)
    dirty_data_profile: Optional[DirtyDataProfile] = Field(default=None)
    business_context: Optional[BusinessContext] = Field(default=None)
    geographical_constraints: Optional[GeographicalConstraints] = Field(default=None)
    name_patterns: Optional[NameCulturalPatterns] = Field(default=None)
    currency_code: str = Field(default="INR")
    locale: Optional[LocaleInfo] = Field(default=None)


# ══════════════════════════════════════════════════════════════════════
# Models 15–18: Schema layer
# ══════════════════════════════════════════════════════════════════════

# 15
class ColumnDefinition(SynthFlowBase):
    name: str = Field(description="Column name")
    data_type: str = Field(default="string", description="string|integer|float|boolean|datetime|date")
    semantic_type: str = Field(default="", description="e.g. 'age', 'salary'")
    nullable: bool = Field(default=True)
    null_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    is_primary_key: bool = Field(default=False)
    is_foreign_key: bool = Field(default=False)
    references_table: Optional[str] = Field(default=None)
    references_column: Optional[str] = Field(default=None)
    min_value: Optional[float] = Field(default=None)
    max_value: Optional[float] = Field(default=None)
    enum_values: list[Any] = Field(default_factory=list)
    generation_order: int = Field(default=0)
    description: str = Field(default="")
    privacy_sensitivity: PrivacySensitivity = Field(default=PrivacySensitivity.PUBLIC)
    unique: bool = Field(default=False)
    format_hint: Optional[str] = Field(default=None)


# 16
class TableRelationship(SynthFlowBase):
    parent_table: str = Field(description="Parent table name")
    parent_column: str = Field(description="Parent PK column")
    child_table: str = Field(description="Child table name")
    child_column: str = Field(description="Child FK column")
    relationship_type: TableRelationshipType = Field(default=TableRelationshipType.ONE_TO_MANY)


# 17
class SchemaTable(SynthFlowBase):
    name: str = Field(description="Table name")
    columns: list[ColumnDefinition] = Field(description="Column definitions")
    description: str = Field(default="")

    @property
    def primary_key_columns(self) -> list[str]:
        return [c.name for c in self.columns if c.is_primary_key]


# 18 — extra="allow"
class SchemaDefinition(SynthFlowBase):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    tables: list[SchemaTable] = Field(description="Tables in this schema")
    version: str = Field(default="1.0")
    description: str = Field(default="")
    relationships: list[TableRelationship] = Field(default_factory=list)

    @model_validator(mode="after")
    def every_table_has_pk(self) -> "SchemaDefinition":
        for table in self.tables:
            pk_cols = [c for c in table.columns if c.is_primary_key]
            if not pk_cols:
                raise ValueError(
                    f"Table '{table.name}' has no primary key. "
                    "Every table must have at least one column with is_primary_key=True."
                )
        return self


# ══════════════════════════════════════════════════════════════════════
# Models 19–23: Constraints, distributions, scenario
# ══════════════════════════════════════════════════════════════════════

# 19
class ConstraintRule(SynthFlowBase):
    name: str = Field(description="Rule name")
    constraint_type: ConstraintType = Field(description="Type of constraint")
    columns: list[str] = Field(description="Columns involved")
    expression: Optional[str] = Field(default=None, description="Python expression string")
    action_on_violation: ViolationAction = Field(default=ViolationAction.FLAG)
    severity: Severity = Field(default=Severity.MEDIUM)
    description: str = Field(default="")
    parameters: dict[str, Any] = Field(default_factory=dict)


# 20
class ConstraintSet(SynthFlowBase):
    rules: list[ConstraintRule] = Field(default_factory=list)
    version: str = Field(default="1.0")
    domain: str = Field(default="")


# 21
class DistributionSpec(SynthFlowBase):
    distribution_type: str = Field(description="normal|lognormal|uniform|truncated_normal|categorical|…")
    params: dict[str, float] = Field(default_factory=dict)
    clip_min: Optional[float] = Field(default=None)
    clip_max: Optional[float] = Field(default=None)
    weights: Optional[list[float]] = Field(default=None, description="Mixture weights")


# 22
class DistributionMap(SynthFlowBase):
    column_distributions: dict[str, DistributionSpec] = Field(default_factory=dict)
    table_name: str = Field(default="")


# 23
class ScenarioParams(SynthFlowBase):
    scenario_name: str = Field(description="e.g. 'recession', 'pandemic', 'boom'")
    multipliers: dict[str, float] = Field(default_factory=dict)
    additive_shifts: dict[str, float] = Field(default_factory=dict)
    affected_columns: list[str] = Field(default_factory=list)
    description: str = Field(default="")
    time_range: Optional[ImpliedTimeRange] = Field(default=None)


# ══════════════════════════════════════════════════════════════════════
# Models 24–27: Reports
# ══════════════════════════════════════════════════════════════════════

# 24
class ValidationCheck(SynthFlowBase):
    check_name: str = Field(description="Name of the check")
    score: float = Field(ge=0.0, le=1.0, description="Score 0.0–1.0")
    violations_found: int = Field(default=0, ge=0)
    severity: Severity = Field(default=Severity.MEDIUM)
    details: str = Field(default="")
    passed: bool = Field(default=True)


# 25
class ValidationReport(SynthFlowBase):
    checks: list[ValidationCheck] = Field(default_factory=list)
    overall_score: float = Field(default=0.0, ge=0.0, le=100.0)
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    generated_at: datetime = Field(default_factory=datetime.utcnow)
    row_count: int = Field(default=0, ge=0)
    column_count: int = Field(default=0, ge=0)


# 26
class PrivacyReport(SynthFlowBase):
    k_anonymity: int = Field(default=1, ge=1)
    l_diversity: float = Field(default=1.0, ge=0.0)
    privacy_score: float = Field(default=40.0, ge=0.0, le=100.0)
    entities_detected: dict[str, int] = Field(default_factory=dict)
    masked_columns: list[str] = Field(default_factory=list)
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    generated_at: datetime = Field(default_factory=datetime.utcnow)


# 27
class QualityReport(SynthFlowBase):
    overall_score: float = Field(default=0.0, ge=0.0, le=100.0)
    sdmetrics_score: float = Field(default=0.0, ge=0.0, le=100.0)
    causal_score: float = Field(default=0.0, ge=0.0, le=100.0)
    privacy_score: float = Field(default=0.0, ge=0.0, le=100.0)
    temporal_score: float = Field(default=0.0, ge=0.0, le=100.0)
    dirty_score: float = Field(default=0.0, ge=0.0, le=100.0)
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    generated_at: datetime = Field(default_factory=datetime.utcnow)
    notes: str = Field(default="")


# ══════════════════════════════════════════════════════════════════════
# Models 28–33: Session, events, conditional params
# ══════════════════════════════════════════════════════════════════════

# 28
class GenerationResult(SynthFlowBase):
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    dataframe: Optional[Any] = Field(default=None, description="Generated pd.DataFrame")
    schema: Optional[SchemaDefinition] = Field(default=None)
    validation_report: Optional[ValidationReport] = Field(default=None)
    quality_report: Optional[QualityReport] = Field(default=None)
    privacy_report: Optional[PrivacyReport] = Field(default=None)
    generated_code: str = Field(default="")
    intent: Optional[IntentObject] = Field(default=None)
    row_count: int = Field(default=0, ge=0)
    seed: Optional[int] = Field(default=None)
    generation_duration_seconds: float = Field(default=0.0, ge=0.0)

    @field_validator("dataframe")
    @classmethod
    def must_be_dataframe(cls, v: Any) -> Any:
        if v is not None and not isinstance(v, pd.DataFrame):
            raise ValueError(
                f"dataframe must be a pandas DataFrame or None, got {type(v).__name__}"
            )
        return v

    def to_dict(self) -> dict[str, Any]:
        d = self.model_dump(exclude={"dataframe"})
        d["dataframe"] = (
            self.dataframe.to_dict(orient="records") if self.dataframe is not None else None
        )
        return d


# 29
class SessionRecord(SynthFlowBase):
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = Field(default_factory=datetime.utcnow)
    domain: str = Field(default="")
    sub_domain: Optional[str] = Field(default=None)
    row_count: int = Field(default=0, ge=0)
    quality_score: float = Field(default=0.0, ge=0.0, le=100.0)
    status: str = Field(default="pending")
    prompt: str = Field(default="")
    seed: Optional[int] = Field(default=None)


# 30
class HealEvent(SynthFlowBase):
    session_id: str = Field(description="Session that required healing")
    attempt: int = Field(ge=1, le=3)
    error_message: str = Field(description="Original error traceback")
    fix_description: str = Field(default="")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    success: bool = Field(default=False)


# 31
class DriftEvent(SynthFlowBase):
    col_a: str = Field(description="First column in pair")
    col_b: str = Field(description="Second column in pair")
    actual_rho: float = Field(ge=-1.0, le=1.0)
    target_rho: float = Field(ge=-1.0, le=1.0)
    delta: float = Field(description="|actual_rho - target_rho|")
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    iteration: int = Field(default=0, ge=0)


# 32
class ConditionalParameterEntry(SynthFlowBase):
    condition: str = Field(description="Python expression, e.g. 'row[\"age\"] > 60'")
    value: Any = Field(description="Value when condition is True")
    priority: int = Field(default=0, ge=0)


# 33
class ConditionalParameterTable(SynthFlowBase):
    column_name: str = Field(description="Column this table applies to")
    entries: list[ConditionalParameterEntry] = Field(default_factory=list)
    default_value: Any = Field(default=None)
