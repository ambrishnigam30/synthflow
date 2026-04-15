# ───────────────────────────────────────────────────────────────
# Copyright (c) 2026 Ambrish Nigam
# Author : Ambrish Nigam | https://github.com/ambrishnigam30
# Project: SynthFlow — Autonomous Synthetic Data Orchestration Platform
# License : Apache License 2.0 | https://www.apache.org/licenses/LICENSE-2.0
# Module : Pytest shared fixtures
# ───────────────────────────────────────────────────────────────

from __future__ import annotations

import uuid

import numpy as np
import pandas as pd
import pytest

from synthflow.llm_client import MockLLMClient
from synthflow.models.schemas import (
    CausalDagRule,
    CausalKnowledgeBundle,
    ColumnDefinition,
    CrossColumnCorrelation,
    DirtyDataProfile,
    IntentObject,
    RegionInfo,
    SchemaDefinition,
    SchemaTable,
    TemporalPatterns,
)


@pytest.fixture
def mock_llm_client() -> MockLLMClient:
    """MockLLMClient — deterministic, zero API calls."""
    return MockLLMClient()


@pytest.fixture
def sample_intent() -> IntentObject:
    """IntentObject for Indian healthcare, 100 rows."""
    return IntentObject(
        domain="healthcare",
        sub_domain="general",
        region=RegionInfo(country="India", state_province="Maharashtra", city="Mumbai"),
        row_count=100,
        seed=42,
        implied_columns=["patient_id", "name", "age", "gender", "diagnosis"],
    )


@pytest.fixture
def sample_schema() -> SchemaDefinition:
    """SchemaDefinition: single 'patients' table with 15 columns incl. PK."""
    columns = [
        ColumnDefinition(name="patient_id", data_type="string", is_primary_key=True,
                         unique=True, nullable=False),
        ColumnDefinition(name="first_name", data_type="string"),
        ColumnDefinition(name="last_name", data_type="string"),
        ColumnDefinition(name="age", data_type="integer", semantic_type="age",
                         min_value=0.0, max_value=120.0),
        ColumnDefinition(name="gender", data_type="string",
                         enum_values=["Male", "Female", "Other"]),
        ColumnDefinition(name="blood_type", data_type="string"),
        ColumnDefinition(name="admission_date", data_type="date"),
        ColumnDefinition(name="discharge_date", data_type="date"),
        ColumnDefinition(name="diagnosis", data_type="string"),
        ColumnDefinition(name="department", data_type="string"),
        ColumnDefinition(name="doctor_id", data_type="string"),
        ColumnDefinition(name="bill_amount", data_type="float", semantic_type="salary",
                         min_value=0.0),
        ColumnDefinition(name="insurance_provider", data_type="string", nullable=True),
        ColumnDefinition(name="city", data_type="string"),
        ColumnDefinition(name="created_at", data_type="datetime"),
    ]
    return SchemaDefinition(
        tables=[SchemaTable(name="patients", columns=columns)],
        version="1.0",
    )


@pytest.fixture
def sample_knowledge_bundle() -> CausalKnowledgeBundle:
    """CausalKnowledgeBundle with DAG rules, correlations, temporal patterns."""
    dag_rules = [
        CausalDagRule(
            parent_column="birth_year",
            child_column="age",
            lambda_str='lambda row, rng: 2024 - row["birth_year"]',
            description="Age derived from birth year",
        ),
        CausalDagRule(
            parent_column="age",
            child_column="is_senior",
            lambda_str='lambda row, rng: row["age"] >= 60',
            description="Senior flag",
        ),
    ]
    return CausalKnowledgeBundle(
        domain="healthcare",
        region=RegionInfo(country="India"),
        temporal_patterns=TemporalPatterns(
            day_of_week_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 0.3],
        ),
        dag_rules=dag_rules,
        correlations=[
            CrossColumnCorrelation(col_a="age", col_b="bill_amount", strength=0.4),
        ],
        dirty_data_profile=DirtyDataProfile(null_rate=0.05),
    )


@pytest.fixture
def sample_dataframe() -> pd.DataFrame:
    """50-row DataFrame matching sample_schema column names."""
    rng = np.random.default_rng(42)
    n = 50
    return pd.DataFrame(
        {
            "patient_id": [str(uuid.uuid4()) for _ in range(n)],
            "first_name": [f"Patient{i}" for i in range(n)],
            "last_name": [f"Surname{i}" for i in range(n)],
            "age": rng.integers(18, 90, size=n).tolist(),
            "gender": rng.choice(["Male", "Female", "Other"], size=n).tolist(),
            "blood_type": rng.choice(["A+", "B+", "O+", "AB+"], size=n).tolist(),
            "admission_date": pd.date_range("2023-01-01", periods=n, freq="3D").tolist(),
            "discharge_date": pd.date_range("2023-01-05", periods=n, freq="3D").tolist(),
            "diagnosis": [f"Diagnosis_{i % 10}" for i in range(n)],
            "department": rng.choice(["Cardiology", "Neurology", "Orthopedics"], size=n).tolist(),
            "doctor_id": [f"DR{i:04d}" for i in range(n)],
            "bill_amount": rng.lognormal(mean=10, sigma=1, size=n).tolist(),
            "insurance_provider": rng.choice(["HDFC", "LIC", None], size=n).tolist(),
            "city": rng.choice(["Mumbai", "Pune", "Nashik"], size=n).tolist(),
            "created_at": pd.date_range("2023-01-01", periods=n, freq="1D").tolist(),
        }
    )
