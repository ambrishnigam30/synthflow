# ───────────────────────────────────────────────────────────────
# Copyright (c) 2026 Ambrish Nigam
# Author : Ambrish Nigam | https://github.com/ambrishnigam30
# Project: SynthFlow — Autonomous Synthetic Data Orchestration Platform
# License : Apache License 2.0 | https://www.apache.org/licenses/LICENSE-2.0
# Module : Tests for PresidioPrivacyGuard
# ───────────────────────────────────────────────────────────────

from __future__ import annotations

import pandas as pd
import numpy as np
import pytest

from synthflow.models.schemas import (
    ColumnDefinition,
    SchemaDefinition,
    SchemaTable,
)


def _make_schema(*col_names: str) -> SchemaDefinition:
    """Helper: build a single-table SchemaDefinition from column names."""
    cols = [
        ColumnDefinition(
            name=name,
            data_type="string",
            is_primary_key=(i == 0),
        )
        for i, name in enumerate(col_names)
    ]
    return SchemaDefinition(tables=[SchemaTable(name="t", columns=cols)])


# ── PII detection tests ────────────────────────────────────────────────────────

def test_presidio_detects_pan_card() -> None:
    """'ABCDE1234F' detected as IN_PAN entity."""
    from synthflow.privacy.presidio_guard import PresidioPrivacyGuard

    guard = PresidioPrivacyGuard()
    df = pd.DataFrame({"pan": ["ABCDE1234F", "normal text", "FGHIJ5678K"]})
    schema = _make_schema("pan")
    _, report = guard.scan_and_mask(df.copy(), schema)
    assert report.entities_detected.get("IN_PAN", 0) > 0, (
        "Expected IN_PAN entities to be detected"
    )


def test_presidio_detects_email() -> None:
    """Email addresses in a column are detected as EMAIL entities."""
    from synthflow.privacy.presidio_guard import PresidioPrivacyGuard

    guard = PresidioPrivacyGuard()
    df = pd.DataFrame({"email": ["user@gmail.com", "test@example.org"]})
    schema = _make_schema("email")
    _, report = guard.scan_and_mask(df.copy(), schema)
    assert report.entities_detected.get("EMAIL", 0) > 0, (
        "Expected EMAIL entities to be detected"
    )


def test_presidio_detects_credit_card() -> None:
    """Luhn-valid Visa test number '4111111111111111' detected as CREDIT_CARD."""
    from synthflow.privacy.presidio_guard import PresidioPrivacyGuard

    guard = PresidioPrivacyGuard()
    df = pd.DataFrame({"card": ["4111111111111111", "some unrelated text"]})
    schema = _make_schema("card")
    _, report = guard.scan_and_mask(df.copy(), schema)
    assert report.entities_detected.get("CREDIT_CARD", 0) > 0, (
        "Expected CREDIT_CARD entity to be detected for '4111111111111111'"
    )


def test_presidio_masked_df_preserves_structure(
    sample_dataframe: pd.DataFrame, sample_schema: SchemaDefinition
) -> None:
    """Masked DataFrame has the same shape and column names as the input."""
    from synthflow.privacy.presidio_guard import PresidioPrivacyGuard

    guard = PresidioPrivacyGuard()
    masked_df, report = guard.scan_and_mask(sample_dataframe.copy(), sample_schema)
    assert masked_df.shape == sample_dataframe.shape, (
        f"Shape mismatch: masked={masked_df.shape}, original={sample_dataframe.shape}"
    )
    assert list(masked_df.columns) == list(sample_dataframe.columns)


# ── k-anonymity tests ──────────────────────────────────────────────────────────

def test_presidio_k_anonymity_k1_scores_40() -> None:
    """k-anonymity=1 (all rows unique) → k-anonymity component scores 40/40."""
    from synthflow.privacy.presidio_guard import PresidioPrivacyGuard

    guard = PresidioPrivacyGuard()
    # 20 fully unique rows → k = 1
    df = pd.DataFrame({
        "name": [f"Person{i}" for i in range(20)],
        "age": list(range(20)),
    })
    quasi_ids = ["name", "age"]
    k = guard.compute_k_anonymity(df, quasi_ids)
    assert k == 1, f"Expected k=1 for fully unique rows, got k={k}"

    # Score k=1 → 40/40 for the k-anonymity component
    score = guard._compute_privacy_score(
        k=1,
        l_diversity=1.0,
        entities_detected={},
        total_cells=40,
    )
    assert score >= 40.0, (
        f"Expected privacy score >= 40.0 for k=1, got {score}"
    )


def test_presidio_k_anonymity_grouped_rows() -> None:
    """k-anonymity correctly identifies minimum group size."""
    from synthflow.privacy.presidio_guard import PresidioPrivacyGuard

    guard = PresidioPrivacyGuard()
    # Group "M/30s" has 5 members → k >= 2 (min of 5 and 2)
    df = pd.DataFrame({
        "gender": ["M", "M", "M", "M", "M", "F", "F"],
        "age_group": ["30s", "30s", "30s", "30s", "30s", "20s", "20s"],
    })
    k = guard.compute_k_anonymity(df, ["gender", "age_group"])
    assert k >= 2, f"Expected k >= 2, got k={k}"


def test_presidio_k_anonymity_empty_quasi_ids() -> None:
    """compute_k_anonymity with empty quasi_ids returns 1 (safe default)."""
    from synthflow.privacy.presidio_guard import PresidioPrivacyGuard

    guard = PresidioPrivacyGuard()
    df = pd.DataFrame({"col": [1, 2, 3]})
    k = guard.compute_k_anonymity(df, [])
    assert k == 1


# ── Privacy score component tests ─────────────────────────────────────────────

def test_presidio_privacy_score_zero_pii_full_score() -> None:
    """With zero PII and k=1, privacy score is at or near 100."""
    from synthflow.privacy.presidio_guard import PresidioPrivacyGuard

    guard = PresidioPrivacyGuard()
    score = guard._compute_privacy_score(
        k=1,
        l_diversity=2.0,
        entities_detected={},
        total_cells=100,
    )
    assert score >= 90.0, f"Expected near-perfect score, got {score}"


def test_presidio_privacy_score_high_pii_penalises() -> None:
    """High PII density reduces the privacy score."""
    from synthflow.privacy.presidio_guard import PresidioPrivacyGuard

    guard = PresidioPrivacyGuard()
    score_clean = guard._compute_privacy_score(
        k=1, l_diversity=2.0, entities_detected={}, total_cells=100
    )
    score_pii = guard._compute_privacy_score(
        k=1, l_diversity=2.0, entities_detected={"EMAIL": 50}, total_cells=100
    )
    assert score_pii < score_clean, (
        "Privacy score with 50 PII entities should be lower than clean score"
    )


def test_presidio_report_has_correct_fields(
    sample_dataframe: pd.DataFrame, sample_schema: SchemaDefinition
) -> None:
    """PrivacyReport returned by scan_and_mask has the required fields."""
    from synthflow.privacy.presidio_guard import PresidioPrivacyGuard
    from synthflow.models.schemas import PrivacyReport

    guard = PresidioPrivacyGuard()
    _, report = guard.scan_and_mask(sample_dataframe.copy(), sample_schema)
    assert isinstance(report, PrivacyReport)
    assert report.k_anonymity >= 1
    assert 0.0 <= report.privacy_score <= 100.0
    assert isinstance(report.entities_detected, dict)
    assert isinstance(report.masked_columns, list)
