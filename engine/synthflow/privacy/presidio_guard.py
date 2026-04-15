# ───────────────────────────────────────────────────────────────
# Copyright (c) 2026 Ambrish Nigam
# Author : Ambrish Nigam | https://github.com/ambrishnigam30
# Project: SynthFlow — Autonomous Synthetic Data Orchestration Platform
# License : Apache License 2.0 | https://www.apache.org/licenses/LICENSE-2.0
# Module : PresidioPrivacyGuard — PII detection, masking, and privacy scoring
# ───────────────────────────────────────────────────────────────

from __future__ import annotations

import re
import uuid
from typing import Optional

import numpy as np
import pandas as pd

from synthflow.models.schemas import PrivacyReport, SchemaDefinition
from synthflow.utils.logger import get_logger

_LOG = get_logger("presidio_guard", component="presidio_guard")

# ── Regex patterns for custom recognisers ───────────────────────────────────

_PAN_PATTERN = re.compile(r"\b[A-Z]{5}[0-9]{4}[A-Z]\b")
_AADHAR_PATTERN = re.compile(r"\b[2-9]{1}[0-9]{3}\s?[0-9]{4}\s?[0-9]{4}\b")
_GST_PATTERN = re.compile(r"\b[0-9]{2}[A-Z]{5}[0-9]{4}[A-Z][1-9A-Z]Z[0-9A-Z]\b")
_US_SSN_PATTERN = re.compile(r"\b(?!000|666|9\d{2})\d{3}-(?!00)\d{2}-(?!0000)\d{4}\b")
_UK_NHS_PATTERN = re.compile(r"\b[0-9]{3}\s?[0-9]{3}\s?[0-9]{4}\b")
_IBAN_PATTERN = re.compile(r"\b[A-Z]{2}[0-9]{2}[A-Z0-9]{4,30}\b")
_CREDIT_CARD_PATTERN = re.compile(r"\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|3[47][0-9]{13}|6(?:011|5[0-9]{2})[0-9]{12})\b")
_IN_PHONE_PATTERN = re.compile(r"(?:\+91[-.\s]?)?(?:0)?[6-9][0-9]{9}\b")
_BR_CPF_PATTERN = re.compile(r"\b[0-9]{3}\.?[0-9]{3}\.?[0-9]{3}-?[0-9]{2}\b")
_EMAIL_PATTERN = re.compile(r"\b[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}\b")

_RECOGNISERS: dict[str, re.Pattern[str]] = {
    "IN_PAN": _PAN_PATTERN,
    "IN_AADHAR": _AADHAR_PATTERN,
    "IN_GST": _GST_PATTERN,
    "US_SSN": _US_SSN_PATTERN,
    "UK_NHS": _UK_NHS_PATTERN,
    "IBAN": _IBAN_PATTERN,
    "CREDIT_CARD": _CREDIT_CARD_PATTERN,
    "IN_PHONE": _IN_PHONE_PATTERN,
    "BR_CPF": _BR_CPF_PATTERN,
    "EMAIL": _EMAIL_PATTERN,
}


def _luhn_checksum(number: str) -> bool:
    """Validate a credit card number using the Luhn algorithm."""
    digits = [int(d) for d in number if d.isdigit()]
    odd_digits = digits[-1::-2]
    even_digits = digits[-2::-2]
    checksum = sum(odd_digits)
    for d in even_digits:
        doubled = d * 2
        checksum += doubled if doubled < 10 else doubled - 9
    return checksum % 10 == 0


def _generate_luhn_valid_cc() -> str:
    """Generate a Luhn-valid 16-digit credit card number (VISA format)."""
    rng = np.random.default_rng()
    while True:
        digits = [4] + list(rng.integers(0, 10, size=14))
        total = 0
        for i, d in enumerate(reversed(digits)):
            if i % 2 == 1:
                d2 = d * 2
                total += d2 if d2 < 10 else d2 - 9
            else:
                total += d
        check = (10 - total % 10) % 10
        number = "".join(str(d) for d in digits) + str(check)
        if _luhn_checksum(number):
            return number


class PresidioPrivacyGuard:
    """
    Privacy guard with custom PII recognisers.

    Scans DataFrames for PII entities (PAN, Aadhar, email, credit card, etc.),
    masks them with realistic replacements, and computes k-anonymity / l-diversity scores.

    Uses regex-based recognisers as a zero-dependency baseline.
    When `presidio-analyzer` is installed, it is used as an additional layer.
    """

    def __init__(self) -> None:
        self._presidio_available = self._check_presidio()

    def _check_presidio(self) -> bool:
        try:
            from presidio_analyzer import AnalyzerEngine  # noqa: F401
            return True
        except ImportError:
            return False

    # ── Primary API ────────────────────────────────────────────────────────

    def scan_and_mask(
        self,
        df: pd.DataFrame,
        schema: Optional[SchemaDefinition] = None,
    ) -> tuple[pd.DataFrame, PrivacyReport]:
        """
        Scan DataFrame for PII, mask it, and compute privacy report.

        Args:
            df:     Input DataFrame (may contain synthetic PII).
            schema: Optional SchemaDefinition for context.

        Returns:
            (masked_df, PrivacyReport) — masked_df has same shape and dtypes.
        """
        df = df.copy()
        entities_detected: dict[str, int] = {}
        masked_columns: list[str] = []

        for col in df.columns:
            if df[col].dtype != object:
                continue
            col_entities, df[col] = self._scan_and_mask_series(df[col])
            if col_entities:
                for entity_type, count in col_entities.items():
                    entities_detected[entity_type] = (
                        entities_detected.get(entity_type, 0) + count
                    )
                masked_columns.append(col)

        # k-anonymity: use string object columns as quasi-identifiers
        quasi_ids = [
            c for c in df.columns
            if df[c].dtype == object and c not in ("patient_id", "id", "uuid")
        ][:5]

        k = self.compute_k_anonymity(df, quasi_ids)
        l_div = self.compute_l_diversity(df, quasi_ids, quasi_ids[0] if quasi_ids else "")

        # Privacy score
        privacy_score = self._compute_privacy_score(
            k=k,
            l_diversity=l_div,
            entities_detected=entities_detected,
            total_cells=df.size,
        )

        report = PrivacyReport(
            k_anonymity=k,
            l_diversity=l_div,
            privacy_score=privacy_score,
            entities_detected=entities_detected,
            masked_columns=masked_columns,
        )

        _LOG.info(
            "Privacy scan: k=%d, l=%.2f, score=%.1f, entities=%s",
            k, l_div, privacy_score, entities_detected,
        )
        return df, report

    def _scan_and_mask_series(
        self, series: pd.Series
    ) -> tuple[dict[str, int], pd.Series]:
        """Scan a single text series for PII and mask matches."""
        entities_found: dict[str, int] = {}
        result = series.copy()

        for entity_type, pattern in _RECOGNISERS.items():
            def _mask_value(val: object, etype: str = entity_type) -> object:
                if not isinstance(val, str):
                    return val
                return pattern.sub(lambda m: self._mask_match(m.group(), etype), val)

            original = result.copy()
            result = result.apply(_mask_value)
            changed = (result != original).sum()
            if changed > 0:
                entities_found[entity_type] = int(changed)

        return entities_found, result

    def _mask_match(self, match: str, entity_type: str) -> str:
        """Replace a PII match with a realistic but fake replacement."""
        if entity_type == "EMAIL":
            # Preserve domain, mask local part
            parts = match.split("@")
            if len(parts) == 2:
                return f"user{abs(hash(match)) % 10000}@{parts[1]}"
            return "user@example.com"

        if entity_type == "IN_PAN":
            # Valid PAN pattern: AAAAA9999A
            letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            rng = np.random.default_rng(abs(hash(match)) % 2**31)
            pan = (
                "".join(rng.choice(list(letters), size=5).tolist())
                + "".join(str(d) for d in rng.integers(0, 10, size=4))
                + rng.choice(list(letters))
            )
            return pan

        if entity_type == "IN_AADHAR":
            rng = np.random.default_rng(abs(hash(match)) % 2**31)
            digits = [str(rng.integers(2, 10))] + [
                str(d) for d in rng.integers(0, 10, size=11)
            ]
            return " ".join(["".join(digits[:4]), "".join(digits[4:8]), "".join(digits[8:12])])

        if entity_type == "CREDIT_CARD":
            return _generate_luhn_valid_cc()

        if entity_type == "IN_PHONE":
            rng = np.random.default_rng(abs(hash(match)) % 2**31)
            num = str(rng.integers(6, 10)) + "".join(
                str(d) for d in rng.integers(0, 10, size=9)
            )
            return f"+91-{num}"

        if entity_type == "US_SSN":
            rng = np.random.default_rng(abs(hash(match)) % 2**31)
            return f"{rng.integers(100,999)}-{rng.integers(10,99)}-{rng.integers(1000,9999)}"

        if entity_type == "UK_NHS":
            rng = np.random.default_rng(abs(hash(match)) % 2**31)
            return f"{rng.integers(100,999)} {rng.integers(100,999)} {rng.integers(1000,9999)}"

        # Generic: return a UUID fragment
        return f"[{entity_type}:{str(uuid.uuid4())[:8]}]"

    # ── k-anonymity ────────────────────────────────────────────────────────

    def compute_k_anonymity(
        self,
        df: pd.DataFrame,
        quasi_ids: list[str],
    ) -> int:
        """
        Compute k-anonymity over the given quasi-identifier columns.

        k=1 means every row is unique — perfect privacy for synthetic data (40/40 score).

        Args:
            df:         DataFrame to analyse.
            quasi_ids:  List of quasi-identifier column names.

        Returns:
            Minimum group size (k). Returns 1 if no quasi-ids or all rows unique.
        """
        if not quasi_ids:
            return 1
        available = [c for c in quasi_ids if c in df.columns]
        if not available:
            return 1
        try:
            group_sizes = df.groupby(available, dropna=False).size()
            return int(group_sizes.min())
        except Exception as exc:
            _LOG.warning("k-anonymity computation failed: %s", exc)
            return 1

    def compute_l_diversity(
        self,
        df: pd.DataFrame,
        quasi_ids: list[str],
        sensitive_col: str,
    ) -> float:
        """
        Compute l-diversity: minimum distinct sensitive values per equivalence class.

        Args:
            df:            DataFrame to analyse.
            quasi_ids:     Quasi-identifier columns.
            sensitive_col: Sensitive column to measure diversity on.

        Returns:
            Minimum distinct values as float. Returns 1.0 if not applicable.
        """
        if not quasi_ids or sensitive_col not in df.columns:
            return 1.0
        available = [c for c in quasi_ids if c in df.columns]
        if not available:
            return 1.0
        try:
            diversities = df.groupby(available, dropna=False)[sensitive_col].nunique()
            return float(diversities.min())
        except Exception as exc:
            _LOG.warning("l-diversity computation failed: %s", exc)
            return 1.0

    # ── Privacy score ──────────────────────────────────────────────────────

    def _compute_privacy_score(
        self,
        k: int,
        l_diversity: float,
        entities_detected: dict[str, int],
        total_cells: int,
    ) -> float:
        """
        Compute composite privacy score 0–100.

        Components:
        - k-anonymity (40 pts): k=1 → 40/40 (every row unique = perfect for synthetic data)
        - l-diversity (20 pts): l≥2 → full score
        - PII density  (40 pts): fraction of cells with PII (0 PII = 40/40)
        """
        # k-anonymity: k=1 (all unique) is the best for synthetic data
        k_score = 40.0 if k == 1 else min(40.0, 40.0 * (1.0 / max(1, k)))

        # l-diversity
        l_score = min(20.0, 20.0 * (l_diversity / 2.0)) if l_diversity < 2.0 else 20.0

        # PII residual density
        total_pii = sum(entities_detected.values())
        pii_rate = total_pii / max(1, total_cells)
        pii_score = 40.0 * max(0.0, 1.0 - pii_rate * 10)

        return min(100.0, k_score + l_score + pii_score)
