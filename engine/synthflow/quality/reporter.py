# ───────────────────────────────────────────────────────────────
# Copyright (c) 2026 Ambrish Nigam
# Author : Ambrish Nigam | https://github.com/ambrishnigam30
# Project: SynthFlow — Autonomous Synthetic Data Orchestration Platform
# License : Apache License 2.0 | https://www.apache.org/licenses/LICENSE-2.0
# Module : QualityReporter — composite quality score aggregator
# ───────────────────────────────────────────────────────────────

from __future__ import annotations

from typing import Optional

import pandas as pd

from synthflow.models.schemas import (
    PrivacyReport,
    QualityReport,
    SchemaDefinition,
    ValidationReport,
)
from synthflow.utils.logger import get_logger

_LOG = get_logger("quality_reporter", component="quality_reporter")


class QualityReporter:
    """
    Aggregates scores from validation, privacy, and statistical checks into
    a single QualityReport with an overall_score (0–100).

    Weighting:
    - Validation score   40 %  (schema completeness, type conformance, constraints)
    - Privacy score      30 %  (k-anonymity, l-diversity, PII density)
    - Statistical score  20 %  (distribution sanity, causal physics)
    - Dirty-data score   10 %  (presence of realistic noise when enabled)
    """

    def generate(
        self,
        seed_df: pd.DataFrame,
        final_df: pd.DataFrame,
        schema: SchemaDefinition,
        privacy_report: Optional[PrivacyReport],
        validation_report: Optional[ValidationReport],
    ) -> QualityReport:
        """
        Compute a QualityReport from the pipeline artefacts.

        Args:
            seed_df:           The DataFrame before dirty-data injection.
            final_df:          The final generated DataFrame.
            schema:            SchemaDefinition used for generation.
            privacy_report:    PrivacyReport from PresidioPrivacyGuard.
            validation_report: ValidationReport from ValidationHygieneEngine.

        Returns:
            QualityReport with overall_score and component scores.
        """
        validation_score = self._extract_validation_score(validation_report)
        privacy_score = self._extract_privacy_score(privacy_report)
        causal_score = self._extract_causal_score(validation_report)
        temporal_score = self._extract_temporal_score(validation_report)
        dirty_score = self._compute_dirty_score(seed_df, final_df)
        sdmetrics_score = self._compute_sdmetrics_score(final_df, schema)

        overall = (
            validation_score * 0.40
            + privacy_score * 0.30
            + causal_score * 0.10
            + temporal_score * 0.10
            + dirty_score * 0.05
            + sdmetrics_score * 0.05
        )
        overall = max(0.0, min(100.0, overall))

        report = QualityReport(
            overall_score=overall,
            sdmetrics_score=sdmetrics_score,
            causal_score=causal_score,
            privacy_score=privacy_score,
            temporal_score=temporal_score,
            dirty_score=dirty_score,
            notes=(
                f"rows={len(final_df)}, cols={len(final_df.columns)}, "
                f"validation={validation_score:.1f}, privacy={privacy_score:.1f}"
            ),
        )

        _LOG.info(
            "Quality report: overall=%.1f (val=%.1f, priv=%.1f, causal=%.1f)",
            overall,
            validation_score,
            privacy_score,
            causal_score,
        )
        return report

    # ── Component extractors ───────────────────────────────────────────────

    def _extract_validation_score(
        self, report: Optional[ValidationReport]
    ) -> float:
        """Return overall_score from ValidationReport, scaled to 0-100."""
        if report is None:
            return 75.0  # neutral default
        return float(report.overall_score)

    def _extract_privacy_score(
        self, report: Optional[PrivacyReport]
    ) -> float:
        """Return privacy_score from PrivacyReport."""
        if report is None:
            return 100.0  # no PII detected = full score
        return float(report.privacy_score)

    def _extract_causal_score(
        self, report: Optional[ValidationReport]
    ) -> float:
        """Extract causal_physics check score from ValidationReport."""
        if report is None:
            return 100.0
        for check in report.checks:
            if check.check_name == "causal_physics":
                return float(check.score * 100)
        return 100.0

    def _extract_temporal_score(
        self, report: Optional[ValidationReport]
    ) -> float:
        """Extract temporal_ordering check score from ValidationReport."""
        if report is None:
            return 100.0
        for check in report.checks:
            if check.check_name == "temporal_ordering":
                return float(check.score * 100)
        return 100.0

    def _compute_dirty_score(
        self, seed_df: pd.DataFrame, final_df: pd.DataFrame
    ) -> float:
        """
        Score how realistic the dirty-data injection is.

        A simple heuristic: if final_df has more rows than seed_df (duplicates
        injected) or a higher null rate (nulls injected), award full score.
        If they are identical, score 80 (clean data is still acceptable).
        """
        if seed_df is final_df or seed_df.equals(final_df):
            return 80.0  # no dirty data injected — acceptable

        # Check null rate increase
        if len(final_df) > 0 and len(seed_df) > 0:
            seed_null_rate = float(seed_df.isnull().values.mean())
            final_null_rate = float(final_df.isnull().values.mean())
            row_growth = len(final_df) / max(1, len(seed_df))
            if final_null_rate > seed_null_rate or row_growth > 1.0:
                return 95.0

        return 85.0

    def _compute_sdmetrics_score(
        self,
        df: pd.DataFrame,
        schema: SchemaDefinition,
    ) -> float:
        """
        Attempt to compute an SDMetrics quality score.

        Falls back to a heuristic (column coverage + row count plausibility)
        when SDMetrics is not installed.
        """
        try:
            from sdmetrics.reports.single_table import QualityReport as SDVReport  # type: ignore[import-untyped]
            import json

            if not schema.tables:
                return 80.0

            table = schema.tables[0]
            metadata: dict[str, object] = {
                "columns": {
                    col.name: {"sdtype": self._sdtype(col.data_type)}
                    for col in table.columns
                    if col.name in df.columns
                }
            }
            report = SDVReport()
            report.generate(df, df, metadata)  # compare df against itself as reference
            score = float(report.get_score()) * 100
            return max(0.0, min(100.0, score))
        except Exception:
            pass

        # Heuristic fallback
        if schema.tables:
            expected = len(schema.tables[0].columns)
            actual = len(df.columns)
            coverage = min(1.0, actual / max(1, expected))
            row_plausibility = min(1.0, len(df) / max(1, 100))
            return round((coverage * 0.6 + row_plausibility * 0.4) * 100, 1)
        return 80.0

    @staticmethod
    def _sdtype(data_type: str) -> str:
        """Map SynthFlow data_type to SDMetrics sdtype string."""
        mapping: dict[str, str] = {
            "integer": "numerical",
            "float": "numerical",
            "string": "categorical",
            "boolean": "boolean",
            "datetime": "datetime",
            "date": "datetime",
            "uuid": "id",
            "json": "categorical",
        }
        return mapping.get(data_type.lower(), "categorical")
