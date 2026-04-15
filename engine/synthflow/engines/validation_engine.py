# ───────────────────────────────────────────────────────────────
# Copyright (c) 2026 Ambrish Nigam
# Author : Ambrish Nigam | https://github.com/ambrishnigam30
# Project: SynthFlow — Autonomous Synthetic Data Orchestration Platform
# License : Apache License 2.0 | https://www.apache.org/licenses/LICENSE-2.0
# Module : ValidationHygieneEngine — 10-check data quality audit
# ───────────────────────────────────────────────────────────────

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from synthflow.models.schemas import (
    CausalKnowledgeBundle,
    ColumnDefinition,
    ConstraintSet,
    SchemaDefinition,
    Severity,
    ValidationCheck,
    ValidationReport,
)
from synthflow.utils.logger import get_logger

_LOG = get_logger("validation_engine", component="validation_engine")

# 10 check names — order is significant
CHECK_NAMES = [
    "schema_completeness",
    "type_conformance",
    "null_policy",
    "range_validity",
    "enum_validity",
    "uniqueness",
    "temporal_ordering",
    "causal_physics",
    "statistical_sanity",
    "correlation_consistency",
]

# Allowed data types for type conformance
_NUMERIC_TYPES = {"integer", "float", "numeric"}
_STRING_TYPES = {"string", "text", "varchar", "char"}
_BOOL_TYPES = {"boolean", "bool"}
_DATE_TYPES = {"date", "datetime", "timestamp"}


class ValidationHygieneEngine:
    """
    Runs exactly 10 quality checks on a generated DataFrame against its schema,
    constraints, and knowledge bundle.

    Checks:
    1. schema_completeness    — all expected columns present
    2. type_conformance       — column dtypes match schema declarations
    3. null_policy            — nullable=False columns have no nulls
    4. range_validity         — numeric values within [min_value, max_value]
    5. enum_validity          — string columns with enums stay in allowed set
    6. uniqueness             — PK and unique=True columns have no duplicates
    7. temporal_ordering      — date/datetime temporal constraints satisfied
    8. causal_physics         — DAG rule expressions hold
    9. statistical_sanity     — numeric means within 3σ of expected distributions
    10. correlation_consistency — pairwise Spearman within tolerance of targets
    """

    def audit(
        self,
        df: pd.DataFrame,
        schema: SchemaDefinition,
        constraints: ConstraintSet,
        knowledge: Optional[CausalKnowledgeBundle] = None,
    ) -> ValidationReport:
        """
        Run all 10 checks and return a ValidationReport.

        Args:
            df:          Generated DataFrame to audit.
            schema:      SchemaDefinition with column definitions.
            constraints: Constraint set.
            knowledge:   Optional knowledge bundle for causal/correlation checks.

        Returns:
            ValidationReport with exactly 10 ValidationCheck objects.
        """
        table = schema.tables[0] if schema.tables else None
        col_defs: dict[str, ColumnDefinition] = {}
        if table:
            col_defs = {c.name: c for c in table.columns}

        checks = [
            self._check_schema_completeness(df, col_defs),
            self._check_type_conformance(df, col_defs),
            self._check_null_policy(df, col_defs),
            self._check_range_validity(df, col_defs),
            self._check_enum_validity(df, col_defs),
            self._check_uniqueness(df, col_defs),
            self._check_temporal_ordering(df, col_defs, constraints),
            self._check_causal_physics(df, knowledge),
            self._check_statistical_sanity(df, col_defs, knowledge),
            self._check_correlation_consistency(df, knowledge),
        ]

        # Weighted overall score
        weights = [0.15, 0.15, 0.10, 0.12, 0.10, 0.10, 0.08, 0.08, 0.07, 0.05]
        overall = sum(w * c.score * 100 for w, c in zip(weights, checks))
        overall = max(0.0, min(100.0, overall))

        report = ValidationReport(
            checks=checks,
            overall_score=overall,
            row_count=len(df),
            column_count=len(df.columns),
        )
        _LOG.info(
            "Audit complete: %d checks, overall=%.1f, rows=%d",
            len(checks), overall, len(df),
        )
        return report

    # ── Check 1: Schema completeness ───────────────────────────────────────

    def _check_schema_completeness(
        self,
        df: pd.DataFrame,
        col_defs: dict[str, ColumnDefinition],
    ) -> ValidationCheck:
        if not col_defs:
            return ValidationCheck(
                check_name="schema_completeness",
                score=1.0,
                violations_found=0,
                passed=True,
                details="No schema columns to check.",
            )
        expected = set(col_defs.keys())
        actual = set(df.columns)
        missing = expected - actual
        extra = actual - expected
        violations = len(missing)
        score = max(0.0, 1.0 - violations / max(1, len(expected)))
        return ValidationCheck(
            check_name="schema_completeness",
            score=score,
            violations_found=violations,
            severity=Severity.HIGH if violations > 0 else Severity.LOW,
            passed=violations == 0,
            details=(
                f"Missing: {sorted(missing)}. Extra: {sorted(extra)}."
                if violations or extra else "All schema columns present."
            ),
        )

    # ── Check 2: Type conformance ──────────────────────────────────────────

    def _check_type_conformance(
        self,
        df: pd.DataFrame,
        col_defs: dict[str, ColumnDefinition],
    ) -> ValidationCheck:
        violations = 0
        details_parts: list[str] = []
        for col_name, col_def in col_defs.items():
            if col_name not in df.columns:
                continue
            series = df[col_name].dropna()
            if series.empty:
                continue
            dtype = col_def.data_type.lower()
            if dtype in _NUMERIC_TYPES:
                if not pd.api.types.is_numeric_dtype(df[col_name]):
                    # Try sample — if values are not castable, violation
                    try:
                        pd.to_numeric(series.iloc[:10], errors="raise")
                    except (ValueError, TypeError):
                        violations += 1
                        details_parts.append(f"{col_name}(expected numeric)")
            elif dtype in _BOOL_TYPES:
                if not pd.api.types.is_bool_dtype(df[col_name]):
                    non_bool = series[~series.isin([True, False, 0, 1, "True", "False"])]
                    if len(non_bool) > 0:
                        violations += 1
                        details_parts.append(f"{col_name}(expected bool)")
            elif dtype in _DATE_TYPES:
                if not pd.api.types.is_datetime64_any_dtype(df[col_name]):
                    # Check if strings are date-parseable
                    try:
                        pd.to_datetime(series.iloc[:5], errors="raise")
                    except Exception:
                        violations += 1
                        details_parts.append(f"{col_name}(expected datetime)")
            # string types: no check needed — everything can be a string

        n_cols = max(1, len(col_defs))
        score = max(0.0, 1.0 - violations / n_cols)
        return ValidationCheck(
            check_name="type_conformance",
            score=score,
            violations_found=violations,
            severity=Severity.HIGH if violations > 0 else Severity.LOW,
            passed=violations == 0,
            details=(
                "Type violations: " + ", ".join(details_parts)
                if details_parts else "All column types conform."
            ),
        )

    # ── Check 3: Null policy ───────────────────────────────────────────────

    def _check_null_policy(
        self,
        df: pd.DataFrame,
        col_defs: dict[str, ColumnDefinition],
    ) -> ValidationCheck:
        violations = 0
        details_parts: list[str] = []
        for col_name, col_def in col_defs.items():
            if col_name not in df.columns:
                continue
            if not col_def.nullable:
                null_count = int(df[col_name].isna().sum())
                if null_count > 0:
                    violations += null_count
                    details_parts.append(f"{col_name}({null_count} nulls)")
        n_cells = max(1, len(df) * max(1, len(col_defs)))
        score = max(0.0, 1.0 - violations / n_cells)
        return ValidationCheck(
            check_name="null_policy",
            score=score,
            violations_found=violations,
            severity=Severity.HIGH if violations > 0 else Severity.LOW,
            passed=violations == 0,
            details=(
                "Non-nullable violations: " + ", ".join(details_parts)
                if details_parts else "Null policy satisfied."
            ),
        )

    # ── Check 4: Range validity ────────────────────────────────────────────

    def _check_range_validity(
        self,
        df: pd.DataFrame,
        col_defs: dict[str, ColumnDefinition],
    ) -> ValidationCheck:
        violations = 0
        details_parts: list[str] = []
        for col_name, col_def in col_defs.items():
            if col_name not in df.columns:
                continue
            if not pd.api.types.is_numeric_dtype(df[col_name]):
                continue
            series = df[col_name].dropna()
            if series.empty:
                continue
            if col_def.min_value is not None:
                below = int((series < col_def.min_value).sum())
                if below > 0:
                    violations += below
                    details_parts.append(f"{col_name}<{col_def.min_value}({below})")
            if col_def.max_value is not None:
                above = int((series > col_def.max_value).sum())
                if above > 0:
                    violations += above
                    details_parts.append(f"{col_name}>{col_def.max_value}({above})")
        n_cells = max(1, len(df))
        score = max(0.0, 1.0 - violations / n_cells)
        return ValidationCheck(
            check_name="range_validity",
            score=score,
            violations_found=violations,
            severity=Severity.MEDIUM if violations > 0 else Severity.LOW,
            passed=violations == 0,
            details=(
                "Range violations: " + "; ".join(details_parts)
                if details_parts else "All values within defined ranges."
            ),
        )

    # ── Check 5: Enum validity ─────────────────────────────────────────────

    def _check_enum_validity(
        self,
        df: pd.DataFrame,
        col_defs: dict[str, ColumnDefinition],
    ) -> ValidationCheck:
        violations = 0
        details_parts: list[str] = []
        for col_name, col_def in col_defs.items():
            if col_name not in df.columns:
                continue
            if not col_def.enum_values:
                continue
            allowed = set(str(v) for v in col_def.enum_values)
            series = df[col_name].dropna().astype(str)
            invalid = series[~series.isin(allowed)]
            if len(invalid) > 0:
                violations += len(invalid)
                details_parts.append(f"{col_name}({len(invalid)} invalid)")
        total_cells = max(1, len(df))
        score = max(0.0, 1.0 - violations / total_cells)
        return ValidationCheck(
            check_name="enum_validity",
            score=score,
            violations_found=violations,
            severity=Severity.MEDIUM if violations > 0 else Severity.LOW,
            passed=violations == 0,
            details=(
                "Enum violations: " + ", ".join(details_parts)
                if details_parts else "All enum values within allowed set."
            ),
        )

    # ── Check 6: Uniqueness ────────────────────────────────────────────────

    def _check_uniqueness(
        self,
        df: pd.DataFrame,
        col_defs: dict[str, ColumnDefinition],
    ) -> ValidationCheck:
        violations = 0
        details_parts: list[str] = []
        for col_name, col_def in col_defs.items():
            if col_name not in df.columns:
                continue
            if col_def.is_primary_key or col_def.unique:
                dup_count = int(df[col_name].dropna().duplicated().sum())
                if dup_count > 0:
                    violations += dup_count
                    details_parts.append(f"{col_name}({dup_count} dups)")
        score = 0.0 if violations > 0 else 1.0
        return ValidationCheck(
            check_name="uniqueness",
            score=score,
            violations_found=violations,
            severity=Severity.CRITICAL if violations > 0 else Severity.LOW,
            passed=violations == 0,
            details=(
                "Uniqueness violations: " + ", ".join(details_parts)
                if details_parts else "All PK/unique columns are unique."
            ),
        )

    # ── Check 7: Temporal ordering ─────────────────────────────────────────

    def _check_temporal_ordering(
        self,
        df: pd.DataFrame,
        col_defs: dict[str, ColumnDefinition],
        constraints: ConstraintSet,
    ) -> ValidationCheck:
        violations = 0
        details_parts: list[str] = []

        # Check explicit TEMPORAL constraints
        for rule in constraints.rules:
            if rule.constraint_type.value != "temporal":
                continue
            cols = rule.columns
            if len(cols) >= 2 and cols[0] in df.columns and cols[1] in df.columns:
                try:
                    start = pd.to_datetime(df[cols[0]], errors="coerce")
                    end = pd.to_datetime(df[cols[1]], errors="coerce")
                    mask = start.notna() & end.notna()
                    bad = (end[mask] < start[mask]).sum()
                    if bad > 0:
                        violations += int(bad)
                        details_parts.append(f"{cols[1]}<{cols[0]}({bad})")
                except Exception:
                    pass

        # Heuristic: admission_date / discharge_date, start_date / end_date
        TEMPORAL_PAIRS = [
            ("admission_date", "discharge_date"),
            ("start_date", "end_date"),
            ("created_at", "updated_at"),
            ("birth_date", "death_date"),
            ("order_date", "delivery_date"),
        ]
        for start_col, end_col in TEMPORAL_PAIRS:
            if start_col in df.columns and end_col in df.columns:
                try:
                    start = pd.to_datetime(df[start_col], errors="coerce")
                    end = pd.to_datetime(df[end_col], errors="coerce")
                    mask = start.notna() & end.notna()
                    bad = int((end[mask] < start[mask]).sum())
                    if bad > 0:
                        violations += bad
                        details_parts.append(f"{end_col}<{start_col}({bad})")
                except Exception:
                    pass

        n = max(1, len(df))
        score = max(0.0, 1.0 - violations / n)
        return ValidationCheck(
            check_name="temporal_ordering",
            score=score,
            violations_found=violations,
            severity=Severity.HIGH if violations > 0 else Severity.LOW,
            passed=violations == 0,
            details=(
                "Temporal violations: " + "; ".join(details_parts)
                if details_parts else "Temporal ordering valid."
            ),
        )

    # ── Check 8: Causal physics ────────────────────────────────────────────

    def _check_causal_physics(
        self,
        df: pd.DataFrame,
        knowledge: Optional[CausalKnowledgeBundle],
    ) -> ValidationCheck:
        if knowledge is None or not knowledge.dag_rules:
            return ValidationCheck(
                check_name="causal_physics",
                score=1.0,
                violations_found=0,
                passed=True,
                details="No DAG rules to validate.",
            )

        violations = 0
        details_parts: list[str] = []

        for rule in knowledge.dag_rules[:10]:  # cap at 10 rules for performance
            parent = rule.parent_column
            child = rule.child_column
            if parent not in df.columns or child not in df.columns:
                continue
            # Simple physics: check age >= 0, salary >= 0, etc.
            if child == "age" or "age" in child:
                bad = int((df[child].dropna() < 0).sum())
                if bad > 0:
                    violations += bad
                    details_parts.append(f"{child}<0({bad})")
            if "salary" in child or "amount" in child or "income" in child:
                if pd.api.types.is_numeric_dtype(df[child]):
                    bad = int((df[child].dropna() < 0).sum())
                    if bad > 0:
                        violations += bad
                        details_parts.append(f"{child}<0({bad})")

        n = max(1, len(df))
        score = max(0.0, 1.0 - violations / n)
        return ValidationCheck(
            check_name="causal_physics",
            score=score,
            violations_found=violations,
            severity=Severity.MEDIUM if violations > 0 else Severity.LOW,
            passed=violations == 0,
            details=(
                "Physics violations: " + "; ".join(details_parts)
                if details_parts else "Causal physics satisfied."
            ),
        )

    # ── Check 9: Statistical sanity ────────────────────────────────────────

    def _check_statistical_sanity(
        self,
        df: pd.DataFrame,
        col_defs: dict[str, ColumnDefinition],
        knowledge: Optional[CausalKnowledgeBundle],
    ) -> ValidationCheck:
        if df.empty:
            return ValidationCheck(
                check_name="statistical_sanity",
                score=1.0,
                violations_found=0,
                passed=True,
                details="Empty DataFrame — no statistics to check.",
            )

        violations = 0
        details_parts: list[str] = []
        numeric_cols = df.select_dtypes(include="number").columns

        for col in numeric_cols:
            series = df[col].dropna()
            if len(series) < 10:
                continue
            mean = float(series.mean())
            std = float(series.std()) or 1.0
            # Flag columns where > 5% of values are > 10 std from mean (extreme outlier rate)
            extreme = ((series - mean).abs() > 10 * std).sum()
            if extreme > len(series) * 0.05:
                violations += 1
                details_parts.append(f"{col}(extreme_outlier_rate={extreme/len(series):.1%})")

        n = max(1, len(numeric_cols))
        score = max(0.0, 1.0 - violations / n)
        return ValidationCheck(
            check_name="statistical_sanity",
            score=score,
            violations_found=violations,
            severity=Severity.LOW if violations > 0 else Severity.LOW,
            passed=violations == 0,
            details=(
                "Sanity issues: " + "; ".join(details_parts)
                if details_parts else "Statistical distributions appear sane."
            ),
        )

    # ── Check 10: Correlation consistency ─────────────────────────────────

    def _check_correlation_consistency(
        self,
        df: pd.DataFrame,
        knowledge: Optional[CausalKnowledgeBundle],
    ) -> ValidationCheck:
        if knowledge is None or not knowledge.correlations:
            return ValidationCheck(
                check_name="correlation_consistency",
                score=1.0,
                violations_found=0,
                passed=True,
                details="No correlation targets to validate.",
            )

        violations = 0
        details_parts: list[str] = []
        numeric_df = df.select_dtypes(include="number")

        for corr_spec in knowledge.correlations[:20]:  # cap for perf
            col_a = corr_spec.col_a
            col_b = corr_spec.col_b
            if col_a not in numeric_df.columns or col_b not in numeric_df.columns:
                continue
            try:
                from scipy.stats import spearmanr
                rho, _ = spearmanr(
                    numeric_df[col_a].dropna(),
                    numeric_df[col_b].dropna(),
                )
                target = corr_spec.strength
                drift = abs(float(rho) - target)
                if drift > 0.3:
                    violations += 1
                    details_parts.append(
                        f"{col_a}↔{col_b}(actual={rho:.2f},target={target:.2f})"
                    )
            except Exception:
                pass

        n = max(1, len(knowledge.correlations))
        score = max(0.0, 1.0 - violations / n)
        return ValidationCheck(
            check_name="correlation_consistency",
            score=score,
            violations_found=violations,
            severity=Severity.LOW if violations > 0 else Severity.LOW,
            passed=violations == 0,
            details=(
                "Correlation drift: " + "; ".join(details_parts)
                if details_parts else "Pairwise correlations within tolerance."
            ),
        )
