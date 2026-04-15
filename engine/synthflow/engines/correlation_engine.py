# ───────────────────────────────────────────────────────────────
# Copyright (c) 2026 Ambrish Nigam
# Author : Ambrish Nigam | https://github.com/ambrishnigam30
# Project: SynthFlow — Autonomous Synthetic Data Orchestration Platform
# License : Apache License 2.0 | https://www.apache.org/licenses/LICENSE-2.0
# Module : CorrelationDriftEngine — Spearman drift detection + correction loop
# ───────────────────────────────────────────────────────────────

from __future__ import annotations

from typing import Any, Optional, Union

import numpy as np
import pandas as pd
from scipy.stats import spearmanr  # type: ignore[import-untyped]

from synthflow.llm_client import LLMClient, MockLLMClient
from synthflow.models.schemas import (
    CausalKnowledgeBundle,
    CrossColumnCorrelation,
    DriftEvent,
)
from synthflow.utils.logger import get_logger

_LOG = get_logger("correlation_engine", component="correlation_engine")


class CorrelationDriftEngine:
    """
    Detects and corrects correlation drift between actual and target correlations.

    Workflow:
    1. compute_actual_correlations(df) → Spearman matrix
    2. detect_drift(actual, target, threshold) → List[DriftEvent]
    3. correct_drift_loop(df, drift_events, ...) → corrected DataFrame
    """

    def __init__(self, llm_client: Optional[Union[LLMClient, MockLLMClient]] = None) -> None:
        self._llm = llm_client

    def compute_actual_correlations(
        self, df: pd.DataFrame
    ) -> dict[tuple[str, str], float]:
        """
        Compute Spearman rank correlations for all numeric column pairs.

        Args:
            df: DataFrame to analyze.

        Returns:
            Dict mapping (col_a, col_b) → Spearman rho.
        """
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        result: dict[tuple[str, str], float] = {}

        for i, col_a in enumerate(numeric_cols):
            for col_b in numeric_cols[i + 1:]:
                mask = df[col_a].notna() & df[col_b].notna()
                if mask.sum() < 3:
                    continue
                try:
                    rho, _ = spearmanr(df.loc[mask, col_a], df.loc[mask, col_b])
                    result[(col_a, col_b)] = float(rho)
                except Exception:
                    result[(col_a, col_b)] = 0.0

        return result

    def detect_drift(
        self,
        actual: dict[tuple[str, str], float],
        target_correlations: list[CrossColumnCorrelation],
        threshold: float = 0.15,
    ) -> list[DriftEvent]:
        """
        Identify pairs where |actual_rho - target_rho| > *threshold*.

        Args:
            actual:               Computed Spearman matrix.
            target_correlations:  Target correlations from knowledge bundle.
            threshold:            Drift tolerance (default 0.15).

        Returns:
            List of DriftEvent objects for pairs that exceed threshold.
        """
        events: list[DriftEvent] = []
        for cor in target_correlations:
            col_a, col_b = cor.col_a, cor.col_b
            actual_rho = actual.get(
                (col_a, col_b),
                actual.get((col_b, col_a), None),
            )
            if actual_rho is None:
                continue
            target_rho = float(cor.strength)
            delta = abs(actual_rho - target_rho)
            if delta > threshold:
                events.append(DriftEvent(
                    col_a=col_a,
                    col_b=col_b,
                    actual_rho=actual_rho,
                    target_rho=target_rho,
                    delta=delta,
                ))
        return events

    async def correct_drift_loop(
        self,
        df: pd.DataFrame,
        drift_events: list[DriftEvent],
        knowledge: CausalKnowledgeBundle,
        script: str,
        max_iterations: int = 3,
    ) -> pd.DataFrame:
        """
        Iteratively attempt to reduce correlation drift.

        For each iteration, re-rank the drifted columns to nudge
        correlations toward their targets using the Iman-Conover method.

        Args:
            df:             DataFrame with correlation drift.
            drift_events:   List of DriftEvent objects.
            knowledge:      Domain knowledge (for context).
            script:         Generation script (not used in simple correction).
            max_iterations: Maximum correction rounds (default 3).

        Returns:
            Corrected DataFrame (best-effort).
        """
        result = df.copy()
        for iteration in range(max_iterations):
            remaining_drift = []
            for event in drift_events:
                corrected = self._nudge_correlation(
                    result, event.col_a, event.col_b,
                    event.target_rho, iteration,
                )
                if corrected is not None:
                    result = corrected
                # Re-check drift
                actual_check = self.compute_actual_correlations(
                    result[[event.col_a, event.col_b]].dropna()
                    if event.col_a in result and event.col_b in result
                    else pd.DataFrame()
                )
                new_rho = actual_check.get(
                    (event.col_a, event.col_b),
                    actual_check.get((event.col_b, event.col_a), None),
                )
                if new_rho is not None:
                    new_delta = abs(new_rho - event.target_rho)
                    if new_delta > 0.15:
                        remaining_drift.append(event)
            drift_events = remaining_drift
            if not drift_events:
                break
        return result

    def _nudge_correlation(
        self,
        df: pd.DataFrame,
        col_a: str,
        col_b: str,
        target_rho: float,
        iteration: int,
    ) -> Optional[pd.DataFrame]:
        """Blend a column toward target correlation using rank manipulation."""
        if col_a not in df.columns or col_b not in df.columns:
            return None
        try:
            step = 0.3 * (0.7 ** iteration)
            a_vals = df[col_a].values.astype(float)
            b_vals = df[col_b].values.astype(float)

            a_ranks = np.argsort(np.argsort(a_vals))
            b_ranks = np.argsort(np.argsort(b_vals))

            if target_rho > 0:
                blended = (1 - step) * b_ranks + step * a_ranks
            else:
                blended = (1 - step) * b_ranks + step * (len(b_ranks) - 1 - a_ranks)

            new_b_order = np.argsort(blended)
            sorted_b = np.sort(b_vals)
            result = df.copy()
            result[col_b] = sorted_b[np.argsort(new_b_order)]
            return result
        except Exception as exc:
            _LOG.debug("Nudge correlation failed for (%s, %s): %s", col_a, col_b, exc)
            return None
