# ───────────────────────────────────────────────────────────────
# Copyright (c) 2026 Ambrish Nigam
# Author : Ambrish Nigam | https://github.com/ambrishnigam30
# Project: SynthFlow — Autonomous Synthetic Data Orchestration Platform
# License : Apache License 2.0 | https://www.apache.org/licenses/LICENSE-2.0
# Module : PatternRhythmLibrary — temporal patterns, autocorrelation, seasonality
# ───────────────────────────────────────────────────────────────

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd

from synthflow.models.schemas import TemporalPatterns
from synthflow.utils.logger import get_logger

_LOG = get_logger("pattern_library", component="pattern_library")


class PatternRhythmLibrary:
    """
    Applies temporal rhythms and autocorrelation patterns to DataFrames.

    Capabilities:
    - Day-of-week weighting on timestamp columns
    - Hour-of-day weighting
    - Monthly seasonality multipliers on numeric columns
    - Special event multipliers
    - Autocorrelation (AR(1) re-ordering)
    """

    def apply_temporal_patterns(
        self,
        df: pd.DataFrame,
        patterns: TemporalPatterns,
        timestamp_columns: list[str],
    ) -> pd.DataFrame:
        """
        Apply temporal patterns to *df*.

        Args:
            df:                DataFrame to modify.
            patterns:          TemporalPatterns from knowledge bundle.
            timestamp_columns: Names of datetime columns to redistribute.

        Returns:
            DataFrame with resampled timestamps and seasonal numeric adjustments.
        """
        result = df.copy()

        for col in timestamp_columns:
            if col not in result.columns:
                continue
            result = self._apply_dow_weights(result, col, patterns)
            result = self._apply_hod_weights(result, col, patterns)

        result = self._apply_monthly_seasonality(result, patterns, timestamp_columns)
        result = self._apply_special_events(result, patterns, timestamp_columns)

        return result

    def apply_autocorrelation(
        self,
        df: pd.DataFrame,
        rho: float,
        ts_columns: list[str],
        sort_by: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Re-order rows to introduce AR(1) autocorrelation in numeric columns.

        Uses the Iman-Conover approach: re-rank a numeric column to match
        a correlated permutation of the row indices.

        Args:
            df:         Input DataFrame.
            rho:        Target autocorrelation coefficient (−1 to 1).
            ts_columns: Timestamp columns to sort by first.
            sort_by:    Optional column to sort by before applying autocorrelation.

        Returns:
            DataFrame with row order adjusted for autocorrelation.
        """
        if abs(rho) < 0.01:
            return df

        result = df.copy()

        # Sort by timestamp if available
        sort_col = sort_by or (ts_columns[0] if ts_columns else None)
        if sort_col and sort_col in result.columns:
            try:
                result = result.sort_values(sort_col).reset_index(drop=True)
            except Exception:
                pass

        # Apply AR(1) to numeric columns
        numeric_cols = result.select_dtypes(include=[np.number]).columns.tolist()
        n = len(result)
        if n < 2:
            return result

        rng = np.random.default_rng(42)
        # Generate correlated index order
        noise = rng.standard_normal(n)
        base = np.arange(n, dtype=float)
        correlated = rho * (base / n) + np.sqrt(1 - rho**2) * noise
        new_order = np.argsort(correlated)

        for col in numeric_cols:
            original_sorted = np.sort(result[col].values)
            result[col] = original_sorted[np.argsort(new_order)]

        return result

    # ── Private helpers ────────────────────────────────────────────────────

    def _apply_dow_weights(
        self,
        df: pd.DataFrame,
        col: str,
        patterns: TemporalPatterns,
    ) -> pd.DataFrame:
        """Redistribute timestamps to match day-of-week weights."""
        weights = patterns.day_of_week_weights
        if not weights or len(weights) != 7:
            return df
        try:
            ts_col = pd.to_datetime(df[col], errors="coerce")
            valid_mask = ts_col.notna()
            if not valid_mask.any():
                return df
            ts_valid = ts_col[valid_mask]
            n = valid_mask.sum()
            days = np.random.default_rng(42).choice(7, size=n, p=weights)
            # Shift timestamps to target days
            new_dates = []
            for orig, target_dow in zip(ts_valid, days):
                delta = (target_dow - orig.weekday()) % 7
                new_dates.append(orig + timedelta(days=int(delta)))
            result = df.copy()
            result.loc[valid_mask, col] = new_dates
            return result
        except Exception as exc:
            _LOG.debug("DOW weight application failed for %s: %s", col, exc)
            return df

    def _apply_hod_weights(
        self,
        df: pd.DataFrame,
        col: str,
        patterns: TemporalPatterns,
    ) -> pd.DataFrame:
        """Set hour-of-day in timestamps to match hour weights."""
        weights = patterns.hour_of_day_weights
        if not weights or len(weights) != 24:
            return df
        try:
            ts_col = pd.to_datetime(df[col], errors="coerce")
            valid_mask = ts_col.notna()
            if not valid_mask.any():
                return df
            ts_valid = ts_col[valid_mask]
            n = valid_mask.sum()
            hours = np.random.default_rng(99).choice(24, size=n, p=weights)
            new_dates = [
                ts.replace(hour=int(h), minute=0, second=0)
                for ts, h in zip(ts_valid, hours)
            ]
            result = df.copy()
            result.loc[valid_mask, col] = new_dates
            return result
        except Exception as exc:
            _LOG.debug("HOD weight application failed for %s: %s", col, exc)
            return df

    def _apply_monthly_seasonality(
        self,
        df: pd.DataFrame,
        patterns: TemporalPatterns,
        timestamp_columns: list[str],
    ) -> pd.DataFrame:
        """Apply monthly multipliers to numeric columns."""
        if not patterns.monthly_seasonality or not timestamp_columns:
            return df
        ts_col = next((c for c in timestamp_columns if c in df.columns), None)
        if ts_col is None:
            return df
        result = df.copy()
        try:
            ts_series = pd.to_datetime(result[ts_col], errors="coerce")
            numeric_cols = result.select_dtypes(include=[np.number]).columns.tolist()
            for month_str, multiplier in patterns.monthly_seasonality.items():
                month = int(month_str)
                mask = ts_series.dt.month == month
                if mask.any():
                    for nc in numeric_cols:
                        if nc in result.columns:
                            result.loc[mask, nc] = result.loc[mask, nc] * multiplier
        except Exception as exc:
            _LOG.debug("Monthly seasonality failed: %s", exc)
        return result

    def _apply_special_events(
        self,
        df: pd.DataFrame,
        patterns: TemporalPatterns,
        timestamp_columns: list[str],
    ) -> pd.DataFrame:
        """Apply special event multipliers (e.g., Diwali sales spike)."""
        if not patterns.special_events or not timestamp_columns:
            return df
        ts_col = next((c for c in timestamp_columns if c in df.columns), None)
        if ts_col is None:
            return df
        result = df.copy()
        try:
            ts_series = pd.to_datetime(result[ts_col], errors="coerce")
            for event in patterns.special_events:
                mask = ts_series.dt.month == event.month
                if event.day is not None:
                    mask &= ts_series.dt.day == event.day
                if mask.any():
                    for ac in event.affected_columns:
                        if ac in result.columns:
                            try:
                                result.loc[mask, ac] = (
                                    pd.to_numeric(result.loc[mask, ac], errors="coerce")
                                    * event.day_multiplier
                                )
                            except Exception:
                                pass
        except Exception as exc:
            _LOG.debug("Special event application failed: %s", exc)
        return result
