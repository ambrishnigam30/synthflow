# ───────────────────────────────────────────────────────────────
# Copyright (c) 2026 Ambrish Nigam
# Author : Ambrish Nigam | https://github.com/ambrishnigam30
# Project: SynthFlow — Autonomous Synthetic Data Orchestration Platform
# License : Apache License 2.0 | https://www.apache.org/licenses/LICENSE-2.0
# Module : AnomalyOutlierEngine — injects realistic dirty data into DataFrames
# ───────────────────────────────────────────────────────────────

from __future__ import annotations

import random
from typing import Optional

import numpy as np
import pandas as pd

from synthflow.models.schemas import DirtyDataProfile, NullMechanism
from synthflow.utils.logger import get_logger

_LOG = get_logger("anomaly_engine", component="anomaly_engine")

# Keyboard adjacency map — all 26 letters + common symbols
KEYBOARD_ADJACENCY_MAP: dict[str, list[str]] = {
    "a": ["q", "w", "s", "z"],
    "b": ["v", "g", "h", "n"],
    "c": ["x", "d", "f", "v"],
    "d": ["s", "e", "r", "f", "c", "x"],
    "e": ["w", "r", "s", "d"],
    "f": ["d", "r", "t", "g", "v", "c"],
    "g": ["f", "t", "y", "h", "b", "v"],
    "h": ["g", "y", "u", "j", "n", "b"],
    "i": ["u", "o", "j", "k"],
    "j": ["h", "u", "i", "k", "n", "m"],
    "k": ["j", "i", "o", "l", "m"],
    "l": ["k", "o", "p"],
    "m": ["n", "j", "k"],
    "n": ["b", "h", "j", "m"],
    "o": ["i", "p", "k", "l"],
    "p": ["o", "l"],
    "q": ["w", "a"],
    "r": ["e", "t", "d", "f"],
    "s": ["a", "w", "e", "d", "x", "z"],
    "t": ["r", "y", "f", "g"],
    "u": ["y", "i", "h", "j"],
    "v": ["c", "f", "g", "b"],
    "w": ["q", "e", "a", "s"],
    "x": ["z", "s", "d", "c"],
    "y": ["t", "u", "g", "h"],
    "z": ["a", "s", "x"],
}


class TypoGenerator:
    """Generates realistic keyboard-error typos in string values."""

    def __init__(self, rng: np.random.Generator) -> None:
        self._rng = rng

    def adjacent_key_typo(self, text: str) -> str:
        """Replace a random char with an adjacent key character."""
        chars = list(text)
        candidates = [
            i for i, c in enumerate(chars)
            if c.lower() in KEYBOARD_ADJACENCY_MAP
        ]
        if not candidates:
            return text
        idx = self._rng.choice(candidates)
        orig = chars[idx]
        adjacents = KEYBOARD_ADJACENCY_MAP[orig.lower()]
        replacement = self._rng.choice(adjacents)
        if orig.isupper():
            replacement = replacement.upper()
        chars[idx] = replacement
        return "".join(chars)

    def transposition_typo(self, text: str) -> str:
        """Swap two adjacent characters."""
        if len(text) < 2:
            return text
        idx = int(self._rng.integers(0, len(text) - 1))
        chars = list(text)
        chars[idx], chars[idx + 1] = chars[idx + 1], chars[idx]
        return "".join(chars)

    def repetition_typo(self, text: str) -> str:
        """Double a random character (e.g. 'Mumbai' → 'Mumbaai')."""
        if not text:
            return text
        idx = int(self._rng.integers(0, len(text)))
        return text[:idx + 1] + text[idx] + text[idx + 1:]

    def deletion_typo(self, text: str) -> str:
        """Delete a random character (e.g. 'Hyderabad' → 'Hydrabad')."""
        if len(text) <= 1:
            return text
        idx = int(self._rng.integers(0, len(text)))
        return text[:idx] + text[idx + 1:]

    def case_error_typo(self, text: str) -> str:
        """Randomly lower-case the whole string."""
        return text.lower()

    def apply_random_typo(self, text: str) -> str:
        """Apply one randomly selected typo mutation."""
        methods = [
            self.adjacent_key_typo,
            self.transposition_typo,
            self.repetition_typo,
            self.deletion_typo,
            self.case_error_typo,
        ]
        method = methods[int(self._rng.integers(0, len(methods)))]
        return method(text)


class AnomalyOutlierEngine:
    """
    Injects realistic dirty data into synthetic DataFrames:
    - Structured nulls (MCAR / MAR / MNAR)
    - Date format inconsistencies
    - Outliers in numeric columns
    - Keyboard-error typos in string columns
    - Exact and near-duplicate rows
    """

    def __init__(self, seed: int = 42) -> None:
        self._seed = seed
        self._rng = np.random.default_rng(seed)
        self._py_rng = random.Random(seed)

    # ── Public API ─────────────────────────────────────────────────────────

    def inject_structured_nulls(
        self,
        df: pd.DataFrame,
        null_patterns: dict[str, float],
        null_conditions: Optional[dict[str, dict[str, object]]] = None,
    ) -> pd.DataFrame:
        """
        Inject nulls into specified columns.

        Args:
            df:              Input DataFrame.
            null_patterns:   {column_name: null_rate} mapping.
            null_conditions: Optional {column_name: {"condition_col": ..., "condition_val": ...}}
                             for MAR (Missing At Random) injection.

        Returns:
            DataFrame with nulls injected (copy).
        """
        df = df.copy()
        null_conditions = null_conditions or {}

        for col, rate in null_patterns.items():
            if col not in df.columns:
                _LOG.warning("inject_structured_nulls: column '%s' not found", col)
                continue
            rate = max(0.0, min(1.0, rate))
            n = len(df)

            if col in null_conditions:
                # MAR: higher null rate conditioned on another column
                cond = null_conditions[col]
                cond_col = cond.get("condition_col", "")
                cond_val = cond.get("condition_val")
                if cond_col in df.columns:
                    mask = df[cond_col] == cond_val
                    null_indices = df.index[mask][
                        self._rng.random(mask.sum()) < (rate * 2)
                    ]
                    df.loc[null_indices, col] = None
                    continue

            # MCAR: completely at random
            null_mask = self._rng.random(n) < rate
            df.loc[null_mask, col] = None

        return df

    def inject_date_format_mix(
        self,
        series: pd.Series,
        rng: Optional[np.random.Generator] = None,
        inconsistency_rate: float = 0.12,
    ) -> pd.Series:
        """
        Mix date format strings in a datetime series.

        A portion of rows get alternative format strings (e.g. DD/MM/YYYY instead of YYYY-MM-DD).

        Args:
            series:             Datetime series.
            rng:                Optional RNG (uses internal if None).
            inconsistency_rate: Fraction of rows to convert to alt format.

        Returns:
            Object (string) series with mixed date formats.
        """
        rng = rng or self._rng
        alt_formats = ["%d/%m/%Y", "%m-%d-%Y", "%d-%b-%Y", "%B %d, %Y"]

        def _format(val: object, use_alt: bool) -> str:
            if pd.isna(val):
                return ""
            try:
                ts = pd.Timestamp(val)  # type: ignore[arg-type]
                if use_alt:
                    fmt = alt_formats[int(rng.integers(0, len(alt_formats)))]
                    return ts.strftime(fmt)
                return ts.strftime("%Y-%m-%d")
            except Exception:
                return str(val)

        use_alt_mask = rng.random(len(series)) < inconsistency_rate
        return pd.Series(
            [_format(v, alt) for v, alt in zip(series, use_alt_mask)],
            index=series.index,
            name=series.name,
            dtype=object,
        )

    def inject_outliers(
        self,
        df: pd.DataFrame,
        outlier_columns: list[str],
        rate: float = 0.01,
    ) -> pd.DataFrame:
        """
        Inject outlier values into numeric columns.

        Uses 5-sigma outliers (mean ± 5 * std) to ensure values are
        statistically extreme but not completely nonsensical.

        Args:
            df:              Input DataFrame.
            outlier_columns: Numeric columns to inject outliers into.
            rate:            Fraction of rows to convert to outliers.

        Returns:
            DataFrame with outliers injected (copy).
        """
        df = df.copy()
        for col in outlier_columns:
            if col not in df.columns:
                continue
            if not pd.api.types.is_numeric_dtype(df[col]):
                continue
            col_mean = float(df[col].mean())
            col_std = float(df[col].std()) or 1.0
            n = len(df)
            outlier_mask = self._rng.random(n) < rate
            outlier_count = outlier_mask.sum()
            if outlier_count == 0:
                continue
            # Generate values 5–10 sigma away, randomly above or below
            direction = self._rng.choice([-1.0, 1.0], size=outlier_count)
            sigma_mult = self._rng.uniform(5.0, 10.0, size=outlier_count)
            outlier_vals = col_mean + direction * sigma_mult * col_std
            df.loc[outlier_mask, col] = outlier_vals
        return df

    def inject_typos(
        self,
        df: pd.DataFrame,
        string_columns: list[str],
        typo_rate: float = 0.02,
    ) -> pd.DataFrame:
        """
        Inject keyboard-error typos into string columns.

        Args:
            df:             Input DataFrame.
            string_columns: Columns to apply typos to.
            typo_rate:      Fraction of non-null values to corrupt.

        Returns:
            DataFrame with typos injected (copy).
        """
        df = df.copy()
        typo_gen = TypoGenerator(self._rng)

        for col in string_columns:
            if col not in df.columns:
                continue
            not_null = df[col].notna()
            indices = df.index[not_null]
            typo_mask = self._rng.random(len(indices)) < typo_rate
            for idx, apply_typo in zip(indices, typo_mask):
                if apply_typo:
                    val = df.at[idx, col]
                    if isinstance(val, str) and len(val) > 1:
                        df.at[idx, col] = typo_gen.apply_random_typo(val)
        return df

    def inject_duplicates(
        self,
        df: pd.DataFrame,
        duplicate_rate: float = 0.005,
        near_duplicate_rate: float = 0.005,
    ) -> pd.DataFrame:
        """
        Inject exact and near-duplicate rows.

        Args:
            df:                  Input DataFrame.
            duplicate_rate:      Fraction of rows to duplicate exactly.
            near_duplicate_rate: Fraction of rows to create near-duplicates
                                 (exact copy with one field slightly altered).

        Returns:
            DataFrame with duplicates appended and shuffled (copy).
        """
        n = len(df)
        extra_rows: list[pd.DataFrame] = []

        # Exact duplicates
        exact_count = max(0, int(n * duplicate_rate))
        if exact_count > 0:
            exact_indices = self._rng.choice(n, size=exact_count, replace=True)
            extra_rows.append(df.iloc[exact_indices])

        # Near-duplicates: copy + alter one numeric field
        near_count = max(0, int(n * near_duplicate_rate))
        if near_count > 0:
            near_indices = self._rng.choice(n, size=near_count, replace=True)
            near_df = df.iloc[near_indices].copy()
            numeric_cols = near_df.select_dtypes(include="number").columns.tolist()
            if numeric_cols:
                alter_col = numeric_cols[int(self._rng.integers(0, len(numeric_cols)))]
                near_df[alter_col] = near_df[alter_col] * self._rng.uniform(
                    0.95, 1.05, size=len(near_df)
                )
            extra_rows.append(near_df)

        if extra_rows:
            result = pd.concat([df] + extra_rows, ignore_index=True)
            # Shuffle
            shuffle_idx = self._rng.permutation(len(result))
            return result.iloc[shuffle_idx].reset_index(drop=True)

        return df.copy()

    def apply_dirty_profile(
        self,
        df: pd.DataFrame,
        profile: DirtyDataProfile,
        schema_columns: Optional[list[str]] = None,
    ) -> pd.DataFrame:
        """
        Apply a full DirtyDataProfile to a DataFrame in one call.

        Applies: nulls, outliers, typos, duplicates.

        Args:
            df:              Input DataFrame.
            profile:         DirtyDataProfile specifying all rates.
            schema_columns:  Column names to use (defaults to all df columns).

        Returns:
            DataFrame with dirty data applied (copy).
        """
        cols = schema_columns or list(df.columns)
        string_cols = [
            c for c in cols
            if c in df.columns and df[c].dtype == object
        ]
        numeric_cols = [
            c for c in cols
            if c in df.columns and pd.api.types.is_numeric_dtype(df[c])
        ]
        datetime_cols = [
            c for c in cols
            if c in df.columns and pd.api.types.is_datetime64_any_dtype(df[c])
        ]

        # 1. Nulls (MCAR on all columns at the profile null_rate)
        null_patterns = {c: profile.null_rate for c in cols if c in df.columns}
        df = self.inject_structured_nulls(df, null_patterns)

        # 2. Date format inconsistency
        for col in datetime_cols:
            df[col] = self.inject_date_format_mix(
                df[col], inconsistency_rate=profile.date_format_inconsistency_rate
            )

        # 3. Outliers
        if numeric_cols:
            df = self.inject_outliers(df, numeric_cols, rate=profile.outlier_rate)

        # 4. Typos
        if string_cols:
            df = self.inject_typos(df, string_cols, typo_rate=profile.typo_rate)

        # 5. Duplicates
        df = self.inject_duplicates(
            df,
            duplicate_rate=profile.duplicate_rate,
            near_duplicate_rate=profile.near_duplicate_rate,
        )

        return df
