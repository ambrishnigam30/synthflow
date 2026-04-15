# ───────────────────────────────────────────────────────────────
# Copyright (c) 2026 Ambrish Nigam
# Author : Ambrish Nigam | https://github.com/ambrishnigam30
# Project: SynthFlow — Autonomous Synthetic Data Orchestration Platform
# License : Apache License 2.0 | https://www.apache.org/licenses/LICENSE-2.0
# Module : SDVMultiplierEngine — optional SDV-based row scaling (BYOK feature)
# ───────────────────────────────────────────────────────────────

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from synthflow.models.schemas import ColumnDefinition, ConstraintSet, SchemaDefinition
from synthflow.utils.logger import get_logger

_LOG = get_logger("sdv_multiplier", component="sdv_multiplier")


class SDVMultiplierEngine:
    """
    OPTIONAL: Scales a seed DataFrame to a target row count using SDV (Synthetic Data Vault).

    Only activated when the user explicitly enables SDV mode via settings.
    Falls back to statistical resampling when SDV is unavailable.

    Usage:
        engine = SDVMultiplierEngine()
        large_df = engine.scale(seed_df, target_rows=50_000, schema=schema)
    """

    def __init__(self, use_sdv: bool = True) -> None:
        self._use_sdv = use_sdv
        self._sdv_available = self._check_sdv_available()

    def _check_sdv_available(self) -> bool:
        """Return True if sdv package is installed."""
        try:
            import sdv  # noqa: F401
            return True
        except ImportError:
            _LOG.info("SDV not installed — will use statistical resampling fallback")
            return False

    def build_sdv_metadata(self, schema: SchemaDefinition) -> Optional[object]:
        """
        Build an SDV SingleTableMetadata object from a SchemaDefinition.

        Args:
            schema: SchemaDefinition with column type information.

        Returns:
            SingleTableMetadata if SDV is available, else None.
        """
        if not self._sdv_available:
            return None

        try:
            from sdv.metadata import SingleTableMetadata

            metadata = SingleTableMetadata()
            table = schema.tables[0] if schema.tables else None
            if not table:
                return metadata

            for col in table.columns:
                sdtype = self._map_sdtype(col)
                if col.is_primary_key:
                    metadata.add_column(col.name, sdtype="id", regex_format="[A-Z0-9]{8}")
                else:
                    metadata.add_column(col.name, sdtype=sdtype)

            _LOG.debug("SDV metadata built: %d columns", len(table.columns))
            return metadata
        except Exception as exc:
            _LOG.warning("Failed to build SDV metadata: %s", exc)
            return None

    def _map_sdtype(self, col: ColumnDefinition) -> str:
        """Map SynthFlow ColumnDefinition data_type to SDV sdtype."""
        dtype = col.data_type.lower()
        if dtype in ("integer", "int"):
            return "numerical"
        if dtype in ("float", "numeric", "decimal"):
            return "numerical"
        if dtype in ("boolean", "bool"):
            return "boolean"
        if dtype in ("datetime", "timestamp"):
            return "datetime"
        if dtype == "date":
            return "datetime"
        if col.enum_values:
            return "categorical"
        return "text"

    def scale(
        self,
        seed_df: pd.DataFrame,
        target_rows: int,
        schema: SchemaDefinition,
        constraints: Optional[ConstraintSet] = None,
        seed: int = 42,
    ) -> pd.DataFrame:
        """
        Scale seed_df to target_rows using SDV (if available) or statistical resampling.

        Args:
            seed_df:     Small seed DataFrame (e.g. 100–1000 rows) with real structure.
            target_rows: Target number of rows in output.
            schema:      SchemaDefinition for type metadata.
            constraints: Optional constraints to re-enforce post-scaling.
            seed:        Random seed.

        Returns:
            DataFrame with target_rows rows.
        """
        if len(seed_df) == 0:
            _LOG.warning("scale(): empty seed_df — returning empty DataFrame")
            return seed_df.copy()

        if target_rows <= len(seed_df):
            # Already at or below target — just sample
            return seed_df.sample(
                n=target_rows, replace=False, random_state=seed
            ).reset_index(drop=True)

        if self._use_sdv and self._sdv_available:
            result = self._sdv_scale(seed_df, target_rows, schema, seed)
        else:
            result = self._statistical_scale(seed_df, target_rows, schema, seed)

        # Re-enforce constraints if provided
        if constraints and constraints.rules:
            result = self._re_enforce_constraints(result, constraints)

        _LOG.info("Scaled %d → %d rows", len(seed_df), len(result))
        return result

    def _sdv_scale(
        self,
        seed_df: pd.DataFrame,
        target_rows: int,
        schema: SchemaDefinition,
        seed: int,
    ) -> pd.DataFrame:
        """Scale using SDV GaussianCopula model."""
        try:
            from sdv.single_table import GaussianCopulaSynthesizer

            metadata = self.build_sdv_metadata(schema)
            if metadata is None:
                return self._statistical_scale(seed_df, target_rows, schema, seed)

            synthesizer = GaussianCopulaSynthesizer(metadata)
            synthesizer.fit(seed_df)
            synthetic = synthesizer.sample(num_rows=target_rows)
            return synthetic.reset_index(drop=True)
        except Exception as exc:
            _LOG.warning("SDV scaling failed: %s — falling back to statistical", exc)
            return self._statistical_scale(seed_df, target_rows, schema, seed)

    def _statistical_scale(
        self,
        seed_df: pd.DataFrame,
        target_rows: int,
        schema: SchemaDefinition,
        seed: int,
    ) -> pd.DataFrame:
        """
        Statistical resampling fallback.

        Strategy:
        - For each numeric column: fit normal distribution from seed, sample from it
        - For each categorical column: resample with frequency weights
        - For datetime columns: extend the time range proportionally
        - PK columns: generate fresh UUIDs/IDs
        """
        rng = np.random.default_rng(seed)
        n = target_rows
        result: dict[str, object] = {}

        table = schema.tables[0] if schema.tables else None
        col_defs: dict[str, ColumnDefinition] = {}
        if table:
            col_defs = {c.name: c for c in table.columns}

        for col in seed_df.columns:
            col_def = col_defs.get(col)
            series = seed_df[col].dropna()

            if col_def and col_def.is_primary_key:
                import uuid
                result[col] = [str(uuid.uuid4()) for _ in range(n)]
                continue

            if pd.api.types.is_numeric_dtype(series) and len(series) > 1:
                mean = float(series.mean())
                std = max(float(series.std()), 1e-9)
                samples = rng.normal(mean, std, size=n)
                # Preserve integer dtype
                if col_def and col_def.data_type.lower() == "integer":
                    samples = np.round(samples).astype(int)
                # Apply clip if defined
                if col_def:
                    if col_def.min_value is not None:
                        samples = np.maximum(samples, col_def.min_value)
                    if col_def.max_value is not None:
                        samples = np.minimum(samples, col_def.max_value)
                result[col] = samples.tolist()

            elif pd.api.types.is_datetime64_any_dtype(series) and len(series) > 1:
                start_ts = series.min().timestamp()
                end_ts = series.max().timestamp()
                span = end_ts - start_ts
                # Scale the time span proportionally
                new_end = end_ts + span * (n / max(1, len(seed_df)) - 1)
                timestamps = rng.uniform(start_ts, max(end_ts, new_end), size=n)
                result[col] = pd.to_datetime(timestamps, unit="s")

            elif series.dtype == object or (col_def and col_def.enum_values):
                # Categorical: resample with frequency weights
                value_counts = series.value_counts(normalize=True)
                categories = value_counts.index.tolist()
                probabilities = value_counts.values.tolist()
                chosen = rng.choice(len(categories), size=n, p=probabilities)
                result[col] = [categories[i] for i in chosen]

            else:
                # Fallback: resample with replacement
                if len(series) > 0:
                    chosen_idx = rng.integers(0, len(series), size=n)
                    result[col] = series.iloc[chosen_idx].tolist()
                else:
                    result[col] = [None] * n

        return pd.DataFrame(result)

    def _re_enforce_constraints(
        self, df: pd.DataFrame, constraints: ConstraintSet
    ) -> pd.DataFrame:
        """Re-apply basic range constraints post-scaling."""
        for rule in constraints.rules:
            if rule.constraint_type.value != "range":
                continue
            for col in rule.columns:
                if col not in df.columns:
                    continue
                if not pd.api.types.is_numeric_dtype(df[col]):
                    continue
                if "min" in rule.parameters:
                    df[col] = df[col].clip(lower=rule.parameters["min"])
                if "max" in rule.parameters:
                    df[col] = df[col].clip(upper=rule.parameters["max"])
        return df
