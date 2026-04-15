# ───────────────────────────────────────────────────────────────
# Copyright (c) 2026 Ambrish Nigam
# Author : Ambrish Nigam | https://github.com/ambrishnigam30
# Project: SynthFlow — Autonomous Synthetic Data Orchestration Platform
# License : Apache License 2.0 | https://www.apache.org/licenses/LICENSE-2.0
# Module : DeterministicRealismEngine — cell-level seeding + locale formatting
# ───────────────────────────────────────────────────────────────

from __future__ import annotations

import hashlib
from typing import Any, Optional

import numpy as np

from synthflow.models.schemas import (
    CausalKnowledgeBundle,
    ColumnDefinition,
    DistributionMap,
    DistributionSpec,
)
from synthflow.utils.logger import get_logger

_LOG = get_logger("realism_engine", component="realism_engine")


class DeterministicRealismEngine:
    """
    Produces deterministic, locale-aware cell values.

    Key design:
    - Cell RNG derived from SHA-256(global_seed || column_name || row_index)
      so each (column, row) pair gets a unique but reproducible random stream.
    - Locale formatting applied post-sampling (Indian lakh vs Western million).
    """

    def get_cell_rng(
        self,
        global_seed: int,
        column_name: str,
        row_index: int,
    ) -> np.random.Generator:
        """
        Derive a reproducible RNG for one cell.

        Hash = SHA-256(str(global_seed) + ":" + column_name + ":" + str(row_index))
        Seed = first 8 bytes of hash interpreted as big-endian uint64.

        Args:
            global_seed:  Top-level session seed.
            column_name:  Column name.
            row_index:    Zero-based row index.

        Returns:
            numpy.random.Generator seeded deterministically.
        """
        raw = f"{global_seed}:{column_name}:{row_index}".encode("utf-8")
        digest = hashlib.sha256(raw).digest()
        cell_seed = int.from_bytes(digest[:8], byteorder="big") % (2**63)
        return np.random.default_rng(cell_seed)

    def sample_column(
        self,
        column: ColumnDefinition,
        n_rows: int,
        global_seed: int,
        context: dict[str, Any],
        dist_map: Optional[DistributionMap] = None,
    ) -> list[Any]:
        """
        Sample *n_rows* values for *column* using deterministic per-cell RNGs.

        For numeric columns, uses the DistributionSpec from *dist_map*.
        For string columns with enum_values, picks randomly from the enum.
        For boolean columns, uses Bernoulli.

        Args:
            column:      ColumnDefinition for the target column.
            n_rows:      Number of rows to generate.
            global_seed: Session-level seed.
            context:     Additional context (e.g., locale).
            dist_map:    DistributionMap (optional).

        Returns:
            List of sampled values.
        """
        from synthflow.engines.stats_engine import _sample_distribution

        spec: Optional[DistributionSpec] = None
        if dist_map and column.name in dist_map.column_distributions:
            spec = dist_map.column_distributions[column.name]

        values: list[Any] = []

        if column.enum_values and column.data_type == "string":
            # Per-row RNG for enum sampling
            for row_idx in range(n_rows):
                rng = self.get_cell_rng(global_seed, column.name, row_idx)
                val = column.enum_values[int(rng.integers(len(column.enum_values)))]
                values.append(val)
            return values

        if column.data_type == "boolean":
            for row_idx in range(n_rows):
                rng = self.get_cell_rng(global_seed, column.name, row_idx)
                values.append(bool(rng.integers(2)))
            return values

        if column.data_type in ("integer", "float") or spec is not None:
            # Use a combined RNG seeded from global_seed + column for batch efficiency
            col_seed_raw = f"{global_seed}:{column.name}:batch".encode("utf-8")
            col_digest = hashlib.sha256(col_seed_raw).digest()
            col_seed = int.from_bytes(col_digest[:8], "big") % (2**63)
            rng = np.random.default_rng(col_seed)

            if spec is None:
                # Default: normal for float, poisson for integer
                if column.data_type == "integer":
                    raw_vals = rng.poisson(lam=10, size=n_rows)
                else:
                    raw_vals = rng.normal(size=n_rows)
            else:
                raw_vals = _sample_distribution(spec, rng, n_rows)

            if column.data_type == "integer":
                raw_vals = np.round(raw_vals).astype(int)

            # Apply locale formatting for salary/amount columns
            locale_fmt = context.get("number_format", "western")
            if column.semantic_type in ("salary", "income", "amount") and locale_fmt == "indian":
                from synthflow.utils.helpers import format_number_indian
                values = [format_number_indian(float(v)) for v in raw_vals]
            else:
                values = raw_vals.tolist()
            return values

        # Fallback: return None for unknown types
        return [None] * n_rows

    def format_value(
        self,
        value: Any,
        semantic_type: str,
        number_format: str = "western",
    ) -> Any:
        """
        Apply locale-aware formatting to a single value.

        Args:
            value:         Raw value.
            semantic_type: Column semantic type.
            number_format: 'indian' or 'western'.

        Returns:
            Formatted value.
        """
        if not isinstance(value, (int, float)):
            return value
        if semantic_type in ("salary", "income", "amount", "revenue"):
            if number_format == "indian":
                from synthflow.utils.helpers import format_number_indian
                return format_number_indian(float(value))
            else:
                from synthflow.utils.helpers import format_number_western
                return format_number_western(float(value))
        return value
