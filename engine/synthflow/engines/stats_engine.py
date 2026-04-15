# ───────────────────────────────────────────────────────────────
# Copyright (c) 2026 Ambrish Nigam
# Author : Ambrish Nigam | https://github.com/ambrishnigam30
# Project: SynthFlow — Autonomous Synthetic Data Orchestration Platform
# License : Apache License 2.0 | https://www.apache.org/licenses/LICENSE-2.0
# Module : StatisticalModelingCore — maps semantic types to distributions
# ───────────────────────────────────────────────────────────────

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from synthflow.models.schemas import (
    CausalKnowledgeBundle,
    ColumnDefinition,
    DistributionMap,
    DistributionSpec,
    SchemaDefinition,
)
from synthflow.utils.logger import get_logger

_LOG = get_logger("stats_engine", component="stats_engine")

# ── Semantic-type → required distribution rules ────────────────────────────
# E-014-01 compliant: no inline list > 20 items

_SEMANTIC_DISTRIBUTION_RULES: dict[str, str] = {
    "salary": "lognormal",
    "income": "lognormal",
    "wage": "lognormal",
    "revenue": "lognormal",
    "price": "lognormal",
    "cost": "lognormal",
    "amount": "lognormal",
    "age": "truncated_normal",
    "years": "truncated_normal",
    "duration": "exponential",
    "tenure": "exponential",
    "count": "poisson",
    "quantity": "poisson",
    "number": "poisson",
    "probability": "beta",
    "rate": "beta",
    "proportion": "beta",
    "boolean": "bernoulli",
    "flag": "bernoulli",
    "credit_score": "beta",
    "score": "beta",
}

# ── Default distribution parameters by type ───────────────────────────────

_DEFAULT_PARAMS: dict[str, dict[str, Any]] = {
    "normal": {"mean": 0.0, "std": 1.0},
    "lognormal": {"mean": 10.5, "sigma": 0.8},
    "truncated_normal": {"mean": 40.0, "std": 15.0, "low": 0.0, "high": 120.0},
    "uniform": {"low": 0.0, "high": 1.0},
    "exponential": {"scale": 1.0},
    "poisson": {"mu": 5.0},
    "beta": {"a": 2.0, "b": 5.0},
    "bernoulli": {"p": 0.5},
    "categorical": {"weights": []},
    "gamma": {"a": 2.0, "scale": 1.0},
    "weibull": {"a": 1.5},
}

# ── Column-name heuristics ─────────────────────────────────────────────────

_NAME_TO_SEMANTIC: dict[str, str] = {
    "salary": "salary", "pay": "salary", "wage": "salary",
    "income": "income", "revenue": "salary", "amount": "salary",
    "age": "age", "years": "years",
    "score": "score", "rating": "score",
    "count": "count", "qty": "count", "quantity": "count",
    "probability": "probability", "rate": "rate",
    "duration": "duration", "tenure": "duration",
    "is_": "boolean", "has_": "boolean",
}


@dataclass
class MixtureDistribution:
    """
    A mixture of two distributions for bimodal or multi-modal columns.
    Stores weights and component specs.
    """
    components: list[DistributionSpec] = field(default_factory=list)
    weights: list[float] = field(default_factory=list)

    def sample(self, rng: np.random.Generator, n: int = 1) -> np.ndarray:
        """Sample *n* values from the mixture."""
        if not self.components or not self.weights:
            return rng.normal(size=n)
        total = sum(self.weights)
        norm_weights = [w / total for w in self.weights]
        counts = rng.multinomial(n, norm_weights)
        parts = []
        for spec, cnt in zip(self.components, counts):
            if cnt > 0:
                parts.append(_sample_distribution(spec, rng, cnt))
        return np.concatenate(parts) if parts else rng.normal(size=n)


class StatisticalModelingCore:
    """
    Maps schema columns to statistical distributions.

    Priority:
    1. Knowledge bundle column overrides (from LLM)
    2. Semantic-type rules (_SEMANTIC_DISTRIBUTION_RULES)
    3. Column-name heuristics
    4. Default by data_type
    """

    def model(
        self,
        schema: SchemaDefinition,
        knowledge: CausalKnowledgeBundle,
    ) -> DistributionMap:
        """
        Build a DistributionMap for all numeric columns in *schema*.

        Args:
            schema:    Schema definition.
            knowledge: Domain knowledge (may contain column overrides).

        Returns:
            DistributionMap keyed by column name.
        """
        # Index knowledge column hints
        col_hints: dict[str, Any] = {}
        for ck in knowledge.column_knowledge:
            col_hints[ck.column_name] = ck

        distributions: dict[str, DistributionSpec] = {}

        for table in schema.tables:
            for col in table.columns:
                spec = self._assign_distribution(col, col_hints.get(col.name))
                distributions[col.name] = spec

        return DistributionMap(
            column_distributions=distributions,
            table_name=schema.tables[0].name if schema.tables else "",
        )

    def _assign_distribution(
        self,
        col: ColumnDefinition,
        hint: Any,
    ) -> DistributionSpec:
        """Return the best DistributionSpec for *col*."""
        # Check enum columns → categorical
        if col.enum_values:
            n = len(col.enum_values)
            return DistributionSpec(
                distribution_type="categorical",
                params={"weights": [1.0 / n] * n, "values": col.enum_values},
            )

        # Boolean / flag
        if col.data_type == "boolean":
            return DistributionSpec(distribution_type="bernoulli", params={"p": 0.5})

        # Semantic type from column definition
        sem = (col.semantic_type or "").lower()
        dist_type = _SEMANTIC_DISTRIBUTION_RULES.get(sem)

        # Name-based fallback
        if not dist_type:
            col_lower = col.name.lower()
            for kw, semantic in _NAME_TO_SEMANTIC.items():
                if col_lower.startswith(kw) or col_lower.endswith(kw):
                    dist_type = _SEMANTIC_DISTRIBUTION_RULES.get(semantic)
                    if dist_type:
                        break

        # Data-type fallback
        if not dist_type:
            dist_type = {
                "integer": "poisson",
                "float": "normal",
                "string": "categorical",
                "datetime": "uniform",
                "date": "uniform",
                "boolean": "bernoulli",
            }.get(col.data_type, "normal")

        # Build params
        params = dict(_DEFAULT_PARAMS.get(dist_type, {}))

        # Incorporate column min/max
        if col.min_value is not None:
            params["clip_min"] = float(col.min_value)
        if col.max_value is not None:
            params["clip_max"] = float(col.max_value)

        # Override from knowledge hint
        if hint is not None:
            if hint.min_value is not None and "mean" in params:
                params["low"] = float(hint.min_value)
            if hint.max_value is not None and "high" in params:
                params["high"] = float(hint.max_value)

        return DistributionSpec(
            distribution_type=dist_type,
            params=params,
            clip_min=col.min_value,
            clip_max=col.max_value,
        )

    def validate_distribution(
        self, semantic_type: str, distribution: str
    ) -> str:
        """
        Check if *distribution* is appropriate for *semantic_type*.
        If not, return the correct distribution type.

        Args:
            semantic_type: e.g. 'salary', 'age'
            distribution:  proposed distribution type

        Returns:
            Validated (possibly corrected) distribution type.
        """
        required = _SEMANTIC_DISTRIBUTION_RULES.get(semantic_type.lower())
        if required is None:
            return distribution  # No rule → accept as-is
        if distribution == required:
            return distribution
        _LOG.warning(
            "Distribution override: semantic_type='%s' requires '%s', got '%s'",
            semantic_type, required, distribution,
        )
        return required


# ── Sampling helper ────────────────────────────────────────────────────────

def _sample_distribution(
    spec: DistributionSpec,
    rng: np.random.Generator,
    n: int = 1,
) -> np.ndarray:
    """Sample *n* values from a DistributionSpec."""
    p = spec.params
    dt = spec.distribution_type
    try:
        if dt == "normal":
            vals = rng.normal(loc=p.get("mean", 0.0), scale=p.get("std", 1.0), size=n)
        elif dt == "lognormal":
            vals = rng.lognormal(mean=p.get("mean", 10.5), sigma=p.get("sigma", 0.8), size=n)
        elif dt in ("truncated_normal", "trunc_normal"):
            from scipy.stats import truncnorm  # type: ignore[import-untyped]
            lo = p.get("low", 0.0)
            hi = p.get("high", 120.0)
            mean = p.get("mean", 40.0)
            std = p.get("std", 15.0)
            a = (lo - mean) / std if std > 0 else -5.0
            b = (hi - mean) / std if std > 0 else 5.0
            vals = truncnorm.rvs(a, b, loc=mean, scale=std, size=n,
                                 random_state=int(rng.integers(2**31)))
        elif dt == "uniform":
            vals = rng.uniform(low=p.get("low", 0.0), high=p.get("high", 1.0), size=n)
        elif dt == "exponential":
            vals = rng.exponential(scale=p.get("scale", 1.0), size=n)
        elif dt == "poisson":
            vals = rng.poisson(lam=p.get("mu", 5.0), size=n).astype(float)
        elif dt == "beta":
            vals = rng.beta(a=p.get("a", 2.0), b=p.get("b", 5.0), size=n)
        elif dt == "bernoulli":
            vals = rng.binomial(1, p=p.get("p", 0.5), size=n).astype(float)
        elif dt == "gamma":
            vals = rng.gamma(shape=p.get("a", 2.0), scale=p.get("scale", 1.0), size=n)
        else:
            vals = rng.normal(size=n)
    except Exception:
        vals = rng.normal(size=n)

    # Apply clip bounds
    if spec.clip_min is not None:
        vals = np.clip(vals, a_min=float(spec.clip_min), a_max=None)
    if spec.clip_max is not None:
        vals = np.clip(vals, a_min=None, a_max=float(spec.clip_max))
    return vals
