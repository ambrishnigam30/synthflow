# ───────────────────────────────────────────────────────────────
# Copyright (c) 2026 Ambrish Nigam
# Author : Ambrish Nigam | https://github.com/ambrishnigam30
# Project: SynthFlow — Autonomous Synthetic Data Orchestration Platform
# License : Apache License 2.0 | https://www.apache.org/licenses/LICENSE-2.0
# Module : ScenarioEngine — applies economic/environmental scenario shifts to DataFrames
# ───────────────────────────────────────────────────────────────

from __future__ import annotations

import re
from typing import Optional, Union

import numpy as np
import pandas as pd

from synthflow.llm_client import LLMClient, MockLLMClient
from synthflow.models.schemas import ImpliedTimeRange, IntentObject, ScenarioParams
from synthflow.utils.helpers import safe_json_loads
from synthflow.utils.logger import get_logger

_LOG = get_logger("scenario_engine", component="scenario_engine")

# ── Known scenario templates ────────────────────────────────────────────────

_SCENARIO_TEMPLATES: dict[str, ScenarioParams] = {
    "recession": ScenarioParams(
        scenario_name="recession",
        multipliers={
            "consumer_spending": 0.72,
            "revenue": 0.78,
            "salary": 0.90,
            "income": 0.88,
            "sales": 0.75,
            "profit": 0.65,
            "investment": 0.60,
            "gdp": 0.85,
            "employment": 0.92,
        },
        additive_shifts={"unemployment_rate": 3.5},
        affected_columns=[
            "consumer_spending", "revenue", "salary", "income", "sales",
            "profit", "investment", "gdp", "employment", "unemployment_rate",
        ],
        description="Economic recession: reduced spending, lower revenue, higher unemployment.",
    ),
    "pandemic": ScenarioParams(
        scenario_name="pandemic",
        multipliers={
            "patient_count": 2.5,
            "hospital_capacity": 1.8,
            "icu_occupancy": 3.2,
            "consumer_spending": 0.65,
            "travel_volume": 0.15,
            "revenue": 0.70,
            "online_sales": 2.1,
        },
        additive_shifts={"mortality_rate": 1.2, "infection_rate": 8.5},
        affected_columns=[
            "patient_count", "hospital_capacity", "icu_occupancy", "consumer_spending",
            "travel_volume", "revenue", "online_sales", "mortality_rate", "infection_rate",
        ],
        description="Pandemic: healthcare surge, reduced mobility, e-commerce spike.",
    ),
    "boom": ScenarioParams(
        scenario_name="boom",
        multipliers={
            "revenue": 1.35,
            "salary": 1.18,
            "consumer_spending": 1.25,
            "investment": 1.40,
            "stock_price": 1.50,
            "gdp": 1.08,
            "employment": 1.05,
            "profit": 1.45,
        },
        additive_shifts={"unemployment_rate": -1.5},
        affected_columns=[
            "revenue", "salary", "consumer_spending", "investment",
            "stock_price", "gdp", "employment", "profit", "unemployment_rate",
        ],
        description="Economic boom: rising revenue, salaries, and consumer confidence.",
    ),
    "demonetization": ScenarioParams(
        scenario_name="demonetization",
        multipliers={
            "cash_transactions": 0.15,
            "consumer_spending": 0.60,
            "retail_sales": 0.55,
            "atm_withdrawals": 4.5,
            "digital_payments": 3.2,
            "bank_deposits": 2.8,
        },
        additive_shifts={},
        affected_columns=[
            "cash_transactions", "consumer_spending", "retail_sales",
            "atm_withdrawals", "digital_payments", "bank_deposits",
        ],
        description="Demonetization: cash crunch, digital payment surge (India 2016 scenario).",
    ),
    "drought": ScenarioParams(
        scenario_name="drought",
        multipliers={
            "crop_yield": 0.45,
            "agricultural_revenue": 0.50,
            "water_availability": 0.35,
            "food_prices": 1.65,
            "farmer_income": 0.55,
            "livestock_count": 0.80,
        },
        additive_shifts={"drought_index": 2.8},
        affected_columns=[
            "crop_yield", "agricultural_revenue", "water_availability",
            "food_prices", "farmer_income", "livestock_count", "drought_index",
        ],
        description="Agricultural drought: reduced yields, higher food prices.",
    ),
}

# ── Keyword → scenario name mapping for fallback parsing ───────────────────

_KEYWORD_MAP: dict[str, str] = {
    "recession": "recession",
    "downturn": "recession",
    "economic crisis": "recession",
    "financial crisis": "recession",
    "pandemic": "pandemic",
    "covid": "pandemic",
    "epidemic": "pandemic",
    "outbreak": "pandemic",
    "boom": "boom",
    "bull market": "boom",
    "expansion": "boom",
    "demonetisation": "demonetization",
    "demonetization": "demonetization",
    "note ban": "demonetization",
    "drought": "drought",
    "dry season": "drought",
    "water crisis": "drought",
}


class ScenarioEngine:
    """
    Parses scenario descriptions and applies distributional shifts to DataFrames.

    Supports known templates (recession, pandemic, boom, demonetization, drought)
    plus LLM-assisted parsing for custom scenarios.
    """

    def __init__(self, llm_client: Optional[Union[LLMClient, MockLLMClient]] = None) -> None:
        self._llm = llm_client

    async def parse_scenario(
        self,
        scenario_text: str,
        intent: Optional[IntentObject] = None,
    ) -> ScenarioParams:
        """
        Parse a natural-language scenario description into ScenarioParams.

        Tries in order:
        1. Exact match against known template names
        2. Keyword matching
        3. LLM parsing (if client available)
        4. Minimal fallback ScenarioParams

        Args:
            scenario_text: Natural-language description of the scenario.
            intent:        Optional IntentObject for context.

        Returns:
            ScenarioParams with multipliers and shifts.
        """
        lower = scenario_text.lower().strip()

        # 1. Exact template match
        if lower in _SCENARIO_TEMPLATES:
            _LOG.info("Scenario template match: %s", lower)
            return _SCENARIO_TEMPLATES[lower]

        # 2. Keyword matching
        for keyword, template_name in _KEYWORD_MAP.items():
            if keyword in lower:
                _LOG.info("Scenario keyword match: '%s' → %s", keyword, template_name)
                return _SCENARIO_TEMPLATES[template_name]

        # 3. LLM parsing
        if self._llm is not None:
            try:
                params = await self._llm_parse(scenario_text, intent)
                if params:
                    return params
            except Exception as exc:
                _LOG.warning("LLM scenario parse failed: %s", exc)

        # 4. Minimal fallback
        _LOG.warning("Unknown scenario '%s' — returning identity ScenarioParams", scenario_text)
        return ScenarioParams(
            scenario_name=scenario_text[:50],
            multipliers={},
            additive_shifts={},
            affected_columns=[],
            description=f"Custom scenario: {scenario_text[:100]}",
        )

    async def _llm_parse(
        self,
        scenario_text: str,
        intent: Optional[IntentObject],
    ) -> Optional[ScenarioParams]:
        """Use LLM to parse a custom scenario description."""
        system = (
            "You are a scenario economist. Given a scenario description, return a JSON object with:\n"
            '- "scenario_name": string\n'
            '- "multipliers": dict[str, float] — column_name → multiplier (e.g. 0.8 = 20% drop)\n'
            '- "additive_shifts": dict[str, float] — column_name → additive shift\n'
            '- "affected_columns": list of column names\n'
            '- "description": string\n'
            "Return ONLY the JSON object."
        )
        domain = intent.domain if intent else "general"
        user = (
            f"Domain: {domain}\n"
            f"Scenario: {scenario_text}\n"
            "Return ScenarioParams JSON."
        )
        raw = await self._llm.complete(user, system_prompt=system, temperature=0.1, max_tokens=512)
        data = safe_json_loads(raw)
        return ScenarioParams(**data)

    def apply(
        self,
        df: pd.DataFrame,
        params: ScenarioParams,
    ) -> pd.DataFrame:
        """
        Apply ScenarioParams to a DataFrame, shifting distributions.

        Args:
            df:     Input DataFrame.
            params: ScenarioParams with multipliers and additive shifts.

        Returns:
            Modified DataFrame (copy) with scenario applied.
        """
        df = df.copy()

        for col, multiplier in params.multipliers.items():
            if col not in df.columns:
                # Try fuzzy match: column name contains the key
                matched = [c for c in df.columns if col.lower() in c.lower()]
                if matched:
                    col = matched[0]
                else:
                    continue
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col] * multiplier
                _LOG.debug("Scenario: %s × %.2f applied to '%s'", params.scenario_name, multiplier, col)

        for col, shift in params.additive_shifts.items():
            if col not in df.columns:
                matched = [c for c in df.columns if col.lower() in c.lower()]
                if matched:
                    col = matched[0]
                else:
                    continue
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col] + shift
                _LOG.debug("Scenario: %s + %.2f applied to '%s'", params.scenario_name, shift, col)

        _LOG.info(
            "Scenario '%s' applied: %d multipliers, %d shifts",
            params.scenario_name, len(params.multipliers), len(params.additive_shifts),
        )
        return df

    @staticmethod
    def list_known_scenarios() -> list[str]:
        """Return names of all built-in scenario templates."""
        return sorted(_SCENARIO_TEMPLATES.keys())

    @staticmethod
    def get_template(name: str) -> Optional[ScenarioParams]:
        """Return a known scenario template by name (None if not found)."""
        return _SCENARIO_TEMPLATES.get(name.lower())
