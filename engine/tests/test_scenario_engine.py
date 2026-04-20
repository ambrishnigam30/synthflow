# ───────────────────────────────────────────────────────────────
# Copyright (c) 2026 Ambrish Nigam
# Author : Ambrish Nigam | https://github.com/ambrishnigam30
# Project: SynthFlow — Autonomous Synthetic Data Orchestration Platform
# License : Apache License 2.0 | https://www.apache.org/licenses/LICENSE-2.0
# Module : Tests for ScenarioEngine
# ───────────────────────────────────────────────────────────────

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from synthflow.engines.scenario_engine import ScenarioEngine
from synthflow.llm_client import MockLLMClient


# ── list_known_scenarios ───────────────────────────────────────────────────


def test_scenario_engine_list_known_scenarios() -> None:
    """list_known_scenarios returns at least recession, pandemic, boom."""
    engine = ScenarioEngine(llm_client=MockLLMClient())
    scenarios = engine.list_known_scenarios()
    assert "recession" in scenarios
    assert "pandemic" in scenarios
    assert "boom" in scenarios


def test_scenario_engine_list_known_scenarios_is_sorted() -> None:
    """list_known_scenarios returns a sorted list."""
    engine = ScenarioEngine(llm_client=MockLLMClient())
    scenarios = engine.list_known_scenarios()
    assert scenarios == sorted(scenarios)


# ── get_template ───────────────────────────────────────────────────────────


def test_scenario_engine_get_template_recession() -> None:
    """get_template('recession') returns a ScenarioParams with scenario_name set."""
    engine = ScenarioEngine(llm_client=MockLLMClient())
    template = engine.get_template("recession")
    assert template is not None
    assert template.scenario_name == "recession"


def test_scenario_engine_get_template_returns_none_for_unknown() -> None:
    """get_template for an unknown scenario returns None."""
    engine = ScenarioEngine(llm_client=MockLLMClient())
    result = engine.get_template("nonexistent_scenario_xyz")
    assert result is None


def test_scenario_engine_get_template_case_insensitive() -> None:
    """get_template is case-insensitive."""
    engine = ScenarioEngine(llm_client=MockLLMClient())
    template = engine.get_template("RECESSION")
    assert template is not None


def test_scenario_engine_get_template_pandemic() -> None:
    engine = ScenarioEngine(llm_client=MockLLMClient())
    template = engine.get_template("pandemic")
    assert template is not None
    assert template.scenario_name == "pandemic"


def test_scenario_engine_get_template_boom() -> None:
    engine = ScenarioEngine(llm_client=MockLLMClient())
    template = engine.get_template("boom")
    assert template is not None
    assert template.scenario_name == "boom"


# ── apply ─────────────────────────────────────────────────────────────────


def test_scenario_engine_apply_recession_multiplies_salary() -> None:
    """Recession template multiplies 'salary' column downward."""
    engine = ScenarioEngine(llm_client=MockLLMClient())
    template = engine.get_template("recession")
    assert template is not None

    rng = np.random.default_rng(42)
    n = 100
    original_salary = rng.lognormal(10, 0.5, size=n)
    df = pd.DataFrame({"salary": original_salary.copy()})
    result = engine.apply(df.copy(), template)
    assert result is not None
    assert len(result) == n
    # Recession multiplier for salary = 0.90 → values should be lower
    assert float(result["salary"].mean()) < float(original_salary.mean())


def test_scenario_engine_apply_boom_raises_revenue() -> None:
    """Boom template multiplies 'revenue' column upward."""
    engine = ScenarioEngine(llm_client=MockLLMClient())
    template = engine.get_template("boom")
    assert template is not None

    rng = np.random.default_rng(1)
    n = 100
    original_rev = rng.lognormal(10, 0.5, size=n)
    df = pd.DataFrame({"revenue": original_rev.copy()})
    result = engine.apply(df.copy(), template)
    assert float(result["revenue"].mean()) > float(original_rev.mean())


def test_scenario_engine_apply_preserves_unaffected_columns() -> None:
    """Columns not in the scenario template are unchanged."""
    engine = ScenarioEngine(llm_client=MockLLMClient())
    template = engine.get_template("recession")
    assert template is not None

    df = pd.DataFrame({
        "patient_id": ["p1", "p2", "p3"],
        "blood_type": ["A+", "B+", "O+"],
    })
    result = engine.apply(df.copy(), template)
    assert list(result["patient_id"]) == ["p1", "p2", "p3"]
    assert list(result["blood_type"]) == ["A+", "B+", "O+"]


def test_scenario_engine_apply_returns_copy() -> None:
    """apply() does not mutate the original DataFrame."""
    engine = ScenarioEngine(llm_client=MockLLMClient())
    template = engine.get_template("recession")
    assert template is not None

    rng = np.random.default_rng(0)
    df = pd.DataFrame({"salary": rng.lognormal(10, 0.5, size=50)})
    original_values = df["salary"].tolist()
    engine.apply(df, template)
    assert df["salary"].tolist() == original_values


# ── parse_scenario (async) ─────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_scenario_engine_parse_known_scenario(sample_intent) -> None:
    """parse_scenario('recession scenario') matches the recession template."""
    engine = ScenarioEngine(llm_client=MockLLMClient())
    params = await engine.parse_scenario("recession scenario", sample_intent)
    assert params is not None
    assert params.scenario_name is not None


@pytest.mark.asyncio
async def test_scenario_engine_parse_keyword_match(sample_intent) -> None:
    """parse_scenario falls back to keyword matching for 'economic crisis'."""
    engine = ScenarioEngine(llm_client=MockLLMClient())
    params = await engine.parse_scenario("economic crisis scenario", sample_intent)
    assert params is not None
    assert params.scenario_name == "recession"


@pytest.mark.asyncio
async def test_scenario_engine_parse_unknown_returns_fallback(sample_intent) -> None:
    """parse_scenario for unknown text returns a ScenarioParams (not None)."""
    engine = ScenarioEngine(llm_client=MockLLMClient())
    params = await engine.parse_scenario("completely unknown scenario xyz123", sample_intent)
    assert params is not None
    assert isinstance(params.scenario_name, str)
