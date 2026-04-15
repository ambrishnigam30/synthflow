# ───────────────────────────────────────────────────────────────
# Copyright (c) 2026 Ambrish Nigam
# Author : Ambrish Nigam | https://github.com/ambrishnigam30
# Project: SynthFlow — Autonomous Synthetic Data Orchestration Platform
# License : Apache License 2.0 | https://www.apache.org/licenses/LICENSE-2.0
# Module : Tests E-006-01 through E-006-05 — UniversalKnowledgeGraph tests
# ───────────────────────────────────────────────────────────────

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from synthflow.engines.knowledge_graph import UniversalKnowledgeGraph, _VALID_CURRENCY_CODES
from synthflow.llm_client import MockLLMClient
from synthflow.models.schemas import (
    CausalKnowledgeBundle,
    IntentObject,
    RegionInfo,
)


# ── Fixtures ───────────────────────────────────────────────────────────────

@pytest.fixture()
def minimal_seeds_dir(tmp_path: Path) -> Path:
    """Create a minimal seeds directory with small JSON files."""
    seeds = tmp_path / "seeds"
    seeds.mkdir()

    geo = {
        "cities": [
            {
                "city": "Mumbai", "state": "Maharashtra", "country": "India",
                "continent": "Asia", "timezone": "Asia/Kolkata",
                "currency_code": "INR", "currency_symbol": "₹",
                "latitude": 19.076, "longitude": 72.878, "population": 20000000,
                "tier": "metro", "gdp_per_capita_usd": 2900,
                "avg_monthly_salary_usd": 850, "cost_of_living_index": 25.4,
                "languages": ["Hindi", "Marathi"], "phone_prefix": "+91",
                "postal_code_format": "######",
            },
            {
                "city": "Delhi", "state": "Delhi", "country": "India",
                "continent": "Asia", "timezone": "Asia/Kolkata",
                "currency_code": "INR", "currency_symbol": "₹",
                "latitude": 28.613, "longitude": 77.209, "population": 32000000,
                "tier": "metro", "gdp_per_capita_usd": 3200,
                "avg_monthly_salary_usd": 900, "cost_of_living_index": 23.0,
                "languages": ["Hindi", "English"], "phone_prefix": "+91",
                "postal_code_format": "######",
            },
        ]
    }
    (seeds / "world_geography.json").write_text(json.dumps(geo), encoding="utf-8")

    econ = {
        "countries": [
            {
                "country": "India", "currency_code": "INR", "currency_symbol": "₹",
                "gdp_per_capita_usd": 2500, "gini_coefficient": 35.7,
                "avg_monthly_salary_usd": 700, "min_wage_usd": 80,
                "inflation_rate": 6.2, "unemployment_rate": 7.9,
                "economic_tier": "lower_middle",
            }
        ]
    }
    (seeds / "economic_indicators.json").write_text(json.dumps(econ), encoding="utf-8")

    ontologies = {
        "domains": [
            {
                "domain": "healthcare",
                "entity_types": ["patient", "doctor"],
                "typical_columns": ["patient_id", "name", "age"],
                "common_relationships": [],
                "regulatory_tags": ["HIPAA"],
            }
        ]
    }
    (seeds / "domain_ontologies.json").write_text(json.dumps(ontologies), encoding="utf-8")

    priors = {
        "priors": [
            {"semantic_type": "salary", "distribution_type": "lognormal",
             "params": {"mean": 10.5, "sigma": 0.8}, "clip_min": 0, "clip_max": None},
            {"semantic_type": "age", "distribution_type": "truncated_normal",
             "params": {"mean": 40, "sigma": 15}, "clip_min": 0, "clip_max": 120},
        ]
    }
    (seeds / "distribution_priors.json").write_text(json.dumps(priors), encoding="utf-8")

    temporal = {
        "patterns": [
            {"domain": "healthcare", "pattern_type": "day_of_week",
             "pattern_data": {"weights": [1.2, 1.2, 1.2, 1.2, 1.2, 0.6, 0.4]}},
        ]
    }
    (seeds / "temporal_patterns.json").write_text(json.dumps(temporal), encoding="utf-8")

    return seeds


@pytest.fixture()
def india_intent() -> IntentObject:
    return IntentObject(
        domain="healthcare",
        row_count=100,
        region=RegionInfo(country="India", state_province="Maharashtra"),
    )


@pytest.fixture()
def kg_with_seeds(minimal_seeds_dir: Path) -> UniversalKnowledgeGraph:
    client = MockLLMClient()
    return UniversalKnowledgeGraph(client, seeds_dir=minimal_seeds_dir)


# ── E-006-01: Loads seed data ──────────────────────────────────────────────

def test_knowledge_graph_loads_seed_data(kg_with_seeds: UniversalKnowledgeGraph) -> None:
    """After _load_seed_data(), DuckDB tables are populated."""
    kg_with_seeds._load_seed_data()
    assert kg_with_seeds.is_seeds_loaded()


def test_knowledge_graph_cities_loaded(kg_with_seeds: UniversalKnowledgeGraph) -> None:
    """Indian cities from seed data are accessible via DuckDB query."""
    kg_with_seeds._load_seed_data()
    cities = kg_with_seeds.get_cities_for_country("India")
    assert len(cities) >= 2
    assert "Mumbai" in cities


def test_knowledge_graph_seeds_not_loaded_before_call(minimal_seeds_dir: Path) -> None:
    """Before any call, is_seeds_loaded() is False."""
    kg = UniversalKnowledgeGraph(MockLLMClient(), seeds_dir=minimal_seeds_dir)
    assert not kg.is_seeds_loaded()


# ── E-006-02: Currency validation ─────────────────────────────────────────

def test_knowledge_graph_corrects_invalid_currency() -> None:
    """Bundle with invalid currency 'XYZ' is corrected."""
    kg = UniversalKnowledgeGraph(MockLLMClient())
    bundle = CausalKnowledgeBundle(
        domain="healthcare",
        region=RegionInfo(country="India"),
        currency_code="XYZ",
    )
    fixed = kg._hallucination_guard(bundle)
    assert fixed.currency_code in _VALID_CURRENCY_CODES
    assert fixed.currency_code == "INR"


def test_knowledge_graph_keeps_valid_currency() -> None:
    """Bundle with valid currency 'USD' is not changed."""
    kg = UniversalKnowledgeGraph(MockLLMClient())
    bundle = CausalKnowledgeBundle(
        domain="banking",
        region=RegionInfo(country="United States"),
        currency_code="USD",
    )
    fixed = kg._hallucination_guard(bundle)
    assert fixed.currency_code == "USD"


def test_knowledge_graph_fixes_pA_IN_locale() -> None:
    """pa_IN locale in bundle is corrected to hi_IN."""
    from synthflow.models.schemas import LocaleInfo
    kg = UniversalKnowledgeGraph(MockLLMClient())
    bundle = CausalKnowledgeBundle(
        domain="hr",
        currency_code="INR",
        locale=LocaleInfo(faker_locale="pa_IN"),
    )
    fixed = kg._hallucination_guard(bundle)
    assert fixed.locale is not None
    assert fixed.locale.faker_locale == "hi_IN"


# ── E-006-03: Geographic grounding ────────────────────────────────────────

def test_knowledge_graph_geographic_grounding_returns_cities(
    kg_with_seeds: UniversalKnowledgeGraph,
    india_intent: IntentObject,
) -> None:
    """Ground geography returns Indian cities from seed data."""
    kg_with_seeds._load_seed_data()
    geo = kg_with_seeds._ground_geography(india_intent)
    assert "cities" in geo
    assert len(geo["cities"]) >= 1
    assert geo["cities"][0]["city"] in ("Mumbai", "Delhi")


def test_knowledge_graph_geographic_grounding_empty_for_no_region() -> None:
    """No region in intent → empty geo context dict."""
    kg = UniversalKnowledgeGraph(MockLLMClient())
    intent = IntentObject(domain="healthcare", row_count=50)
    geo = kg._ground_geography(intent)
    assert geo == {}


def test_knowledge_graph_geographic_grounding_economics(
    kg_with_seeds: UniversalKnowledgeGraph,
    india_intent: IntentObject,
) -> None:
    """Ground geography returns economics data for India."""
    kg_with_seeds._load_seed_data()
    geo = kg_with_seeds._ground_geography(india_intent)
    if "economics" in geo:
        assert geo["economics"]["currency_code"] == "INR"


# ── E-006-04: Distribution priors from seed ───────────────────────────────

def test_knowledge_graph_distribution_prior_salary(
    kg_with_seeds: UniversalKnowledgeGraph,
) -> None:
    """get_distribution_prior('salary') returns lognormal."""
    kg_with_seeds._load_seed_data()
    prior = kg_with_seeds.get_distribution_prior("salary")
    if prior is not None:  # seeds loaded
        assert prior["distribution_type"] == "lognormal"


def test_knowledge_graph_distribution_prior_age(
    kg_with_seeds: UniversalKnowledgeGraph,
) -> None:
    """get_distribution_prior('age') returns truncated_normal."""
    kg_with_seeds._load_seed_data()
    prior = kg_with_seeds.get_distribution_prior("age")
    if prior is not None:
        assert prior["distribution_type"] == "truncated_normal"


# ── E-006-05: Returns valid bundle ────────────────────────────────────────

@pytest.mark.asyncio
async def test_knowledge_graph_returns_valid_bundle(
    kg_with_seeds: UniversalKnowledgeGraph,
    india_intent: IntentObject,
) -> None:
    """activate() always returns a CausalKnowledgeBundle with required fields."""
    bundle = await kg_with_seeds.activate(india_intent)
    assert isinstance(bundle, CausalKnowledgeBundle)
    assert bundle.domain == "healthcare"
    assert bundle.currency_code in _VALID_CURRENCY_CODES
    assert bundle.temporal_patterns is not None
    assert bundle.dirty_data_profile is not None


@pytest.mark.asyncio
async def test_knowledge_graph_bundle_has_india_currency(
    kg_with_seeds: UniversalKnowledgeGraph,
    india_intent: IntentObject,
) -> None:
    """Bundle for India has INR currency."""
    bundle = await kg_with_seeds.activate(india_intent)
    assert bundle.currency_code == "INR"


@pytest.mark.asyncio
async def test_knowledge_graph_minimal_bundle_fallback(minimal_seeds_dir: Path) -> None:
    """_minimal_bundle always returns a valid CausalKnowledgeBundle."""
    kg = UniversalKnowledgeGraph(MockLLMClient(), seeds_dir=minimal_seeds_dir)
    intent = IntentObject(domain="agriculture", row_count=50)
    bundle = kg._minimal_bundle(intent)
    assert isinstance(bundle, CausalKnowledgeBundle)
    assert bundle.domain == "agriculture"
