# ───────────────────────────────────────────────────────────────
# Copyright (c) 2026 Ambrish Nigam
# Author : Ambrish Nigam | https://github.com/ambrishnigam30
# Project: SynthFlow — Autonomous Synthetic Data Orchestration Platform
# License : Apache License 2.0 | https://www.apache.org/licenses/LICENSE-2.0
# Module : Tests for ConstraintPhysicsEngine
# ───────────────────────────────────────────────────────────────

from __future__ import annotations

import pandas as pd
import numpy as np
import pytest


@pytest.mark.asyncio
async def test_constraint_engine_builds_constraint_set(
    mock_llm_client, sample_schema, sample_knowledge_bundle
) -> None:
    """build_constraint_set returns a ConstraintSet instance."""
    from synthflow.engines.constraint_engine import ConstraintPhysicsEngine
    from synthflow.models.schemas import ConstraintSet

    engine = ConstraintPhysicsEngine(llm_client=mock_llm_client)
    constraints = await engine.build_constraint_set(sample_schema, sample_knowledge_bundle)
    assert isinstance(constraints, ConstraintSet)


@pytest.mark.asyncio
async def test_constraint_engine_produces_rules(
    mock_llm_client, sample_schema, sample_knowledge_bundle
) -> None:
    """ConstraintSet built from sample_schema has at least one rule."""
    from synthflow.engines.constraint_engine import ConstraintPhysicsEngine

    engine = ConstraintPhysicsEngine(llm_client=mock_llm_client)
    constraints = await engine.build_constraint_set(sample_schema, sample_knowledge_bundle)
    # sample_schema has columns with min_value/max_value/enum/unique — expect rules
    assert len(constraints.rules) > 0, "Expected at least one constraint rule"


@pytest.mark.asyncio
async def test_constraint_engine_includes_range_rule_for_age(
    mock_llm_client, sample_schema, sample_knowledge_bundle
) -> None:
    """age column (min=0, max=120) produces a RANGE constraint rule."""
    from synthflow.engines.constraint_engine import ConstraintPhysicsEngine
    from synthflow.models.schemas import ConstraintType

    engine = ConstraintPhysicsEngine(llm_client=mock_llm_client)
    constraints = await engine.build_constraint_set(sample_schema, sample_knowledge_bundle)
    range_rules = [
        r for r in constraints.rules
        if r.constraint_type == ConstraintType.RANGE
        and "age" in r.columns
    ]
    assert len(range_rules) >= 1, "Expected a RANGE rule for 'age' column"


@pytest.mark.asyncio
async def test_constraint_engine_includes_enum_rule_for_gender(
    mock_llm_client, sample_schema, sample_knowledge_bundle
) -> None:
    """gender column (enum_values set) produces an ENUM constraint rule."""
    from synthflow.engines.constraint_engine import ConstraintPhysicsEngine
    from synthflow.models.schemas import ConstraintType

    engine = ConstraintPhysicsEngine(llm_client=mock_llm_client)
    constraints = await engine.build_constraint_set(sample_schema, sample_knowledge_bundle)
    enum_rules = [
        r for r in constraints.rules
        if r.constraint_type == ConstraintType.ENUM
        and "gender" in r.columns
    ]
    assert len(enum_rules) >= 1, "Expected an ENUM rule for 'gender' column"


@pytest.mark.asyncio
async def test_constraint_engine_includes_unique_rule_for_pk(
    mock_llm_client, sample_schema, sample_knowledge_bundle
) -> None:
    """patient_id (is_primary_key=True) produces a UNIQUE constraint rule."""
    from synthflow.engines.constraint_engine import ConstraintPhysicsEngine
    from synthflow.models.schemas import ConstraintType

    engine = ConstraintPhysicsEngine(llm_client=mock_llm_client)
    constraints = await engine.build_constraint_set(sample_schema, sample_knowledge_bundle)
    unique_rules = [
        r for r in constraints.rules
        if r.constraint_type == ConstraintType.UNIQUE
        and "patient_id" in r.columns
    ]
    assert len(unique_rules) >= 1, "Expected a UNIQUE rule for 'patient_id' primary key"


@pytest.mark.asyncio
async def test_constraint_engine_dag_rules_produce_causal_constraints(
    mock_llm_client, sample_schema, sample_knowledge_bundle
) -> None:
    """DAG rules in knowledge bundle produce CAUSAL constraint rules."""
    from synthflow.engines.constraint_engine import ConstraintPhysicsEngine
    from synthflow.models.schemas import ConstraintType

    engine = ConstraintPhysicsEngine(llm_client=mock_llm_client)
    constraints = await engine.build_constraint_set(sample_schema, sample_knowledge_bundle)
    causal_rules = [
        r for r in constraints.rules
        if r.constraint_type == ConstraintType.CAUSAL
    ]
    # sample_knowledge_bundle has 2 DAG rules → expect 2 causal constraints
    assert len(causal_rules) >= 2, (
        f"Expected at least 2 CAUSAL rules from DAG, found {len(causal_rules)}"
    )


@pytest.mark.asyncio
async def test_constraint_engine_no_duplicate_rule_names(
    mock_llm_client, sample_schema, sample_knowledge_bundle
) -> None:
    """All constraint rule names are unique within the ConstraintSet."""
    from synthflow.engines.constraint_engine import ConstraintPhysicsEngine

    engine = ConstraintPhysicsEngine(llm_client=mock_llm_client)
    constraints = await engine.build_constraint_set(sample_schema, sample_knowledge_bundle)
    names = [r.name for r in constraints.rules]
    assert len(names) == len(set(names)), (
        "Duplicate constraint rule names found: "
        + str([n for n in names if names.count(n) > 1])
    )


@pytest.mark.asyncio
async def test_constraint_engine_domain_set_on_constraint_set(
    mock_llm_client, sample_schema, sample_knowledge_bundle
) -> None:
    """ConstraintSet.domain is populated from the knowledge bundle."""
    from synthflow.engines.constraint_engine import ConstraintPhysicsEngine

    engine = ConstraintPhysicsEngine(llm_client=mock_llm_client)
    constraints = await engine.build_constraint_set(sample_schema, sample_knowledge_bundle)
    assert constraints.domain == "healthcare"


def test_constraint_rule_action_on_violation_defaults() -> None:
    """ConstraintRule default action_on_violation is FLAG."""
    from synthflow.models.schemas import ConstraintRule, ConstraintType, ViolationAction

    rule = ConstraintRule(
        name="test_rule",
        constraint_type=ConstraintType.RANGE,
        columns=["salary"],
    )
    assert rule.action_on_violation == ViolationAction.FLAG


def test_constraint_set_empty_init() -> None:
    """ConstraintSet can be created with zero rules."""
    from synthflow.models.schemas import ConstraintSet

    cs = ConstraintSet(rules=[])
    assert cs.rules == []
    assert cs.version == "1.0"
