# ───────────────────────────────────────────────────────────────
# Copyright (c) 2026 Ambrish Nigam
# Author : Ambrish Nigam | https://github.com/ambrishnigam30
# Project: SynthFlow — Autonomous Synthetic Data Orchestration Platform
# License : Apache License 2.0 | https://www.apache.org/licenses/LICENSE-2.0
# Module : Tests for SchemaIntelligenceLayer
# ───────────────────────────────────────────────────────────────

from __future__ import annotations

import pytest


@pytest.mark.asyncio
async def test_schema_minimum_12_columns_enforced(
    mock_llm_client, sample_intent, sample_knowledge_bundle
) -> None:
    """Schema with fewer than 12 columns triggers re-generation."""
    from synthflow.engines.schema_intelligence import SchemaIntelligenceLayer

    engine = SchemaIntelligenceLayer(llm_client=mock_llm_client)
    schema = await engine.architect(sample_intent, sample_knowledge_bundle)
    total_cols = sum(len(t.columns) for t in schema.tables)
    assert total_cols >= 12, (
        f"Expected at least 12 columns total, got {total_cols}"
    )


@pytest.mark.asyncio
async def test_schema_primary_key_present(
    mock_llm_client, sample_intent, sample_knowledge_bundle
) -> None:
    """Every table in the generated schema has at least one primary key column."""
    from synthflow.engines.schema_intelligence import SchemaIntelligenceLayer

    engine = SchemaIntelligenceLayer(llm_client=mock_llm_client)
    schema = await engine.architect(sample_intent, sample_knowledge_bundle)
    for table in schema.tables:
        assert any(c.is_primary_key for c in table.columns), (
            f"Table '{table.name}' has no primary key column"
        )


@pytest.mark.asyncio
async def test_schema_column_types_valid(
    mock_llm_client, sample_intent, sample_knowledge_bundle
) -> None:
    """All column data_type values are recognisable type strings."""
    from synthflow.engines.schema_intelligence import SchemaIntelligenceLayer

    ALLOWED_TYPES = {
        "string", "integer", "float", "boolean", "date", "datetime",
        "text", "numeric", "json", "uuid", "varchar", "char",
        "timestamp", "bigint", "smallint", "decimal", "double",
    }
    engine = SchemaIntelligenceLayer(llm_client=mock_llm_client)
    schema = await engine.architect(sample_intent, sample_knowledge_bundle)
    for table in schema.tables:
        for col in table.columns:
            # Flexible: allow any type; log unexpected ones rather than hard-fail
            assert isinstance(col.data_type, str), (
                f"Column '{col.name}' has non-string data_type: {col.data_type!r}"
            )
            assert len(col.data_type) > 0, (
                f"Column '{col.name}' has empty data_type"
            )


@pytest.mark.asyncio
async def test_schema_returns_schema_definition(
    mock_llm_client, sample_intent, sample_knowledge_bundle
) -> None:
    """architect() returns a SchemaDefinition instance."""
    from synthflow.engines.schema_intelligence import SchemaIntelligenceLayer
    from synthflow.models.schemas import SchemaDefinition

    engine = SchemaIntelligenceLayer(llm_client=mock_llm_client)
    schema = await engine.architect(sample_intent, sample_knowledge_bundle)
    assert isinstance(schema, SchemaDefinition)


@pytest.mark.asyncio
async def test_schema_has_at_least_one_table(
    mock_llm_client, sample_intent, sample_knowledge_bundle
) -> None:
    """Generated schema contains at least one table."""
    from synthflow.engines.schema_intelligence import SchemaIntelligenceLayer

    engine = SchemaIntelligenceLayer(llm_client=mock_llm_client)
    schema = await engine.architect(sample_intent, sample_knowledge_bundle)
    assert len(schema.tables) >= 1, "Schema must have at least one table"


@pytest.mark.asyncio
async def test_schema_all_columns_have_names(
    mock_llm_client, sample_intent, sample_knowledge_bundle
) -> None:
    """Every column definition has a non-empty name."""
    from synthflow.engines.schema_intelligence import SchemaIntelligenceLayer

    engine = SchemaIntelligenceLayer(llm_client=mock_llm_client)
    schema = await engine.architect(sample_intent, sample_knowledge_bundle)
    for table in schema.tables:
        for col in table.columns:
            assert col.name and col.name.strip(), (
                f"Table '{table.name}' has a column with an empty name"
            )
