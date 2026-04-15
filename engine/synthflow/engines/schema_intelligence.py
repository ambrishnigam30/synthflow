# ───────────────────────────────────────────────────────────────
# Copyright (c) 2026 Ambrish Nigam
# Author : Ambrish Nigam | https://github.com/ambrishnigam30
# Project: SynthFlow — Autonomous Synthetic Data Orchestration Platform
# License : Apache License 2.0 | https://www.apache.org/licenses/LICENSE-2.0
# Module : SchemaIntelligenceLayer — designs optimal table schemas via LLM
# ───────────────────────────────────────────────────────────────

from __future__ import annotations

import uuid
from typing import Any, Optional, Union

from synthflow.causal_dag import CausalDAG
from synthflow.llm_client import LLMClient, MockLLMClient
from synthflow.models.schemas import (
    CausalKnowledgeBundle,
    ColumnDefinition,
    IntentObject,
    SchemaDefinition,
    SchemaTable,
)
from synthflow.utils.helpers import safe_json_loads
from synthflow.utils.logger import get_logger

_LOG = get_logger("schema_intelligence", component="schema_intelligence")

_VALID_DATA_TYPES = frozenset(
    {"string", "integer", "float", "boolean", "datetime", "date", "json", "uuid"}
)

_SCHEMA_SYSTEM_PROMPT = (
    "You are SynthFlow's schema architect. Design a database table schema optimized for "
    "synthetic data generation. Rules:\n"
    "1. Minimum 12 columns — more is better for realistic data\n"
    "2. Always include at least one primary key column (UUID)\n"
    "3. Column ordering: IDs → demographics → measures → metadata\n"
    "4. Infer FK relationships where logical\n"
    "5. Return ONLY valid JSON"
)

_SCHEMA_USER_TEMPLATE = (
    "Design a schema for: domain='{domain}', region='{region}', row_count={row_count}.\n"
    "Implied columns hint: {implied}.\n"
    "Return JSON: {{\"table_name\": string, \"columns\": [{{\"name\": string, "
    "\"data_type\": string, \"semantic_type\": string, \"is_primary_key\": bool, "
    "\"nullable\": bool, \"null_rate\": float, \"min_value\": number|null, "
    "\"max_value\": number|null, \"enum_values\": list, \"unique\": bool, "
    "\"description\": string}}]}}\n"
    "data_type must be one of: string, integer, float, boolean, datetime, date, json, uuid\n"
    "Ensure at least 12 columns."
)

# Minimum required columns when LLM returns too few
_MIN_COLUMNS = 12


class SchemaIntelligenceLayer:
    """
    Designs SchemaDefinition objects via LLM, with minimum-12-column enforcement
    and automatic PK/FK inference.
    """

    def __init__(self, llm_client: Union[LLMClient, MockLLMClient]) -> None:
        self._llm = llm_client

    async def architect(
        self,
        intent: IntentObject,
        knowledge: CausalKnowledgeBundle,
    ) -> SchemaDefinition:
        """
        Design a SchemaDefinition for *intent* enriched by *knowledge*.

        Args:
            intent:    Parsed user intent.
            knowledge: Domain knowledge bundle.

        Returns:
            Validated SchemaDefinition with ≥ 12 columns.
        """
        schema = await self._llm_architect(intent, knowledge)
        schema = self._ensure_minimum_columns(schema, intent, knowledge)
        schema = self._assign_generation_order(schema, knowledge)
        return schema

    # ── LLM call ──────────────────────────────────────────────────────────

    async def _llm_architect(
        self,
        intent: IntentObject,
        knowledge: CausalKnowledgeBundle,
    ) -> SchemaDefinition:
        region_str = (
            intent.region.country if intent.region else "global"
        )
        implied = ", ".join(intent.implied_columns) if intent.implied_columns else "auto"
        user = _SCHEMA_USER_TEMPLATE.format(
            domain=intent.domain,
            region=region_str,
            row_count=intent.row_count,
            implied=implied,
        )
        try:
            raw = await self._llm.complete(
                user,
                system_prompt=_SCHEMA_SYSTEM_PROMPT,
                json_mode=True,
                temperature=0.3,
                max_tokens=2048,
            )
            data = safe_json_loads(raw)
            return self._parse_schema_response(data, intent)
        except Exception as exc:
            _LOG.warning("LLM schema generation failed: %s — using fallback schema", exc)
            return self._fallback_schema(intent)

    def _parse_schema_response(
        self, data: dict[str, Any], intent: IntentObject
    ) -> SchemaDefinition:
        if not isinstance(data, dict):
            return self._fallback_schema(intent)

        table_name = str(data.get("table_name", intent.domain + "_records")).lower()
        raw_cols: list[dict[str, Any]] = data.get("columns", [])

        columns: list[ColumnDefinition] = []
        has_pk = False

        for rc in raw_cols:
            if not isinstance(rc, dict):
                continue
            name = str(rc.get("name", "")).strip()
            if not name:
                continue
            dtype = str(rc.get("data_type", "string")).lower()
            if dtype not in _VALID_DATA_TYPES:
                dtype = "string"
            is_pk = bool(rc.get("is_primary_key", False))
            if is_pk:
                has_pk = True
            try:
                col = ColumnDefinition(
                    name=name,
                    data_type=dtype,
                    semantic_type=str(rc.get("semantic_type", "")),
                    is_primary_key=is_pk,
                    nullable=bool(rc.get("nullable", True)),
                    null_rate=float(rc.get("null_rate", 0.0)),
                    min_value=rc.get("min_value"),
                    max_value=rc.get("max_value"),
                    enum_values=list(rc.get("enum_values", [])),
                    unique=bool(rc.get("unique", False) or is_pk),
                    description=str(rc.get("description", "")),
                )
                columns.append(col)
            except Exception:
                continue

        if not has_pk:
            pk_col = ColumnDefinition(
                name=f"{table_name}_id",
                data_type="uuid",
                semantic_type="id",
                is_primary_key=True,
                unique=True,
                nullable=False,
                description="Auto-generated primary key",
            )
            columns.insert(0, pk_col)

        table = SchemaTable(name=table_name, columns=columns)
        return SchemaDefinition(tables=[table])

    # ── Minimum columns enforcement ───────────────────────────────────────

    def _ensure_minimum_columns(
        self,
        schema: SchemaDefinition,
        intent: IntentObject,
        knowledge: CausalKnowledgeBundle,
    ) -> SchemaDefinition:
        """Re-generate or pad the schema until it has ≥ _MIN_COLUMNS columns."""
        updated_tables = []
        for table in schema.tables:
            if len(table.columns) >= _MIN_COLUMNS:
                updated_tables.append(table)
                continue
            # Pad with generic metadata columns
            padded = list(table.columns)
            filler_cols = _make_filler_columns(len(padded), _MIN_COLUMNS, intent.domain)
            padded.extend(filler_cols)
            updated_tables.append(SchemaTable(name=table.name, columns=padded, description=table.description))
        return SchemaDefinition(
            tables=updated_tables,
            version=schema.version,
            description=schema.description,
            relationships=schema.relationships,
        )

    def _assign_generation_order(
        self,
        schema: SchemaDefinition,
        knowledge: CausalKnowledgeBundle,
    ) -> SchemaDefinition:
        """Assign generation_order based on DAG topology."""
        dag = CausalDAG()
        if knowledge.dag_rules:
            dag.build_from_rules(knowledge.dag_rules)
            try:
                topo = dag.topological_sort()
            except Exception:
                topo = []
        else:
            topo = []

        updated_tables = []
        for table in schema.tables:
            cols_by_name = {c.name: c for c in table.columns}
            updated_cols = []
            for order_idx, col_name in enumerate(topo):
                if col_name in cols_by_name:
                    col = cols_by_name[col_name]
                    updated_cols.append(col.model_copy(update={"generation_order": order_idx + 1}))
                    del cols_by_name[col_name]
            # Remaining columns (not in DAG) get order 0
            for col in cols_by_name.values():
                updated_cols.append(col)
            # Sort: PK first, then by generation_order, then rest
            updated_cols.sort(key=_col_sort_key)
            updated_tables.append(
                SchemaTable(name=table.name, columns=updated_cols, description=table.description)
            )
        return SchemaDefinition(
            tables=updated_tables,
            version=schema.version,
            description=schema.description,
            relationships=schema.relationships,
        )

    def _fallback_schema(self, intent: IntentObject) -> SchemaDefinition:
        """Return a generic schema when the LLM completely fails."""
        table_name = f"{intent.domain}_records"
        base_cols = [
            ColumnDefinition(name=f"{table_name}_id", data_type="uuid", semantic_type="id",
                             is_primary_key=True, unique=True, nullable=False),
            ColumnDefinition(name="name", data_type="string", semantic_type="name"),
            ColumnDefinition(name="created_at", data_type="datetime", semantic_type="timestamp"),
            ColumnDefinition(name="status", data_type="string",
                             enum_values=["active", "inactive", "pending"]),
        ]
        filler = _make_filler_columns(len(base_cols), _MIN_COLUMNS, intent.domain)
        table = SchemaTable(name=table_name, columns=base_cols + filler)
        return SchemaDefinition(tables=[table])


# ── Helpers ────────────────────────────────────────────────────────────────

def _make_filler_columns(
    current: int, target: int, domain: str
) -> list[ColumnDefinition]:
    """Generate generic metadata filler columns to reach *target* count."""
    fillers: list[ColumnDefinition] = []
    generic_pool: list[tuple[str, str, str]] = [
        ("notes", "string", "notes"),
        ("category", "string", "category"),
        ("sub_category", "string", "category"),
        ("region", "string", "region"),
        ("source", "string", "source"),
        ("is_active", "boolean", "flag"),
        ("priority", "integer", "priority"),
        ("updated_at", "datetime", "timestamp"),
        ("external_id", "string", "id"),
        ("metadata_tags", "string", "tags"),
        ("confidence_score", "float", "score"),
        ("version", "integer", "version"),
    ]
    for i in range(target - current):
        if i < len(generic_pool):
            col_name, dtype, stype = generic_pool[i]
        else:
            col_name = f"field_{i + 1}"
            dtype = "string"
            stype = ""
        fillers.append(ColumnDefinition(
            name=col_name,
            data_type=dtype,
            semantic_type=stype,
            nullable=True,
            null_rate=0.05,
            description=f"Auto-generated filler column for {domain}",
        ))
    return fillers


def _col_sort_key(col: ColumnDefinition) -> tuple[int, int, str]:
    """Sort: PK=0, then by generation_order, then alphabetically."""
    return (0 if col.is_primary_key else 1, col.generation_order, col.name)
