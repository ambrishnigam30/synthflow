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
    "5. Return ONLY valid JSON\n"
    "6. Every numeric column MUST have realistic min_value and max_value constraints\n"
    "7. Age is always INTEGER data_type with min_value=0 and max_value=100\n"
    "8. Monetary columns are FLOAT data_type with 2 decimal precision implied\n"
    "9. String columns representing categories MUST include enum_values with real domain values\n"
    "10. If both date_of_birth and age columns exist, mark age as computed (add note in description)\n"
    "11. Data types must be precise: use integer for counts/ages, float for continuous measures, "
    "string for text/codes, datetime for timestamps, date for calendar dates, uuid for IDs\n"
    "12. Primary key column must use uuid data_type and unique=true and nullable=false\n"
    "13. Healthcare patient datasets MUST include at minimum: patient_id, patient_name, age, "
    "gender, date_of_birth, diagnosis, admission_date. These are mandatory for any healthcare "
    "patient record system."
)

_SCHEMA_USER_TEMPLATE = (
    "Design a schema for: domain='{domain}', region='{region}', row_count={row_count}.\n"
    "Implied columns hint: {implied}.\n"
    "{column_design_hint}"
    "Return JSON: {{\"table_name\": string, \"columns\": [{{\"name\": string, "
    "\"data_type\": string, \"semantic_type\": string, \"is_primary_key\": bool, "
    "\"nullable\": bool, \"null_rate\": float, \"min_value\": number|null, "
    "\"max_value\": number|null, \"enum_values\": list, \"unique\": bool, "
    "\"description\": string}}]}}\n"
    "data_type must be one of: string, integer, float, boolean, datetime, date, json, uuid\n"
    "Ensure at least 12 columns. Every numeric column must have min_value and max_value."
)

# Minimum parseable columns from knowledge bundle to use as foundation
_MIN_KNOWLEDGE_COLUMNS = 8
# Final minimum column count (enforced via padding if needed)
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
        """
        Build a SchemaDefinition from the knowledge bundle's column_design if available,
        merging with a schema LLM call when the knowledge bundle has 8–11 columns.
        Falls back to a full LLM schema call when fewer than 8 columns are parseable.
        """
        partial_schema: Optional[SchemaDefinition] = None

        if knowledge.column_design:
            try:
                partial_schema = self._build_from_column_design(knowledge.column_design, intent)
                # If the knowledge bundle already provides 12+ columns, use it directly.
                total = sum(len(t.columns) for t in partial_schema.tables)
                if total >= _MIN_COLUMNS:
                    _LOG.info(
                        "Using knowledge bundle column_design directly (%d columns)", total
                    )
                    return partial_schema
                # 8–11 columns: use as foundation and let the LLM fill the rest.
                _LOG.info(
                    "Knowledge bundle has %d columns (< %d) — merging with schema LLM",
                    total, _MIN_COLUMNS,
                )
            except ValueError as exc:
                _LOG.warning(
                    "column_design too sparse (%s) — falling back to full schema LLM", exc
                )
                partial_schema = None
            except Exception as exc:
                _LOG.warning(
                    "column_design from knowledge bundle unusable (%s) — falling back to LLM", exc
                )
                partial_schema = None

        llm_schema = await self._schema_llm_call(intent, knowledge, partial_schema)

        if partial_schema is not None:
            return self._merge_schemas(partial_schema, llm_schema)
        return llm_schema

    async def _schema_llm_call(
        self,
        intent: IntentObject,
        knowledge: CausalKnowledgeBundle,
        partial_schema: Optional[SchemaDefinition],
    ) -> SchemaDefinition:
        """Invoke the schema LLM to design (or extend) a schema."""
        region_str = intent.region.country if intent.region else "global"
        implied = ", ".join(intent.implied_columns) if intent.implied_columns else "auto"

        # Build hint from already-known columns so LLM adds new ones, not duplicates.
        column_design_hint = ""
        if partial_schema is not None:
            existing = [c.name for t in partial_schema.tables for c in t.columns]
            column_design_hint = (
                f"These columns are already defined — DO NOT duplicate them, only ADD new ones "
                f"to reach 12 total: {', '.join(existing)}.\n"
            )
        elif knowledge.column_design:
            col_names = [
                c.get("column_name", "") for c in knowledge.column_design
                if isinstance(c, dict) and c.get("column_name")
            ]
            if col_names:
                column_design_hint = (
                    f"Use these columns from domain knowledge analysis: {', '.join(col_names)}.\n"
                )

        user = _SCHEMA_USER_TEMPLATE.format(
            domain=intent.domain,
            region=region_str,
            row_count=intent.row_count,
            implied=implied,
            column_design_hint=column_design_hint,
        )
        try:
            raw = await self._llm.complete(
                user,
                system_prompt=_SCHEMA_SYSTEM_PROMPT,
                json_mode=True,
                temperature=0.3,
                max_tokens=3072,
            )
            _LOG.debug(
                "Schema LLM raw response (first 500 chars): %s",
                (raw or "")[:500],
            )
            data = safe_json_loads(raw)
            if data is None:
                _LOG.error(
                    "Schema LLM returned unparseable JSON. Full response: %s", raw
                )
                raise RuntimeError(
                    "Schema design failed: LLM returned content that could not be parsed as JSON. "
                    "Response preview: " + (raw or "")[:200]
                )
            return self._parse_schema_response(data, intent)
        except RuntimeError:
            raise
        except Exception as exc:
            raise RuntimeError(
                f"Schema design failed: error parsing LLM response ({type(exc).__name__}: {exc}). "
                "This is a parsing error, not an LLM connectivity error."
            ) from exc

    def _build_from_column_design(
        self, column_design: list[dict[str, Any]], intent: IntentObject
    ) -> SchemaDefinition:
        """
        Build a SchemaDefinition directly from the master prompt's column_design list.
        Avoids a second LLM call when column_design is already rich enough.
        """
        from typing import Any as _Any

        table_name = (intent.domain + "_records").lower().replace(" ", "_")
        columns: list[ColumnDefinition] = []
        has_pk = False

        _VALID = frozenset({"string", "integer", "float", "boolean", "datetime", "date", "json", "uuid"})

        for cd in column_design:
            if not isinstance(cd, dict):
                continue
            name = str(cd.get("column_name", "")).strip()
            if not name:
                continue
            dtype = str(cd.get("data_type", "string")).lower()
            if dtype not in _VALID:
                dtype = "string"
            is_pk = bool(cd.get("is_primary_key", False))
            if is_pk:
                has_pk = True

            enum_vals: list[_Any] = []
            if cd.get("enum_values") and isinstance(cd["enum_values"], list):
                enum_vals = list(cd["enum_values"])

            # Also gather enum_values from value pools if semantic_type matches
            try:
                col = ColumnDefinition(
                    name=name,
                    data_type=dtype,
                    semantic_type=str(cd.get("semantic_type", "")),
                    is_primary_key=is_pk,
                    nullable=bool(cd.get("nullable", True)),
                    null_rate=float(cd.get("null_rate", 0.0)),
                    min_value=cd.get("min_value"),
                    max_value=cd.get("max_value"),
                    enum_values=enum_vals,
                    unique=bool(cd.get("unique", False) or is_pk),
                    description=str(cd.get("description", "")),
                    format_hint=cd.get("format_hint"),
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

        if len(columns) < _MIN_KNOWLEDGE_COLUMNS:
            raise ValueError(
                f"column_design has only {len(columns)} parseable columns "
                f"(min {_MIN_KNOWLEDGE_COLUMNS} to use as foundation)"
            )

        table = SchemaTable(name=table_name, columns=columns)
        return SchemaDefinition(tables=[table])

    def _merge_schemas(
        self,
        partial: SchemaDefinition,
        full: SchemaDefinition,
    ) -> SchemaDefinition:
        """
        Merge knowledge-bundle columns (partial) with LLM schema (full).
        Knowledge-bundle columns take precedence; LLM-only columns are appended
        to reach the minimum column count without discarding either source.
        """
        merged_tables: list[SchemaTable] = []
        for p_table, f_table in zip(partial.tables, full.tables):
            known_names = {c.name for c in p_table.columns}
            extra = [c for c in f_table.columns if c.name not in known_names]
            merged_cols = list(p_table.columns) + extra
            merged_tables.append(
                SchemaTable(
                    name=p_table.name,
                    columns=merged_cols,
                    description=p_table.description or f_table.description,
                )
            )
        # If full schema has more tables than partial, append them.
        if len(full.tables) > len(partial.tables):
            merged_tables.extend(full.tables[len(partial.tables):])
        return SchemaDefinition(
            tables=merged_tables,
            version=full.version,
            description=full.description,
            relationships=full.relationships,
        )

    def _parse_schema_response(
        self, data: dict[str, Any], intent: IntentObject
    ) -> SchemaDefinition:
        if not isinstance(data, dict):
            raise RuntimeError(
                f"Schema validation failed: expected JSON object from LLM, got {type(data).__name__}. "
                "The LLM may have returned a list or primitive instead of {{table_name, columns}}."
            )

        table_name = str(data.get("table_name", intent.domain + "_records")).lower()
        raw_cols: list[dict[str, Any]] = data.get("columns", [])

        columns: list[ColumnDefinition] = []

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

        # Safety net: ensure exactly one PK column exists in the parsed list.
        # We check the ACTUAL list (not the LLM flag) because a PK column may have
        # been parsed but its ColumnDefinition constructor silently failed above.
        actual_pks = [c for c in columns if c.is_primary_key]
        if not actual_pks:
            # Try to promote an existing id-like column first
            promoted = False
            for i, col in enumerate(columns):
                name_lower = col.name.lower()
                is_id_like = (
                    name_lower.endswith("_id")
                    or name_lower == "id"
                    or col.semantic_type in ("identifier", "id", "uuid")
                )
                if is_id_like:
                    _LOG.warning(
                        "Schema validation: no primary key found in LLM response for table '%s'. "
                        "Auto-promoting column '%s' to primary key.",
                        table_name, col.name,
                    )
                    columns[i] = col.model_copy(
                        update={"is_primary_key": True, "unique": True, "nullable": False}
                    )
                    promoted = True
                    break
            if not promoted:
                # Last resort: inject a UUID PK as the first column
                _LOG.warning(
                    "Schema validation: no id-like column found in table '%s'. "
                    "Injecting auto-generated UUID primary key.",
                    table_name,
                )
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
