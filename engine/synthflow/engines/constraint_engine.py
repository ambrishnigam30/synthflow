# ───────────────────────────────────────────────────────────────
# Copyright (c) 2026 Ambrish Nigam
# Author : Ambrish Nigam | https://github.com/ambrishnigam30
# Project: SynthFlow — Autonomous Synthetic Data Orchestration Platform
# License : Apache License 2.0 | https://www.apache.org/licenses/LICENSE-2.0
# Module : ConstraintPhysicsEngine — builds and enforces domain constraint sets
# ───────────────────────────────────────────────────────────────

from __future__ import annotations

from typing import Any, Optional, Union

import pandas as pd

from synthflow.llm_client import LLMClient, MockLLMClient
from synthflow.models.schemas import (
    CausalKnowledgeBundle,
    ColumnDefinition,
    ConstraintRule,
    ConstraintSet,
    ConstraintType,
    SchemaDefinition,
    Severity,
    ViolationAction,
)
from synthflow.utils.helpers import safe_json_loads
from synthflow.utils.logger import get_logger

_LOG = get_logger("constraint_engine", component="constraint_engine")

_CONSTRAINT_SYSTEM_PROMPT = (
    "You are SynthFlow's constraint engineer. Given a domain schema, generate additional "
    "domain physics rules as JSON list. Each rule: {\"name\": str, \"constraint_type\": str, "
    "\"columns\": [str], \"expression\": str, \"description\": str, \"severity\": str}. "
    "constraint_type in: range, enum, temporal, causal, referential, expression. "
    "severity in: low, medium, high, critical. Return ONLY a JSON array."
)

_CONSTRAINT_USER_TEMPLATE = (
    "Generate domain physics constraints for domain='{domain}', "
    "table='{table}', columns={columns}. Include temporal (date ordering), "
    "range (realistic value bounds), and causal rules."
)


class ConstraintPhysicsEngine:
    """
    Builds a ConstraintSet from three sources:
    1. Schema-derived (min/max/enum/unique from ColumnDefinition)
    2. DAG-derived (causal ordering from CausalKnowledgeBundle)
    3. LLM-augmented (domain physics rules)

    Also provides enforce_batch() to apply constraints to a DataFrame.
    """

    def __init__(self, llm_client: Union[LLMClient, MockLLMClient]) -> None:
        self._llm = llm_client

    async def build_constraint_set(
        self,
        schema: SchemaDefinition,
        knowledge: CausalKnowledgeBundle,
    ) -> ConstraintSet:
        """
        Build a ConstraintSet from schema + knowledge + LLM augmentation.

        Args:
            schema:    Schema for the data to generate.
            knowledge: Domain knowledge bundle.

        Returns:
            ConstraintSet with all rules merged.
        """
        rules: list[ConstraintRule] = []

        for table in schema.tables:
            rules.extend(self._schema_derived_constraints(table.columns))
            rules.extend(self._dag_derived_constraints(knowledge))
            rules.extend(await self._llm_augmented_constraints(schema, knowledge))

        # Deduplicate by rule name
        seen: set[str] = set()
        deduped: list[ConstraintRule] = []
        for rule in rules:
            if rule.name not in seen:
                seen.add(rule.name)
                deduped.append(rule)

        return ConstraintSet(
            rules=deduped,
            domain=knowledge.domain,
        )

    # ── Schema-derived ────────────────────────────────────────────────────

    def _schema_derived_constraints(
        self, columns: list[ColumnDefinition]
    ) -> list[ConstraintRule]:
        rules: list[ConstraintRule] = []
        for col in columns:
            # Range constraint
            if col.min_value is not None or col.max_value is not None:
                rules.append(ConstraintRule(
                    name=f"{col.name}_range",
                    constraint_type=ConstraintType.RANGE,
                    columns=[col.name],
                    parameters={
                        "min": col.min_value,
                        "max": col.max_value,
                    },
                    action_on_violation=ViolationAction.CLIP,
                    severity=Severity.HIGH,
                    description=f"{col.name} must be within [{col.min_value}, {col.max_value}]",
                ))
            # Enum constraint
            if col.enum_values:
                rules.append(ConstraintRule(
                    name=f"{col.name}_enum",
                    constraint_type=ConstraintType.ENUM,
                    columns=[col.name],
                    parameters={"values": col.enum_values},
                    action_on_violation=ViolationAction.FLAG,
                    severity=Severity.MEDIUM,
                    description=f"{col.name} must be one of {col.enum_values}",
                ))
            # Uniqueness constraint
            if col.unique or col.is_primary_key:
                rules.append(ConstraintRule(
                    name=f"{col.name}_unique",
                    constraint_type=ConstraintType.UNIQUE,
                    columns=[col.name],
                    action_on_violation=ViolationAction.REJECT,
                    severity=Severity.CRITICAL,
                    description=f"{col.name} must be unique",
                ))
            # Null rate enforcement
            if not col.nullable:
                rules.append(ConstraintRule(
                    name=f"{col.name}_not_null",
                    constraint_type=ConstraintType.EXPRESSION,
                    columns=[col.name],
                    expression=f'df["{col.name}"].notna()',
                    action_on_violation=ViolationAction.FLAG,
                    severity=Severity.HIGH,
                    description=f"{col.name} must not be null",
                ))
        return rules

    # ── DAG-derived ───────────────────────────────────────────────────────

    def _dag_derived_constraints(
        self, knowledge: CausalKnowledgeBundle
    ) -> list[ConstraintRule]:
        rules: list[ConstraintRule] = []
        for rule in knowledge.dag_rules:
            rules.append(ConstraintRule(
                name=f"causal_{rule.parent_column}_to_{rule.child_column}",
                constraint_type=ConstraintType.CAUSAL,
                columns=[rule.parent_column, rule.child_column],
                expression=rule.lambda_str,
                action_on_violation=ViolationAction.FLAG,
                severity=Severity.MEDIUM,
                description=rule.description or f"{rule.child_column} derived from {rule.parent_column}",
            ))
        return rules

    # ── LLM-augmented ─────────────────────────────────────────────────────

    async def _llm_augmented_constraints(
        self,
        schema: SchemaDefinition,
        knowledge: CausalKnowledgeBundle,
    ) -> list[ConstraintRule]:
        if not schema.tables:
            return []
        table = schema.tables[0]
        col_names = [c.name for c in table.columns[:15]]  # cap to avoid huge prompt
        user = _CONSTRAINT_USER_TEMPLATE.format(
            domain=knowledge.domain,
            table=table.name,
            columns=col_names,
        )
        try:
            raw = await self._llm.complete(
                user,
                system_prompt=_CONSTRAINT_SYSTEM_PROMPT,
                json_mode=True,
                temperature=0.2,
                max_tokens=1024,
            )
            data = safe_json_loads(raw)
            return self._parse_llm_rules(data)
        except Exception as exc:
            _LOG.debug("LLM constraint augmentation failed: %s", exc)
            return []

    def _parse_llm_rules(self, data: Any) -> list[ConstraintRule]:
        if not isinstance(data, list):
            if isinstance(data, dict) and "rules" in data:
                data = data["rules"]
            else:
                return []
        rules: list[ConstraintRule] = []
        for item in data:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name", "")).strip()
            if not name:
                continue
            ct_str = str(item.get("constraint_type", "expression")).lower()
            try:
                ct = ConstraintType(ct_str)
            except ValueError:
                ct = ConstraintType.EXPRESSION
            sev_str = str(item.get("severity", "medium")).lower()
            try:
                sev = Severity(sev_str)
            except ValueError:
                sev = Severity.MEDIUM
            cols = list(item.get("columns", []))
            if not cols:
                continue
            try:
                rules.append(ConstraintRule(
                    name=name,
                    constraint_type=ct,
                    columns=cols,
                    expression=item.get("expression"),
                    severity=sev,
                    action_on_violation=ViolationAction.FLAG,
                    description=str(item.get("description", "")),
                ))
            except Exception:
                continue
        return rules

    # ── Enforcement ───────────────────────────────────────────────────────

    def enforce_batch(
        self,
        df: pd.DataFrame,
        constraints: ConstraintSet,
    ) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
        """
        Apply *constraints* to *df*, fixing or flagging violations.

        Args:
            df:          DataFrame to enforce constraints on.
            constraints: ConstraintSet to apply.

        Returns:
            (fixed_df, violation_log) where violation_log is a list of dicts.
        """
        result = df.copy()
        violations: list[dict[str, Any]] = []

        for rule in constraints.rules:
            if not rule.columns:
                continue

            if rule.constraint_type == ConstraintType.RANGE:
                result, viol = self._enforce_range(result, rule)
                violations.extend(viol)

            elif rule.constraint_type == ConstraintType.ENUM:
                result, viol = self._enforce_enum(result, rule)
                violations.extend(viol)

            elif rule.constraint_type == ConstraintType.EXPRESSION:
                result, viol = self._enforce_expression(result, rule)
                violations.extend(viol)

        return result, violations

    def _enforce_range(
        self, df: pd.DataFrame, rule: ConstraintRule
    ) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
        col = rule.columns[0]
        if col not in df.columns:
            return df, []
        mn = rule.parameters.get("min")
        mx = rule.parameters.get("max")
        violations: list[dict[str, Any]] = []
        try:
            series = pd.to_numeric(df[col], errors="coerce")
            mask = pd.Series([False] * len(df), index=df.index)
            if mn is not None:
                mask |= series < float(mn)
            if mx is not None:
                mask |= series > float(mx)
            if mask.any():
                count = int(mask.sum())
                violations.append({
                    "rule": rule.name,
                    "column": col,
                    "count": count,
                    "action": rule.action_on_violation,
                })
                if rule.action_on_violation == ViolationAction.CLIP:
                    df = df.copy()
                    df[col] = pd.to_numeric(df[col], errors="coerce")
                    if mn is not None:
                        df[col] = df[col].clip(lower=float(mn))
                    if mx is not None:
                        df[col] = df[col].clip(upper=float(mx))
        except Exception as exc:
            _LOG.debug("Range enforcement failed for %s: %s", col, exc)
        return df, violations

    def _enforce_enum(
        self, df: pd.DataFrame, rule: ConstraintRule
    ) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
        col = rule.columns[0]
        if col not in df.columns:
            return df, []
        allowed = set(str(v) for v in rule.parameters.get("values", []))
        if not allowed:
            return df, []
        violations: list[dict[str, Any]] = []
        try:
            mask = ~df[col].astype(str).isin(allowed)
            if mask.any():
                violations.append({
                    "rule": rule.name,
                    "column": col,
                    "count": int(mask.sum()),
                    "action": rule.action_on_violation,
                })
        except Exception as exc:
            _LOG.debug("Enum enforcement failed for %s: %s", col, exc)
        return df, violations

    def _enforce_expression(
        self, df: pd.DataFrame, rule: ConstraintRule
    ) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
        col = rule.columns[0]
        if col not in df.columns or not rule.expression:
            return df, []
        violations: list[dict[str, Any]] = []
        try:
            # Only support not-null check
            if "notna" in (rule.expression or ""):
                mask = df[col].isna()
                if mask.any():
                    violations.append({
                        "rule": rule.name,
                        "column": col,
                        "count": int(mask.sum()),
                        "action": rule.action_on_violation,
                    })
        except Exception as exc:
            _LOG.debug("Expression enforcement failed for %s: %s", col, exc)
        return df, violations
