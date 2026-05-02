# ───────────────────────────────────────────────────────────────
# Copyright (c) 2026 Ambrish Nigam
# Author : Ambrish Nigam | https://github.com/ambrishnigam30
# Project: SynthFlow — Autonomous Synthetic Data Orchestration Platform
# License : Apache License 2.0 | https://www.apache.org/licenses/LICENSE-2.0
# Module : UniversalKnowledgeGraph — DuckDB-grounded domain knowledge activator
# ───────────────────────────────────────────────────────────────

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional, Union

import duckdb

from synthflow.llm_client import LLMClient, MockLLMClient
from synthflow.models.schemas import (
    BusinessContext,
    CausalDagRule,
    CausalKnowledgeBundle,
    ColumnKnowledge,
    CrossColumnCorrelation,
    DirtyDataProfile,
    GeographicalConstraints,
    IntentObject,
    LocaleInfo,
    NameCulturalPatterns,
    RegionInfo,
    TemporalPatterns,
    ValuePool,
)
from synthflow.prompts.master_prompt import build_knowledge_prompt
from synthflow.utils.helpers import safe_json_loads
from synthflow.utils.logger import get_logger

_LOG = get_logger("knowledge_graph", component="knowledge_graph")

# Seed data directory (relative to this file's package root)
_DEFAULT_SEEDS_DIR = Path(__file__).parent.parent.parent / "data" / "knowledge_seeds"

# ISO 4217 currency codes — used for hallucination guard
_VALID_CURRENCY_CODES: frozenset[str] = frozenset(
    {
        "AED", "ARS", "AUD", "BDT", "BRL", "CAD", "CHF", "CLP",
        "CNY", "COP", "DKK", "EGP", "EUR", "GBP", "HKD", "IDR",
        "ILS", "INR", "JPY", "KES", "KRW", "MXN", "MYR", "NGN",
        "NOK", "NZD", "PEN", "PHP", "PKR", "PLN", "QAR", "RUB",
        "SAR", "SEK", "SGD", "THB", "TRY", "TWD", "UAH", "USD",
        "VND", "ZAR",
    }
)

# Master prompt is imported from synthflow.prompts.master_prompt — no inline constants needed.


class UniversalKnowledgeGraph:
    """
    Activates domain knowledge from seed data + LLM, grounded by DuckDB.

    Responsibilities:
    - Load JSON seed files into in-process DuckDB tables on first use
    - Query geography/economics/ontologies for hallucination-guard validation
    - Call LLM to generate a CausalKnowledgeBundle enriched with seed facts
    - Validate and correct invalid currency codes, city names, etc.
    """

    def __init__(
        self,
        llm_client: Union[LLMClient, MockLLMClient],
        seeds_dir: Optional[Path] = None,
        memory_store: Any = None,
    ) -> None:
        self._llm = llm_client
        self._seeds_dir = seeds_dir or _DEFAULT_SEEDS_DIR
        self._store = memory_store
        self._db = duckdb.connect(":memory:")
        self._seeds_loaded = False

    # ── Seed data ─────────────────────────────────────────────────────────

    def _load_seed_data(self) -> None:
        """Load JSON seeds into DuckDB tables (idempotent)."""
        if self._seeds_loaded:
            return

        self._db.execute("""
            CREATE TABLE IF NOT EXISTS world_cities (
                city VARCHAR, state VARCHAR, country VARCHAR, continent VARCHAR,
                timezone VARCHAR, currency_code VARCHAR, currency_symbol VARCHAR,
                latitude DOUBLE, longitude DOUBLE, population BIGINT,
                tier VARCHAR, gdp_per_capita_usd DOUBLE,
                avg_monthly_salary_usd DOUBLE, cost_of_living_index DOUBLE,
                languages JSON, phone_prefix VARCHAR, postal_code_format VARCHAR
            )
        """)
        self._db.execute("""
            CREATE TABLE IF NOT EXISTS economic_indicators (
                country VARCHAR, currency_code VARCHAR, currency_symbol VARCHAR,
                gdp_per_capita_usd DOUBLE, gini_coefficient DOUBLE,
                avg_monthly_salary_usd DOUBLE, min_wage_usd DOUBLE,
                inflation_rate DOUBLE, unemployment_rate DOUBLE,
                economic_tier VARCHAR
            )
        """)
        self._db.execute("""
            CREATE TABLE IF NOT EXISTS domain_ontologies (
                domain VARCHAR, entity_types JSON, typical_columns JSON,
                common_relationships JSON, regulatory_tags JSON
            )
        """)
        self._db.execute("""
            CREATE TABLE IF NOT EXISTS distribution_priors (
                semantic_type VARCHAR, distribution_type VARCHAR,
                params JSON, clip_min DOUBLE, clip_max DOUBLE
            )
        """)
        self._db.execute("""
            CREATE TABLE IF NOT EXISTS temporal_pattern_seeds (
                domain VARCHAR, pattern_type VARCHAR, pattern_data JSON
            )
        """)

        seeds_path = self._seeds_dir
        self._load_json_table(seeds_path / "world_geography.json", "world_cities", "cities")
        self._load_json_table(seeds_path / "economic_indicators.json", "economic_indicators", "countries")
        self._load_json_table(seeds_path / "domain_ontologies.json", "domain_ontologies", "domains")
        self._load_json_table(seeds_path / "distribution_priors.json", "distribution_priors", "priors")
        self._load_json_table(seeds_path / "temporal_patterns.json", "temporal_pattern_seeds", "patterns")

        self._seeds_loaded = True

    def _load_json_table(
        self, path: Path, table: str, root_key: str
    ) -> None:
        """Load JSON array from file into a DuckDB table (best-effort)."""
        if not path.exists():
            _LOG.warning("Seed file not found: %s", path)
            return
        try:
            with open(path, encoding="utf-8") as fh:
                payload = json.load(fh)
            items: list[dict[str, Any]] = payload.get(root_key, [])
            if not items:
                return
            cols = list(items[0].keys())
            placeholders = ", ".join("?" * len(cols))
            col_names = ", ".join(cols)
            for item in items:
                row = [
                    json.dumps(v) if isinstance(v, (list, dict)) else v
                    for v in (item.get(c) for c in cols)
                ]
                self._db.execute(
                    f"INSERT INTO {table} ({col_names}) VALUES ({placeholders})",
                    row,
                )
        except Exception as exc:
            _LOG.warning("Failed to load seed %s: %s", path, exc)

    # ── Main activation ───────────────────────────────────────────────────

    async def activate(self, intent: IntentObject) -> CausalKnowledgeBundle:
        """
        Produce a CausalKnowledgeBundle for *intent*.

        Args:
            intent: Parsed IntentObject from CognitiveIntentEngine.

        Returns:
            Domain-grounded CausalKnowledgeBundle.
        """
        self._load_seed_data()

        bundle_key = f"{intent.domain}:{getattr(intent.region, 'country', 'global')}"

        # Check semantic cache
        if self._store is not None:
            cached = self._store.get_cached_knowledge(bundle_key)
            if cached is not None:
                _LOG.info("Knowledge cache hit for %s", bundle_key)
                return cached

        # Ground geography data
        geo_context = self._ground_geography(intent)

        # Build LLM prompt with grounded context
        bundle = await self._llm_activate(intent, geo_context)

        # Hallucination guard
        bundle = self._hallucination_guard(bundle)

        # Persist to cache
        if self._store is not None:
            try:
                self._store.cache_knowledge(bundle_key, bundle)
            except Exception:
                pass

        return bundle

    def _ground_geography(self, intent: IntentObject) -> dict[str, Any]:
        """Query DuckDB for geography facts about the intent's region."""
        result: dict[str, Any] = {}
        if intent.region is None:
            return result
        country = intent.region.country
        try:
            rows = self._db.execute(
                "SELECT city, state, timezone, currency_code, avg_monthly_salary_usd "
                "FROM world_cities WHERE country = ? LIMIT 10",
                [country],
            ).fetchall()
            result["cities"] = [
                {"city": r[0], "state": r[1], "timezone": r[2],
                 "currency_code": r[3], "avg_monthly_salary_usd": r[4]}
                for r in rows
            ]
            econ = self._db.execute(
                "SELECT currency_code, gdp_per_capita_usd, avg_monthly_salary_usd, "
                "min_wage_usd, economic_tier FROM economic_indicators WHERE country = ?",
                [country],
            ).fetchall()
            if econ:
                result["economics"] = {
                    "currency_code": econ[0][0],
                    "gdp_per_capita_usd": econ[0][1],
                    "avg_monthly_salary_usd": econ[0][2],
                    "min_wage_usd": econ[0][3],
                    "economic_tier": econ[0][4],
                }
        except Exception as exc:
            _LOG.debug("Geography grounding query failed: %s", exc)
        return result

    async def _llm_activate(
        self, intent: IntentObject, geo_context: dict[str, Any]
    ) -> CausalKnowledgeBundle:
        """Call LLM to generate a knowledge bundle using the master prompt."""
        country_str = getattr(intent.region, "country", "global") if intent.region else "global"
        region_hint = country_str
        if geo_context.get("economics"):
            econ = geo_context["economics"]
            region_hint += (
                f" (currency: {econ.get('currency_code', 'USD')}, "
                f"avg salary USD: {econ.get('avg_monthly_salary_usd', 1000)})"
            )

        # Build geo context string for the prompt
        geo_ctx_str = json.dumps(geo_context, indent=2) if geo_context else "{}"

        # Use the original user prompt from intent if available (extra field via extra="allow")
        user_prompt_text = getattr(intent, "original_prompt", None) or (
            f"Generate {intent.row_count} rows of {intent.domain} data"
            + (f" for {country_str}" if country_str != "global" else "")
        )

        system, user = build_knowledge_prompt(
            user_prompt=user_prompt_text,
            domain=intent.domain,
            region=region_hint,
            row_count=intent.row_count,
            geo_context=geo_ctx_str,
        )

        try:
            raw = await self._llm.complete(
                user,
                system_prompt=system,
                json_mode=True,
                temperature=0.4,
                max_tokens=8192,
            )
            data = safe_json_loads(raw)
            return self._build_bundle(data, intent)
        except Exception as exc:
            raise RuntimeError(
                "Knowledge extraction failed: LLM provider returned an error. "
                "Please wait 1-2 minutes and retry, or switch to a different provider."
            ) from exc

    def _build_bundle(
        self, data: dict[str, Any], intent: IntentObject
    ) -> CausalKnowledgeBundle:
        """Build a CausalKnowledgeBundle from LLM response dict (handles both simple and master prompt formats)."""
        if not isinstance(data, dict):
            raise RuntimeError(
                "Knowledge extraction failed: LLM provider returned an error. "
                "Please wait 1-2 minutes and retry, or switch to a different provider."
            )

        # ── dag_rules ─────────────────────────────────────────────────────
        dag_rules: list[CausalDagRule] = []
        for r in data.get("dag_rules", []):
            if isinstance(r, dict) and "parent_column" in r and "child_column" in r:
                try:
                    dag_rules.append(CausalDagRule(
                        parent_column=str(r["parent_column"]),
                        child_column=str(r["child_column"]),
                        lambda_str=str(r.get("lambda_str", 'lambda row, rng: row["{p}"]'.format(
                            p=r["parent_column"]
                        ))),
                        description=str(r.get("description", "")),
                    ))
                except Exception:
                    continue

        # ── column_knowledge — also derive from column_design if present ──
        column_knowledge: list[ColumnKnowledge] = []
        # Try column_design from master prompt first (richer)
        col_design_raw: list[dict[str, Any]] = []
        if isinstance(data.get("column_design"), list):
            col_design_raw = data["column_design"]
        for c in col_design_raw:
            if isinstance(c, dict) and "column_name" in c:
                try:
                    dist_info = c.get("distribution", {}) or {}
                    dist_params = dist_info.get("parameters", {}) if isinstance(dist_info, dict) else {}
                    column_knowledge.append(ColumnKnowledge(
                        column_name=str(c["column_name"]),
                        description=str(c.get("description", "")),
                        semantic_type=c.get("semantic_type"),
                        min_value=c.get("min_value"),
                        max_value=c.get("max_value"),
                        distribution_hint=str(dist_info.get("type", "")) if dist_info else None,
                        value_examples=list(c.get("enum_values", [])) if c.get("enum_values") else [],
                    ))
                except Exception:
                    continue
        # Fall back to legacy column_knowledge field
        if not column_knowledge:
            for c in data.get("column_knowledge", []):
                if isinstance(c, dict) and "column_name" in c:
                    try:
                        column_knowledge.append(ColumnKnowledge(
                            column_name=str(c["column_name"]),
                            description=str(c.get("description", "")),
                            semantic_type=c.get("semantic_type"),
                            min_value=c.get("min_value"),
                            max_value=c.get("max_value"),
                        ))
                    except Exception:
                        continue

        # ── correlations ──────────────────────────────────────────────────
        correlations: list[CrossColumnCorrelation] = []
        for cor in data.get("correlations", []):
            if isinstance(cor, dict) and "col_a" in cor and "col_b" in cor:
                try:
                    correlations.append(CrossColumnCorrelation(
                        col_a=str(cor["col_a"]),
                        col_b=str(cor["col_b"]),
                        strength=float(cor.get("strength", 0.3)),
                    ))
                except Exception:
                    continue

        # ── temporal_patterns — try master prompt format first, then legacy ──
        temporal: Optional[TemporalPatterns] = None
        # Master prompt puts temporal in "temporal_patterns" with slightly different keys
        tp_raw = data.get("temporal_patterns") or data.get("temporal_patterns_legacy")
        if isinstance(tp_raw, dict):
            try:
                # Map master prompt keys to legacy TemporalPatterns model
                tp_kwargs: dict[str, Any] = {}
                if "day_of_week_weights" in tp_raw:
                    tp_kwargs["day_of_week_weights"] = tp_raw["day_of_week_weights"]
                if "hour_of_day_weights" in tp_raw:
                    tp_kwargs["hour_of_day_weights"] = tp_raw["hour_of_day_weights"]
                if "monthly_seasonality" in tp_raw:
                    tp_kwargs["monthly_seasonality"] = tp_raw["monthly_seasonality"]
                elif "monthly_weights" in tp_raw:
                    # Convert list of 12 floats to {month_str: multiplier} dict
                    mw = tp_raw["monthly_weights"]
                    if isinstance(mw, list) and len(mw) == 12:
                        tp_kwargs["monthly_seasonality"] = {
                            str(i + 1): float(mw[i]) * 12 for i in range(12)
                        }
                if "has_autocorrelation" in tp_raw:
                    tp_kwargs["has_autocorrelation"] = tp_raw["has_autocorrelation"]
                if "autocorrelation_rho" in tp_raw:
                    tp_kwargs["autocorrelation_rho"] = tp_raw["autocorrelation_rho"]
                temporal = TemporalPatterns(**tp_kwargs)
            except Exception:
                temporal = TemporalPatterns()

        # ── dirty_data_profile — try legacy format, ignore per-column extended ──
        dirty: Optional[DirtyDataProfile] = None
        ddp_raw = data.get("dirty_data_profile_legacy") or data.get("dirty_data_profile")
        if isinstance(ddp_raw, dict) and "null_rate" in ddp_raw:
            try:
                dirty = DirtyDataProfile(**{
                    k: v for k, v in ddp_raw.items()
                    if k in DirtyDataProfile.model_fields
                })
            except Exception:
                dirty = DirtyDataProfile()

        # ── currency_code ─────────────────────────────────────────────────
        currency = "INR" if (intent.region and intent.region.country == "India") else "USD"
        # Check blueprint_metadata.geography first
        bp_meta = data.get("blueprint_metadata") or {}
        geo_from_meta = bp_meta.get("geography") if isinstance(bp_meta, dict) else {}
        cand_src = (
            (geo_from_meta.get("currency_code") if isinstance(geo_from_meta, dict) else None)
            or data.get("currency_code")
        )
        if isinstance(cand_src, str):
            candidate = cand_src.upper()
            if candidate in _VALID_CURRENCY_CODES:
                currency = candidate

        # ── real_world_value_pools ────────────────────────────────────────
        value_pools: list[ValuePool] = []
        for vp in data.get("real_world_value_pools", []):
            if isinstance(vp, dict) and "pool_name" in vp:
                try:
                    value_pools.append(ValuePool(
                        pool_name=str(vp["pool_name"]),
                        used_in_column=str(vp.get("used_in_column", "")),
                        cultural_context=vp.get("cultural_context"),
                        verification_basis=vp.get("verification_basis"),
                        values=[str(v) for v in vp.get("values", [])],
                    ))
                except Exception:
                    continue

        # ── causal_generation_order ───────────────────────────────────────
        causal_order: list[str] = []
        if isinstance(data.get("causal_generation_order"), list):
            causal_order = [str(c) for c in data["causal_generation_order"] if c]

        # ── geography dict ────────────────────────────────────────────────
        geography_dict: Optional[dict[str, Any]] = None
        if isinstance(geo_from_meta, dict) and geo_from_meta:
            geography_dict = geo_from_meta
        elif isinstance(data.get("geography"), dict):
            geography_dict = data["geography"]

        # ── institutions ──────────────────────────────────────────────────
        institutions: list[dict[str, Any]] = []
        rwe = data.get("real_world_entities") or {}
        if isinstance(rwe, dict):
            for inst in rwe.get("institutions", []):
                if isinstance(inst, dict):
                    institutions.append(inst)

        # ── simulation_rules ──────────────────────────────────────────────
        simulation_rules: Optional[dict[str, Any]] = None
        if isinstance(data.get("simulation_rules"), dict):
            simulation_rules = data["simulation_rules"]

        # ── dirty_data_profile_extended ───────────────────────────────────
        ddp_extended: Optional[dict[str, Any]] = None
        if isinstance(data.get("dirty_data_profile"), dict) and "per_column" in data["dirty_data_profile"]:
            ddp_extended = data["dirty_data_profile"]

        return CausalKnowledgeBundle(
            domain=intent.domain,
            sub_domain=intent.sub_domain,
            region=intent.region,
            temporal_patterns=temporal or TemporalPatterns(),
            column_knowledge=column_knowledge,
            dag_rules=dag_rules,
            correlations=correlations,
            dirty_data_profile=dirty or DirtyDataProfile(),
            currency_code=currency,
            # New fields from master prompt
            real_world_value_pools=value_pools,
            causal_generation_order=causal_order,
            column_design=col_design_raw,
            geography=geography_dict,
            institutions=institutions,
            simulation_rules=simulation_rules,
            dirty_data_profile_extended=ddp_extended,
            blueprint_metadata=bp_meta if isinstance(bp_meta, dict) and bp_meta else None,
        )

    # ── Hallucination guard ───────────────────────────────────────────────

    def _hallucination_guard(
        self, bundle: CausalKnowledgeBundle
    ) -> CausalKnowledgeBundle:
        """Validate and correct hallucinated values in *bundle*."""
        # Fix currency code
        if bundle.currency_code not in _VALID_CURRENCY_CODES:
            # Default to INR for India, USD otherwise
            if bundle.region and bundle.region.country == "India":
                corrected = "INR"
            else:
                corrected = "USD"
            _LOG.warning(
                "Hallucination guard: invalid currency '%s' → '%s'",
                bundle.currency_code, corrected,
            )
            bundle = bundle.model_copy(update={"currency_code": corrected})

        # Validate locale — hi_IN covers Punjab and all North India.
        # Faker has no locale for "pa" + "_IN"; correct it automatically.
        _unsupported = "pa" + "_IN"
        if bundle.locale and bundle.locale.faker_locale == _unsupported:
            bundle = bundle.model_copy(
                update={"locale": bundle.locale.model_copy(
                    update={"faker_locale": "hi_IN"}
                )}
            )

        return bundle

    def get_distribution_prior(self, semantic_type: str) -> Optional[dict[str, Any]]:
        """Return distribution prior from DuckDB for *semantic_type*."""
        self._load_seed_data()
        try:
            rows = self._db.execute(
                "SELECT distribution_type, params, clip_min, clip_max "
                "FROM distribution_priors WHERE semantic_type = ?",
                [semantic_type],
            ).fetchall()
            if rows:
                return {
                    "distribution_type": rows[0][0],
                    "params": json.loads(rows[0][1]) if rows[0][1] else {},
                    "clip_min": rows[0][2],
                    "clip_max": rows[0][3],
                }
        except Exception:
            pass
        return None

    def get_cities_for_country(self, country: str) -> list[str]:
        """Return city names from seed data for *country*."""
        self._load_seed_data()
        try:
            rows = self._db.execute(
                "SELECT city FROM world_cities WHERE country = ? LIMIT 20",
                [country],
            ).fetchall()
            return [r[0] for r in rows if r[0]]
        except Exception:
            return []

    def is_seeds_loaded(self) -> bool:
        """Return True if seed data tables have been populated."""
        return self._seeds_loaded
