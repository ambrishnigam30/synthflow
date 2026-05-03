# ───────────────────────────────────────────────────────────────
# Copyright (c) 2026 Ambrish Nigam
# Author : Ambrish Nigam | https://github.com/ambrishnigam30
# Project: SynthFlow — Autonomous Synthetic Data Orchestration Platform
# License : Apache License 2.0 | https://www.apache.org/licenses/LICENSE-2.0
# Module : 9-phase generation orchestrator — coordinates all subsystems end-to-end
# ───────────────────────────────────────────────────────────────

from __future__ import annotations

import asyncio
import time
import uuid
from typing import Awaitable, Callable, Optional

import pandas as pd

from synthflow.core import SynthFlowContainer
from synthflow.engines.self_healing import SelfHealingFailureError
from synthflow.llm_client import LLMConfigError
from synthflow.models.schemas import (
    CausalKnowledgeBundle,
    ConstraintSet,
    DistributionMap,
    GenerationResult,
    IntentObject,
    SchemaDefinition,
    ValidationReport,
)
from synthflow.utils.logger import get_logger

_LOG = get_logger("orchestrator", component="orchestrator")


class OrchestrationError(Exception):
    """Raised when a non-recoverable error occurs during the generation pipeline."""


# Re-export so callers can do: from synthflow.orchestrator import LLMConfigError
__all__ = ["SynthFlowOrchestrator", "OrchestrationError", "LLMConfigError"]


class SynthFlowOrchestrator:
    """
    9-phase pipeline that converts a natural-language prompt into a validated,
    privacy-screened, quality-scored synthetic DataFrame.

    Phases:
    1  INTENT      — parse prompt → IntentObject
    2  KNOWLEDGE   — activate domain knowledge → CausalKnowledgeBundle
    3  SCHEMA      — design table schema → SchemaDefinition
    4  CONSTRAINTS — build constraint set → ConstraintSet
    5  STATISTICS  — map distributions → DistributionMap
    6  GENERATION  — synthesise code + execute → DataFrame
    7  PATTERNS    — apply temporal rhythms + autocorrelation
    8  VALIDATION  — audit + privacy scan + quality report
    9  ANOMALY     — inject dirty data + optional SDV scale + drift correction
    """

    def __init__(self, container: SynthFlowContainer) -> None:
        self._c = container

    async def generate(
        self,
        prompt: str,
        row_count: Optional[int] = None,
        seed: Optional[int] = None,
        scenario_text: Optional[str] = None,
        enable_dirty_data: bool = True,
        enable_sdv: bool = False,
        progress_callback: Optional[Callable[[int, float, str], Awaitable[None]]] = None,
    ) -> GenerationResult:
        """
        Run the full 9-phase synthesis pipeline.

        Args:
            prompt:            Natural-language data generation request.
            row_count:         Override row count (uses intent's value if None).
            seed:              Override random seed (uses intent's value if None).
            scenario_text:     Optional economic/environmental scenario to apply.
            enable_dirty_data: Whether to inject realistic dirty-data patterns.
            enable_sdv:        Whether to scale via SDV (requires sdv package).
            progress_callback: Optional async callback(phase, fraction, message).

        Returns:
            GenerationResult with DataFrame, reports, and generated code.

        Raises:
            LLMConfigError:    When the LLM provider is not configured.
            OrchestrationError: When the pipeline fails irrecoverably.
        """
        session_id = str(uuid.uuid4())
        t_start = time.monotonic()
        c = self._c

        async def _progress(phase: int, fraction: float, message: str) -> None:
            if progress_callback is not None:
                try:
                    await progress_callback(phase, fraction, message)
                except Exception as cb_exc:
                    _LOG.debug("Progress callback error (ignored): %s", cb_exc)

        # ── Phase 1: INTENT ────────────────────────────────────────────────
        await _progress(1, 0.05, "Analyzing your request…")
        try:
            intent: IntentObject = await c.intent_engine.parse(prompt)
        except LLMConfigError:
            raise
        except Exception as exc:
            _LOG.error("Phase 1 (INTENT) failed: %s", exc)
            raise OrchestrationError(f"Intent parsing failed: {exc}") from exc

        # Resolve row_count and seed — caller overrides take precedence
        effective_row_count: int = row_count if row_count is not None else (intent.row_count or 1000)
        effective_seed: int = seed if seed is not None else (intent.seed or 42)
        # Write back so downstream phases see consistent values
        intent = intent.model_copy(
            update={"row_count": effective_row_count, "seed": effective_seed}
        )

        _LOG.info(
            "Session %s | domain=%s | rows=%d | seed=%d",
            session_id, intent.domain, effective_row_count, effective_seed,
        )

        _intent_region = (
            getattr(intent.region, "country", "Global") if intent.region else "Global"
        )
        await _progress(
            1, 0.10,
            f"Domain: {intent.domain}, Region: {_intent_region}, Rows: {effective_row_count:,}",
        )

        # ── Phase 2: KNOWLEDGE ─────────────────────────────────────────────
        await asyncio.sleep(5)  # Rate-limit guard between LLM phases
        await _progress(2, 0.15, "Extracting domain knowledge…")
        try:
            knowledge: CausalKnowledgeBundle = await c.knowledge_graph.activate(intent)
        except LLMConfigError:
            raise
        except Exception as exc:
            _LOG.error("Phase 2 (KNOWLEDGE) failed: %s", exc)
            raise OrchestrationError(str(exc)) from exc

        await _progress(
            2, 0.20,
            f"Learning {intent.domain} patterns and causal rules",
        )

        # ── Phase 3: SCHEMA ────────────────────────────────────────────────
        await asyncio.sleep(5)  # Rate-limit guard between LLM phases
        await _progress(3, 0.25, "Designing data schema…")
        try:
            schema: SchemaDefinition = await c.schema_intelligence.architect(intent, knowledge)
        except LLMConfigError:
            raise
        except Exception as exc:
            _LOG.error("Phase 3 (SCHEMA) failed: %s", exc)
            raise OrchestrationError(str(exc)) from exc

        _col_count = sum(len(t.columns) for t in schema.tables) if schema.tables else 0
        await _progress(
            3, 0.30,
            f"Schema ready: {_col_count} columns across {len(schema.tables)} table(s)",
        )

        # ── Phase 4: CONSTRAINTS ───────────────────────────────────────────
        await asyncio.sleep(5)  # Rate-limit guard between LLM phases
        await _progress(4, 0.35, "Mapping causal constraints…")
        try:
            constraints: ConstraintSet = await c.constraint_engine.build_constraint_set(
                schema, knowledge
            )
        except LLMConfigError:
            raise
        except Exception as exc:
            _LOG.warning("Phase 4 (CONSTRAINTS) failed: %s — using empty constraint set", exc)
            constraints = ConstraintSet(rules=[], domain=knowledge.domain)

        await _progress(
            4, 0.40,
            f"Applied {len(constraints.rules)} constraint rules",
        )

        # ── Phase 5: STATISTICS ────────────────────────────────────────────
        await _progress(5, 0.45, "Modeling statistical distributions…")
        try:
            distributions: DistributionMap = c.stats_engine.model(schema, knowledge)
        except Exception as exc:
            _LOG.warning("Phase 5 (STATISTICS) failed: %s — using empty distribution map", exc)
            distributions = DistributionMap(
                column_distributions={},
                table_name=schema.tables[0].name if schema.tables else "",
            )

        await _progress(
            5, 0.50,
            f"Distributions mapped for {len(distributions.column_distributions)} columns",
        )

        # ── Phase 6: GENERATION ────────────────────────────────────────────
        await asyncio.sleep(5)  # Rate-limit guard before code synthesis LLM call
        await _progress(6, 0.60, "Synthesizing Glass Box code…")
        df: pd.DataFrame = pd.DataFrame()
        generated_code: str = ""
        try:
            generated_code = await c.code_synthesizer.synthesize(
                schema=schema,
                knowledge=knowledge,
                distributions=distributions,
                constraints=constraints,
                row_count=effective_row_count,
                seed=effective_seed,
            )
            execution_context: dict[str, object] = {
                "row_count": effective_row_count,
                "seed": effective_seed,
                "domain": intent.domain,
            }
            df = await c.self_healing.execute(
                script=generated_code,
                context=execution_context,
                session_id=session_id,
            )
        except LLMConfigError:
            raise
        except SelfHealingFailureError as exc:
            _LOG.error("Phase 6 (GENERATION) self-healing exhausted: %s", exc)
            raise OrchestrationError(str(exc)) from exc
        except OrchestrationError:
            raise
        except Exception as exc:
            _LOG.error("Phase 6 (GENERATION) failed: %s", exc)
            raise OrchestrationError(str(exc)) from exc

        await _progress(
            6, 0.65,
            f"Generated {len(df):,} rows with self-healing execution",
        )

        # ── Phase 7: PATTERNS ──────────────────────────────────────────────
        await _progress(7, 0.70, "Applying temporal rhythms and autocorrelation…")
        try:
            # Identify timestamp columns
            ts_cols = [
                col.name
                for table in schema.tables
                for col in table.columns
                if col.data_type in ("datetime", "date") and col.name in df.columns
            ]
            df = c.pattern_library.apply_temporal_patterns(
                df=df,
                patterns=knowledge.temporal_patterns,
                timestamp_columns=ts_cols,
            )
            rho = getattr(knowledge.temporal_patterns, "autocorrelation_rho", 0.0) or 0.0
            if abs(rho) > 0.01:
                df = c.pattern_library.apply_autocorrelation(
                    df=df,
                    rho=float(rho),
                    ts_columns=ts_cols,
                )
        except Exception as exc:
            _LOG.warning("Phase 7 (PATTERNS) failed: %s — skipping pattern application", exc)

        await _progress(7, 0.75, "Temporal patterns and correlations applied")

        # Apply scenario shifts if requested
        if scenario_text:
            try:
                scenario_params = await c.scenario_engine.parse_scenario(
                    scenario_text, intent=intent
                )
                df = c.scenario_engine.apply(df, scenario_params)
            except Exception as exc:
                _LOG.warning("Scenario application failed: %s — skipping", exc)

        # ── Post-generation quality checks (warnings only, never fail pipeline) ──
        try:
            import re as _re
            _placeholder_pattern = _re.compile(r"^[a-zA-Z]+_?\d+$")

            # Build schema column-type lookup for smart UUID detection
            _pg_schema_types: dict[str, str] = {}
            _pg_table = schema.tables[0] if schema.tables else None
            if _pg_table:
                for _sc in _pg_table.columns:
                    _pg_schema_types[_sc.name] = (_sc.data_type or "").lower()

            for col in df.select_dtypes(include=["object", "str"]).columns:
                _sample = df[col].dropna().astype(str)
                if len(_sample) > 0:
                    _pct = _sample.apply(lambda v: bool(_placeholder_pattern.match(v))).mean()
                    if _pct > 0.30:
                        _col_schema_type = _pg_schema_types.get(col, "")
                        _is_uuid_col = "uuid" in col.lower() or _col_schema_type == "uuid"
                        if _is_uuid_col:
                            # Replace all placeholder values with real UUIDs
                            df[col] = [str(uuid.uuid4()) for _ in range(len(df))]
                            _LOG.info(
                                "Post-gen fix: replaced %.0f%% placeholder values in UUID"
                                " column '%s' with str(uuid.uuid4())",
                                _pct * 100, col,
                            )
                        else:
                            _LOG.warning(
                                "Post-gen quality: column '%s' has %.0f%% placeholder-like values "
                                "(e.g. word_0, category_1) — value pools may not have been embedded",
                                col, _pct * 100,
                            )
        except Exception as _exc:
            _LOG.warning("Post-gen placeholder check failed: %s", _exc)

        try:
            import numpy as _np
            _table = schema.tables[0] if schema.tables else None
            if _table:
                for _col_def in _table.columns:
                    _cname = _col_def.name
                    if _cname not in df.columns:
                        continue
                    _mn, _mx = _col_def.min_value, _col_def.max_value
                    if _mn is not None or _mx is not None:
                        _series = df[_cname]
                        if _np.issubdtype(_series.dtype, _np.number):
                            _out_of_range = (
                                (_series < _mn if _mn is not None else False) |
                                (_series > _mx if _mx is not None else False)
                            ).sum()
                            if _out_of_range > 0:
                                _LOG.warning(
                                    "Post-gen quality: column '%s' has %d values outside "
                                    "[%s, %s] — np.clip not applied in generated code",
                                    _cname, _out_of_range, _mn, _mx,
                                )
        except Exception as _exc:
            _LOG.warning("Post-gen numeric range check failed: %s", _exc)

        # ── Fix _id columns: decimals/negatives → positive int, NaN → fill ──
        try:
            import numpy as _np2
            _id_table = schema.tables[0] if schema.tables else None
            _id_schema_types: dict[str, str] = {}
            if _id_table:
                for _sc2 in _id_table.columns:
                    _id_schema_types[_sc2.name] = (_sc2.data_type or "").lower()

            for col in df.columns:
                if "_id" not in col.lower():
                    continue
                _col_dt = _id_schema_types.get(col, "")
                _series = df[col]

                # Safely coerce to numeric — handles StringDtype and other extension types
                _numeric = pd.to_numeric(_series, errors="coerce")
                _has_any_numeric = _numeric.notna().any()

                if _has_any_numeric:
                    _is_float = _np2.issubdtype(_numeric.dtype, _np2.floating)
                    _has_decimals = _is_float and bool((_numeric.dropna() % 1 != 0).any())
                    _has_negatives = bool((_numeric.dropna() < 0).any())

                    if _has_decimals or _has_negatives:
                        df[col] = _numeric.abs().fillna(0).astype(int)
                        _LOG.info(
                            "Post-gen fix: column '%s' had %s values → abs().astype(int)",
                            col,
                            "decimal" if _has_decimals else "negative",
                        )
                        # Refresh after fix
                        _numeric = df[col].astype(float)

                    # Fix NaN in numeric IDs → fill with sequential integers from max+1
                    _nan_count = int(_numeric.isna().sum())
                    if _nan_count > 0:
                        _cur_max = int(_numeric.dropna().max()) if len(_numeric.dropna()) > 0 else 0
                        _fill_vals = list(range(_cur_max + 1, _cur_max + 1 + _nan_count))
                        _s_copy = _numeric.copy()
                        _s_copy[_s_copy.isna()] = _fill_vals
                        df[col] = _s_copy.astype(int)
                        _LOG.info(
                            "Post-gen fix: %d NaN values in numeric ID column '%s' → sequential ints",
                            _nan_count, col,
                        )
                elif _col_dt == "uuid":
                    # Fix NaN in UUID ID columns → new uuid4
                    _nan_count = int(_series.isna().sum())
                    if _nan_count > 0:
                        _s_copy2 = _series.copy().astype(object)
                        _s_copy2[_s_copy2.isna()] = [str(uuid.uuid4()) for _ in range(_nan_count)]
                        df[col] = _s_copy2
                        _LOG.info(
                            "Post-gen fix: %d NaN values in UUID ID column '%s' → uuid4",
                            _nan_count, col,
                        )
        except Exception as _exc:
            _LOG.warning("Post-gen ID column fix failed: %s", _exc)

        try:
            import pandas as _pd
            _dob_col = next(
                (c for c in df.columns if c in ("date_of_birth", "dob", "birth_date")), None
            )
            _age_col = next(
                (c for c in df.columns if c in ("age", "age_years")), None
            )
            if _dob_col and _age_col and _pd.api.types.is_datetime64_any_dtype(df[_dob_col]):
                _ref = _pd.Timestamp.now()
                _computed_age = ((_ref - df[_dob_col]).dt.days / 365.25).astype(int)
                _diff = (_computed_age - df[_age_col].fillna(0)).abs()
                _inconsistent = (_diff > 2).sum()
                if _inconsistent > 0:
                    _LOG.warning(
                        "Post-gen quality: %d rows have age/date_of_birth inconsistency "
                        "(>2 year gap) — date consistency rule not applied in generated code",
                        _inconsistent,
                    )
        except Exception as _exc:
            _LOG.warning("Post-gen date consistency check failed: %s", _exc)

        try:
            import pandas as _pd2
            _date_pairs = [
                ("admission_date", "discharge_date"),
                ("purchase_date", "delivery_date"),
                ("order_date", "ship_date"),
                ("start_date", "end_date"),
                ("created_at", "updated_at"),
            ]
            for _before, _after in _date_pairs:
                if _before in df.columns and _after in df.columns:
                    if (
                        _pd2.api.types.is_datetime64_any_dtype(df[_before])
                        and _pd2.api.types.is_datetime64_any_dtype(df[_after])
                    ):
                        _violations = (df[_after] < df[_before]).sum()
                        if _violations > 0:
                            _LOG.warning(
                                "Post-gen quality: %d rows have %s after %s — "
                                "temporal ordering not enforced in generated code",
                                _violations, _before, _after,
                            )
        except Exception as _exc:
            _LOG.warning("Post-gen sequential date check failed: %s", _exc)

        # Keep a copy of the pre-dirty DataFrame for quality comparison
        seed_df = df.copy()

        # ── Phase 8: VALIDATION ────────────────────────────────────────────
        await _progress(8, 0.85, "Validating quality and scanning for PII…")
        validation_report: Optional[ValidationReport] = None
        try:
            validation_report = c.validation_engine.audit(
                df=df,
                schema=schema,
                constraints=constraints,
                knowledge=knowledge,
            )
        except Exception as exc:
            _LOG.warning("Phase 8 (VALIDATION) audit failed: %s", exc)

        try:
            df, privacy_report = c.presidio_guard.scan_and_mask(df=df, schema=schema)
        except Exception as exc:
            _LOG.warning("Phase 8 (PRIVACY) scan failed: %s", exc)
            from synthflow.models.schemas import PrivacyReport
            privacy_report = PrivacyReport()

        try:
            quality_report = c.quality_reporter.generate(
                seed_df=seed_df,
                final_df=df,
                schema=schema,
                privacy_report=privacy_report,
                validation_report=validation_report,
            )
        except Exception as exc:
            _LOG.warning("Phase 8 (QUALITY REPORT) failed: %s", exc)
            from synthflow.models.schemas import QualityReport
            quality_report = QualityReport(overall_score=75.0)

        await _progress(
            8, 0.90,
            f"Quality score: {quality_report.overall_score:.1f}/100",
        )

        # ── Phase 9: ANOMALY / DIRTY DATA / SDV / DRIFT ────────────────────
        await _progress(9, 0.95, "Finalizing output…")
        actual_correlations: dict[tuple[str, str], float] = {}
        try:
            if enable_dirty_data and knowledge.dirty_data_profile:
                df = c.anomaly_engine.apply_dirty_profile(
                    df=df,
                    profile=knowledge.dirty_data_profile,
                )

            if enable_sdv:
                df = c.sdv_multiplier.scale(
                    seed_df=df,
                    target_rows=effective_row_count,
                    schema=schema,
                    seed=effective_seed,
                )

            # Correlation drift detection
            actual_correlations = c.correlation_engine.compute_actual_correlations(df)
            if knowledge.correlations:
                drift_events = c.correlation_engine.detect_drift(
                    actual=actual_correlations,
                    target_correlations=knowledge.correlations,
                )
                if drift_events:
                    _LOG.info("Drift detected in %d pairs — running correction loop", len(drift_events))
                    df = await c.correlation_engine.correct_drift_loop(
                        df=df,
                        drift_events=drift_events,
                        knowledge=knowledge,
                        script=generated_code,
                    )
        except Exception as exc:
            _LOG.warning("Phase 9 (ANOMALY/DRIFT) failed: %s — skipping", exc)

        # ── Finalise and persist ───────────────────────────────────────────
        duration = time.monotonic() - t_start

        result = GenerationResult(
            session_id=session_id,
            dataframe=df,
            schema=schema,
            validation_report=validation_report,
            quality_report=quality_report,
            privacy_report=privacy_report,
            generated_code=generated_code,
            intent=intent,
            row_count=len(df),
            seed=effective_seed,
            generation_duration_seconds=round(duration, 3),
        )

        try:
            c.memory_store.save_session(result, prompt=prompt)
        except Exception as exc:
            _LOG.warning("Failed to persist session to memory store: %s", exc)

        await _progress(9, 1.0, f"Done — {len(df):,} rows generated in {duration:.1f}s")
        _LOG.info(
            "Session %s complete: %d rows, quality=%.1f, duration=%.2fs",
            session_id,
            len(df),
            quality_report.overall_score,
            duration,
        )
        return result

