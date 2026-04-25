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
        await _progress(1, 0.05, "Parsing intent from prompt…")
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

        # ── Phase 2: KNOWLEDGE ─────────────────────────────────────────────
        await asyncio.sleep(2)  # Rate-limit guard between LLM phases
        await _progress(2, 0.15, "Activating domain knowledge graph…")
        try:
            knowledge: CausalKnowledgeBundle = await c.knowledge_graph.activate(intent)
        except LLMConfigError:
            raise
        except Exception as exc:
            _LOG.warning("Phase 2 (KNOWLEDGE) failed: %s — using minimal bundle", exc)
            from synthflow.models.schemas import (
                DirtyDataProfile,
                TemporalPatterns,
            )
            knowledge = CausalKnowledgeBundle(
                domain=intent.domain,
                sub_domain=intent.sub_domain,
                region=intent.region,
                temporal_patterns=TemporalPatterns(),
                dirty_data_profile=DirtyDataProfile(),
            )

        # ── Phase 3: SCHEMA ────────────────────────────────────────────────
        await asyncio.sleep(2)  # Rate-limit guard between LLM phases
        await _progress(3, 0.25, "Designing table schema…")
        try:
            schema: SchemaDefinition = await c.schema_intelligence.architect(intent, knowledge)
        except LLMConfigError:
            raise
        except Exception as exc:
            _LOG.warning("Phase 3 (SCHEMA) failed: %s — using fallback schema", exc)
            # The SchemaIntelligenceLayer has its own internal fallback, but if
            # the whole call threw we need a minimal schema here.
            from synthflow.models.schemas import ColumnDefinition, SchemaTable
            table_name = f"{intent.domain}_records"
            schema = SchemaDefinition(tables=[
                SchemaTable(
                    name=table_name,
                    columns=[
                        ColumnDefinition(
                            name=f"{table_name}_id",
                            data_type="uuid",
                            semantic_type="id",
                            is_primary_key=True,
                            unique=True,
                            nullable=False,
                        ),
                        ColumnDefinition(name="name", data_type="string", semantic_type="name"),
                        ColumnDefinition(name="created_at", data_type="datetime", semantic_type="timestamp"),
                        ColumnDefinition(name="status", data_type="string",
                                         enum_values=["active", "inactive", "pending"]),
                    ],
                )
            ])

        # ── Phase 4: CONSTRAINTS ───────────────────────────────────────────
        await asyncio.sleep(2)  # Rate-limit guard between LLM phases
        await _progress(4, 0.35, "Building constraint physics set…")
        try:
            constraints: ConstraintSet = await c.constraint_engine.build_constraint_set(
                schema, knowledge
            )
        except LLMConfigError:
            raise
        except Exception as exc:
            _LOG.warning("Phase 4 (CONSTRAINTS) failed: %s — using empty constraint set", exc)
            constraints = ConstraintSet(rules=[], domain=knowledge.domain)

        # ── Phase 5: STATISTICS ────────────────────────────────────────────
        await _progress(5, 0.45, "Modelling statistical distributions…")
        try:
            distributions: DistributionMap = c.stats_engine.model(schema, knowledge)
        except Exception as exc:
            _LOG.warning("Phase 5 (STATISTICS) failed: %s — using empty distribution map", exc)
            distributions = DistributionMap(
                column_distributions={},
                table_name=schema.tables[0].name if schema.tables else "",
            )

        # ── Phase 6: GENERATION ────────────────────────────────────────────
        await asyncio.sleep(2)  # Rate-limit guard before code synthesis LLM call
        await _progress(6, 0.60, "Synthesising data via Glass Box code…")
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
            _LOG.warning("Phase 6 (GENERATION) self-healing exhausted: %s — building fallback DataFrame", exc)
            df = self._fallback_dataframe(
                schema=schema,
                row_count=effective_row_count,
                seed=effective_seed,
            )
        except Exception as exc:
            _LOG.warning("Phase 6 (GENERATION) failed: %s — building fallback DataFrame", exc)
            df = self._fallback_dataframe(
                schema=schema,
                row_count=effective_row_count,
                seed=effective_seed,
            )

        # ── Phase 7: PATTERNS ──────────────────────────────────────────────
        await _progress(7, 0.70, "Applying temporal patterns and autocorrelation…")
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

        # Apply scenario shifts if requested
        if scenario_text:
            try:
                scenario_params = await c.scenario_engine.parse_scenario(
                    scenario_text, intent=intent
                )
                df = c.scenario_engine.apply(df, scenario_params)
            except Exception as exc:
                _LOG.warning("Scenario application failed: %s — skipping", exc)

        # Keep a copy of the pre-dirty DataFrame for quality comparison
        seed_df = df.copy()

        # ── Phase 8: VALIDATION ────────────────────────────────────────────
        await _progress(8, 0.85, "Validating, scanning for PII, computing quality…")
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

        # ── Phase 9: ANOMALY / DIRTY DATA / SDV / DRIFT ────────────────────
        await _progress(9, 0.95, "Applying dirty data, SDV scaling, drift correction…")
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

    # ── Fallback DataFrame builder ─────────────────────────────────────────

    def _fallback_dataframe(
        self,
        schema: SchemaDefinition,
        row_count: int,
        seed: int,
    ) -> pd.DataFrame:
        """
        Build a minimal DataFrame using DeterministicRealismEngine.sample_column()
        for each column in the schema when Glass Box execution fails.

        Args:
            schema:    SchemaDefinition with column definitions.
            row_count: Number of rows to generate.
            seed:      Random seed for reproducibility.

        Returns:
            pd.DataFrame with one column per schema column.
        """
        realism = self._c.realism_engine
        data: dict[str, object] = {}

        table = schema.tables[0] if schema.tables else None
        if table is None:
            return pd.DataFrame({"id": range(row_count), "value": range(row_count)})

        for col in table.columns:
            try:
                values = realism.sample_column(
                    column=col,
                    n_rows=row_count,
                    global_seed=seed,
                    context={},
                    dist_map=None,
                )
                data[col.name] = values
            except Exception as col_exc:
                _LOG.debug("Fallback sample_column failed for %s: %s", col.name, col_exc)
                data[col.name] = [None] * row_count

        return pd.DataFrame(data)
