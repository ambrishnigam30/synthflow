# ───────────────────────────────────────────────────────────────
# Copyright (c) 2026 Ambrish Nigam
# Author : Ambrish Nigam | https://github.com/ambrishnigam30
# Project: SynthFlow — Autonomous Synthetic Data Orchestration Platform
# License : Apache License 2.0 | https://www.apache.org/licenses/LICENSE-2.0
# Module : DI container — wires together all engine subsystems
# ───────────────────────────────────────────────────────────────

from __future__ import annotations

from typing import Optional, Union

from synthflow.llm_client import LLMClient, MockLLMClient
from synthflow.engines.intent_engine import CognitiveIntentEngine
from synthflow.engines.knowledge_graph import UniversalKnowledgeGraph
from synthflow.engines.schema_intelligence import SchemaIntelligenceLayer
from synthflow.engines.constraint_engine import ConstraintPhysicsEngine
from synthflow.engines.stats_engine import StatisticalModelingCore
from synthflow.engines.realism_engine import DeterministicRealismEngine
from synthflow.engines.code_synthesizer import GlassBoxCodeSynthesizer
from synthflow.engines.self_healing import SelfHealingRuntime
from synthflow.engines.pattern_library import PatternRhythmLibrary
from synthflow.engines.anomaly_engine import AnomalyOutlierEngine
from synthflow.engines.correlation_engine import CorrelationDriftEngine
from synthflow.engines.validation_engine import ValidationHygieneEngine
from synthflow.engines.scenario_engine import ScenarioEngine
from synthflow.engines.sdv_multiplier import SDVMultiplierEngine
from synthflow.engines.memory_store import MemoryContextStore
from synthflow.privacy.presidio_guard import PresidioPrivacyGuard
from synthflow.quality.reporter import QualityReporter


class SynthFlowContainer:
    """
    Dependency-injection container that lazily initialises all 15 engine
    subsystems, the privacy guard, quality reporter, and memory store.

    All subsystems share a single LLM client and a single DuckDB path so
    that the in-process DuckDB connection (used by the knowledge graph and
    the memory store) is consistent across the session.
    """

    def __init__(
        self,
        llm_client: Union[LLMClient, MockLLMClient],
        duckdb_path: str = ":memory:",
    ) -> None:
        self._llm = llm_client
        self._duckdb_path = duckdb_path

        # Backing fields for lazily created subsystems
        self._intent_engine: Optional[CognitiveIntentEngine] = None
        self._knowledge_graph: Optional[UniversalKnowledgeGraph] = None
        self._schema_intelligence: Optional[SchemaIntelligenceLayer] = None
        self._constraint_engine: Optional[ConstraintPhysicsEngine] = None
        self._stats_engine: Optional[StatisticalModelingCore] = None
        self._realism_engine: Optional[DeterministicRealismEngine] = None
        self._code_synthesizer: Optional[GlassBoxCodeSynthesizer] = None
        self._self_healing: Optional[SelfHealingRuntime] = None
        self._pattern_library: Optional[PatternRhythmLibrary] = None
        self._anomaly_engine: Optional[AnomalyOutlierEngine] = None
        self._correlation_engine: Optional[CorrelationDriftEngine] = None
        self._validation_engine: Optional[ValidationHygieneEngine] = None
        self._scenario_engine: Optional[ScenarioEngine] = None
        self._sdv_multiplier: Optional[SDVMultiplierEngine] = None
        self._memory_store: Optional[MemoryContextStore] = None
        self._presidio_guard: Optional[PresidioPrivacyGuard] = None
        self._quality_reporter: Optional[QualityReporter] = None

    # ── Subsystem properties ───────────────────────────────────────────────

    @property
    def memory_store(self) -> MemoryContextStore:
        """Lazily initialised DuckDB-backed session/cache store."""
        if self._memory_store is None:
            self._memory_store = MemoryContextStore(db_path=self._duckdb_path)
        return self._memory_store

    @property
    def intent_engine(self) -> CognitiveIntentEngine:
        """Lazily initialised intent parser; shares the memory store for caching."""
        if self._intent_engine is None:
            self._intent_engine = CognitiveIntentEngine(
                llm_client=self._llm,
                memory_store=self.memory_store,
            )
        return self._intent_engine

    @property
    def knowledge_graph(self) -> UniversalKnowledgeGraph:
        """Lazily initialised knowledge graph; shares the memory store for caching."""
        if self._knowledge_graph is None:
            self._knowledge_graph = UniversalKnowledgeGraph(
                llm_client=self._llm,
                memory_store=self.memory_store,
            )
        return self._knowledge_graph

    @property
    def schema_intelligence(self) -> SchemaIntelligenceLayer:
        """Lazily initialised schema architect."""
        if self._schema_intelligence is None:
            self._schema_intelligence = SchemaIntelligenceLayer(
                llm_client=self._llm,
            )
        return self._schema_intelligence

    @property
    def constraint_engine(self) -> ConstraintPhysicsEngine:
        """Lazily initialised constraint builder."""
        if self._constraint_engine is None:
            self._constraint_engine = ConstraintPhysicsEngine(
                llm_client=self._llm,
            )
        return self._constraint_engine

    @property
    def stats_engine(self) -> StatisticalModelingCore:
        """Lazily initialised statistical modelling core."""
        if self._stats_engine is None:
            self._stats_engine = StatisticalModelingCore()
        return self._stats_engine

    @property
    def realism_engine(self) -> DeterministicRealismEngine:
        """Lazily initialised deterministic realism engine."""
        if self._realism_engine is None:
            self._realism_engine = DeterministicRealismEngine()
        return self._realism_engine

    @property
    def code_synthesizer(self) -> GlassBoxCodeSynthesizer:
        """Lazily initialised Glass Box code synthesizer."""
        if self._code_synthesizer is None:
            self._code_synthesizer = GlassBoxCodeSynthesizer(
                llm_client=self._llm,
            )
        return self._code_synthesizer

    @property
    def self_healing(self) -> SelfHealingRuntime:
        """Lazily initialised self-healing script executor."""
        if self._self_healing is None:
            self._self_healing = SelfHealingRuntime(
                llm_client=self._llm,
            )
        return self._self_healing

    @property
    def pattern_library(self) -> PatternRhythmLibrary:
        """Lazily initialised temporal pattern library."""
        if self._pattern_library is None:
            self._pattern_library = PatternRhythmLibrary()
        return self._pattern_library

    @property
    def anomaly_engine(self) -> AnomalyOutlierEngine:
        """Lazily initialised dirty-data injection engine."""
        if self._anomaly_engine is None:
            self._anomaly_engine = AnomalyOutlierEngine()
        return self._anomaly_engine

    @property
    def correlation_engine(self) -> CorrelationDriftEngine:
        """Lazily initialised Spearman drift detection engine."""
        if self._correlation_engine is None:
            self._correlation_engine = CorrelationDriftEngine(
                llm_client=self._llm,
            )
        return self._correlation_engine

    @property
    def validation_engine(self) -> ValidationHygieneEngine:
        """Lazily initialised 10-check validation engine."""
        if self._validation_engine is None:
            self._validation_engine = ValidationHygieneEngine()
        return self._validation_engine

    @property
    def scenario_engine(self) -> ScenarioEngine:
        """Lazily initialised scenario shift engine."""
        if self._scenario_engine is None:
            self._scenario_engine = ScenarioEngine(
                llm_client=self._llm,
            )
        return self._scenario_engine

    @property
    def sdv_multiplier(self) -> SDVMultiplierEngine:
        """Lazily initialised SDV row-scaling engine."""
        if self._sdv_multiplier is None:
            self._sdv_multiplier = SDVMultiplierEngine()
        return self._sdv_multiplier

    @property
    def presidio_guard(self) -> PresidioPrivacyGuard:
        """Lazily initialised PII detection and masking guard."""
        if self._presidio_guard is None:
            self._presidio_guard = PresidioPrivacyGuard()
        return self._presidio_guard

    @property
    def quality_reporter(self) -> QualityReporter:
        """Lazily initialised composite quality reporter."""
        if self._quality_reporter is None:
            self._quality_reporter = QualityReporter()
        return self._quality_reporter
