# ───────────────────────────────────────────────────────────────
# Copyright (c) 2026 Ambrish Nigam
# Author : Ambrish Nigam | https://github.com/ambrishnigam30
# Project: SynthFlow — Autonomous Synthetic Data Orchestration Platform
# License : Apache License 2.0 | https://www.apache.org/licenses/LICENSE-2.0
# Module : MemoryContextStore — DuckDB-backed session + cache store
# ───────────────────────────────────────────────────────────────

from __future__ import annotations

import json
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

import duckdb

from synthflow.models.schemas import (
    CausalKnowledgeBundle,
    DriftEvent,
    GenerationResult,
    HealEvent,
    IntentObject,
)

# Optional ChromaDB integration
try:
    import chromadb  # type: ignore[import-untyped]

    _CHROMA_AVAILABLE = True
except ImportError:
    _CHROMA_AVAILABLE = False


_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS generation_sessions (
    session_id     VARCHAR PRIMARY KEY,
    created_at     TIMESTAMP DEFAULT NOW(),
    domain         VARCHAR  DEFAULT '',
    sub_domain     VARCHAR  DEFAULT '',
    row_count      INTEGER  DEFAULT 0,
    quality_score  DOUBLE   DEFAULT 0.0,
    status         VARCHAR  DEFAULT 'completed',
    prompt         VARCHAR  DEFAULT '',
    seed           INTEGER,
    metadata       JSON     DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS intent_cache (
    prompt_hash    VARCHAR PRIMARY KEY,
    intent_json    JSON     NOT NULL,
    cached_at      TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS knowledge_cache (
    bundle_key     VARCHAR PRIMARY KEY,
    bundle_json    JSON     NOT NULL,
    cached_at      TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS heal_events (
    id             VARCHAR  PRIMARY KEY,
    session_id     VARCHAR  NOT NULL,
    attempt        INTEGER  NOT NULL,
    error_message  VARCHAR  DEFAULT '',
    fix_description VARCHAR DEFAULT '',
    timestamp      TIMESTAMP DEFAULT NOW(),
    success        BOOLEAN  DEFAULT FALSE
);

CREATE TABLE IF NOT EXISTS constraint_violations (
    id             VARCHAR  PRIMARY KEY,
    session_id     VARCHAR  NOT NULL,
    rule_name      VARCHAR  DEFAULT '',
    column_name    VARCHAR  DEFAULT '',
    violation_count INTEGER DEFAULT 0,
    recorded_at    TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS correlation_drift_events (
    id             VARCHAR  PRIMARY KEY,
    session_id     VARCHAR  DEFAULT '',
    col_a          VARCHAR  NOT NULL,
    col_b          VARCHAR  NOT NULL,
    actual_rho     DOUBLE   NOT NULL,
    target_rho     DOUBLE   NOT NULL,
    delta          DOUBLE   NOT NULL,
    iteration      INTEGER  DEFAULT 0,
    recorded_at    TIMESTAMP DEFAULT NOW()
);
"""


class MemoryContextStore:
    """
    DuckDB-backed context store for generation sessions, intent cache,
    knowledge cache, heal events, constraint violations, and drift events.

    ChromaDB semantic cache is used when available (optional dependency).
    """

    def __init__(self, db_path: str = ":memory:") -> None:
        self._conn = duckdb.connect(db_path)
        self._init_tables()
        self._chroma_collection: Any = None
        if _CHROMA_AVAILABLE:
            try:
                client = chromadb.Client()
                self._chroma_collection = client.get_or_create_collection(
                    "synthflow_schemas"
                )
            except Exception:
                self._chroma_collection = None

    def _init_tables(self) -> None:
        for stmt in _SCHEMA_SQL.strip().split(";"):
            stmt = stmt.strip()
            if stmt:
                self._conn.execute(stmt)

    # ── Intent cache ───────────────────────────────────────────────────────

    def cache_intent(self, prompt_hash: str, intent: IntentObject) -> None:
        """Upsert an IntentObject into the cache keyed on prompt_hash."""
        data = json.dumps(intent.model_dump())
        self._conn.execute(
            """
            INSERT INTO intent_cache (prompt_hash, intent_json, cached_at)
            VALUES (?, ?, NOW())
            ON CONFLICT (prompt_hash)
            DO UPDATE SET intent_json = excluded.intent_json,
                          cached_at   = excluded.cached_at
            """,
            [prompt_hash, data],
        )

    def get_cached_intent(
        self, prompt_hash: str, ttl_days: int = 7
    ) -> Optional[IntentObject]:
        """Return cached IntentObject if within TTL, else None."""
        cutoff = datetime.now(timezone.utc) - timedelta(days=ttl_days)
        rows = self._conn.execute(
            "SELECT intent_json FROM intent_cache WHERE prompt_hash = ? AND cached_at > ?",
            [prompt_hash, cutoff],
        ).fetchall()
        if not rows:
            return None
        try:
            return IntentObject.model_validate(json.loads(rows[0][0]))
        except Exception:
            return None

    # ── Knowledge cache ────────────────────────────────────────────────────

    def cache_knowledge(
        self, bundle_key: str, bundle: CausalKnowledgeBundle
    ) -> None:
        data = json.dumps(bundle.model_dump())
        self._conn.execute(
            """
            INSERT INTO knowledge_cache (bundle_key, bundle_json, cached_at)
            VALUES (?, ?, NOW())
            ON CONFLICT (bundle_key)
            DO UPDATE SET bundle_json = excluded.bundle_json,
                          cached_at   = excluded.cached_at
            """,
            [bundle_key, data],
        )

    def get_cached_knowledge(
        self, bundle_key: str, ttl_days: int = 7
    ) -> Optional[CausalKnowledgeBundle]:
        cutoff = datetime.now(timezone.utc) - timedelta(days=ttl_days)
        rows = self._conn.execute(
            "SELECT bundle_json FROM knowledge_cache WHERE bundle_key = ? AND cached_at > ?",
            [bundle_key, cutoff],
        ).fetchall()
        if not rows:
            return None
        try:
            return CausalKnowledgeBundle.model_validate(json.loads(rows[0][0]))
        except Exception:
            return None

    # ── Session management ─────────────────────────────────────────────────

    def save_session(
        self,
        result: GenerationResult,
        prompt: str = "",
    ) -> None:
        """Persist a completed GenerationResult to the sessions table."""
        domain = ""
        sub_domain = ""
        seed_val = result.seed
        if result.intent:
            domain = result.intent.domain
            sub_domain = result.intent.sub_domain or ""
        quality = 0.0
        if result.quality_report:
            quality = result.quality_report.overall_score
        metadata: dict[str, Any] = {}

        self._conn.execute(
            """
            INSERT INTO generation_sessions
                (session_id, domain, sub_domain, row_count, quality_score,
                 status, prompt, seed, metadata)
            VALUES (?, ?, ?, ?, ?, 'completed', ?, ?, ?)
            ON CONFLICT (session_id) DO NOTHING
            """,
            [
                result.session_id,
                domain,
                sub_domain,
                result.row_count,
                quality,
                prompt,
                seed_val,
                json.dumps(metadata),
            ],
        )

    def list_sessions(self, limit: int = 50) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            """
            SELECT session_id, created_at, domain, sub_domain,
                   row_count, quality_score, status, prompt, seed
            FROM generation_sessions
            ORDER BY created_at DESC
            LIMIT ?
            """,
            [limit],
        ).fetchall()
        keys = [
            "session_id", "created_at", "domain", "sub_domain",
            "row_count", "quality_score", "status", "prompt", "seed",
        ]
        return [dict(zip(keys, row)) for row in rows]

    def get_session(self, session_id: str) -> Optional[dict[str, Any]]:
        rows = self._conn.execute(
            """
            SELECT session_id, created_at, domain, sub_domain,
                   row_count, quality_score, status, prompt, seed, metadata
            FROM generation_sessions WHERE session_id = ?
            """,
            [session_id],
        ).fetchall()
        if not rows:
            return None
        keys = [
            "session_id", "created_at", "domain", "sub_domain",
            "row_count", "quality_score", "status", "prompt", "seed", "metadata",
        ]
        return dict(zip(keys, rows[0]))

    def delete_session(self, session_id: str) -> None:
        self._conn.execute(
            "DELETE FROM generation_sessions WHERE session_id = ?",
            [session_id],
        )

    # ── Event logging ──────────────────────────────────────────────────────

    def log_heal_event(self, event: HealEvent) -> None:
        self._conn.execute(
            """
            INSERT INTO heal_events
                (id, session_id, attempt, error_message, fix_description,
                 timestamp, success)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            [
                str(uuid.uuid4()),
                event.session_id,
                event.attempt,
                event.error_message,
                event.fix_description,
                event.timestamp,
                event.success,
            ],
        )

    def log_drift_event(self, event: DriftEvent) -> None:
        self._conn.execute(
            """
            INSERT INTO correlation_drift_events
                (id, session_id, col_a, col_b, actual_rho, target_rho,
                 delta, iteration)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                str(uuid.uuid4()),
                event.session_id,
                event.col_a,
                event.col_b,
                event.actual_rho,
                event.target_rho,
                event.delta,
                event.iteration,
            ],
        )

    def log_violation(
        self,
        session_id: str,
        rule_name: str,
        column_name: str,
        count: int,
    ) -> None:
        self._conn.execute(
            """
            INSERT INTO constraint_violations
                (id, session_id, rule_name, column_name, violation_count)
            VALUES (?, ?, ?, ?, ?)
            """,
            [str(uuid.uuid4()), session_id, rule_name, column_name, count],
        )

    # ── ChromaDB semantic cache ────────────────────────────────────────────

    def semantic_cache_add(
        self,
        doc_id: str,
        text: str,
        metadata: dict[str, Any],
    ) -> None:
        if self._chroma_collection is None:
            return
        try:
            self._chroma_collection.add(
                documents=[text],
                metadatas=[metadata],
                ids=[doc_id],
            )
        except Exception:
            pass

    def semantic_cache_query(
        self,
        text: str,
        n_results: int = 1,
        similarity_threshold: float = 0.92,
    ) -> Optional[dict[str, Any]]:
        if self._chroma_collection is None:
            return None
        try:
            results = self._chroma_collection.query(
                query_texts=[text],
                n_results=n_results,
            )
            distances = results.get("distances", [[]])[0]
            if not distances:
                return None
            # ChromaDB distance: 0 = identical, smaller = more similar
            # Convert distance to similarity: sim = 1 - distance
            similarity = 1.0 - float(distances[0])
            if similarity >= similarity_threshold:
                metas = results.get("metadatas", [[]])[0]
                return metas[0] if metas else None
        except Exception:
            pass
        return None

    def close(self) -> None:
        """Close the DuckDB connection."""
        self._conn.close()
