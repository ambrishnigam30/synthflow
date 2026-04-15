# SynthFlow v2.0 — Development Plan

**Timeline**: 24 working days (5 weeks)
**Developer**: Ambrish Nigam
**Tool**: Claude Code in VS Code (primary), Claude.ai chat (architecture/review)

---

## Development Principles

1. **Build → Test → Verify → Proceed**: Never move to the next layer until current layer's tests pass
2. **One component at a time**: Write component, write its tests, run tests, fix, commit
3. **Git discipline**: One commit per component, descriptive messages, feature branches
4. **No dead code**: Every function must be called. Every import must be used.
5. **Type everything**: TypeScript strict mode (frontend), Python type hints on all signatures (backend)

---

## Layer 0: Project Scaffold [Day 1]

### Tasks

```
0.1  Create GitHub repo: github.com/ambrishnigam/synthflow
0.2  Initialize monorepo structure:
     synthflow/
     ├── frontend/     (Next.js)
     ├── backend/      (FastAPI)
     ├── engine/       (SynthFlow core)
     └── docs/

0.3  Frontend initialization:
     - npx create-next-app@latest frontend --typescript --tailwind --app --src-dir
     - npm install @radix-ui/react-* class-variance-authority clsx tailwind-merge
     - npm install lucide-react zustand @tanstack/react-query socket.io-client
     - npm install @supabase/supabase-js
     - Configure tailwind.config.ts with SynthFlow design tokens
     - Set up path aliases (@/components, @/lib, etc.)
     - Install fonts: Satoshi, Instrument Serif, JetBrains Mono, Plus Jakarta Sans
     - Create global.css with design system variables

0.4  Backend initialization:
     - Create Python 3.11 virtual environment
     - pip install fastapi uvicorn sqlalchemy alembic pydantic pydantic-settings
     - pip install python-jose[cryptography] passlib[bcrypt] python-multipart
     - pip install celery redis aiofiles httpx python-socketio
     - pip install supabase
     - Create app/ directory structure
     - Create config.py with pydantic-settings
     - Create main.py with CORS, routers, error handlers
     - Create database.py with SQLAlchemy async engine
     - Set up Alembic for migrations

0.5  Engine initialization:
     - Create engine/synthflow/ package structure
     - Copy models/schemas.py skeleton
     - pip install pandas numpy scipy pydantic faker tenacity duckdb

0.6  Docker Compose for local dev:
     - PostgreSQL 16
     - Redis 7
     - (Frontend and backend run directly for hot reload)

0.7  Create .env.example files for frontend and backend
0.8  Create CLAUDE.md (Claude Code instruction file)
```

### Verification Checklist
- [ ] `cd frontend && npm run dev` → Next.js starts on :3000
- [ ] `cd backend && uvicorn app.main:app --reload` → FastAPI starts on :8000
- [ ] `cd backend && alembic upgrade head` → tables created
- [ ] Docker Compose `docker compose up` → PostgreSQL + Redis running
- [ ] Frontend shows placeholder page with correct fonts and colors
- [ ] FastAPI /docs shows Swagger UI

---

## Layer 1: Engine Foundation [Day 2-3]

### Tasks

```
1.1  engine/synthflow/models/schemas.py
     - All 33 Pydantic v2 models
     - Full type hints, Field descriptions, validators
     - extra="allow" on: IntentObject, CausalKnowledgeBundle, SchemaDefinition
     - Key validators:
       * IntentObject: row_count <= 10M, domain normalized to lowercase
       * CausalKnowledgeBundle: temporal weights normalized to sum=1.0
       * SchemaDefinition: every table must have a primary key

1.2  engine/synthflow/utils/logger.py
     - Structured logging via Python logging + JSON formatter
     - Log levels: DEBUG, INFO, WARNING, ERROR
     - Context fields: session_id, phase, component

1.3  engine/synthflow/utils/helpers.py
     - detect_timestamp_columns(df): match "_at" suffix ONLY, not "at" substring
     - format_number_indian(n): 1,23,456.78 lakh system
     - format_number_western(n): 1,234,567.89
     - safe_json_loads(text): 6-strategy fallback chain
     - compute_prompt_hash(prompt): MD5 hex digest
     - generate_session_id(): str(uuid4())

1.4  engine/synthflow/llm_client.py
     - LLMClient class: unified Gemini/OpenAI/Groq interface
     - MockLLMClient class: deterministic responses for testing
     - MODEL_REGISTRY: provider → models, defaults, context windows
     - async complete(prompt, system, json_mode, temperature, max_tokens)
     - parse_json_response(text): 6-strategy fallback
     - tenacity retry: 3 attempts, exponential backoff (2s, 4s, 8s)

1.5  engine/synthflow/causal_dag.py
     - CausalDAG class
     - build_from_rules(rules: List[CausalDagRule]): build adjacency list
     - topological_sort(): Kahn's algorithm, raises CycleError on cycles
     - _compile_lambda(lambda_str): AST whitelist validation
       * FORBIDDEN: Import, ImportFrom, Global, Nonlocal AST nodes
       * FORBIDDEN calls: exec, eval, compile, open, __import__, etc.
       * FORBIDDEN attributes: anything starting with __
     - resolve_row(partial_row, rng, columns): walk DAG, apply rules
```

### Test Cases — Layer 1

```python
# tests/test_models.py

def test_all_33_models_can_be_instantiated():
    """Every model can be created with minimal valid data."""

def test_intent_object_rejects_row_count_above_10m():
    """IntentObject raises ValueError if row_count > 10_000_000."""

def test_intent_object_normalizes_domain_to_lowercase():
    """domain='Healthcare' becomes 'healthcare'."""

def test_knowledge_bundle_normalizes_temporal_weights():
    """day_of_week_weights [2,2,2,2,2,2,2] normalized to [0.142,...,0.142]."""

def test_schema_definition_rejects_table_without_primary_key():
    """SchemaDefinition raises if any table has no PK column."""

def test_intent_object_allows_extra_fields():
    """IntentObject(extra_field="x") does not raise."""

def test_region_info_serialization_roundtrip():
    """RegionInfo → dict → RegionInfo produces identical object."""


# tests/test_helpers.py

def test_detect_timestamp_columns_matches_created_at():
    """Column 'created_at' is detected as timestamp."""

def test_detect_timestamp_columns_ignores_marks_math():
    """Column 'marks_math' is NOT detected as timestamp."""

def test_detect_timestamp_columns_matches_updated_at():
    """Column 'updated_at' is detected as timestamp."""

def test_detect_timestamp_columns_matches_date_types():
    """Columns with datetime64 dtype are detected regardless of name."""

def test_format_number_indian():
    """1234567.89 → '12,34,567.89'"""

def test_format_number_western():
    """1234567.89 → '1,234,567.89'"""

def test_safe_json_loads_parses_clean_json():
    """Direct JSON string parses correctly."""

def test_safe_json_loads_strips_markdown_fences():
    """```json\n{...}\n``` parses correctly."""

def test_safe_json_loads_extracts_json_from_text():
    """'Here is the result: {...} Hope this helps' extracts the JSON."""

def test_safe_json_loads_raises_on_garbage():
    """Completely invalid text raises LLMParseError."""

def test_compute_prompt_hash_deterministic():
    """Same prompt always produces same hash."""

def test_generate_session_id_is_string():
    """Returns string, not UUID object."""


# tests/test_llm_client.py

def test_mock_llm_returns_deterministic_response():
    """MockLLMClient returns consistent JSON for same prompt."""

def test_llm_client_retries_3_times_on_failure(mock_provider):
    """LLMClient retries exactly 3 times before raising."""

def test_llm_client_parses_json_from_markdown_fences(mock_provider):
    """Response wrapped in ```json ... ``` is parsed correctly."""

def test_llm_client_parses_json_with_preamble(mock_provider):
    """'Here is the output: {...}' is parsed correctly."""

def test_parse_json_response_strategy_chain():
    """All 6 strategies are tried in order."""


# tests/test_causal_dag.py

def test_dag_builds_correct_adjacency_from_rules():
    """3 rules create correct adjacency list."""

def test_dag_topological_sort_parents_before_children():
    """In sorted order, every parent appears before its children."""

def test_dag_raises_on_cycle_detection():
    """A→B→C→A raises CycleError."""

def test_dag_resolves_age_from_birthdate():
    """Lambda computing age from birth_date produces correct integer."""

def test_dag_safe_lambda_allows_math_operations():
    """'lambda row, rng: row[\"age\"] * 12' compiles and executes."""

def test_dag_rejects_lambda_with_import_statement():
    """'lambda row, rng: __import__(\"os\").system(\"rm -rf /\")' raises UnsafeLambdaError."""

def test_dag_rejects_lambda_with_exec_call():
    """'lambda row, rng: exec(\"print(1)\")' raises UnsafeLambdaError."""

def test_dag_rejects_lambda_with_dunder_attribute():
    """'lambda row, rng: row.__class__.__bases__' raises UnsafeLambdaError."""

def test_dag_resolve_row_skips_columns_with_missing_parents():
    """Column depending on unresolved parent is skipped gracefully."""

def test_dag_empty_rules_returns_empty_dag():
    """No rules → empty adjacency, topological sort returns empty list."""
```

### Verification Checklist
- [ ] `cd engine && pytest tests/ -v` → all tests pass
- [ ] Coverage > 90% for models, helpers, llm_client, causal_dag
- [ ] `python -c "from synthflow.models import IntentObject"` works

---

## Layer 2: Engine Subsystems [Day 4-7]

### Tasks

```
2.1  engines/intent_engine.py
     - CognitiveIntentEngine class
     - async parse(prompt) → IntentObject
     - DuckDB cache check (7-day TTL)
     - LLM call with Stage 1 prompt template
     - Fallback: keyword domain classifier (DOMAIN_KEYWORD_MAP)
     - Fallback: regex row count extractor
     - Deterministic seed: hash(prompt) % 2^31

2.2  engines/knowledge_graph.py
     - UniversalKnowledgeGraph class
     - _load_seed_data(): JSON seeds → DuckDB tables
     - async activate(intent) → CausalKnowledgeBundle
     - ChromaDB semantic cache (similarity > 0.92 = cache hit)
     - DuckDB grounding query (geography, economics, ontologies)
     - Hallucination guard: validate cities against DuckDB
     - Combined Knowledge + Schema output (single LLM call)

2.3  engines/schema_intelligence.py
     - SchemaIntelligenceLayer class
     - async architect(intent, knowledge) → SchemaDefinition
     - Minimum 12 columns enforced in prompt
     - Column ordering: IDs → demographics → measures
     - PK/FK relationship inference
     - Fallback: used when combined call in 2.2 fails

2.4  engines/constraint_engine.py
     - ConstraintPhysicsEngine class
     - async build_constraint_set(schema, knowledge) → ConstraintSet
     - Schema-derived constraints (min/max/enum/unique)
     - DAG-derived constraints
     - LLM-augmented domain physics rules
     - enforce_batch(df, constraints) → (fixed_df, violation_log)

2.5  engines/stats_engine.py
     - StatisticalModelingCore class
     - MixtureDistribution inner class
     - model(schema, knowledge) → DistributionMap
     - validate_distribution(semantic_type, distribution): override LLM if wrong
     - Semantic validator rules (salary=lognormal, age=truncated_normal, etc.)

2.6  engines/realism_engine.py
     - DeterministicRealismEngine class
     - get_cell_rng(global_seed, column_name, row_index): SHA-256 cell seeding
     - sample_column(column, row_index, global_seed, context, dist_map)
     - Locale-specific formatting (Indian lakh, Western million)

2.7  engines/code_synthesizer.py
     - GlassBoxCodeSynthesizer class
     - async synthesize(schema, knowledge, distributions, constraints, row_count, seed) → str
     - LLM Call 2: generates standalone Python generate(row_count, seed)
     - Generated code embeds ALL knowledge as Python constants
     - Zero Faker for domain entities, zero API calls
     - Base64 script embedding (never triple-quote)

2.8  engines/self_healing.py
     - SelfHealingRuntime class
     - async execute(script, context, session_id, max_attempts=3) → DataFrame
     - Base64 encode → subprocess → decode → temp .py → execute
     - On failure: parse traceback → LLM fix → retry
     - Timeout: 120s, memory limit: 2GB
     - _generate_subprocess_wrapper(script, row_count, output_path)

2.9  engines/pattern_library.py
     - PatternRhythmLibrary class
     - apply_temporal_patterns(df, patterns, timestamp_columns)
     - apply_autocorrelation(df, rho, ts_columns, sort_by)
     - Day-of-week, hour-of-day, monthly seasonality, special events

2.10 engines/anomaly_engine.py
     - AnomalyOutlierEngine class
     - TypoGenerator with KEYBOARD_ADJACENCY_MAP (26 letters + keys)
     - inject_structured_nulls(df, null_patterns, null_conditions)
     - inject_date_format_mix(series, rng, inconsistency_rate)
     - inject_outliers(df, outlier_columns, rate)
     - inject_duplicates(df, duplicate_rate, near_duplicate_rate)

2.11 engines/correlation_engine.py
     - CorrelationDriftEngine class
     - compute_actual_correlations(df) → Spearman matrix
     - detect_drift(actual, target, threshold=0.15) → List[DriftEvent]
     - async correct_drift_loop(df, drift_events, knowledge, script, max_iter=3)

2.12 engines/validation_engine.py
     - ValidationHygieneEngine class
     - audit(df, schema, constraints, knowledge) → ValidationReport
     - 10 checks: schema completeness, type conformance, null policy, range,
       enum, uniqueness, temporal, causal physics, statistical sanity, correlation

2.13 engines/scenario_engine.py
     - ScenarioEngine class
     - Known templates: recession, pandemic, boom, demonetization, drought
     - async parse_scenario(scenario_text, intent) → ScenarioParams
     - apply(df, params) → DataFrame with shifted distributions

2.14 engines/memory_store.py
     - MemoryContextStore class
     - DuckDB tables: generation_sessions, intent_cache, knowledge_cache,
       heal_events, constraint_violations, correlation_drift_events
     - ChromaDB collection: synthflow_schemas
     - save_session(), list_sessions(), get_session(), delete_session()

2.15 engines/sdv_multiplier.py
     - SDVMultiplierEngine class
     - OPTIONAL: only used when user explicitly enables
     - build_sdv_metadata(schema) → SingleTableMetadata
     - scale(seed_df, target_rows, schema) → DataFrame
     - Post-SDV constraint re-enforcement

2.16 privacy/presidio_guard.py
     - PresidioPrivacyGuard class
     - Custom recognizers: IN_PAN, IN_AADHAR, IN_GST, US_SSN, UK_NHS,
       IBAN, CREDIT_CARD, IN_PHONE, BR_CPF
     - scan_and_mask(df, schema) → (masked_df, PrivacyReport)
     - compute_k_anonymity(df, quasi_ids) → int (k=1 scores 40/40)
     - compute_l_diversity(df, quasi_ids, sensitive_col) → float

2.17 quality/reporter.py
     - QualityReporter class
     - generate(seed_df, full_df, schema, privacy_report, validation_report) → QualityReport
     - Composite score: 30% sdmetrics + 25% causal + 20% privacy + 15% temporal + 10% dirty

2.18 core.py
     - SynthFlowContainer: DI container
     - Protocol classes for each subsystem interface
     - Wire all subsystems together
     - Configuration from environment variables

2.19 orchestrator.py
     - SynthFlowOrchestrator class
     - async generate(prompt, row_count, ...) → GenerationResult
     - 9-phase pipeline with progress_callback
     - Error handling at each phase

2.20 5 JSON seed files
     - data/knowledge_seeds/world_geography.json (200+ cities)
     - data/knowledge_seeds/economic_indicators.json (60+ countries)
     - data/knowledge_seeds/domain_ontologies.json (15+ domains)
     - data/knowledge_seeds/distribution_priors.json (40+ semantic types)
     - data/knowledge_seeds/temporal_patterns.json (15+ domains)
```

### Test Cases — Layer 2

```python
# tests/test_engines.py

def test_intent_engine_extracts_domain_from_prompt():
    """'Generate Indian healthcare records' → domain='healthcare'"""

def test_intent_engine_extracts_row_count():
    """'Generate 5000 records' → row_count=5000"""

def test_intent_engine_falls_back_on_llm_failure():
    """When LLM returns garbage, keyword classifier produces valid IntentObject."""

def test_intent_engine_computes_deterministic_seed():
    """Same prompt always produces same seed."""

def test_knowledge_graph_validates_currency_codes():
    """Invalid currency 'XYZ' is corrected to nearest valid."""

def test_schema_minimum_12_columns_enforced():
    """Schema with fewer than 12 columns triggers re-generation."""

def test_constraint_engine_rejects_discharge_before_admission():
    """discharge_date < admission_date flagged as violation."""

def test_constraint_engine_rejects_negative_salary():
    """salary < 0 flagged and clipped to 0."""

def test_realism_engine_same_seed_same_value():
    """get_cell_rng(42, 'age', 0) always produces same random stream."""

def test_realism_engine_different_rows_different_values():
    """Row 0 and row 1 of same column produce different values."""

def test_stats_engine_salary_must_be_lognormal():
    """semantic_type='salary' with distribution='normal' gets corrected to 'lognormal'."""

def test_stats_engine_age_must_be_truncated_normal():
    """semantic_type='age' with distribution='pareto' gets corrected."""

def test_correlation_engine_detects_drift():
    """Pair with actual_rho=0.1 and target_rho=0.6 triggers DriftEvent."""

def test_pattern_library_applies_seasonal_multiplier():
    """Revenue values in Diwali months are multiplied correctly."""

def test_anomaly_engine_null_rate_within_tolerance():
    """Injected null rate is within 5% of target."""

def test_anomaly_engine_typo_adjacent_key():
    """'gmail' can become 'gmqil' (q is adjacent to a)."""

def test_anomaly_engine_typo_transposition():
    """'Bangalore' can become 'Banglore'."""

def test_anomaly_engine_date_format_inconsistency():
    """12% of dates get alternative format strings."""

def test_scenario_engine_recession_reduces_spending():
    """Recession scenario multiplies consumer_spending by 0.72."""

def test_self_healing_uses_base64_not_triple_quotes():
    """Generated wrapper script uses base64 import, not triple-quote embedding."""

def test_validation_engine_10_checks():
    """audit() returns ValidationReport with exactly 10 checks."""

def test_no_hardcoded_domain_lists_in_source_files():
    """AST scan of all engine .py files finds no inline list > 20 items of strings."""

def test_code_synthesizer_generates_valid_python():
    """Generated code compiles without syntax errors."""

def test_privacy_k1_scores_40_of_40():
    """When every row is unique (k=1), privacy score component is 40/40."""

def test_presidio_detects_indian_pan_card():
    """'ABCDE1234F' detected as IN_PAN entity."""

def test_reproducibility_same_seed_same_output():
    """Two runs with same seed produce identical DataFrames."""


# tests/test_orchestrator.py

def test_full_pipeline_healthcare_50_rows(mock_llm):
    """End-to-end: healthcare prompt → 50-row DataFrame with quality score."""

def test_full_pipeline_returns_generation_result(mock_llm):
    """Result has all fields: dataframe, schema, validation, quality, code."""

def test_progress_callback_fires_9_times(mock_llm):
    """progress_callback called at least 9 times with increasing progress."""

def test_invalid_api_key_raises_error():
    """Missing or invalid API key raises LLMConfigError."""
```

### Verification Checklist
- [ ] `cd engine && pytest tests/ -v --cov` → all tests pass, coverage > 85%
- [ ] Full pipeline runs with MockLLMClient
- [ ] Generated Python code executes independently and produces DataFrame

---

## Layer 3: Backend — Auth + Database [Day 8-9]

### Tasks

```
3.1  backend/app/models/ — SQLAlchemy models
     - user.py, conversation.py, generation.py, dataset.py, billing.py, webhook.py
     - All models from database schema in architecture doc

3.2  Alembic migration: create all tables

3.3  backend/app/core/security.py
     - JWT creation/validation (python-jose)
     - Password hashing (passlib bcrypt)
     - AES-256 encryption for API keys
     - API key generation (sf_live_ prefix)

3.4  backend/app/core/database.py
     - SQLAlchemy async engine with Supabase PostgreSQL
     - Session dependency for FastAPI

3.5  backend/app/core/storage.py
     - Supabase Storage client
     - upload_file, download_file, delete_file, get_signed_url

3.6  backend/app/api/auth.py
     - POST /api/auth/signup
     - POST /api/auth/login
     - POST /api/auth/google (Supabase OAuth)
     - POST /api/auth/refresh
     - GET /api/auth/me
     - PATCH /api/auth/me

3.7  backend/app/api/llm_config.py
     - POST /api/llm-config (save encrypted key)
     - GET /api/llm-config (list providers, keys masked)
     - DELETE /api/llm-config/:provider
     - POST /api/llm-config/:provider/test

3.8  backend/app/dependencies.py
     - get_current_user: JWT → User
     - get_db: async session
     - check_plan_limit: enforce plan restrictions
```

### Test Cases — Layer 3

```python
def test_signup_creates_user():
def test_signup_rejects_duplicate_email():
def test_login_returns_jwt():
def test_login_rejects_wrong_password():
def test_jwt_validation_succeeds():
def test_jwt_expired_returns_401():
def test_refresh_token_returns_new_access():
def test_google_oauth_creates_user():
def test_llm_config_encrypts_api_key():
def test_llm_config_test_connection():
def test_api_key_never_returned_in_full():
```

---

## Layer 4: Backend — Core API [Day 10-12]

### Tasks

```
4.1  backend/app/api/conversations.py — CRUD
4.2  backend/app/services/chat_service.py — intent classify + route
4.3  backend/app/api/chat.py — WebSocket + REST
4.4  backend/app/services/generation_service.py — wraps SynthFlow engine
4.5  backend/app/api/generate.py — trigger + status + download
4.6  backend/app/services/dataset_service.py — upload + parse + query
4.7  backend/app/api/datasets.py — upload + list + query
4.8  backend/app/services/usage_service.py — track + enforce limits
```

### Test Cases — Layer 4

```python
def test_create_conversation():
def test_send_message_gets_response():
def test_generation_request_triggers_pipeline():
def test_generation_progress_via_websocket():
def test_generation_result_has_download_url():
def test_dataset_upload_parses_schema():
def test_dataset_query_returns_result():
def test_usage_tracking_increments():
def test_plan_limit_enforced():
```

---

## Layer 5: Backend — Business Features [Day 13-14]

### Tasks

```
5.1  backend/app/services/billing_service.py
5.2  backend/app/api/billing.py — plans, subscribe, cancel, usage, webhooks
5.3  backend/app/api/teams.py — CRUD + invite + roles
5.4  backend/app/api/api_keys.py — create, list, revoke
5.5  backend/app/api/webhooks.py — CRUD + deliveries + test
5.6  backend/app/services/webhook_service.py — HMAC signing, delivery, retry
5.7  backend/app/api/public_api.py — /api/v1/* (API key auth)
```

---

## Layer 6: Frontend — Foundation [Day 15-16]

### Tasks

```
6.1  Design system implementation:
     - tailwind.config.ts with SynthFlow tokens
     - globals.css with font imports + CSS variables
     - Shadcn component customization (buttons, cards, inputs)
     - Animation utilities (scale hover, fade-in, phase pulse)

6.2  Landing page (/) — all sections per wireframe spec

6.3  Auth pages:
     - /login — email + password + Google OAuth button
     - /signup — registration form + Google OAuth

6.4  App shell:
     - Sidebar component (260px, collapsible on mobile)
     - Header component
     - Layout with auth check (redirect to /login if not authenticated)
```

---

## Layer 7: Frontend — Core Product [Day 17-20]

### Tasks

```
7.1  Dashboard (/app/dashboard)
     - Usage stats cards
     - Usage chart (recharts bar chart)
     - Recent generations list
     - Quick-start input

7.2  Generate page (/app/generate) — THE MAIN PRODUCT
     - Conversation sidebar (left panel)
     - Chat message thread (center)
     - ChatMessage component (user, assistant, system variants)
     - ChatInput component (text input + submit + advanced options)
     - GenerationCard component (9-phase progress)
     - ResultCard component (preview table + download + code + report)
     - WebSocket integration for real-time streaming
     - Empty state with 6 example prompt cards
     - Provider/model selector in sidebar

7.3  Explore page (/app/explore)
     - Upload dropzone (drag & drop)
     - Dataset list sidebar
     - Dataset tabs: Chat, Preview, Schema, Stats, Profile
     - QueryResult component (table + chart inline)
     - ChartRenderer component (recharts)

7.4  History page (/app/history)
     - DataTable with search, filter, sort
     - Detail panel (slide-out)
     - Compare mode
     - Regenerate button
```

---

## Layer 8: Frontend — Settings + Billing [Day 21-22]

### Tasks

```
8.1  Settings — Profile tab
8.2  Settings — LLM Providers tab (BYOK key management)
8.3  Settings — Team management tab
8.4  Settings — Billing tab (current plan, upgrade, usage meters)
8.5  Settings — API Keys tab
8.6  Settings — Webhooks tab
8.7  Pricing page (/pricing) — 4 plan cards + feature matrix
8.8  Razorpay / Stripe checkout integration
```

---

## Layer 9: Integration + Polish [Day 23-24]

### Tasks

```
9.1  End-to-end testing: signup → configure LLM → generate → download
9.2  Error handling: every API call has loading, error, empty states
9.3  Mobile responsiveness (425px, 640px, 768px, 896px breakpoints)
9.4  Performance: lazy loading, code splitting, image optimization
9.5  SEO: meta tags, OG images, sitemap
9.6  Deployment:
     - Vercel: frontend (connect GitHub repo)
     - Render: backend (Dockerfile)
     - Supabase: database + auth + storage
     - Upstash: Redis
9.7  README.md: installation, architecture, API docs, contributing
9.8  Final smoke test on production URLs
```

---

## Git Branch Strategy

```
main                    ← production-ready, deployed
├── develop             ← integration branch
│   ├── feat/engine-foundation     (Layer 1)
│   ├── feat/engine-subsystems     (Layer 2)
│   ├── feat/backend-auth          (Layer 3)
│   ├── feat/backend-api           (Layer 4)
│   ├── feat/backend-business      (Layer 5)
│   ├── feat/frontend-foundation   (Layer 6)
│   ├── feat/frontend-core         (Layer 7)
│   ├── feat/frontend-settings     (Layer 8)
│   └── feat/integration           (Layer 9)
```

Each feature branch merges into develop after tests pass.
develop merges into main for deployment.

---

## Quality Gates

Each layer must pass these before proceeding:

| Gate | Criteria |
|------|----------|
| Tests pass | 100% of layer's tests green |
| Coverage | > 85% for engine, > 70% for backend |
| No lint errors | ruff (Python), eslint (TypeScript) |
| Type check | mypy --strict (Python), tsc --noEmit (TypeScript) |
| No hardcoded secrets | No API keys, passwords, or tokens in code |
| Builds cleanly | Docker image builds without errors |

---
