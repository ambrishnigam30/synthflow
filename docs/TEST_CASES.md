# SynthFlow v2.0 — Complete Test Case Specification

## Test Infrastructure

### Fixtures (conftest.py)

```
mock_llm_client          — MockLLMClient returning deterministic JSON, zero API calls
sample_intent            — IntentObject for Indian healthcare, 100 rows
sample_schema            — SchemaDefinition with 15 columns, patients table
sample_knowledge_bundle  — CausalKnowledgeBundle with DAG rules, distributions, temporal
sample_dataframe         — 50-row DataFrame matching sample_schema
in_memory_duckdb         — DuckDB :memory: with seed data loaded
mock_supabase_client     — Mock for Supabase auth + storage
mock_redis               — Mock for Upstash Redis
authenticated_client     — FastAPI TestClient with valid JWT
```

---

## Engine Tests

### E-001: Models (tests/test_models.py)

| ID | Test | Priority | Input | Expected |
|----|------|----------|-------|----------|
| E-001-01 | All 33 models instantiate | P0 | Minimal valid data for each | No exceptions |
| E-001-02 | IntentObject rejects >10M rows | P0 | row_count=20_000_000 | ValueError |
| E-001-03 | IntentObject normalizes domain | P0 | domain="Healthcare" | domain="healthcare" |
| E-001-04 | IntentObject allows extra fields | P0 | extra_field="test" | No exception |
| E-001-05 | Knowledge bundle normalizes weights | P0 | weights=[2,2,2,2,2,2,2] | Each ≈ 0.142 |
| E-001-06 | Schema rejects table without PK | P0 | Table with no is_primary_key=True | ValueError |
| E-001-07 | RegionInfo round-trip serialization | P1 | RegionInfo → dict → RegionInfo | Identical |
| E-001-08 | ColumnDefinition defaults | P1 | Minimal ColumnDefinition | nullable=True, null_rate=0.0 |
| E-001-09 | GenerationResult validates types | P1 | Invalid dataframe type | ValidationError |
| E-001-10 | Enum values are correct strings | P2 | NullMechanism.MCAR | "MCAR" |

### E-002: Helpers (tests/test_helpers.py)

| ID | Test | Priority | Input | Expected |
|----|------|----------|-------|----------|
| E-002-01 | Timestamp detects _at suffix | P0 | columns=["created_at","age"] | ["created_at"] |
| E-002-02 | Timestamp ignores marks_math | P0 | columns=["marks_math","updated_at"] | ["updated_at"] |
| E-002-03 | Timestamp detects datetime dtype | P0 | DataFrame with datetime64 col | Detected |
| E-002-04 | Indian number format | P0 | 1234567.89 | "12,34,567.89" |
| E-002-05 | Western number format | P0 | 1234567.89 | "1,234,567.89" |
| E-002-06 | JSON parse clean | P0 | '{"a":1}' | {"a": 1} |
| E-002-07 | JSON parse markdown fences | P0 | '```json\n{"a":1}\n```' | {"a": 1} |
| E-002-08 | JSON parse with preamble | P0 | 'Result: {"a":1} done' | {"a": 1} |
| E-002-09 | JSON parse raises on garbage | P0 | 'hello world' | LLMParseError |
| E-002-10 | Prompt hash deterministic | P1 | Same prompt twice | Same hash |
| E-002-11 | Session ID is string | P1 | generate_session_id() | isinstance str |

### E-003: LLM Client (tests/test_llm_client.py)

| ID | Test | Priority | Input | Expected |
|----|------|----------|-------|----------|
| E-003-01 | Mock returns deterministic JSON | P0 | Same prompt | Same response |
| E-003-02 | Retry 3 times on failure | P0 | Provider always fails | Retried 3x, then raises |
| E-003-03 | Parse JSON from markdown | P0 | ```json {...} ``` | Parsed dict |
| E-003-04 | Parse JSON with preamble | P0 | "Here is: {...}" | Parsed dict |
| E-003-05 | Temperature parameter passed | P1 | temperature=0.5 | Forwarded to provider |
| E-003-06 | Max tokens parameter passed | P1 | max_tokens=4096 | Forwarded |
| E-003-07 | Provider registry has all 3 | P1 | REGISTRY keys | gemini, openai, groq |
| E-003-08 | Invalid provider raises | P1 | provider="invalid" | LLMConfigError |

### E-004: Causal DAG (tests/test_causal_dag.py)

| ID | Test | Priority | Input | Expected |
|----|------|----------|-------|----------|
| E-004-01 | Correct adjacency from 3 rules | P0 | A→B, B→C, A→C | Correct adj list |
| E-004-02 | Topological sort: parents first | P0 | A→B→C | Order: A, B, C |
| E-004-03 | Cycle detection raises | P0 | A→B→C→A | CycleError |
| E-004-04 | Age from birthdate | P0 | Lambda computing age | Correct integer |
| E-004-05 | Safe lambda: math allowed | P0 | row["age"] * 12 | Compiles + runs |
| E-004-06 | Unsafe: import blocked | P0 | __import__("os") | UnsafeLambdaError |
| E-004-07 | Unsafe: exec blocked | P0 | exec("print(1)") | UnsafeLambdaError |
| E-004-08 | Unsafe: dunder blocked | P0 | row.__class__ | UnsafeLambdaError |
| E-004-09 | Missing parents skipped | P0 | Parent not in row | Column skipped |
| E-004-10 | Empty rules → empty DAG | P1 | No rules | Empty adj, empty sort |
| E-004-11 | Multiple roots handled | P1 | A→C, B→C | Sort includes A,B before C |
| E-004-12 | Lambda with numpy allowed | P1 | np.sqrt(row["x"]) | Compiles + runs |

### E-005: Intent Engine (tests/test_intent_engine.py)

| ID | Test | Priority | Input | Expected |
|----|------|----------|-------|----------|
| E-005-01 | Extracts domain | P0 | "Indian healthcare records" | domain="healthcare" |
| E-005-02 | Extracts row count | P0 | "Generate 5000 records" | row_count=5000 |
| E-005-03 | Extracts region | P0 | "Indian healthcare" | country="India" |
| E-005-04 | Falls back on LLM failure | P0 | LLM returns garbage | Valid IntentObject via keywords |
| E-005-05 | Deterministic seed | P0 | Same prompt | Same seed |
| E-005-06 | Cache hit on same prompt | P1 | Same prompt twice | Second call uses cache |
| E-005-07 | Default row count 1000 | P1 | "Generate healthcare data" | row_count=1000 |
| E-005-08 | Multi-word domain | P1 | "real estate listings" | domain="real_estate" |

### E-006: Knowledge Graph (tests/test_knowledge_graph.py)

| ID | Test | Priority | Input | Expected |
|----|------|----------|-------|----------|
| E-006-01 | Loads seed data | P0 | Init with seeds | DuckDB tables populated |
| E-006-02 | Validates currency codes | P0 | Currency "XYZ" in bundle | Corrected to valid |
| E-006-03 | Geographic grounding query | P0 | country="India" | Returns Indian cities |
| E-006-04 | ChromaDB cache on similar prompt | P1 | Same domain+region | Cache hit |
| E-006-05 | Returns valid bundle | P0 | Healthcare India | All required fields present |

### E-007: Schema Intelligence (tests/test_schema_intelligence.py)

| ID | Test | Priority | Input | Expected |
|----|------|----------|-------|----------|
| E-007-01 | Minimum 12 columns | P0 | Any intent | Schema has ≥ 12 columns |
| E-007-02 | Primary key present | P0 | Any schema | At least 1 PK column |
| E-007-03 | Column types valid | P0 | Any schema | All types in allowed set |
| E-007-04 | Generation order assigned | P1 | Schema with DAG | generation_order populated |

### E-008: Code Synthesizer (tests/test_code_synthesizer.py)

| ID | Test | Priority | Input | Expected |
|----|------|----------|-------|----------|
| E-008-01 | Generates valid Python | P0 | Any schema+knowledge | No SyntaxError |
| E-008-02 | Code has generate() function | P0 | Generated code | "def generate" present |
| E-008-03 | Code uses no Faker for domain | P0 | Generated code | No "Faker" for entity names |
| E-008-04 | Code embeds constants | P0 | Generated code | Contains list/dict constants |
| E-008-05 | Code is stateless | P0 | Run twice same seed | Identical output |
| E-008-06 | Uses base64 not triple-quotes | P0 | Wrapper code | "import base64" present |

### E-009: Self-Healing (tests/test_self_healing.py)

| ID | Test | Priority | Input | Expected |
|----|------|----------|-------|----------|
| E-009-01 | Successful execution | P0 | Valid script | DataFrame returned |
| E-009-02 | Heals on first failure | P0 | Script with fixable bug | Healed on attempt 2 |
| E-009-03 | Fails after 3 attempts | P0 | Unfixable script | SelfHealingFailureError |
| E-009-04 | Uses base64 encoding | P0 | Any script | Wrapper uses base64 |
| E-009-05 | Timeout enforced | P1 | Infinite loop script | Timeout after 120s |
| E-009-06 | Traceback sent to LLM | P1 | Failing script | Heal prompt includes traceback |

### E-010: Anomaly Engine (tests/test_anomaly_engine.py)

| ID | Test | Priority | Input | Expected |
|----|------|----------|-------|----------|
| E-010-01 | Null rate within 5% | P0 | Target 0.10 | Actual 0.05-0.15 |
| E-010-02 | Adjacent key typo | P0 | "gmail" | "gmqil" possible |
| E-010-03 | Transposition typo | P0 | "Bangalore" | "Banglore" possible |
| E-010-04 | Repetition typo | P0 | "Mumbai" | "Mumbaai" possible |
| E-010-05 | Deletion typo | P0 | "Hyderabad" | "Hydrabad" possible |
| E-010-06 | Case error typo | P0 | "Mumbai" | "mumbai" possible |
| E-010-07 | Date format mix | P0 | datetime series | ~12% alternative formats |
| E-010-08 | MCAR null injection | P0 | Random column | Nulls uniformly distributed |
| E-010-09 | MAR null injection | P1 | Conditioned column | Higher null rate in condition |
| E-010-10 | Duplicate injection | P1 | 0.1% rate | ≈ 0.1% exact duplicates |

### E-011: Validation Engine (tests/test_validation_engine.py)

| ID | Test | Priority | Input | Expected |
|----|------|----------|-------|----------|
| E-011-01 | Returns 10 checks | P0 | Valid df + schema | ValidationReport with 10 checks |
| E-011-02 | Schema completeness | P0 | Missing column | Score < 1.0 |
| E-011-03 | Type conformance | P0 | String in int column | Score < 1.0 |
| E-011-04 | Range violation detected | P0 | Age = 200 | Flagged |
| E-011-05 | Temporal violation | P0 | end < start | Flagged |
| E-011-06 | Unique violation | P0 | Duplicate PK | Score = 0.0 for that check |
| E-011-07 | Overall score computed | P0 | Any df | 0-100 weighted score |

### E-012: Privacy (tests/test_privacy.py)

| ID | Test | Priority | Input | Expected |
|----|------|----------|-------|----------|
| E-012-01 | Detects email | P0 | "user@gmail.com" | Entity detected |
| E-012-02 | Masks email preserves domain | P0 | "user@gmail.com" | "xxx@gmail.com" |
| E-012-03 | Detects Indian phone | P0 | "+91-9876543210" | Entity detected |
| E-012-04 | Masks phone valid format | P0 | Indian phone | Valid format replacement |
| E-012-05 | Detects PAN card | P0 | "ABCDE1234F" | IN_PAN detected |
| E-012-06 | Masks PAN valid pattern | P0 | PAN | Valid PAN pattern |
| E-012-07 | Detects credit card | P0 | Luhn-valid 16 digits | Detected |
| E-012-08 | Masks CC Luhn-valid | P0 | Credit card | Luhn-valid replacement |
| E-012-09 | k-anonymity k=1 scores 40/40 | P0 | Unique rows | Privacy score 40/40 |
| E-012-10 | k-anonymity k=5 computed | P1 | Grouped data | Correct k value |
| E-012-11 | Masked df preserves structure | P0 | Any df | Same columns, dtypes, shape |
| E-012-12 | Privacy report lists entities | P1 | df with PII | Entity types listed |

### E-013: Orchestrator (tests/test_orchestrator.py)

| ID | Test | Priority | Input | Expected |
|----|------|----------|-------|----------|
| E-013-01 | Full pipeline 50 rows | P0 | Healthcare prompt | DataFrame with quality score |
| E-013-02 | Returns GenerationResult | P0 | Any prompt | All fields populated |
| E-013-03 | Progress callback 9 times | P0 | With callback | ≥ 9 callback invocations |
| E-013-04 | Invalid key raises error | P0 | Bad API key | LLMConfigError |
| E-013-05 | Same seed same output | P0 | seed=42 twice | Identical DataFrames |
| E-013-06 | Quality score > 0 | P1 | Any successful gen | quality_score > 0 |
| E-013-07 | Generated code is valid | P1 | Any successful gen | code compiles |

### E-014: Cross-Cutting (tests/test_cross_cutting.py)

| ID | Test | Priority | Input | Expected |
|----|------|----------|-------|----------|
| E-014-01 | No hardcoded domain lists | P0 | AST scan all .py | No inline list > 20 strings |
| E-014-02 | No Faker pa_IN locale | P0 | Grep all .py | No "pa_IN" |
| E-014-03 | No bare uuid.uuid4() | P0 | Grep all .py | Always str(uuid.uuid4()) |
| E-014-04 | All files have copyright | P1 | Check all .py | Header present |
| E-014-05 | No TODO/FIXME in release | P2 | Grep all .py | None found |

---

## Backend Tests

### B-001: Auth (tests/test_auth.py)

| ID | Test | Priority | Input | Expected |
|----|------|----------|-------|----------|
| B-001-01 | Signup creates user | P0 | Valid email+password | 201 + user object |
| B-001-02 | Signup rejects duplicate | P0 | Same email twice | 409 Conflict |
| B-001-03 | Login returns JWT | P0 | Valid credentials | 200 + access_token |
| B-001-04 | Login rejects wrong password | P0 | Wrong password | 401 |
| B-001-05 | JWT validates | P0 | Valid token | User returned |
| B-001-06 | Expired JWT returns 401 | P0 | Expired token | 401 |
| B-001-07 | Refresh returns new access | P0 | Valid refresh token | New access_token |
| B-001-08 | Protected route requires auth | P0 | No token | 401 |

### B-002: Conversations (tests/test_conversations.py)

| ID | Test | Priority | Input | Expected |
|----|------|----------|-------|----------|
| B-002-01 | Create conversation | P0 | Valid data | 201 + conversation |
| B-002-02 | List conversations | P0 | Authenticated user | Array of conversations |
| B-002-03 | Get conversation + messages | P0 | Valid ID | Conversation with messages |
| B-002-04 | Delete conversation | P0 | Valid ID | 204 |
| B-002-05 | Cannot access other user's conv | P0 | Wrong user | 404 |

### B-003: Generation (tests/test_generation.py)

| ID | Test | Priority | Input | Expected |
|----|------|----------|-------|----------|
| B-003-01 | Trigger generation | P0 | Valid prompt | 202 + generation_id |
| B-003-02 | Get generation status | P0 | Valid ID | Status object |
| B-003-03 | Download completed generation | P0 | Completed ID | File response |
| B-003-04 | Get Glass Box code | P0 | Completed ID | Python source |
| B-003-05 | Plan limit enforced | P0 | Exceeds limit | 429 |

### B-004: Datasets (tests/test_datasets.py)

| ID | Test | Priority | Input | Expected |
|----|------|----------|-------|----------|
| B-004-01 | Upload CSV | P0 | Valid CSV file | 201 + dataset |
| B-004-02 | Upload detects schema | P0 | CSV with columns | schema_summary populated |
| B-004-03 | Query returns result | P0 | "average salary" | Query result + data |
| B-004-04 | Upload size limit | P0 | Over-limit file | 413 |
| B-004-05 | Delete dataset | P0 | Valid ID | 204 + file removed |

### B-005: Billing (tests/test_billing.py)

| ID | Test | Priority | Input | Expected |
|----|------|----------|-------|----------|
| B-005-01 | List plans | P0 | GET /plans | 4 plans |
| B-005-02 | Subscribe | P0 | Plan + payment | Subscription created |
| B-005-03 | Cancel subscription | P0 | Active sub | Status = cancelled |
| B-005-04 | Usage tracking | P0 | After generation | Usage incremented |
| B-005-05 | Webhook Razorpay | P1 | Valid payload | Subscription updated |

### B-006: API Keys (tests/test_api_keys.py)

| ID | Test | Priority | Input | Expected |
|----|------|----------|-------|----------|
| B-006-01 | Create API key | P0 | Name + scopes | Key returned (once) |
| B-006-02 | List shows prefix only | P0 | List keys | Only prefix visible |
| B-006-03 | Revoke key | P0 | Key ID | is_active = false |
| B-006-04 | API auth with key | P0 | sf_live_xxx header | Authenticated |
| B-006-05 | Revoked key rejected | P0 | Revoked key | 401 |

---

## Frontend Tests

### F-001: Components (vitest + testing-library)

| ID | Test | Priority | Description |
|----|------|----------|-------------|
| F-001-01 | ChatMessage renders user variant | P0 | Right-aligned, dark bg |
| F-001-02 | ChatMessage renders assistant variant | P0 | Left-aligned, white bg |
| F-001-03 | ChatInput submits on Enter | P0 | Calls onSubmit |
| F-001-04 | GenerationCard shows 9 phases | P0 | 9 phase items rendered |
| F-001-05 | GenerationCard updates progress | P0 | Progress bar width matches |
| F-001-06 | ResultCard shows preview table | P0 | Table with correct rows |
| F-001-07 | ResultCard download buttons | P0 | 4 format buttons |
| F-001-08 | UploadZone accepts CSV | P0 | onUpload called with file |
| F-001-09 | UploadZone rejects invalid type | P0 | Error displayed |
| F-001-10 | PricingCard shows all plans | P0 | 4 plan cards rendered |

### F-002: Pages (integration)

| ID | Test | Priority | Description |
|----|------|----------|-------------|
| F-002-01 | Landing page renders | P0 | All sections visible |
| F-002-02 | Login form submits | P0 | API called with credentials |
| F-002-03 | Dashboard shows stats | P1 | Usage numbers displayed |
| F-002-04 | Generate sends message | P0 | Message appears in thread |
| F-002-05 | Generate shows progress | P0 | Phase cards update |
| F-002-06 | Explore uploads file | P0 | Dataset appears in list |
| F-002-07 | History lists generations | P1 | Table populated |
| F-002-08 | Settings saves provider | P1 | API key saved confirmation |

---

## End-to-End Tests (Playwright)

| ID | Test | Priority | Description |
|----|------|----------|-------------|
| E2E-01 | Signup → Dashboard | P0 | New user sees dashboard |
| E2E-02 | Configure LLM provider | P0 | Save API key, test passes |
| E2E-03 | Generate from prompt | P0 | Type prompt → see progress → download |
| E2E-04 | Upload dataset + query | P1 | Upload CSV → ask question → see result |
| E2E-05 | Upgrade plan | P1 | Click upgrade → payment → limits increased |
| E2E-06 | API key generation flow | P1 | Create key → use in curl → get result |

---

## Test Execution Schedule

| Layer | When | Tests Run | Pass Criteria |
|-------|------|-----------|---------------|
| 1 | Day 2-3 | E-001 through E-004 | 100% green, >90% coverage |
| 2 | Day 4-7 | E-005 through E-014 | 100% green, >85% coverage |
| 3 | Day 8-9 | B-001 | 100% green |
| 4 | Day 10-12 | B-002 through B-004 | 100% green |
| 5 | Day 13-14 | B-005, B-006 | 100% green |
| 6-8 | Day 15-22 | F-001, F-002 | 100% green |
| 9 | Day 23-24 | E2E-01 through E2E-06 | 100% green |
