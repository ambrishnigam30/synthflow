# SynthFlow v2.0 — Product Requirements Document

**Version**: 2.0.0
**Author**: Ambrish Nigam
**Last Updated**: April 15, 2026
**Status**: Pre-Development

---

## 1. Executive Summary

SynthFlow is a commercial SaaS platform for autonomous synthetic data generation. Users describe the data they need in plain English — "Generate 5000 Indian healthcare patient records" — and SynthFlow produces causally realistic, statistically accurate, privacy-safe synthetic datasets.

The platform makes exactly two LLM API calls per generation request, then executes a standalone Python script locally to produce any number of rows at zero additional token cost. The generated code is fully transparent ("Glass Box") — users can read, edit, and trust every line.

**Customer-facing tagline**: "Data that understands the real world"

**Business model**: Freemium SaaS with BYOK (Bring Your Own Key) for LLM costs. Users configure their own Gemini, OpenAI, or Groq API keys. SynthFlow charges for platform access, not AI usage.

---

## 2. Target Users

### Primary Personas

**Data Scientist / ML Engineer** (P1)
- Needs synthetic training data that preserves causal relationships
- Cares about: distributional accuracy, reproducibility, schema correctness
- Pain point: existing tools produce statistically plausible but causally hollow data
- Usage: 50-200 generations/month, 1K-100K rows each

**QA / Test Engineer** (P2)
- Needs realistic test datasets for software testing
- Cares about: edge cases, dirty data patterns, referential integrity
- Pain point: hand-writing test data is slow, Faker produces obviously fake data
- Usage: 20-50 generations/month, 100-10K rows each

**Privacy / Compliance Officer** (P3)
- Needs synthetic replacements for production data containing PII
- Cares about: privacy guarantees (k-anonymity), regulatory compliance
- Pain point: can't share real data for development/analytics
- Usage: 5-20 generations/month, 10K-1M rows each

**Data Analyst / Product Manager** (P4)
- Needs to prototype dashboards and analytics before real data exists
- Cares about: realistic distributions, easy export formats
- Pain point: waiting for real data to build prototypes
- Usage: 10-30 generations/month, 1K-50K rows each

### Secondary Personas

**Startup Founder**: Needs demo data for investor presentations
**Educator**: Needs teaching datasets for data science courses
**Researcher**: Needs synthetic datasets for reproducible research

---

## 3. Product Features

### 3.1 Core Feature: Chat + Generate (P0 — Must Have)

A unified conversational interface where users interact with an AI assistant that can both answer questions and trigger synthetic data generation.

**User Stories:**

- As a user, I want to describe data I need in plain English and receive a generated dataset
- As a user, I want to ask follow-up questions about my generated data ("add a BMI column", "make it 10K rows")
- As a user, I want to ask general data science questions without triggering generation
- As a user, I want to see real-time progress as my data is being generated (9 phases)
- As a user, I want to download generated data in CSV, Excel, JSON, or Parquet format
- As a user, I want to view and download the Python code that generated my data (Glass Box)
- As a user, I want to view a quality report showing statistical accuracy and privacy scores
- As a user, I want to configure advanced options (row count, seed, scenario, output format)
- As a user, I want my chat history preserved so I can return to past conversations

**Acceptance Criteria:**

1. The system correctly classifies user messages as: generation request, follow-up modification, or general conversation
2. Generation requests trigger the 9-phase pipeline with real-time WebSocket progress updates
3. Each phase update includes: phase number, phase name, progress percentage, status message
4. Completed generations display inline: data preview (first 10 rows), quality score, download buttons
5. The Glass Box code is a valid standalone Python script that reproduces the exact same output when run independently
6. Same seed + same prompt = identical output (deterministic reproducibility)
7. Chat responses stream token-by-token (like ChatGPT/Claude)
8. Conversation history persists across sessions

### 3.2 Core Feature: Data Explorer (P0 — Must Have)

A separate interface where users upload datasets and interact with them through natural language.

**User Stories:**

- As a user, I want to upload CSV, Excel, Parquet, or JSON files
- As a user, I want to ask questions about my data in plain English ("What's the average salary by department?")
- As a user, I want to see query results as tables and charts inline in the chat
- As a user, I want to view data preview, schema, column statistics, and distribution plots
- As a user, I want to generate synthetic data that matches my uploaded dataset's schema and distributions
- As a user, I want to clean or transform my data through conversation ("fix the date column", "remove duplicates")

**Acceptance Criteria:**

1. Supports CSV, XLSX, Parquet, JSON uploads up to plan limits
2. Schema auto-detection: column names, types, nulls, uniques, sample values
3. Natural language queries execute pandas operations and return results
4. Results include appropriate visualizations (bar charts, histograms, scatter plots)
5. "Generate synthetic version" creates data matching the uploaded schema
6. Data transformations generate downloadable modified datasets

### 3.3 User Authentication (P0 — Must Have)

**User Stories:**

- As a visitor, I want to sign up with email/password or Google SSO
- As a user, I want to log in and have my data preserved
- As a user, I want to reset my password via email

**Acceptance Criteria:**

1. Email + password registration with email verification
2. Google OAuth (one-click signup/login)
3. JWT-based session management (access token 15min, refresh token 7 days)
4. Password reset via email link
5. Account deletion with data cleanup

### 3.4 LLM Provider Configuration — BYOK (P0 — Must Have)

**User Stories:**

- As a user, I want to add my own API keys for Gemini, OpenAI, or Groq
- As a user, I want to test that my API key works before saving
- As a user, I want to select which provider/model to use for each generation
- As a user, I want to set a default provider

**Acceptance Criteria:**

1. API keys encrypted with AES-256 before database storage
2. "Test Connection" validates the key against the provider's API
3. Provider selector in chat sidebar, persists as user preference
4. Keys never logged, never sent to frontend after initial save
5. Support for: Gemini (gemini-1.5-pro, gemini-1.5-flash, gemini-2.0-flash), OpenAI (gpt-4o, gpt-4o-mini), Groq (llama-3.1-70b, mixtral-8x7b)

### 3.5 Dashboard (P1 — Should Have)

**User Stories:**

- As a user, I want to see my usage stats (generations, rows, tokens this month)
- As a user, I want to see my recent generations and conversations
- As a user, I want quick-start actions

**Acceptance Criteria:**

1. Usage stats: generation count, total rows, storage used — for current billing period
2. Usage trend chart (last 6 months, bar chart)
3. Recent 5 generations with status, quality score, click-to-open
4. Recent 5 conversations with last message preview
5. Quick-start prompt input

### 3.6 Generation History (P1 — Should Have)

**User Stories:**

- As a user, I want to browse all my past generations with search and filters
- As a user, I want to re-download past outputs
- As a user, I want to regenerate a past request
- As a user, I want to compare two generations side-by-side

**Acceptance Criteria:**

1. Sortable table: date, prompt (truncated), domain, region, rows, columns, quality score, status
2. Search by prompt text
3. Filter by: domain, status, date range
4. Click row to open detail panel (full intent, schema, quality report, download)
5. "Regenerate" re-runs the same prompt
6. "Open in Chat" opens the conversation that created this generation
7. Multi-select + "Compare" shows side-by-side statistics
8. Delete with confirmation

### 3.7 Pricing & Billing (P1 — Should Have)

**User Stories:**

- As a visitor, I want to see pricing plans and compare features
- As a user, I want to upgrade/downgrade my plan
- As a user, I want to see my current usage against plan limits
- As a user, I want to view past invoices

**Acceptance Criteria:**

1. 4 plans: Free, Pro ($19/mo), Business ($49/mo), Enterprise (custom)
2. Razorpay integration for India, Stripe for international
3. Plan enforcement: generation count, row limit, upload limit, API call limit
4. Upgrade flow: select plan → payment → instant activation
5. Downgrade: takes effect at end of current billing period
6. Usage meters with visual progress bars
7. Invoice history with download

### 3.8 Team Management (P2 — Nice to Have for v1)

**User Stories:**

- As a user, I want to create a team and invite members by email
- As a team admin, I want to manage member roles (admin, member)
- As a team member, I want to share generations and datasets with my team

**Acceptance Criteria:**

1. Create team with name
2. Invite by email (sends invitation link)
3. Roles: owner (1 per team), admin, member
4. Team members share: generation history, uploaded datasets
5. Team billing: one subscription covers all members
6. Available on Business plan and above

### 3.9 API Access for Developers (P1 — Should Have)

**User Stories:**

- As a developer, I want programmatic access to SynthFlow via REST API
- As a developer, I want to create/manage API keys
- As a developer, I want clear API documentation

**Acceptance Criteria:**

1. API keys with prefix `sf_live_` (production) and `sf_test_` (sandbox)
2. Key shown once on creation, stored as SHA-256 hash
3. Scopes: generate, read, admin
4. Rate limiting per plan
5. API endpoints:
   - POST /api/v1/generate — trigger generation (returns job_id)
   - GET /api/v1/generate/:id — check status
   - GET /api/v1/generate/:id/download — download result
6. Webhook delivery on generation.completed and generation.failed

### 3.10 Webhooks (P2 — Nice to Have for v1)

**User Stories:**

- As a developer, I want to receive HTTP callbacks when generation completes
- As a developer, I want to verify webhook signatures for security

**Acceptance Criteria:**

1. Configure webhook URL + events to subscribe
2. Events: generation.completed, generation.failed
3. HMAC-SHA256 signature in X-SynthFlow-Signature header
4. Retry logic: 3 attempts with exponential backoff
5. Delivery log with response status
6. "Send Test Event" button

### 3.11 Landing Page (P0 — Must Have)

**Sections:**

1. **Hero**: Tagline, subtitle, CTA buttons, animated demo
2. **How It Works**: 3-step explanation (Understand → Architect → Generate)
3. **Five Pillars**: Zero Hardcoding, Causal Realism, Distributional Accuracy, Temporal Realism, Dirty Data
4. **Domain Showcase**: Horizontal scroll of domain cards (Healthcare, Banking, Retail, etc.)
5. **Glass Box**: Dark section showing generated code preview
6. **Pricing**: Inline plan cards
7. **CTA**: Final conversion section
8. **Footer**: Links, attribution, legal

**No competitor comparisons on any customer-facing page.**

---

## 4. Technical Architecture

### 4.1 System Components

```
┌─────────────────┐     ┌──────────────────┐     ┌───────────────┐
│   Next.js App    │────▶│   FastAPI App     │────▶│  SynthFlow    │
│   (Vercel)       │◀────│   (Render)        │◀────│  Engine       │
│                  │ WS  │                   │     │  (in-process) │
└─────────────────┘     └──────────────────┘     └───────────────┘
                              │    │    │
                    ┌─────────┘    │    └──────────┐
                    ▼              ▼               ▼
             ┌───────────┐  ┌──────────┐    ┌───────────┐
             │ Supabase  │  │ Upstash  │    │ LLM APIs  │
             │ (PG+Auth  │  │ (Redis)  │    │ (BYOK)    │
             │  +Storage)│  │          │    │           │
             └───────────┘  └──────────┘    └───────────┘
```

### 4.2 SynthFlow Engine — 15-Subsystem Architecture

**THE ARCHITECT (Cognitive) — 5 subsystems:**
1. Intent Engine — prompt → IntentObject
2. Knowledge Graph — DuckDB + LLM → CausalKnowledgeBundle
3. Schema Intelligence — intent + knowledge → SchemaDefinition
4. Constraint Engine — schema + knowledge → ConstraintSet
5. Memory Store — DuckDB + ChromaDB session persistence

**THE DIRECTOR (Simulation) — 4 subsystems:**
6. Realism Engine — causal deterministic sampler
7. Stats Engine — distribution selector + MixtureDistribution
8. Correlation Engine — drift detection + correction
9. Pattern Library — temporal rhythm injection

**THE BUILDER (Execution) — 2 subsystems:**
10. Code Synthesizer — Glass Box Python generator (LLM Call 2)
11. Self-Healing Runtime — base64 subprocess + error-patch loop

**THE AUDITOR (Trust) — 4 subsystems:**
12. Anomaly Engine — dirty data injection (nulls, typos, outliers)
13. Validation Engine — 10-check data audit
14. Scenario Engine — what-if distribution shifting
15. SDV Multiplier — optional SDV enrichment

### 4.3 9-Phase Pipeline

```
Phase 1: Intent Parsing        → IntentObject
Phase 2: Knowledge + Schema    → CausalKnowledgeBundle + SchemaDefinition
         (LLM Call 1)             (combined, fallback to 2 calls)
Phase 3: Constraints + Stats   → ConstraintSet + DistributionMap
Phase 4: Code Synthesis        → kernel.py (stateless, deterministic)
         (LLM Call 2)
Phase 5: Self-Healing Exec     → DataFrame (full row_count)
Phase 6: Enhancement           → temporal + anomalies + scenarios + correlation
Phase 7: Privacy Audit         → PrivacyReport (safety net)
Phase 8: Validation            → ValidationReport (10-check audit)
Phase 9: Delivery              → GenerationResult + QualityReport
```

### 4.4 Key Technical Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Primary scaling | Glass Box code re-execution | Preserves causal DAG; SDV is optional |
| Script embedding | Base64 encoding | Prevents quote-breaking; real file for tracebacks |
| Timestamp detection | `_at` suffix match | Prevents `marks_math` misclassification |
| Privacy scoring | k=1 scores 40/40 | Unique synthetic rows = perfect privacy |
| Faker locale | `hi_IN` for North India | `pa_IN` does not exist |
| UUID handling | Always `str(uuid.uuid4())` | Prevents serialization failures |
| Schema minimum | 12 columns enforced | LLM prompt explicitly requires this |
| Domain data | Zero Python dicts/lists | LLM generates all domain values |

---

## 5. Pricing Plans

| Feature | Free | Pro ($19/mo) | Business ($49/mo) | Enterprise |
|---------|------|-------------|-------------------|-----------|
| Generations/month | 10 | 200 | Unlimited | Unlimited |
| Max rows/generation | 1,000 | 100,000 | 1,000,000 | Unlimited |
| LLM providers | 1 (BYOK) | All (BYOK) | All (BYOK) | All (BYOK) |
| Chat + Generate | ✓ | ✓ | ✓ | ✓ |
| Data Explorer | 1 upload, 50MB | 10 uploads, 1GB | 50 uploads, 10GB | Unlimited |
| Export formats | CSV, JSON | All | All | All |
| Glass Box code | — | ✓ | ✓ | ✓ |
| Quality report | Basic | Full | Full | Full |
| API access | — | 1K calls/mo | 10K calls/mo | Unlimited |
| Teams | — | — | 10 members | Unlimited |
| Webhooks | — | — | ✓ | ✓ |
| Scenario engine | — | — | ✓ | ✓ |
| History retention | 7 days | 90 days | Unlimited | Unlimited |
| Support | Community | Email | Priority | Dedicated |

---

## 6. Non-Functional Requirements

### Performance
- Chat response: first token < 500ms (excluding LLM latency)
- Generation: < 60s for 1,000 rows, < 300s for 100,000 rows
- Page load: < 2s (LCP), < 100ms (FID)
- WebSocket reconnection: automatic within 5s

### Security
- API keys encrypted AES-256 at rest
- HTTPS everywhere
- JWT tokens: access 15min, refresh 7 days
- API key hashed SHA-256, prefix shown, full key shown once
- CORS: frontend domain only
- Rate limiting: per user, per plan

### Reliability
- Self-healing: 3 retry attempts with LLM-powered code fixes
- Graceful degradation: if WebSocket fails, fall back to REST polling
- Generation timeout: 300s max
- Error messages: user-friendly, never expose stack traces

### Scalability (future)
- Celery worker pool for concurrent generations
- Redis queue for job management
- Database connection pooling
- CDN for static assets (Vercel handles this)

---

## 7. Out of Scope for v1

- Mobile native app
- SSO/SAML
- Self-hosted deployment
- Audit logs
- Data marketplace
- Collaborative real-time editing
- Multi-language UI (English only)
- Custom model fine-tuning

---

## 8. Success Metrics

| Metric | Target (Month 1) | Target (Month 6) |
|--------|-------------------|-------------------|
| Registered users | 100 | 2,000 |
| Weekly active users | 30 | 500 |
| Generations completed | 500 | 15,000 |
| Free → Pro conversion | 5% | 8% |
| Generation success rate | 85% | 95% |
| Average quality score | 80/100 | 90/100 |
| NPS | — | 40+ |

---

## 9. Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|-----------|
| LLM generates invalid code frequently | Medium | High | Self-healing with 3 retry attempts; extensive prompt engineering |
| Free hosting limits hit quickly | High | Medium | Monitor usage; upgrade to paid tiers when revenue arrives |
| Users confused by BYOK model | Medium | Medium | Clear onboarding; link to free-tier LLM provider signup guides |
| Complex prompts produce poor schemas | Medium | High | Few-shot examples in LLM prompts; schema validation; minimum 12 columns |
| Competitor copies architecture | Low | Low | Speed of execution; community building; brand |

---
