# CLAUDE.md — SynthFlow v2.0 Development Instructions

This file is the instruction manual for Claude Code working on the SynthFlow project.
Read this COMPLETELY before writing any code.

## Project Overview

SynthFlow is a commercial SaaS platform for autonomous synthetic data generation.
Monorepo with three packages: frontend (Next.js), backend (FastAPI), engine (Python).

## Repository Structure

```
synthflow/
├── frontend/          Next.js 14 (App Router) + TypeScript + Tailwind
├── backend/           FastAPI + SQLAlchemy + Alembic
├── engine/            SynthFlow core engine (15 subsystems)
│   └── synthflow/
│       ├── models/    Pydantic v2 data models
│       ├── utils/     Logger, helpers
│       ├── engines/   All 15 subsystems
│       ├── privacy/   Presidio guard
│       ├── quality/   Quality reporter
│       ├── core.py    DI container
│       └── orchestrator.py  9-phase pipeline
├── docs/              PRD, architecture, API docs
└── docker-compose.yml
```

## Absolute Rules — Never Violate

### Engine Rules
1. ZERO hardcoded domain data. No Python lists of city names, school names, salary ranges, job titles, or any domain-specific values. The LLM generates all domain knowledge.
2. Every engine file starts with the copyright header:
   ```python
   # ───────────────────────────────────────────────────────────────
   # Copyright (c) 2026 Ambrish Nigam
   # Author : Ambrish Nigam | https://github.com/ambrishnigam
   # Project: SynthFlow — Autonomous Synthetic Data Orchestration Platform
   # License : Apache License 2.0 | https://www.apache.org/licenses/LICENSE-2.0
   # Module : [one-line description]
   # ───────────────────────────────────────────────────────────────
   ```
3. self_healing.py and code_synthesizer.py MUST use base64 encoding for script embedding. NEVER use triple-quote string substitution.
4. helpers.py timestamp detection MUST match `_at` suffix, NOT `at` substring. `marks_math` must NOT be classified as datetime.
5. privacy score: k_anonymity=1 means every row is unique = perfect privacy for synthetic data. Score it 40/40, not 0/40.
6. NEVER use Faker locale `pa_IN` (does not exist). Use `hi_IN` for Punjab and all North India.
7. Always `str(uuid.uuid4())`, never bare `uuid.uuid4()`.
8. Schema generation must enforce minimum 12 columns.
9. Generated Glass Box code must be stateless: `generate(row_count, seed) -> pd.DataFrame` is a pure function.
10. Type hints on ALL function signatures (input AND output).
11. async/await on all I/O-bound operations (LLM calls, file I/O, DB queries).
12. No bare `except Exception`. Use specific exception types.

### Frontend Rules
1. Design system: Intercom-inspired warm geometry
   - Canvas: #faf9f6 (warm cream)
   - Text: #111111 (off-black)
   - Accent: #ff5600 (SynthFlow orange) — AI features only
   - Border: #dedbd6 (warm oat)
   - Button radius: 4px (sharp, not rounded)
   - Card radius: 8px
   - No box-shadows. Depth via borders and surface tints.
2. Fonts: Satoshi (headings + body), Instrument Serif (editorial), JetBrains Mono (code/labels), Plus Jakarta Sans (bold UI)
3. Headings: line-height 1.00, negative letter-spacing (-2.4px at 80px scaling down)
4. Button hover: scale(1.1) with transition. Active: scale(0.85).
5. No competitor mentions on any customer-facing page. No "MOSTLY AI" comparison.
6. Chat interface styled like Claude AI / ChatGPT — clean, spacious, message bubbles.
7. TypeScript strict mode. No `any` types.
8. All API calls via React Query with proper loading/error/empty states.

### Backend Rules
1. API keys encrypted AES-256 before database storage.
2. JWT access tokens: 15 min expiry. Refresh tokens: 7 days.
3. All endpoints return consistent JSON: `{"data": ..., "error": null}` or `{"data": null, "error": {"message": "...", "code": "..."}}`
4. Rate limiting per user per plan tier.
5. CORS: frontend domain only.
6. No stack traces in API responses. Log them server-side.

## Running Tests

```bash
# Engine tests
cd engine && pytest tests/ -v --cov=synthflow --cov-report=term-missing

# Backend tests
cd backend && pytest tests/ -v

# Frontend tests
cd frontend && npm test

# Type checking
cd engine && mypy synthflow/ --strict
cd frontend && npx tsc --noEmit
```

## Build Order

Follow the Development Plan exactly. Build one component, write its tests, run tests, fix until green, then commit and proceed.

Current progress: [UPDATE THIS AS YOU BUILD]
- [ ] Layer 0: Scaffold
- [ ] Layer 1: Engine Foundation
- [ ] Layer 2: Engine Subsystems
- [ ] Layer 3: Backend Auth + DB
- [ ] Layer 4: Backend Core API
- [ ] Layer 5: Backend Business
- [ ] Layer 6: Frontend Foundation
- [ ] Layer 7: Frontend Core
- [ ] Layer 8: Frontend Settings
- [ ] Layer 9: Integration

## Key Data Contracts

### IntentObject (engine input)
```python
{
    "domain": "healthcare",
    "sub_domain": "cardiology",
    "region": {"country": "India", "state_province": "Maharashtra"},
    "row_count": 5000,
    "seed": 42,
    "implied_columns": ["patient_id", "name", "age", ...]
}
```

### GenerationResult (engine output)
```python
{
    "session_id": "uuid",
    "dataframe": pd.DataFrame,      # the generated data
    "schema": SchemaDefinition,      # what was designed
    "validation_report": {...},      # 10-check audit
    "quality_report": {...},         # composite score
    "privacy_report": {...},         # k-anonymity, PII scan
    "generated_code": "string",      # Glass Box Python
    "intent": IntentObject           # what was understood
}
```

### WebSocket Messages (frontend ↔ backend)
```json
// Client → Server
{"type": "message", "content": "Generate 5000 records", "conversation_id": "uuid"}

// Server → Client (streaming text)
{"type": "text_chunk", "content": "I'll generate..."}

// Server → Client (generation progress)
{"type": "phase_update", "phase": 3, "progress": 0.33, "message": "Designing schema..."}

// Server → Client (generation complete)
{"type": "generation_done", "generation_id": "uuid", "quality_score": 94.2, "preview_rows": [...]}
```

## Environment Variables

### Backend (.env)
```
DATABASE_URL=postgresql+asyncpg://user:pass@host:5432/synthflow
REDIS_URL=redis://default:pass@host:6379
SUPABASE_URL=https://xxx.supabase.co
SUPABASE_KEY=eyJ...
JWT_SECRET=random-256-bit-secret
ENCRYPTION_KEY=random-256-bit-key
RAZORPAY_KEY_ID=rzp_...
RAZORPAY_KEY_SECRET=...
STRIPE_SECRET_KEY=sk_...
```

### Frontend (.env.local)
```
NEXT_PUBLIC_API_URL=http://localhost:8000
NEXT_PUBLIC_WS_URL=ws://localhost:8000
NEXT_PUBLIC_SUPABASE_URL=https://xxx.supabase.co
NEXT_PUBLIC_SUPABASE_ANON_KEY=eyJ...
```
