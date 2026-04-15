# SynthFlow

**Data that understands the real world**

SynthFlow is a commercial SaaS platform for autonomous synthetic data generation. Describe the data you need in plain English — *"Generate 5,000 Indian healthcare patient records"* — and SynthFlow produces causally realistic, statistically accurate, privacy-safe synthetic datasets.

## Architecture

```
synthflow/
├── frontend/     Next.js 14 (App Router) + TypeScript + Tailwind
├── backend/      FastAPI + SQLAlchemy + Alembic
├── engine/       SynthFlow core — 15 generation subsystems, 9-phase pipeline
└── docs/         PRD, architecture, API docs
```

## Quick Start (local dev)

### Prerequisites
- Python 3.11+
- Node.js 18+
- Docker (for PostgreSQL + Redis)

### 1. Start infrastructure
```bash
docker compose up -d
```

### 2. Engine
```bash
cd engine
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

### 3. Backend
```bash
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill in secrets
uvicorn app.main:app --reload
```

### 4. Frontend
```bash
cd frontend
npm install
cp .env.local.example .env.local   # fill in values
npm run dev
```

Visit [http://localhost:3000](http://localhost:3000) — API at [http://localhost:8000/docs](http://localhost:8000/docs).

## License

Apache 2.0 — Copyright (c) 2026 Ambrish Nigam
