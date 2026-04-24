#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# SynthFlow Development Environment Startup
# Starts Backend (FastAPI :8000) + Frontend (Next.js :3000) in background
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo ""
echo "  SynthFlow Development Environment"
echo ""

# ── Env file checks ───────────────────────────────────────────────────────────
[[ ! -f backend/.env ]]        && echo "  WARNING: backend/.env not found"
[[ ! -f frontend/.env.local ]] && echo "  WARNING: frontend/.env.local not found"

# ── Backend ───────────────────────────────────────────────────────────────────
echo "[1/2] Starting backend on http://localhost:8000 ..."
(
  cd backend
  if [[ -d .venv ]]; then
    source .venv/bin/activate
  elif [[ -d venv ]]; then
    source venv/bin/activate
  fi
  uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
) &
BACKEND_PID=$!
echo "      Backend PID: $BACKEND_PID"

# ── Frontend ──────────────────────────────────────────────────────────────────
echo "[2/2] Starting frontend on http://localhost:3000 ..."
(
  cd frontend
  npm run dev
) &
FRONTEND_PID=$!
echo "      Frontend PID: $FRONTEND_PID"

echo ""
echo "  Services:"
echo "    Backend API:  http://localhost:8000"
echo "    API Docs:     http://localhost:8000/docs"
echo "    Frontend:     http://localhost:3000"
echo ""
echo "  Press Ctrl+C to stop all services."
echo ""

# ── Cleanup on exit ───────────────────────────────────────────────────────────
cleanup() {
  echo ""
  echo "Stopping SynthFlow..."
  kill "$BACKEND_PID" "$FRONTEND_PID" 2>/dev/null || true
  exit 0
}
trap cleanup INT TERM

wait
