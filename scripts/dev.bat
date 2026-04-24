@echo off
REM ─────────────────────────────────────────────────────────────────────────────
REM SynthFlow Development Environment Startup
REM Starts Backend (FastAPI :8000) + Frontend (Next.js :3000) in separate windows
REM ─────────────────────────────────────────────────────────────────────────────

echo.
echo  ┌─────────────────────────────────────────┐
echo  │   SynthFlow Development Environment     │
echo  └─────────────────────────────────────────┘
echo.

REM Navigate to repo root (one level up from scripts/)
cd /d "%~dp0.."

echo [1/3] Checking .env files...
if not exist "backend\.env" (
    echo   WARNING: backend\.env not found. Copy backend\.env.example and fill in values.
)
if not exist "frontend\.env.local" (
    echo   WARNING: frontend\.env.local not found. Copy frontend\.env.local.example and fill in values.
)
echo.

echo [2/3] Starting Backend (FastAPI on http://localhost:8000)...
start "SynthFlow Backend" cmd /k "cd backend && call .venv\Scripts\activate.bat && uvicorn app.main:app --reload --host 0.0.0.0 --port 8000"

REM Give the backend a moment to start
timeout /t 2 /nobreak > nul

echo [3/3] Starting Frontend (Next.js on http://localhost:3000)...
start "SynthFlow Frontend" cmd /k "cd frontend && npm run dev"

echo.
echo  ✓ SynthFlow is starting up.
echo.
echo  Services:
echo    Backend API:  http://localhost:8000
echo    API Docs:     http://localhost:8000/docs
echo    Frontend:     http://localhost:3000
echo.
echo  To stop: close the "SynthFlow Backend" and "SynthFlow Frontend" windows.
echo.
