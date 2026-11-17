#!/usr/bin/env bash
# Levanta el sistema completo en un solo paso: dependencias, frontend y API.
#   ./run.sh            -> http://127.0.0.1:8000
#   PORT=9000 ./run.sh
set -euo pipefail
cd "$(dirname "$0")"

PORT="${PORT:-8000}"
PY="${PYTHON:-python3}"
command -v "$PY" >/dev/null 2>&1 || PY=python

if [ ! -d .venv ]; then
  echo "==> creando entorno virtual .venv"
  "$PY" -m venv .venv
fi
if [ -x .venv/bin/python ]; then VPY=.venv/bin/python; else VPY=.venv/Scripts/python; fi

echo "==> instalando dependencias de Python"
"$VPY" -m pip install -q -r requirements.txt

echo "==> construyendo el frontend"
(cd frontend && { [ -d node_modules ] || npm ci; } && npm run build)

mkdir -p data
echo "==> API + cliente web en http://127.0.0.1:${PORT}"
exec "$VPY" -m uvicorn backend.api.main:app --host 127.0.0.1 --port "$PORT"
