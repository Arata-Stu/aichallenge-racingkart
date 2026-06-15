#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

export E2E_RECORD_ROOT="${E2E_RECORD_ROOT:-/aichallenge/record}"
HOST="${E2E_STUDIO_HOST:-0.0.0.0}"
PORT="${E2E_STUDIO_PORT:-8765}"

echo "Starting E2E Learning Studio"
echo "  URL:    http://localhost:${PORT}"
echo "  Record: ${E2E_RECORD_ROOT}"

exec python3 "${SCRIPT_DIR}/learning_studio/server.py" \
  --host "${HOST}" \
  --port "${PORT}"
