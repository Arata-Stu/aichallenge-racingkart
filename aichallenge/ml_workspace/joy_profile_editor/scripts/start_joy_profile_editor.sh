#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
EDITOR_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8767}"

exec python3 "${EDITOR_ROOT}/server.py" --host "${HOST}" --port "${PORT}" "$@"
