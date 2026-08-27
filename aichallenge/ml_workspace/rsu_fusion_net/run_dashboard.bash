#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
port="${AIC_RSU_FUSION_DASHBOARD_PORT:-8766}"
host="${AIC_RSU_FUSION_DASHBOARD_HOST:-127.0.0.1}"
runtime_dir="${AIC_RSU_FUSION_DASHBOARD_RUNTIME_DIR:-/output/rsu-fusion-dashboard}"
pid_file="${runtime_dir}/dashboard.pid"
mkdir -p "${runtime_dir}"

echo "[INFO] RSU Fusion Studio backend"
echo "[INFO] Open in your browser: http://localhost:${port}"
echo "[INFO] PID file: ${pid_file}"
echo "[INFO] Stop with Ctrl+C"
exec python3 "${script_dir}/dashboard_backend.py" --host "${host}" --port "${port}" --pid-file "${pid_file}" "$@"
