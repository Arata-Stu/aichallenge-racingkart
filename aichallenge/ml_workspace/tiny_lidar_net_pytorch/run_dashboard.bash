#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
port="${AIC_TINY_LIDAR_DASHBOARD_PORT:-8765}"
host="${AIC_TINY_LIDAR_DASHBOARD_HOST:-127.0.0.1}"
runtime_dir="${AIC_TINY_LIDAR_DASHBOARD_RUNTIME_DIR:-/output/tiny-lidar-dashboard}"
pid_file="${runtime_dir}/dashboard.pid"

mkdir -p "${runtime_dir}"

echo "[INFO] TinyLiDAR Studio backend"
echo "[INFO] Open in your browser: http://localhost:${port}"
echo "[INFO] PID file: ${pid_file}"
echo "[INFO] Stop with Ctrl+C"

exec python3 "${script_dir}/dashboard_backend.py" \
    --host "${host}" \
    --port "${port}" \
    --pid-file "${pid_file}" \
    "$@"
