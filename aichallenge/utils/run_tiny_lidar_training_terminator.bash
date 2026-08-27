#!/usr/bin/env bash

set -euo pipefail

if (( $# != 0 )); then
    echo "Usage: /aichallenge/utils/run_tiny_lidar_training_terminator.bash" >&2
    exit 2
fi
if [[ -z "${DISPLAY:-}" ]]; then
    echo "[ERROR] DISPLAY is empty. Start this from 'make autoware-bash'." >&2
    exit 1
fi
if ! command -v terminator >/dev/null 2>&1; then
    echo "[ERROR] terminator is not installed in this Docker image." >&2
    exit 1
fi

export AIC_TINY_LIDAR_TRAINING_DIR="/aichallenge/ml_workspace/tiny_lidar_net_pytorch"
export AIC_TINY_LIDAR_TRAINING_HISTORY_DIR="${AIC_TINY_LIDAR_TRAINING_HISTORY_DIR:-/output/terminator-history/tiny-lidar-training}"
mkdir -p "${AIC_TINY_LIDAR_TRAINING_HISTORY_DIR}"

echo "[INFO] TinyLiDARNet training workspace: ${AIC_TINY_LIDAR_TRAINING_DIR}"
echo "[INFO] Main pane: interactive preprocessing and training"
echo "[INFO] Utility panes: GPU monitor, dataset/checkpoint status, free shell"
echo "[INFO] History: ${AIC_TINY_LIDAR_TRAINING_HISTORY_DIR}"

exec terminator \
    --no-dbus \
    --maximize \
    --config /aichallenge/utils/tiny_lidar_training_terminator.config \
    --layout tiny-lidar-training
