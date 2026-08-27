#!/usr/bin/env bash

set -euo pipefail

if (( $# != 0 )); then
    echo "Usage: /aichallenge/utils/run_virtual_scan_rl_terminator.bash" >&2
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

export AIC_PLAYER_ID=1
export AIC_PLAYER_COUNT=1
export AIC_VEHICLE_COUNT=1
export AIC_GAMEPAD_PLAYER=0
export AIC_RVIZ_PLAYER=1
export AIC_SIM_MODE="${AIC_SIM_MODE:-dev}"
export AIC_WALL_RECOVERY="${AIC_WALL_RECOVERY:-off}"
export AIC_VIRTUAL_SCAN_RL_DIR="/aichallenge/ml_workspace/virtual_scan_rl"
export AIC_TERMINATOR_LOG_DIR="${AIC_TERMINATOR_LOG_DIR:-/output/terminator-virtual-scan-rl-$(date +%Y%m%d-%H%M%S)}"
export AIC_TERMINATOR_HISTORY_DIR="${AIC_TERMINATOR_HISTORY_DIR:-/output/terminator-history/virtual-scan-rl}"

# Avoid X11 shared-memory failures seen when Terminator runs through Docker.
export GDK_DISABLE_SHM="${GDK_DISABLE_SHM:-1}"
export QT_X11_NO_MITSHM="${QT_X11_NO_MITSHM:-1}"

mkdir -p "${AIC_TERMINATOR_LOG_DIR}" "${AIC_TERMINATOR_HISTORY_DIR}"

echo "[INFO] Virtual Scan RL Terminator"
echo "[INFO] Player: 1 / ROS_DOMAIN_ID=1 / RViz enabled"
echo "[INFO] Simulator: one vehicle / Domain 0; mode is selected when it runs"
echo "[INFO] RL workspace: ${AIC_VIRTUAL_SCAN_RL_DIR}"
echo "[INFO] Logs: ${AIC_TERMINATOR_LOG_DIR}"
echo "[INFO] Start order: (1) Autoware, (2) AWSIM, (3) SAC runner"
echo "[INFO] Training/resume/evaluation, checkpoint, and Joy are selected in step 3."

exec terminator \
    --no-dbus \
    --maximize \
    --config /aichallenge/utils/virtual_scan_rl_terminator.config \
    --layout virtual-scan-rl
