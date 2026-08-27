#!/usr/bin/env bash

set -euo pipefail

if (( $# != 0 )); then
    echo "[ERROR] command-line arguments are not supported; answer the interactive prompts instead." >&2
    echo "Usage: /aichallenge/utils/run_terminator.bash" >&2
    exit 2
fi

while true; do
    if ! read -r -p "Number of players [1-4] (default: 3): " player_count; then
        echo >&2
        echo "[ERROR] failed to read the player count." >&2
        exit 1
    fi
    player_count="${player_count:-3}"
    if [[ "${player_count}" =~ ^[1-4]$ ]]; then
        break
    fi
    echo "Please enter a number from 1 to 4." >&2
done

layout="aichallenge-${player_count}p"

while true; do
    if ! read -r -p "Gamepad player [0-${player_count}] (0: none, default: 0): " gamepad_player; then
        echo >&2
        echo "[ERROR] failed to read the Gamepad player." >&2
        exit 1
    fi
    gamepad_player="${gamepad_player:-0}"
    if [[ "${gamepad_player}" =~ ^[0-9]+$ ]] && \
       (( gamepad_player >= 0 && gamepad_player <= player_count )); then
        break
    fi
    echo "Please enter a number from 0 to ${player_count}." >&2
done

if (( gamepad_player > 0 )); then
    default_rviz_player="${gamepad_player}"
else
    default_rviz_player=1
fi

while true; do
    if ! read -r -p "RViz player [1-${player_count}] (default: ${default_rviz_player}): " rviz_player; then
        echo >&2
        echo "[ERROR] failed to read the RViz player." >&2
        exit 1
    fi
    rviz_player="${rviz_player:-${default_rviz_player}}"
    if [[ "${rviz_player}" =~ ^[1-9][0-9]*$ ]] && \
       (( rviz_player >= 1 && rviz_player <= player_count )); then
        break
    fi
    echo "Please enter a number from 1 to ${player_count}." >&2
done

if [[ -z "${DISPLAY:-}" ]]; then
    echo "[ERROR] DISPLAY is empty. Start this from 'make autoware-bash' with X11 available." >&2
    exit 1
fi
if ! command -v terminator >/dev/null 2>&1; then
    echo "[ERROR] terminator is not installed in this Docker image." >&2
    exit 1
fi

export AIC_PLAYER_ID=1
export AIC_PLAYER_COUNT="${player_count}"
export AIC_VEHICLE_COUNT="${player_count}"
export AIC_GAMEPAD_PLAYER="${gamepad_player}"
export AIC_RVIZ_PLAYER="${rviz_player}"
export AIC_SIM_MODE="${AIC_SIM_MODE:-dev}"
export AIC_WALL_RECOVERY="${AIC_WALL_RECOVERY:-off}"
export AIC_MPC_OBSTACLE_AVOIDANCE="${AIC_MPC_OBSTACLE_AVOIDANCE:-true}"
export AIC_TERMINATOR_LOG_DIR="${AIC_TERMINATOR_LOG_DIR:-/output/terminator-$(date +%Y%m%d-%H%M%S)}"
export AIC_TERMINATOR_HISTORY_DIR="${AIC_TERMINATOR_HISTORY_DIR:-/output/terminator-history}"

mkdir -p "${AIC_TERMINATOR_LOG_DIR}" "${AIC_TERMINATOR_HISTORY_DIR}"

echo "[INFO] Terminator layout: ${layout}"
echo "[INFO] Simulator: domain=0, vehicles=${AIC_VEHICLE_COUNT}; mode is selected when it runs"
echo "[INFO] Players: domains 1-${AIC_PLAYER_COUNT}"
if (( AIC_PLAYER_COUNT < 2 )); then
    echo "[INFO] Player control is selected when the Player pane runs"
else
    echo "[INFO] Player 2-${AIC_PLAYER_COUNT}: control method and checkpoint are selected per run"
fi
if (( AIC_GAMEPAD_PLAYER == 0 )); then
    echo "[INFO] Gamepad: disabled"
    echo "[INFO] Bag Manager: disabled until a Gamepad player is selected"
else
    echo "[INFO] Gamepad: player ${AIC_GAMEPAD_PLAYER} / domain ${AIC_GAMEPAD_PLAYER}"
    echo "[INFO] Bag Manager: Joycon player ${AIC_GAMEPAD_PLAYER} / domain ${AIC_GAMEPAD_PLAYER}"
fi
echo "[INFO] RViz: player ${AIC_RVIZ_PLAYER} / domain ${AIC_RVIZ_PLAYER} only"
echo "[INFO] Logs: ${AIC_TERMINATOR_LOG_DIR}"
echo "[INFO] History: ${AIC_TERMINATOR_HISTORY_DIR}"
if (( AIC_PLAYER_COUNT > 1 )); then
    echo "[IMPORTANT] Start Player 1-${AIC_PLAYER_COUNT} panes first; start the AWSIM pane last."
fi

exec terminator \
    --no-dbus \
    --maximize \
    --config /aichallenge/utils/terminator.config \
    --layout "${layout}"
