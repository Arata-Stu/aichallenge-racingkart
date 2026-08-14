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

while true; do
    if ! read -r -p "Simulator mode (default: dev): " sim_mode; then
        echo >&2
        echo "[ERROR] failed to read the simulator mode." >&2
        exit 1
    fi
    sim_mode="${sim_mode:-dev}"
    mode_script="${sim_mode}"
    if [[ "${mode_script}" =~ ^(dev|gate)[0-9]+$ ]]; then
        mode_script="${BASH_REMATCH[1]}"
    fi
    if [[ -f "/aichallenge/simulator_scripts/${mode_script}.sh" ]]; then
        break
    fi
    echo "Unknown simulator mode: ${sim_mode}" >&2
done

wall_recovery=off
if grep -q -- '--wall-recovery' "/aichallenge/simulator_scripts/${mode_script}.sh"; then
    while true; do
        if ! read -r -p "Wall recovery [off/on] (default: off): " wall_recovery; then
            echo >&2
            echo "[ERROR] failed to read the wall recovery setting." >&2
            exit 1
        fi
        wall_recovery="${wall_recovery:-off}"
        case "${wall_recovery,,}" in
        off|on)
            wall_recovery="${wall_recovery,,}"
            break
            ;;
        *)
            echo "Please enter off or on." >&2
            ;;
        esac
    done
else
    echo "[INFO] Simulator mode '${sim_mode}' does not expose --wall-recovery; using off."
fi

while true; do
    if ! read -r -p "Enable MPC obstacle avoidance? [Y/n] (default: Y): " avoidance_answer; then
        echo >&2
        echo "[ERROR] failed to read the obstacle avoidance setting." >&2
        exit 1
    fi
    case "${avoidance_answer:-y}" in
    y|Y|yes|YES|Yes)
        mpc_obstacle_avoidance=true
        break
        ;;
    n|N|no|NO|No)
        mpc_obstacle_avoidance=false
        break
        ;;
    *)
        echo "Please enter y or n." >&2
        ;;
    esac
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
export AIC_SIM_MODE="${sim_mode}"
export AIC_WALL_RECOVERY="${wall_recovery}"
export AIC_MPC_OBSTACLE_AVOIDANCE="${mpc_obstacle_avoidance}"
export AIC_TERMINATOR_LOG_DIR="${AIC_TERMINATOR_LOG_DIR:-/output/terminator-$(date +%Y%m%d-%H%M%S)}"
export AIC_TERMINATOR_HISTORY_DIR="${AIC_TERMINATOR_HISTORY_DIR:-/output/terminator-history}"

mkdir -p "${AIC_TERMINATOR_LOG_DIR}" "${AIC_TERMINATOR_HISTORY_DIR}"

echo "[INFO] Terminator layout: ${layout}"
echo "[INFO] Simulator: domain=0, mode=${AIC_SIM_MODE}, vehicles=${AIC_VEHICLE_COUNT}"
echo "[INFO] Simulator wall recovery: ${AIC_WALL_RECOVERY}"
echo "[INFO] Players: domains 1-${AIC_PLAYER_COUNT}"
echo "[INFO] MPC obstacle avoidance: ${AIC_MPC_OBSTACLE_AVOIDANCE}"
if (( AIC_GAMEPAD_PLAYER == 0 )); then
    echo "[INFO] Gamepad: disabled"
else
    echo "[INFO] Gamepad: player ${AIC_GAMEPAD_PLAYER} / domain ${AIC_GAMEPAD_PLAYER}"
fi
echo "[INFO] RViz: player ${AIC_RVIZ_PLAYER} / domain ${AIC_RVIZ_PLAYER} only"
echo "[INFO] Logs: ${AIC_TERMINATOR_LOG_DIR}"
echo "[INFO] History: ${AIC_TERMINATOR_HISTORY_DIR}"

exec terminator \
    --no-dbus \
    --maximize \
    --config /aichallenge/utils/terminator.config \
    --layout "${layout}"
