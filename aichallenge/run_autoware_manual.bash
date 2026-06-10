#!/bin/bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_LIDAR_TRAJECTORY_CKPT="/aichallenge/ml_workspace/lidar_trajectory_net/checkpoints/best_model.pth"

MODES=(
    "awsim"
    "awsim-no-viz"
    "awsim-joycon"
    "awsim-joycon-no-viz"
    "awsim-lidar-trajectory-net"
    "awsim-lidar-trajectory-net-no-viz"
    "vehicle"
    "vehicle-joycon"
    "rosbag"
)

MODE_DESCRIPTIONS=(
    "AWSIM + RViz"
    "AWSIM without RViz"
    "AWSIM + Joy-Con data collection + RViz"
    "AWSIM + Joy-Con data collection without RViz"
    "AWSIM + LiDAR Trajectory Net + Pure Pursuit + RViz"
    "AWSIM + LiDAR Trajectory Net + Pure Pursuit without RViz"
    "Real vehicle"
    "Real vehicle + Joy-Con"
    "Rosbag playback + RViz"
)

select_mode() {
    local choice
    local index

    echo "Select Autoware mode:" >&2
    for index in "${!MODES[@]}"; do
        printf "  %d) %-39s %s\n" \
            "$((index + 1))" "${MODES[$index]}" "${MODE_DESCRIPTIONS[$index]}" >&2
    done

    while true; do
        read -r -p "Mode [1]: " choice
        choice="${choice:-1}"

        if [[ "${choice}" =~ ^[0-9]+$ ]] &&
            (( choice >= 1 && choice <= ${#MODES[@]} )); then
            printf "%s" "${MODES[$((choice - 1))]}"
            return 0
        fi

        for index in "${!MODES[@]}"; do
            if [ "${choice}" = "${MODES[$index]}" ]; then
                printf "%s" "${MODES[$index]}"
                return 0
            fi
        done

        echo "Invalid selection: ${choice}" >&2
    done
}

if [ "$#" -eq 0 ]; then
    if [ ! -t 0 ]; then
        echo "Interactive mode requires a terminal." >&2
        echo "Usage: $0 <mode> [domain_id] [output_root] [checkpoint_path]" >&2
        exit 2
    fi

    mode="$(select_mode)"

    default_domain_id="${ROS_DOMAIN_ID:-1}"
    read -r -p "ROS_DOMAIN_ID [${default_domain_id}]: " domain_id
    domain_id="${domain_id:-${default_domain_id}}"

    default_out_root="/output/manual"
    read -r -p "Output root [${default_out_root}]: " out_root
    out_root="${out_root:-${default_out_root}}"

    checkpoint_path=""
    if [[ "${mode}" == awsim-lidar-trajectory-net* ]]; then
        read -r -p "Checkpoint path [${DEFAULT_LIDAR_TRAJECTORY_CKPT}]: " checkpoint_path
        checkpoint_path="${checkpoint_path:-${DEFAULT_LIDAR_TRAJECTORY_CKPT}}"
    fi

    echo
    echo "Starting mode=${mode}, ROS_DOMAIN_ID=${domain_id}, output=${out_root}/d${domain_id}"
    if [ -n "${checkpoint_path}" ]; then
        echo "Checkpoint=${checkpoint_path}"
    fi
else
    mode="${1}"
    domain_id="${2:-${ROS_DOMAIN_ID:-1}}"
    out_root="${3:-/output/manual}"
    checkpoint_path="${4:-}"
fi

child_pid=""
child_pgid=""
stopping=0

is_child_alive() {
    [ -n "${child_pid}" ] && kill -0 "${child_pid}" 2>/dev/null
}

kill_child_group() {
    local signal="$1"

    if [ -n "${child_pgid}" ]; then
        kill "-${signal}" -- "-${child_pgid}" 2>/dev/null || true
    elif [ -n "${child_pid}" ]; then
        kill "-${signal}" "${child_pid}" 2>/dev/null || true
    fi
}

cleanup_child() {
    [ "${stopping}" -eq 0 ] || return 0
    stopping=1

    if ! is_child_alive; then
        return 0
    fi

    kill_child_group INT

    for _ in $(seq 1 50); do
        is_child_alive || return 0
        sleep 0.1
    done

    kill_child_group TERM

    for _ in $(seq 1 50); do
        is_child_alive || return 0
        sleep 0.1
    done

    kill_child_group KILL
}

on_int() {
    trap - INT TERM
    cleanup_child
    exit 130
}

on_term() {
    trap - INT TERM
    cleanup_child
    exit 143
}

trap on_int INT
trap on_term TERM
trap cleanup_child EXIT

if command -v setsid >/dev/null 2>&1; then
    setsid bash "${SCRIPT_DIR}/run_autoware.bash" \
        "${mode}" "${domain_id}" "${out_root}" "${checkpoint_path}" &
else
    bash "${SCRIPT_DIR}/run_autoware.bash" \
        "${mode}" "${domain_id}" "${out_root}" "${checkpoint_path}" &
fi

child_pid="$!"
child_pgid="$(ps -o pgid= -p "${child_pid}" 2>/dev/null | tr -d '[:space:]' || true)"

set +e
wait "${child_pid}"
status="$?"
set -e

trap - EXIT INT TERM
exit "${status}"
