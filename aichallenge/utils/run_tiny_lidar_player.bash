#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
player_id="${1:-}"
run_mode="${2:-awsim-no-viz}"

if [[ -z "${player_id}" ]]; then
    IFS= read -r -p "Player/ROS domain [1-4] (default: 2): " player_id
    player_id="${player_id:-2}"
fi
if ! [[ "${player_id}" =~ ^[1-4]$ ]]; then
    echo "[ERROR] Player must be 1, 2, 3, or 4." >&2
    exit 2
fi
case "${run_mode}" in
awsim|awsim-no-viz|vehicle|rosbag) ;;
*) echo "[ERROR] run mode must be awsim, awsim-no-viz, vehicle, or rosbag." >&2; exit 2 ;;
esac

checkpoint="$(${script_dir}/select_tiny_lidar_checkpoint.bash)"
export TINY_LIDAR_NET_PYTORCH_CHECKPOINT="${checkpoint}"
export ROS_DOMAIN_ID="${player_id}"

if ! command -v ros2 >/dev/null 2>&1; then
    if [[ ! -f /aichallenge/workspace/install/setup.bash ]]; then
        echo "[ERROR] ROS workspace is not built: /aichallenge/workspace/install/setup.bash" >&2
        exit 1
    fi
    set +u
    # shellcheck disable=SC1091
    source /aichallenge/workspace/install/setup.bash
    set -u
fi

log_root="${AIC_TINY_LIDAR_LOG_ROOT:-/output/tiny-lidar-$(date +%Y%m%d-%H%M%S)}"
echo "[INFO] Player ${player_id} / ROS_DOMAIN_ID=${player_id}"
echo "[INFO] checkpoint=${TINY_LIDAR_NET_PYTORCH_CHECKPOINT}"
echo "[INFO] mode=${run_mode} logs=${log_root}/d${player_id}"
exec /aichallenge/run_autoware.bash \
    "${run_mode}" \
    "${player_id}" \
    "${log_root}" \
    tiny_lidar_net_pytorch \
    false
