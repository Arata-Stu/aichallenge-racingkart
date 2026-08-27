#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
player_id="${1:-}"
run_mode="${2:-awsim-no-viz}"

if [[ -z "${player_id}" ]]; then
    IFS= read -r -p "Player/ROS domain [1-4] (default: 2): " player_id
    player_id="${player_id:-2}"
fi
[[ "${player_id}" =~ ^[1-4]$ ]] || { echo "[ERROR] Player must be 1, 2, 3, or 4." >&2; exit 2; }
case "${run_mode}" in awsim|awsim-no-viz|vehicle|rosbag) ;; *) echo "[ERROR] Invalid run mode: ${run_mode}" >&2; exit 2 ;; esac

checkpoint="$(${script_dir}/select_rsu_fusion_checkpoint.bash)"
export RSU_FUSION_NET_PYTORCH_CHECKPOINT="${checkpoint}"
export ROS_DOMAIN_ID="${player_id}"

if [[ -f /aichallenge/workspace/install/setup.bash ]]; then
    set +u
    # shellcheck disable=SC1091
    source /aichallenge/workspace/install/setup.bash
    set -u
elif ! command -v ros2 >/dev/null 2>&1; then
    echo "[ERROR] Run make autoware-build first." >&2
    exit 1
fi

log_root="${AIC_RSU_FUSION_LOG_ROOT:-/output/rsu-fusion-$(date +%Y%m%d-%H%M%S)}"
echo "[INFO] Player ${player_id} / ROS_DOMAIN_ID=${player_id}"
echo "[INFO] checkpoint=${RSU_FUSION_NET_PYTORCH_CHECKPOINT}"
echo "[INFO] mode=${run_mode} logs=${log_root}/d${player_id}"
exec /aichallenge/run_autoware.bash "${run_mode}" "${player_id}" "${log_root}" rsu_fusion_net_pytorch false
