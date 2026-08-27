#!/bin/bash

mode="${1}"
id="${2:-${ROS_DOMAIN_ID:-0}}"
out_dir="${3:+${3}/d${id}}"
out_dir="${out_dir:-/output/$(date +%Y%m%d-%H%M%S)/d${id}}"
control_method="${4:-mpc}"
use_obstacle_avoidance="${5:-false}"

case "${mode}" in
"awsim")
    opts=("simulation:=true" "use_sim_time:=true" "run_rviz:=true")
    ;;
"awsim-no-viz")
    opts=("simulation:=true" "use_sim_time:=true" "run_rviz:=false")
    ;;
"vehicle")
    opts=("simulation:=false" "use_sim_time:=false" "run_rviz:=false")
    ;;
"rosbag")
    opts=("simulation:=false" "use_sim_time:=true" "run_rviz:=true")
    ;;
*)
    echo "invalid argument (use 'awsim' or 'vehicle' or 'rosbag')"
    exit 1
    ;;
esac

export ROS_DOMAIN_ID=$id

# Refresh the overlay for every launch. A long-running Terminator container may
# have been opened before a newly added package was built, leaving its inherited
# AMENT_PREFIX_PATH stale even though workspace/install already contains it.
workspace_setup="/aichallenge/workspace/install/setup.bash"
if [[ ! -f "${workspace_setup}" ]]; then
    echo "[ERROR] ROS workspace setup not found: ${workspace_setup}" >&2
    echo "[ERROR] Run 'make autoware-build' on the host first." >&2
    exit 3
fi
# shellcheck disable=SC1091
source "${workspace_setup}"

mkdir -p "${out_dir}"
exec >"${out_dir}/autoware.log" 2>&1

cd "${out_dir}" || exit
# Persist ROS node logs under the run output directory (so autostart_orchestrator logs are collectible).
export ROS_HOME="${out_dir}/ros"
export ROS_LOG_DIR="${ROS_HOME}/log"
mkdir -p "${ROS_LOG_DIR}"

if [[ "${control_method}" == "tiny_lidar_net_pytorch" ]] && \
   ! ros2 pkg prefix tiny_lidar_net_pytorch >/dev/null 2>&1; then
    echo "[ERROR] ROS package 'tiny_lidar_net_pytorch' is not installed in the overlay."
    echo "[ERROR] Run 'make autoware-build' on the host, then press Enter in this pane again."
    exit 3
fi

if [[ "${control_method}" == "rsu_fusion_net_pytorch" ]] && \
   ! ros2 pkg prefix rsu_fusion_net_pytorch >/dev/null 2>&1; then
    echo "[ERROR] ROS package 'rsu_fusion_net_pytorch' is not installed in the overlay."
    echo "[ERROR] Run 'make autoware-build' on the host, then press Enter in this pane again."
    exit 3
fi

# set -m keeps bash from setting SIGINT to SIG_IGN on the backgrounded child (then the forwarded INT would be a no-op).
set -m
ros2 launch aichallenge_system_launch aichallenge_system.launch.xml \
    "${opts[@]}" \
    "domain_id:=$id" \
    "control_method:=${control_method}" \
    "use_obstacle_avoidance:=${use_obstacle_avoidance}" &
launch_pid=$!
launch_status=0
trap 'kill -INT "${launch_pid}" 2>/dev/null' TERM INT
while kill -0 "${launch_pid}" 2>/dev/null; do
    wait "${launch_pid}"
    launch_status=$?
done
trap - TERM INT
exit "${launch_status}"
