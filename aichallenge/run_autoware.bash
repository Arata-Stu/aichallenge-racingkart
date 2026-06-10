#!/bin/bash

mode="${1}"
id="${2:-${ROS_DOMAIN_ID:-0}}"
out_dir="${3:+${3}/d${id}}"
out_dir="${out_dir:-/output/$(date +%Y%m%d-%H%M%S)/d${id}}"
checkpoint_path="${4:-}"
launch_file="aichallenge_system.launch.xml"

case "${mode}" in
"awsim")
    opts=("simulation:=true" "use_sim_time:=true" "run_rviz:=true")
    ;;
"awsim-no-viz")
    opts=("simulation:=true" "use_sim_time:=true" "run_rviz:=false")
    ;;
"awsim-joycon")
    opts=("simulation:=true" "use_sim_time:=true" "run_rviz:=true")
    launch_file="joycon_data_collection.launch.xml"
    ;;
"awsim-joycon-no-viz")
    opts=("simulation:=true" "use_sim_time:=true" "run_rviz:=false")
    launch_file="joycon_data_collection.launch.xml"
    ;;
"awsim-lidar-trajectory-net")
    opts=("simulation:=true" "use_sim_time:=true" "run_rviz:=true" "control_method:=lidar_trajectory_net")
    if [ -n "${checkpoint_path}" ]; then
        opts+=("lidar_trajectory_ckpt_path:=${checkpoint_path}")
    fi
    ;;
"awsim-lidar-trajectory-net-no-viz")
    opts=("simulation:=true" "use_sim_time:=true" "run_rviz:=false" "control_method:=lidar_trajectory_net")
    if [ -n "${checkpoint_path}" ]; then
        opts+=("lidar_trajectory_ckpt_path:=${checkpoint_path}")
    fi
    ;;
"vehicle")
    opts=("simulation:=false" "use_sim_time:=false" "run_rviz:=false")
    ;;
"vehicle-joycon")
    opts=("simulation:=false" "use_sim_time:=false" "run_rviz:=false")
    launch_file="joycon_data_collection.launch.xml"
    ;;
"rosbag")
    opts=("simulation:=false" "use_sim_time:=true" "run_rviz:=true")
    ;;
*)
    echo "invalid argument (use 'awsim', 'awsim-no-viz', 'awsim-joycon', 'awsim-joycon-no-viz', 'awsim-lidar-trajectory-net', 'awsim-lidar-trajectory-net-no-viz', 'vehicle', 'vehicle-joycon', or 'rosbag')"
    exit 1
    ;;
esac

export ROS_DOMAIN_ID=$id

mkdir -p "${out_dir}"
exec >"${out_dir}/autoware.log" 2>&1
trap 'bash /aichallenge/utils/fix_ownership.bash "${HOST_UID}" "${HOST_GID}" /output "$(dirname "${out_dir}")"' EXIT

cd "${out_dir}" || exit
# Persist ROS node logs under the run output directory (so autostart_orchestrator logs are collectible).
export ROS_HOME="${out_dir}/ros"
export ROS_LOG_DIR="${ROS_HOME}/log"
mkdir -p "${ROS_LOG_DIR}"

ros2 launch aichallenge_system_launch "${launch_file}" "${opts[@]}" "domain_id:=$id"
