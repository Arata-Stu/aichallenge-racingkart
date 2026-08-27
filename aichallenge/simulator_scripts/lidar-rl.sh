#!/bin/bash

set -eu

AWSIM_DIRECTORY=/aichallenge/simulator/AWSIM
export ROS_DOMAIN_ID=0

# Development defaults are one vehicle and CPU LiDAR. Override with the first
# argument and LIDAR_MODE=gpu on a supported Ubuntu/NVIDIA host.
vehicles="${1:-${LIDAR_RL_VEHICLES:-1}}"
lidar_mode="${LIDAR_MODE:-cpu}"
laps="${LIDAR_RL_LAPS:-unlimited}"
timeout="${LIDAR_RL_TIMEOUT_SECONDS:-10000000.0}"

case "${vehicles}" in
    1 | 2 | 3 | 4) ;;
    *)
        echo "[ERROR] vehicle count must be between 1 and 4: ${vehicles}" >&2
        exit 2
        ;;
esac

case "${lidar_mode}" in
    cpu | gpu) ;;
    *)
        echo "[ERROR] LIDAR_MODE must be cpu or gpu: ${lidar_mode}" >&2
        exit 2
        ;;
esac

exec "${AWSIM_DIRECTORY}/AWSIM.x86_64" \
    --venue citycircuit \
    --start-mode count \
    --start-count-seconds 5 \
    --vehicles "${vehicles}" \
    --npcs 0 \
    --boosts 2 \
    --laps "${laps}" \
    --timeout "${timeout}" \
    --steer-source ackermann \
    --sound off \
    --collisions on \
    --handicap off \
    --wall-recovery off \
    --start-random off \
    --ranking off \
    --camera off \
    --lidar "${lidar_mode}" \
    --imu off \
    --gnss off \
    --v2x off
