#!/bin/bash

AWSIM_DIRECTORY=/aichallenge/simulator/AWSIM
export ROS_DOMAIN_ID=0

control_method="${CONTROL_METHOD:-mpc}"
lidar_mode="${LIDAR_MODE:-}"

# Preserve the existing MPC evaluation cost while making LiDAR controllers
# usable without editing this sealed-image script for each submission.
if [[ -z "${lidar_mode}" ]]; then
    case "${control_method}" in
        lidar_racing | tiny_lidar_net) lidar_mode=cpu ;;
        *) lidar_mode=off ;;
    esac
fi

case "${lidar_mode}" in
    off | cpu | gpu) ;;
    *)
        echo "[ERROR] LIDAR_MODE must be off, cpu, or gpu: ${lidar_mode}" >&2
        exit 2
        ;;
esac

echo "[INFO] AWSIM eval sensor mode: control=${control_method}, lidar=${lidar_mode}"

exec "$AWSIM_DIRECTORY/AWSIM.x86_64" \
    --venue citycircuit \
    --start-mode sync \
    --start-count-seconds 5 \
    --vehicles 1 \
    --npcs 0 \
    --boosts 2 \
    --laps 6 \
    --timeout 600 \
    --steer-source ackermann \
    --sound off \
    --collisions on \
    --handicap off \
    --wall-recovery off \
    --ranking off \
    --camera off \
    --lidar "${lidar_mode}"

# Cameraを使う場合 : --camera cpu or gpu
# LiDARを使う場合 : --lidar cpu or gpu
# GPUがない場合 -headlessを末尾に追加
