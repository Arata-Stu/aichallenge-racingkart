#!/bin/bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

DEFAULT_TARGET="dev"
DEFAULT_NAME_PREFIX="aichallenge-2025"

usage() {
    cat <<'EOF'
Usage:
  ./my_run_docker.sh dev [auto|gpu|cpu] [--name NAME] [--no-exec] [--recreate]
  ./my_run_docker.sh eval [auto|gpu|cpu] [--name NAME] [--no-exec] [--recreate]
  ./my_run_docker.sh exec [NAME]
  ./my_run_docker.sh stop [NAME]
  ./my_run_docker.sh down [NAME]
  ./my_run_docker.sh rm

Examples:
  ./my_run_docker.sh dev
  ./my_run_docker.sh dev cpu
  ./my_run_docker.sh dev --no-exec
  ./my_run_docker.sh exec

Manual launch flow inside the dev container:
  # terminal 1
  bash /aichallenge/run_simulator.bash dev

  # terminal 2
  bash /aichallenge/run_autoware.bash awsim 1 /output/manual
EOF
}

log_file=""
log() {
    local message="$*"
    printf '%s\n' "${message}"
    if [ -n "${log_file}" ]; then
        printf '%s\n' "${message}" >>"${log_file}"
    fi
}

die() {
    log "[ERROR] $*"
    exit 1
}

init_log() {
    local ts
    ts="$(date +%Y%m%d-%H%M%S)"
    log_file="output/docker/${ts}-my_run_docker-$$.log"
    mkdir -p output/docker output/latest
    ln -sfn "${SCRIPT_DIR}/${log_file}" output/latest/my_run_docker.log
}

container_exists() {
    docker container inspect "$1" >/dev/null 2>&1
}

container_running() {
    [ "$(docker inspect -f '{{.State.Running}}' "$1" 2>/dev/null || true)" = "true" ]
}

default_name() {
    local target="${1:-${DEFAULT_TARGET}}"
    printf '%s-%s-manual\n' "${DEFAULT_NAME_PREFIX}" "${target}"
}

exec_shell() {
    local name="${1:-$(default_name dev)}"

    container_exists "${name}" || die "container not found: ${name}"
    container_running "${name}" || die "container is not running: ${name}"

    log "[INFO] Entering container: ${name}"

    local tty_args=()
    if [ -t 0 ] && [ -t 1 ]; then
        tty_args=(-it)
    else
        tty_args=(-i)
    fi

    docker exec "${tty_args[@]}" "${name}" bash -lc \
        'source /docker-entrypoint.sh; cd /aichallenge 2>/dev/null || cd /; exec bash -i'
}

stop_container() {
    local name="${1:-$(default_name dev)}"

    if ! container_exists "${name}"; then
        log "[INFO] container not found: ${name}"
        return 0
    fi

    if container_running "${name}"; then
        log "[INFO] Stopping container: ${name}"
        docker stop "${name}" >/dev/null
    else
        log "[INFO] container already stopped: ${name}"
    fi
}

remove_container() {
    local name="${1:-$(default_name dev)}"

    if container_exists "${name}"; then
        log "[INFO] Removing container: ${name}"
        docker rm -f "${name}" >/dev/null
    else
        log "[INFO] container not found: ${name}"
    fi
}

append_if_exists_device() {
    local path="$1"

    if [ -c "${path}" ] || [ -b "${path}" ]; then
        run_args+=(--device "${path}")
    fi
}

append_if_exists_volume() {
    local spec="$1"
    local path="${spec%%:*}"

    if [ -e "${path}" ]; then
        run_args+=(-v "${spec}")
    fi
}

target="${1:-}"
if [ -z "${target}" ]; then
    usage
    exit 2
fi
shift || true

init_log

case "${target}" in
exec)
    exec_shell "${1:-$(default_name dev)}"
    exit 0
    ;;
stop)
    stop_container "${1:-$(default_name dev)}"
    exit 0
    ;;
down)
    remove_container "${1:-$(default_name dev)}"
    exit 0
    ;;
rm)
    log "[INFO] Pruning dangling Docker images"
    docker image prune -f
    exit 0
    ;;
dev | eval)
    ;;
-h | --help | help)
    usage
    exit 0
    ;;
*)
    usage
    die "invalid argument: ${target}"
    ;;
esac

device="auto"
container_name="$(default_name "${target}")"
exec_after_start=1
recreate=0

while [ "$#" -gt 0 ]; do
    case "$1" in
    auto | gpu | cpu)
        device="$1"
        shift
        ;;
    --name)
        container_name="${2:-}"
        [ -n "${container_name}" ] || die "--name requires a value"
        shift 2
        ;;
    --no-exec)
        exec_after_start=0
        shift
        ;;
    --recreate)
        recreate=1
        shift
        ;;
    -h | --help)
        usage
        exit 0
        ;;
    *)
        usage
        die "unknown option: $1"
        ;;
    esac
done

image="aichallenge-2025-${target}"
docker image inspect "${image}" >/dev/null 2>&1 || die "Docker image not found: ${image} (run ./docker_build.sh ${target} first)"

mkdir -p output

run_args=(
    --name "${container_name}"
    --privileged
    --network host
    --ipc host
    --stop-signal SIGINT
    -e "DISPLAY=${DISPLAY:-}"
    -e "USER=${USER:-}"
    -e "ROS_DISTRO=humble"
    -e "QT_X11_NO_MITSHM=1"
    -e "TZ=Asia/Tokyo"
    -e "RUN_MODE=${RUN_MODE:-}"
    -e "SIM_MODE=${SIM_MODE:-}"
    -e "CMD=${CMD:-}"
    -e "COLCON_TRACE=${COLCON_TRACE:-}"
    -e "ROS_HOME=${ROS_HOME:-}"
    -e "ROS_LOG_DIR=${ROS_LOG_DIR:-}"
    -e "HOST_UID=$(id -u)"
    -e "HOST_GID=$(id -g)"
    -e "ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-1}"
    -v "${SCRIPT_DIR}/output:/output"
    -v "${SCRIPT_DIR}/vehicle/cyclonedds.xml:/opt/autoware/cyclonedds.xml:ro"
    -w /aichallenge
)

if [ "${target}" = "dev" ]; then
    run_args+=(
        -v "${SCRIPT_DIR}/aichallenge:/aichallenge"
        -v "${SCRIPT_DIR}/remote:/remote"
        -v "${SCRIPT_DIR}/vehicle:/vehicle"
    )
fi

append_if_exists_volume "/tmp/.X11-unix:/tmp/.X11-unix:rw"
append_if_exists_volume "/run/user:/run/user:rw"
append_if_exists_volume "/dev/dri:/dev/dri"

if [ -n "${XAUTHORITY:-}" ] && [ -e "${XAUTHORITY}" ]; then
    run_args+=(-e "XAUTHORITY=${XAUTHORITY}" -v "${XAUTHORITY}:${XAUTHORITY}:rw")
elif [ -n "${DISPLAY:-}" ]; then
    log "[WARN] XAUTHORITY is not set or does not exist; X11 may require 'xhost +local:docker'"
fi

append_if_exists_device "/dev/video0"

if [ -d "/run/user/$(id -u)" ]; then
    run_args+=(-e "PULSE_SERVER=unix:/run/user/$(id -u)/pulse/native")
fi

case "${device}" in
cpu)
    log "[INFO] Running in CPU mode (forced by argument)"
    ;;
gpu)
    log "[INFO] Running in GPU mode (forced by argument)"
    run_args+=(--gpus all -e NVIDIA_VISIBLE_DEVICES=all -e NVIDIA_DRIVER_CAPABILITIES=all)
    ;;
auto)
    if [ -e /dev/nvidia0 ]; then
        log "[INFO] NVIDIA device node detected (/dev/nvidia0) -> enabling --gpus all"
        run_args+=(--gpus all -e NVIDIA_VISIBLE_DEVICES=all -e NVIDIA_DRIVER_CAPABILITIES=all)
    else
        log "[INFO] No NVIDIA GPU detected -> running on CPU"
    fi
    ;;
esac

if container_exists "${container_name}" && [ "${recreate}" -eq 1 ]; then
    remove_container "${container_name}"
fi

if container_exists "${container_name}"; then
    if container_running "${container_name}"; then
        log "[INFO] Reusing running container: ${container_name}"
    else
        log "[INFO] Starting existing container: ${container_name}"
        docker start "${container_name}" >/dev/null
    fi
else
    log "[INFO] Creating detached container: ${container_name}"
    log "[INFO] Image: ${image}"
    docker run -d "${run_args[@]}" "${image}" sleep infinity >>"${log_file}" 2>&1
    log "[INFO] Container started: ${container_name}"
fi

log "[INFO] To enter later: ./my_run_docker.sh exec ${container_name}"
log "[INFO] To stop/remove: ./my_run_docker.sh down ${container_name}"
log "[INFO] Log: ${log_file}"

if [ "${exec_after_start}" -eq 1 ]; then
    exec_shell "${container_name}"
fi
