#!/bin/bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

mode="${1:-awsim}"
domain_id="${2:-${ROS_DOMAIN_ID:-1}}"
out_root="${3:-/output/manual}"

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
    setsid bash "${SCRIPT_DIR}/run_autoware.bash" "${mode}" "${domain_id}" "${out_root}" &
else
    bash "${SCRIPT_DIR}/run_autoware.bash" "${mode}" "${domain_id}" "${out_root}" &
fi

child_pid="$!"
child_pgid="$(ps -o pgid= -p "${child_pid}" 2>/dev/null | tr -d '[:space:]' || true)"

set +e
wait "${child_pid}"
status="$?"
set -e

trap - EXIT INT TERM
exit "${status}"
