#!/usr/bin/env bash

set -uo pipefail

session_dir="${1:-${AIC_TERMINATOR_LOG_DIR:-}}"
if [[ -z "${session_dir}" ]]; then
    echo "[ERROR] session directory is required." >&2
    echo "Usage: stop_terminator.bash <terminator-log-directory>" >&2
    exit 2
fi

pid_dir="${session_dir}/sessions"
if [[ ! -d "${pid_dir}" ]]; then
    echo "[INFO] No active session directory: ${pid_dir}"
    exit 0
fi

shopt -s nullglob
session_files=("${pid_dir}"/*.sid)
if (( ${#session_files[@]} == 0 )); then
    echo "[INFO] No registered AWSIM/Autoware sessions."
    exit 0
fi

active_sessions=()
for session_file in "${session_files[@]}"; do
    read -r session_id < "${session_file}" || true
    if ! [[ "${session_id:-}" =~ ^[1-9][0-9]*$ ]]; then
        echo "[WARN] Ignoring invalid session file: ${session_file}" >&2
        continue
    fi
    if pgrep -s "${session_id}" >/dev/null 2>&1; then
        active_sessions+=("${session_id}")
        echo "[INFO] Registered session ${session_id} ($(basename "${session_file}" .sid))"
    else
        rm -f "${session_file}"
    fi
done

if (( ${#active_sessions[@]} == 0 )); then
    echo "[INFO] No active AWSIM/Autoware sessions."
    exit 0
fi

signal_sessions()
{
    local signal_name="$1"
    local session_id
    for session_id in "${active_sessions[@]}"; do
        if pgrep -s "${session_id}" >/dev/null 2>&1; then
            pkill "-${signal_name}" -s "${session_id}" 2>/dev/null || true
        fi
    done
}

wait_for_sessions()
{
    local attempts="$1"
    local session_id
    local alive
    local i
    for ((i = 0; i < attempts; ++i)); do
        alive=0
        for session_id in "${active_sessions[@]}"; do
            if pgrep -s "${session_id}" >/dev/null 2>&1; then
                alive=1
                break
            fi
        done
        if (( alive == 0 )); then
            return 0
        fi
        sleep 0.2
    done
    return 1
}

echo "[INFO] Stopping all registered sessions with SIGINT..."
signal_sessions INT
if ! wait_for_sessions 40; then
    echo "[WARN] Some sessions ignored SIGINT; sending SIGTERM..." >&2
    signal_sessions TERM
    if ! wait_for_sessions 20; then
        echo "[WARN] Some sessions ignored SIGTERM; sending SIGKILL..." >&2
        signal_sessions KILL
        wait_for_sessions 10 || true
    fi
fi

for session_file in "${session_files[@]}"; do
    rm -f "${session_file}"
done

echo "[INFO] All registered AWSIM/Autoware sessions stopped."
