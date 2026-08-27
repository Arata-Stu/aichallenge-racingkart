#!/usr/bin/env bash

set -uo pipefail

role="${1:-shell}"
rl_dir="${AIC_VIRTUAL_SCAN_RL_DIR:-/aichallenge/ml_workspace/virtual_scan_rl}"
history_dir="${AIC_TERMINATOR_HISTORY_DIR:-/output/terminator-history/virtual-scan-rl}"
log_dir="${AIC_TERMINATOR_LOG_DIR:-/output/terminator-virtual-scan-rl}"

open_rl_shell()
{
    local history_name="${1:-shell_history}"
    export ROS_DOMAIN_ID=1
    export HISTFILE="${history_dir}/${history_name}"
    mkdir -p "${history_dir}"
    touch "${HISTFILE}"
    cd "${rl_dir}" || exit 1
    printf '\033]0;%s\007' "Virtual Scan RL Shell / Domain 1"
    exec bash --rcfile /aichallenge/utils/terminator.bashrc -i
}

status_once()
{
    local session_file session_id state checkpoint_count intervention_count
    printf '\033]0;%s\007' "Virtual Scan RL Status"
    echo "Virtual Scan RL status - $(date '+%F %T')"
    echo

    echo "Registered sessions"
    shopt -s nullglob
    for session_file in "${log_dir}/sessions"/*.sid; do
        read -r session_id < "${session_file}" || session_id=""
        state="stale"
        if [[ "${session_id}" =~ ^[1-9][0-9]*$ ]] && pgrep -s "${session_id}" >/dev/null 2>&1; then
            state="running"
        fi
        printf '  %-8s %-28s SID=%s\n' "${state}" "$(basename "${session_file}" .sid)" "${session_id:-?}"
    done
    if ! compgen -G "${log_dir}/sessions/*.sid" >/dev/null; then
        echo "  none"
    fi

    checkpoint_count="$(find "${rl_dir}/checkpoints" -type f -name '*.zip' 2>/dev/null | wc -l)"
    intervention_count="$(find "${rl_dir}/interventions" -type f -name '*.npz' 2>/dev/null | wc -l)"
    echo
    echo "Artifacts"
    echo "  checkpoints:  ${checkpoint_count}"
    echo "  interventions: ${intervention_count}"
    echo
    echo "Newest checkpoints"
    find "${rl_dir}/checkpoints" -type f -name '*.zip' -printf '%T@ %p\n' 2>/dev/null \
        | sort -nr | head -n 5 | cut -d' ' -f2- \
        | sed "s#${rl_dir}/#  #"
}

monitor_status()
{
    printf '\033]0;%s\007' "Virtual Scan RL Status"
    echo "[INFO] Status refreshes every 5 seconds. Ctrl+C opens an RL shell."
    watch -n 5 /aichallenge/utils/virtual_scan_rl_terminator_pane.bash status-once
    open_rl_shell status_history
}

monitor_gpu()
{
    printf '\033]0;%s\007' "Virtual Scan RL GPU"
    cd "${rl_dir}" || exit 1
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        echo "[WARN] nvidia-smi is unavailable. Opening an RL shell."
        open_rl_shell gpu_history
    fi
    echo "[INFO] GPU monitor refreshes every 2 seconds. Ctrl+C opens an RL shell."
    watch -n 2 nvidia-smi
    open_rl_shell gpu_history
}

case "${role}" in
gpu) monitor_gpu ;;
status) monitor_status ;;
status-once) status_once ;;
shell) open_rl_shell shell_history ;;
*) echo "[ERROR] Unknown Virtual Scan RL pane role: ${role}" >&2; exit 2 ;;
esac
