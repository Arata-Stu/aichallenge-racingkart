#!/usr/bin/env bash

set -uo pipefail

role="${1:-shell}"
player_id="${AIC_PLAYER_ID:-1}"
vehicle_count="${AIC_VEHICLE_COUNT:-1}"
sim_mode="${AIC_SIM_MODE:-dev}"
log_dir="${AIC_TERMINATOR_LOG_DIR:-/output/terminator}"
history_dir="${AIC_TERMINATOR_HISTORY_DIR:-/output/terminator-history}"

open_shell()
{
    local domain_id="$1"
    export ROS_DOMAIN_ID="${domain_id}"
    exec bash --rcfile /aichallenge/utils/terminator.bashrc -i
}

open_free_shell()
{
    local domain_id="${1:-1}"
    export HISTFILE="${history_dir}/free_shell_history"
    mkdir -p "${history_dir}"
    touch "${HISTFILE}"
    echo "[READY] Free ROS 2 shell / Domain ${domain_id}"
    echo "[READY] Example: ros2 topic list"
    echo "[READY] Change domain with: aic_domain <0-${AIC_PLAYER_COUNT:-4}>"
    open_shell "${domain_id}"
}

run_and_stream()
{
    local title="$1"
    local log_file="$2"
    local session_file="$3"
    shift 3

    mkdir -p "$(dirname "${log_file}")"
    touch "${log_file}"
    printf '\033]0;%s\007' "${title}"
    echo "[INFO] ${title}"
    echo "[INFO] log: ${log_file}"
    echo "[INFO] command: $*"

    mkdir -p "$(dirname "${session_file}")"
    setsid "$@" &
    process_pid=$!
    session_id="${process_pid}"
    printf '%s\n' "${session_id}" > "${session_file}"
    tail -n +1 -F "${log_file}" &
    tail_pid=$!
    interrupted=0

    stop_children()
    {
        pkill -INT -s "${session_id}" 2>/dev/null || true
        kill "${tail_pid}" 2>/dev/null || true
        wait "${process_pid}" 2>/dev/null || true
        wait "${tail_pid}" 2>/dev/null || true
    }

    handle_interrupt()
    {
        interrupted=1
        stop_children
    }

    handle_shutdown()
    {
        stop_children
        exit 143
    }

    trap handle_interrupt INT
    trap handle_shutdown TERM HUP
    wait "${process_pid}"
    status=$?
    kill "${tail_pid}" 2>/dev/null || true
    wait "${tail_pid}" 2>/dev/null || true
    rm -f "${session_file}"
    trap - INT TERM HUP

    if (( interrupted )); then
        status=130
    fi

    echo
    echo "[INFO] ${title} exited with status ${status}. Opening an interactive shell."
    return "${status}"
}

aic_simulator()
{
    export ROS_DOMAIN_ID=0
    export LOG_DIR="${log_dir}"
    run_and_stream \
        "AWSIM / Domain 0" \
        "${log_dir}/awsim.log" \
        "${log_dir}/sessions/domain0-awsim.sid" \
        /aichallenge/run_simulator.bash "${sim_mode}" "${vehicle_count}"
}

aic_autoware()
{
    local control_method="${1:-mpc}"
    local run_mode="${2:-}"
    local obstacle_avoidance="${AIC_MPC_OBSTACLE_AVOIDANCE:-true}"
    if [[ -z "${run_mode}" ]]; then
        if [[ "${player_id}" == "${AIC_RVIZ_PLAYER:-1}" ]]; then
            run_mode=awsim
        else
            run_mode=awsim-no-viz
        fi
    fi
    export ROS_DOMAIN_ID="${player_id}"
    run_and_stream \
        "Autoware / Player ${player_id} / Domain ${player_id} / ${control_method} / ${run_mode}" \
        "${log_dir}/d${player_id}/autoware.log" \
        "${log_dir}/sessions/domain${player_id}-autoware.sid" \
        /aichallenge/run_autoware.bash \
        "${run_mode}" \
        "${player_id}" \
        "${log_dir}" \
        "${control_method}" \
        "${obstacle_avoidance}"
}

aic_stop_all()
{
    /aichallenge/utils/stop_terminator.bash "${log_dir}"
}

prepare_command()
{
    local title="$1"
    local domain_id="$2"
    local default_command="$3"
    local history_file="${history_dir}/domain${domain_id}_history"
    local command_line
    local read_status

    read_prepared_line()
    {
        trap 'return 130' INT
        IFS= read -e -r -i "${default_command}" -p "(Domain ${domain_id}) $ " command_line
    }

    export ROS_DOMAIN_ID="${domain_id}"
    export HISTFILE="${history_file}"
    mkdir -p "${history_dir}"
    touch "${history_file}"
    set -o history
    history -r "${history_file}"

    printf '\033]0;%s\007' "${title}"
    echo "[READY] ${title}"
    echo "[READY] The command is prefilled. Edit it if needed, then press Enter."
    echo "[READY] Use the Up/Down keys to recall commands from ${history_file}."

    while true; do
        read_prepared_line
        read_status=$?
        trap - INT

        if (( read_status == 130 )); then
            echo
            continue
        fi
        if (( read_status != 0 )); then
            echo
            echo "[INFO] Opening an interactive Domain ${domain_id} shell."
            open_shell "${domain_id}"
        fi
        if [[ -z "${command_line}" ]]; then
            continue
        fi

        history -s "${command_line}"
        history -a
        eval "${command_line}"
        status=$?
        echo
        echo "[INFO] Command exited with status ${status}. Ready to run again."
    done
}

case "${role}" in
simulator)
    aic_simulator
    open_shell 0
    ;;
autoware)
    player_id="${2:-${player_id}}"
    aic_autoware
    open_shell "${player_id}"
    ;;
prepared)
    case "${2:-}" in
    simulator)
        prepare_command "AWSIM / Domain 0" 0 aic_simulator
        ;;
    autoware)
        player_id="${3:-${player_id}}"
        if [[ "${player_id}" == "${AIC_GAMEPAD_PLAYER:-0}" ]]; then
            default_control_command="aic_autoware joycon"
        else
            default_control_command="aic_autoware mpc"
        fi
        prepare_command \
            "Autoware / Player ${player_id} / Domain ${player_id}" \
            "${player_id}" \
            "${default_control_command}"
        ;;
    stop-all)
        prepare_command \
            "Session Control / Stop All" \
            0 \
            aic_stop_all
        ;;
    *)
        echo "[ERROR] unknown prepared command: ${2:-}" >&2
        exit 2
        ;;
    esac
    ;;
shell)
    open_shell "${2:-${player_id}}"
    ;;
free-shell)
    open_free_shell "${2:-1}"
    ;;
*)
    echo "[ERROR] unknown pane role: ${role}" >&2
    exit 2
    ;;
esac
