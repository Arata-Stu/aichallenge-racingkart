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

run_registered()
{
    local title="$1"
    local session_file="$2"
    shift 2

    printf '\033]0;%s\007' "${title}"
    echo "[INFO] ${title}"
    echo "[INFO] command: $*"

    mkdir -p "$(dirname "${session_file}")"
    setsid "$@" &
    process_pid=$!
    session_id="${process_pid}"
    printf '%s\n' "${session_id}" > "${session_file}"
    interrupted=0

    stop_child()
    {
        pkill -INT -s "${session_id}" 2>/dev/null || true
        wait "${process_pid}" 2>/dev/null || true
    }

    handle_interrupt()
    {
        interrupted=1
        stop_child
    }

    handle_shutdown()
    {
        stop_child
        exit 143
    }

    trap handle_interrupt INT
    trap handle_shutdown TERM HUP
    wait "${process_pid}"
    status=$?
    rm -f "${session_file}"
    trap - INT TERM HUP

    if (( interrupted )); then
        status=130
    fi

    echo
    echo "[INFO] ${title} exited with status ${status}."
    return "${status}"
}

run_registered_interactive()
{
    local title="$1"
    local session_file="$2"
    shift 2
    local interrupted=0
    local session_id=""
    local status

    printf '\033]0;%s\007' "${title}"
    echo "[INFO] ${title}"
    echo "[INFO] command: $*"

    mkdir -p "$(dirname "${session_file}")"

    stop_interactive_child()
    {
        if [[ -f "${session_file}" ]]; then
            read -r session_id < "${session_file}" || session_id=""
        fi
        if [[ "${session_id}" =~ ^[1-9][0-9]*$ ]] && \
           pgrep -s "${session_id}" >/dev/null 2>&1; then
            pkill -INT -s "${session_id}" 2>/dev/null || true
        fi
    }

    handle_interactive_interrupt()
    {
        interrupted=1
        stop_interactive_child
    }

    handle_interactive_shutdown()
    {
        stop_interactive_child
        rm -f "${session_file}"
        exit 143
    }

    trap handle_interactive_interrupt INT
    trap handle_interactive_shutdown TERM HUP

    # Keep the command in the foreground so menus can read from this terminal.
    # The small wrapper writes its session ID before exec, allowing Stop All to
    # terminate the complete command tree without relying on process names.
    setsid bash -c '
        session_file="$1"
        shift
        printf "%s\n" "$$" > "${session_file}"
        exec "$@"
    ' aic-registered-interactive "${session_file}" "$@"
    status=$?

    rm -f "${session_file}"
    trap - INT TERM HUP
    if (( interrupted )); then
        status=130
    fi

    echo
    echo "[INFO] ${title} exited with status ${status}."
    return "${status}"
}

aic_simulator()
{
    local selected_mode="${1:-${AIC_SIM_MODE:-dev}}"
    local wall_recovery="${2:-${AIC_WALL_RECOVERY:-off}}"
    export ROS_DOMAIN_ID=0
    export LOG_DIR="${log_dir}"
    export AIC_WALL_RECOVERY="${wall_recovery}"
    run_and_stream \
        "AWSIM / Domain 0 / ${selected_mode} / wall-recovery=${wall_recovery}" \
        "${log_dir}/awsim.log" \
        "${log_dir}/sessions/domain0-awsim.sid" \
        /aichallenge/run_simulator.bash "${selected_mode}" "${vehicle_count}"
}

aic_simulator_menu()
{
    local selected_mode mode_script wall_recovery=off selection default_selection=1
    local index marker
    local -a simulator_modes=()
    while IFS= read -r mode_script; do
        simulator_modes+=("${mode_script}")
    done < <(
        find /aichallenge/simulator_scripts -maxdepth 1 -type f -name '*.sh' -printf '%f\n' \
            | sed 's/\.sh$//' \
            | sort
    )
    if (( ${#simulator_modes[@]} == 0 )); then
        echo "[ERROR] No simulator mode scripts were found." >&2
        return 2
    fi
    for index in "${!simulator_modes[@]}"; do
        if [[ "${simulator_modes[index]}" == "${AIC_SIM_MODE:-dev}" ]]; then
            default_selection=$((index + 1))
            break
        fi
    done
    while true; do
        echo "Simulator mode:"
        for index in "${!simulator_modes[@]}"; do
            marker=""
            (( index + 1 == default_selection )) && marker=" [default]"
            printf '  %2d) %s%s\n' \
                "$((index + 1))" "${simulator_modes[index]}" "${marker}"
        done
        IFS= read -r -p "Select [1-${#simulator_modes[@]}] (default: ${default_selection}): " selection
        selection="${selection:-${default_selection}}"
        if [[ "${selection}" =~ ^[0-9]+$ ]] && \
           (( selection >= 1 && selection <= ${#simulator_modes[@]} )); then
            selected_mode="${simulator_modes[selection - 1]}"
            break
        fi
        echo "Please enter a displayed number." >&2
    done
    mode_script="${selected_mode}"
    if [[ "${mode_script}" =~ ^(dev|gate)[0-9]+$ ]]; then
        mode_script="${BASH_REMATCH[1]}"
    fi
    if [[ ! -f "/aichallenge/simulator_scripts/${mode_script}.sh" ]]; then
        echo "[ERROR] Unknown simulator mode: ${selected_mode}" >&2
        return 2
    fi
    if grep -q -- '--wall-recovery' "/aichallenge/simulator_scripts/${mode_script}.sh"; then
        default_selection=1
        [[ "${AIC_WALL_RECOVERY:-off}" == on ]] && default_selection=2
        while true; do
            echo "Wall recovery:"
            echo "  1) off"
            echo "  2) on"
            IFS= read -r -p "Select [1-2] (default: ${default_selection}): " selection
            selection="${selection:-${default_selection}}"
            case "${selection}" in
            1) wall_recovery=off; break ;;
            2) wall_recovery=on; break ;;
            *) echo "Please enter 1 or 2." >&2 ;;
            esac
        done
    else
        echo "[INFO] ${selected_mode} does not expose wall recovery; using off."
    fi
    export AIC_SIM_MODE="${selected_mode}"
    export AIC_WALL_RECOVERY="${wall_recovery}"
    aic_simulator "${selected_mode}" "${wall_recovery}"
}

aic_autoware()
{
    local control_method="${1:-mpc}"
    local run_mode="${2:-}"
    local obstacle_avoidance="${3:-${AIC_MPC_OBSTACLE_AVOIDANCE:-true}}"
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

aic_tiny_lidar()
{
    local run_mode="${1:-}"
    local checkpoint="${2:-}"
    local setup_status
    if [[ -z "${checkpoint}" ]]; then
        checkpoint="$(/aichallenge/utils/select_tiny_lidar_checkpoint.bash)" || return $?
    elif [[ ! -f "${checkpoint}" ]]; then
        echo "[ERROR] TinyLiDARNet checkpoint does not exist: ${checkpoint}" >&2
        return 2
    fi
    export TINY_LIDAR_NET_PYTORCH_CHECKPOINT="$(realpath "${checkpoint}")"
    set +u
    # Refresh package discovery when the workspace was rebuilt after Terminator started.
    # shellcheck disable=SC1091
    source /aichallenge/workspace/install/setup.bash
    setup_status=$?
    set -u
    if (( setup_status != 0 )); then
        echo "[ERROR] Failed to source /aichallenge/workspace/install/setup.bash" >&2
        return "${setup_status}"
    fi
    if ! ros2 pkg prefix tiny_lidar_net_pytorch >/dev/null 2>&1; then
        echo "[ERROR] tiny_lidar_net_pytorch is not built." >&2
        echo "[ERROR] Run 'make autoware-build' on the host, then retry aic_tiny_lidar." >&2
        return 3
    fi
    echo "[INFO] TinyLiDARNet checkpoint: ${TINY_LIDAR_NET_PYTORCH_CHECKPOINT}"
    aic_autoware tiny_lidar_net_pytorch "${run_mode}" false
}

aic_rsu_fusion()
{
    local run_mode="${1:-}"
    local checkpoint="${2:-}"
    local setup_status
    if [[ -z "${checkpoint}" ]]; then
        checkpoint="$(/aichallenge/utils/select_rsu_fusion_checkpoint.bash)" || return $?
    elif [[ ! -f "${checkpoint}" ]]; then
        echo "[ERROR] RSU Fusion checkpoint does not exist: ${checkpoint}" >&2
        return 2
    fi
    export RSU_FUSION_NET_PYTORCH_CHECKPOINT="$(realpath "${checkpoint}")"
    set +u
    # shellcheck disable=SC1091
    source /aichallenge/workspace/install/setup.bash
    setup_status=$?
    set -u
    if (( setup_status != 0 )); then
        echo "[ERROR] Failed to source /aichallenge/workspace/install/setup.bash" >&2
        return "${setup_status}"
    fi
    if ! ros2 pkg prefix rsu_fusion_net_pytorch >/dev/null 2>&1; then
        echo "[ERROR] rsu_fusion_net_pytorch is not built." >&2
        echo "[ERROR] Run 'make autoware-build' on the host, then retry aic_rsu_fusion." >&2
        return 3
    fi
    echo "[INFO] RSU Fusion checkpoint: ${RSU_FUSION_NET_PYTORCH_CHECKPOINT}"
    aic_autoware rsu_fusion_net_pytorch "${run_mode}" false
}

aic_player_menu()
{
    local default_control_selection control_selection control_method
    local default_run_mode_selection run_mode_selection run_mode
    local avoidance_selection obstacle_avoidance=false
    if [[ "${player_id}" == "${AIC_GAMEPAD_PLAYER:-0}" ]]; then
        default_control_selection=1
    elif (( player_id >= 2 )); then
        default_control_selection=2
    else
        default_control_selection=4
    fi
    while true; do
        echo "Player ${player_id} control:"
        echo "  1) Joycon"
        echo "  2) TinyLiDARNet PyTorch"
        echo "  3) RSU Fusion PyTorch"
        echo "  4) MPC"
        echo "  5) Virtual Scan RL (SAC training/evaluation)"
        IFS= read -r -p "Select [1-5] (default: ${default_control_selection}): " control_selection
        control_selection="${control_selection:-${default_control_selection}}"
        case "${control_selection}" in
        1) control_method=joycon; break ;;
        2) control_method=tiny_lidar_net_pytorch; break ;;
        3) control_method=rsu_fusion_net_pytorch; break ;;
        4) control_method=mpc; break ;;
        5) control_method=virtual_scan_rl; break ;;
        *) echo "Please enter 1, 2, 3, 4, or 5." >&2 ;;
        esac
    done

    if [[ "${player_id}" == "${AIC_RVIZ_PLAYER:-1}" ]]; then
        default_run_mode_selection=1
    else
        default_run_mode_selection=2
    fi
    while true; do
        echo "Run mode:"
        echo "  1) awsim (RViz on)"
        echo "  2) awsim-no-viz (RViz off)"
        echo "  3) vehicle"
        echo "  4) rosbag"
        IFS= read -r -p "Select [1-4] (default: ${default_run_mode_selection}): " run_mode_selection
        run_mode_selection="${run_mode_selection:-${default_run_mode_selection}}"
        case "${run_mode_selection}" in
        1) run_mode=awsim; break ;;
        2) run_mode=awsim-no-viz; break ;;
        3) run_mode=vehicle; break ;;
        4) run_mode=rosbag; break ;;
        *) echo "Please enter 1, 2, 3, or 4." >&2 ;;
        esac
    done

    if [[ "${control_method}" == mpc ]]; then
        while true; do
            echo "MPC obstacle avoidance:"
            echo "  1) on"
            echo "  2) off"
            IFS= read -r -p "Select [1-2] (default: 1): " avoidance_selection
            avoidance_selection="${avoidance_selection:-1}"
            case "${avoidance_selection}" in
            1) obstacle_avoidance=true; break ;;
            2) obstacle_avoidance=false; break ;;
            *) echo "Please enter 1 or 2." >&2 ;;
            esac
        done
        export AIC_MPC_OBSTACLE_AVOIDANCE="${obstacle_avoidance}"
    fi

    case "${control_method}" in
    tiny_lidar_net_pytorch) aic_tiny_lidar "${run_mode}" ;;
    rsu_fusion_net_pytorch) aic_rsu_fusion "${run_mode}" ;;
    *) aic_autoware "${control_method}" "${run_mode}" "${obstacle_avoidance}" ;;
    esac
}

aic_stop_all()
{
    /aichallenge/utils/stop_terminator.bash "${log_dir}"
}

aic_bag_manager()
{
    local domain_id="${AIC_GAMEPAD_PLAYER:-0}"
    local setup_status
    if ! [[ "${domain_id}" =~ ^[1-9][0-9]*$ ]]; then
        echo "[ERROR] No Joycon player is selected." >&2
        echo "[ERROR] Restart run_terminator.bash and select a Gamepad player." >&2
        return 2
    fi

    export ROS_DOMAIN_ID="${domain_id}"
    export ROS_LOG_DIR="${log_dir}/d${domain_id}/ros/bag-manager"
    mkdir -p "${ROS_LOG_DIR}"
    # ROS/ament setup files legitimately probe unset variables. Temporarily
    # disable nounset, then restore this script's strict mode immediately.
    set +u
    # shellcheck disable=SC1091
    source /aichallenge/workspace/install/setup.bash
    setup_status=$?
    set -u
    if (( setup_status != 0 )); then
        echo "[ERROR] Failed to source /aichallenge/workspace/install/setup.bash" >&2
        return "${setup_status}"
    fi

    run_registered \
        "Bag Manager / Joycon Player ${domain_id} / Domain ${domain_id}" \
        "${log_dir}/sessions/domain${domain_id}-bag-manager.sid" \
        ros2 launch bag_manager_py bag_manager.launch.xml
}

aic_virtual_scan_rl()
{
    local rl_dir="${AIC_VIRTUAL_SCAN_RL_DIR:-/aichallenge/ml_workspace/virtual_scan_rl}"
    if [[ ! -x "${rl_dir}/run.sh" ]]; then
        echo "[ERROR] Virtual Scan RL runner is unavailable: ${rl_dir}/run.sh" >&2
        return 2
    fi
    export ROS_DOMAIN_ID=1
    cd "${rl_dir}" || return 1
    run_registered_interactive \
        "Virtual Scan RL / SAC / Domain 1" \
        "${log_dir}/sessions/domain1-virtual-scan-rl.sid" \
        bash "${rl_dir}/run.sh"
}

prepare_command()
{
    local title="$1"
    local domain_id="$2"
    local default_command="$3"
    local history_name="${4:-domain${domain_id}_history}"
    local history_file="${history_dir}/${history_name}"
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
    if [[ "${default_command}" == "aic_player_menu" ]]; then
        echo "[READY] Control method, checkpoint, run mode, and relevant parameters are selected after Enter."
    elif [[ "${default_command}" == "aic_simulator_menu" ]]; then
        echo "[READY] Simulator mode and wall recovery are selected after Enter."
    fi
    echo "[READY] Use the Up/Down keys to recall commands from ${history_file}."
    if [[ "${title}" == "AWSIM / Domain 0" ]] && (( ${AIC_PLAYER_COUNT:-1} > 1 )); then
        echo "[IMPORTANT] Start every Player pane first, then press Enter here last."
        echo "[IMPORTANT] Late Player startup can miss AWSIM's one-time Ready/Grounded state."
    fi

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
        prepare_command "AWSIM / Domain 0" 0 aic_simulator_menu
        ;;
    autoware)
        player_id="${3:-${player_id}}"
        prepare_command \
            "Autoware / Player ${player_id} / Domain ${player_id}" \
            "${player_id}" \
            aic_player_menu
        ;;
    virtual-scan-rl-autoware)
        player_id=1
        prepare_command \
            "Autoware / Player 1 / Virtual Scan RL / Domain 1" \
            1 \
            "aic_autoware virtual_scan_rl awsim false" \
            virtual_scan_rl_autoware_history
        ;;
    virtual-scan-rl-runner)
        prepare_command \
            "Virtual Scan RL / SAC / Domain 1" \
            1 \
            aic_virtual_scan_rl \
            virtual_scan_rl_runner_history
        ;;
    stop-all)
        prepare_command \
            "Session Control / Stop All" \
            0 \
            aic_stop_all
        ;;
    bag-manager)
        bag_manager_domain="${AIC_GAMEPAD_PLAYER:-0}"
        if [[ "${bag_manager_domain}" =~ ^[1-9][0-9]*$ ]]; then
            prepare_command \
                "Bag Manager / Joycon Player ${bag_manager_domain} / Domain ${bag_manager_domain}" \
                "${bag_manager_domain}" \
                aic_bag_manager \
                bag_manager_history
        else
            prepare_command \
                "Bag Manager / Joycon player not selected" \
                1 \
                "echo '[ERROR] Restart Terminator and select a Gamepad player.'" \
                bag_manager_history
        fi
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
