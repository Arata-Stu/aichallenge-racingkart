#!/usr/bin/env bash

set -eo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${script_dir}"

if [[ -f /opt/ros/humble/setup.bash ]]; then
    # ROS setup scripts may inspect unset variables. Keep nounset disabled.
    set +u
    # shellcheck disable=SC1091
    source /opt/ros/humble/setup.bash
fi
if [[ -f /aichallenge/workspace/install/setup.bash ]]; then
    # shellcheck disable=SC1091
    source /aichallenge/workspace/install/setup.bash
elif [[ -f "${script_dir}/../../workspace/install/setup.bash" ]]; then
    # shellcheck disable=SC1091
    source "${script_dir}/../../workspace/install/setup.bash"
fi

export PYTHONPATH="${script_dir}:${PYTHONPATH:-}"
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-1}"

echo "Virtual Scan RL / SAC"
echo "ROS_DOMAIN_ID=${ROS_DOMAIN_ID}"
echo "  1) Lap training (new)"
echo "  2) Lap training (resume)"
echo "  3) Evaluate checkpoint"
echo "  4) Check environment"
while true; do
    IFS= read -r -p "Select [1-4]: " selection
    case "${selection:-1}" in
    1|2|3|4) selection="${selection:-1}"; break ;;
    *) echo "Please enter 1, 2, 3, or 4." ;;
    esac
done

config="${script_dir}/configs/lap.yaml"
model=""
replay_buffer=""
if [[ "${selection}" == 2 || "${selection}" == 3 ]]; then
    mapfile -t models < <(find "${script_dir}/checkpoints" -type f -name '*.zip' 2>/dev/null | sort -r)
    if (( ${#models[@]} == 0 )); then
        echo "[ERROR] No .zip checkpoint found under ${script_dir}/checkpoints" >&2
        exit 1
    fi
    echo
    echo "Checkpoints"
    for index in "${!models[@]}"; do
        printf '  %2d) %s\n' "$((index + 1))" "${models[index]#"${script_dir}/"}"
    done
    while true; do
        IFS= read -r -p "Select checkpoint [1-${#models[@]}]: " checkpoint_number
        if [[ "${checkpoint_number}" =~ ^[0-9]+$ ]] && \
           (( checkpoint_number >= 1 && checkpoint_number <= ${#models[@]} )); then
            model="${models[checkpoint_number - 1]}"
            break
        fi
        echo "Please enter a valid checkpoint number."
    done
    model_dir="$(dirname -- "${model}")"
    model_stem="$(basename -- "${model}" .zip)"
    if [[ "${model_stem}" == last_model ]]; then
        candidate="${model_dir}/last_replay_buffer.pkl"
    elif [[ "${model_stem}" =~ ^(.+)_([0-9]+)_steps$ ]]; then
        candidate="${model_dir}/${BASH_REMATCH[1]}_replay_buffer_${BASH_REMATCH[2]}_steps.pkl"
    else
        candidate="${model%.zip}_replay_buffer.pkl"
    fi
    [[ -f "${candidate}" ]] && replay_buffer="${candidate}"
fi

if command -v ros2 >/dev/null 2>&1 && ros2 node list 2>/dev/null | grep -q '/teleop_manager_node'; then
    echo "[ERROR] teleop_manager_node is running and would publish competing control commands." >&2
    echo "[ERROR] Stop it; RL uses joy_node directly for hold-to-intervene." >&2
    exit 1
fi

joy_pid=""
learner_pid=""
stop_session()
{
    local session_pid="$1"
    if [[ -n "${session_pid}" ]] && kill -0 "${session_pid}" 2>/dev/null; then
        pkill -INT -s "${session_pid}" 2>/dev/null || true
        wait "${session_pid}" 2>/dev/null || true
    fi
}
cleanup()
{
    stop_session "${learner_pid}"
    stop_session "${joy_pid}"
}
handle_signal()
{
    cleanup
    exit 130
}
trap cleanup EXIT
trap handle_signal INT TERM HUP

if [[ "${selection}" != 4 ]]; then
    IFS= read -r -p "Start joy_node for human intervention? [Y/n]: " start_joy
    case "${start_joy:-y}" in
    y|Y|yes|YES)
        setsid ros2 run joy joy_node --ros-args -r __node:=virtual_scan_rl_joy &
        joy_pid=$!
        echo "[INFO] joy_node PID=${joy_pid}; hold button index 2 to intervene."
        ;;
    esac
fi

case "${selection}" in
1) command=(python3 -m virtual_scan_rl --config "${config}" train) ;;
2)
    command=(python3 -m virtual_scan_rl --config "${config}" resume --model "${model}")
    [[ -n "${replay_buffer}" ]] && command+=(--replay-buffer "${replay_buffer}")
    ;;
3) command=(python3 -m virtual_scan_rl --config "${config}" evaluate --model "${model}") ;;
4) command=(python3 -m virtual_scan_rl --config "${config}" check) ;;
esac

setsid "${command[@]}" &
learner_pid=$!
wait "${learner_pid}"
status=$?
learner_pid=""
exit "${status}"
