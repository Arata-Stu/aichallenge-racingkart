#!/usr/bin/env bash

set -uo pipefail

role="${1:-shell}"
training_dir="${AIC_TINY_LIDAR_TRAINING_DIR:-/aichallenge/ml_workspace/tiny_lidar_net_pytorch}"
history_dir="${AIC_TINY_LIDAR_TRAINING_HISTORY_DIR:-/output/terminator-history/tiny-lidar-training}"

open_training_shell()
{
    local history_name="${1:-shell_history}"
    export HISTFILE="${history_dir}/${history_name}"
    mkdir -p "${history_dir}"
    touch "${HISTFILE}"
    cd "${training_dir}" || exit 1
    printf '\033]0;%s\007' "TinyLiDARNet Training Shell"
    exec bash --rcfile /aichallenge/utils/tiny_lidar_training.bashrc -i
}

prepare_pipeline()
{
    local history_file="${history_dir}/pipeline_history"
    local default_command="./run_pipeline.sh"
    local command_line read_status status
    export HISTFILE="${history_file}"
    mkdir -p "${history_dir}"
    touch "${history_file}"
    cd "${training_dir}" || exit 1
    set -o history
    history -r "${history_file}"
    printf '\033]0;%s\007' "TinyLiDARNet Preprocess + Train"

    read_pipeline_line()
    {
        trap 'return 130' INT
        IFS= read -e -r -i "${default_command}" -p "(training) $ " command_line
    }

    echo "[READY] TinyLiDARNet interactive preprocessing and training"
    echo "[READY] Press Enter to run: ${default_command}"
    echo "[READY] Ctrl+C returns to this prompt after the active process exits."
    while true; do
        read_pipeline_line
        read_status=$?
        trap - INT
        if (( read_status == 130 )); then
            echo
            continue
        fi
        if (( read_status != 0 )); then
            open_training_shell pipeline_history
        fi
        command_line="${command_line:-${default_command}}"
        history -s "${command_line}"
        history -a
        eval "${command_line}"
        status=$?
        echo
        echo "[INFO] Pipeline exited with status ${status}. Ready to run again."
    done
}

monitor_gpu()
{
    printf '\033]0;%s\007' "TinyLiDARNet GPU Monitor"
    cd "${training_dir}" || exit 1
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        echo "[WARN] nvidia-smi is unavailable. Opening a training shell."
        open_training_shell gpu_history
    fi
    echo "[INFO] GPU monitor refreshes every 2 seconds. Ctrl+C opens a shell."
    watch -n 2 nvidia-smi
    open_training_shell gpu_history
}

monitor_status()
{
    printf '\033]0;%s\007' "TinyLiDARNet Dataset + Checkpoints"
    cd "${training_dir}" || exit 1
    echo "[INFO] Dataset/checkpoint status refreshes every 5 seconds. Ctrl+C opens a shell."
    watch -n 5 ./run_pipeline.sh --list
    open_training_shell status_history
}

case "${role}" in
pipeline) prepare_pipeline ;;
gpu) monitor_gpu ;;
status) monitor_status ;;
shell) open_training_shell shell_history ;;
*) echo "[ERROR] Unknown training pane role: ${role}" >&2; exit 2 ;;
esac
