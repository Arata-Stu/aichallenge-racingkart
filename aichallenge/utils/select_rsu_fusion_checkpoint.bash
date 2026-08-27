#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
checkpoint_root="${AIC_RSU_FUSION_CHECKPOINT_ROOT:-${script_dir}/../ml_workspace/rsu_fusion_net/checkpoints}"
mode=interactive

usage()
{
    cat <<EOF
Usage: $(basename "$0") [--root PATH] [--latest|--list]

Select an RSU Fusion .pth checkpoint without typing its full path.
The selected absolute path is printed to stdout; prompts are printed to stderr.
EOF
}

while (( $# > 0 )); do
    case "$1" in
    --root) checkpoint_root="${2:?--root requires PATH}"; shift 2 ;;
    --latest) mode=latest; shift ;;
    --list) mode=list; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "[ERROR] Unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done

declare -a checkpoints=()
while IFS= read -r line; do
    path="${line#*|}"
    [[ -n "${path}" ]] && checkpoints+=("${path}")
done < <(
    if [[ -d "${checkpoint_root}" ]]; then
        find "${checkpoint_root}" -type f -name best_model.pth -printf '%T@|%p\n' | sort -t '|' -k1,1nr
    fi
)
if (( ${#checkpoints[@]} == 0 )) && [[ -d "${checkpoint_root}" ]]; then
    while IFS= read -r line; do
        path="${line#*|}"
        [[ -n "${path}" ]] && checkpoints+=("${path}")
    done < <(find "${checkpoint_root}" -type f -name '*.pth' -printf '%T@|%p\n' | sort -t '|' -k1,1nr)
fi

print_checkpoints()
{
    echo "RSU Fusion checkpoints: ${checkpoint_root}" >&2
    local index relative
    for index in "${!checkpoints[@]}"; do
        relative="${checkpoints[index]#"${checkpoint_root%/}/"}"
        printf '  %2d) %s\n' "$((index + 1))" "${relative}" >&2
    done
}

if [[ "${mode}" == list ]]; then print_checkpoints; exit 0; fi
if [[ "${mode}" == latest ]]; then
    (( ${#checkpoints[@]} > 0 )) || { echo "[ERROR] No .pth checkpoint found under ${checkpoint_root}" >&2; exit 1; }
    realpath "${checkpoints[0]}"; exit 0
fi
if [[ ! -r /dev/tty ]]; then
    echo "[ERROR] Interactive checkpoint selection requires a terminal." >&2; exit 1
fi

if (( ${#checkpoints[@]} > 0 )); then
    print_checkpoints
    echo "  c) Enter a custom path" >&2
    while true; do
        IFS= read -r -p "Select checkpoint [1-${#checkpoints[@]}] (default: 1): " selection </dev/tty
        selection="${selection:-1}"
        [[ "${selection,,}" == c ]] && break
        if [[ "${selection}" =~ ^[0-9]+$ ]] && (( selection >= 1 && selection <= ${#checkpoints[@]} )); then
            realpath "${checkpoints[selection - 1]}"; exit 0
        fi
        echo "Please enter a displayed number or c." >&2
    done
else
    echo "[WARN] No checkpoint found under ${checkpoint_root}" >&2
fi

while true; do
    IFS= read -e -r -p "Custom .pth path: " custom_path </dev/tty
    custom_path="${custom_path/#\~/${HOME}}"
    if [[ -f "${custom_path}" && "${custom_path}" == *.pth ]]; then realpath "${custom_path}"; exit 0; fi
    echo "Checkpoint does not exist or is not .pth: ${custom_path}" >&2
done
