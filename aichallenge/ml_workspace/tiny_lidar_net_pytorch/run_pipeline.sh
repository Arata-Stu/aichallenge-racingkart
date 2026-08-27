#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
record_root="${AIC_RECORD_ROOT:-${script_dir}/../../record}"
dataset_root="${AIC_TINY_LIDAR_DATASET_ROOT:-${script_dir}/datasets}"
checkpoint_root="${AIC_TINY_LIDAR_CHECKPOINT_ROOT:-${script_dir}/checkpoints}"
preprocessor="${script_dir}/preprocess_bag.py"
trainer="${script_dir}/train.py"
list_only=false

usage()
{
    cat <<EOF
Usage: $(basename "$0") [--record-root PATH] [--dataset-root PATH] [--checkpoint-root PATH] [--list]

Interactively select Bag Manager recordings, preprocess train/validation data,
and launch GPU training without typing long paths.

Environment overrides:
  AIC_RECORD_ROOT                 recording search root
  AIC_TINY_LIDAR_DATASET_ROOT    dataset root containing train/ and val/
  AIC_TINY_LIDAR_CHECKPOINT_ROOT checkpoint output and search root
EOF
}

while (( $# > 0 )); do
    case "$1" in
    --record-root) record_root="${2:?--record-root requires PATH}"; shift 2 ;;
    --dataset-root) dataset_root="${2:?--dataset-root requires PATH}"; shift 2 ;;
    --checkpoint-root) checkpoint_root="${2:?--checkpoint-root requires PATH}"; shift 2 ;;
    --list) list_only=true; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "[ERROR] Unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done

declare -a bag_dirs=()

discover_bags()
{
    local metadata
    bag_dirs=()
    [[ -d "${record_root}" ]] || return 1
    while IFS= read -r -d '' metadata; do
        bag_dirs+=("${metadata%/metadata.yaml}")
    done < <(find "${record_root}" -type f -name metadata.yaml -print0 | sort -z)
}

print_bags()
{
    local index bag_dir relative size
    echo
    echo "Bag Manager recordings: ${record_root}"
    for index in "${!bag_dirs[@]}"; do
        bag_dir="${bag_dirs[index]}"
        relative="${bag_dir#"${record_root%/}/"}"
        size="$(du -sh "${bag_dir}" 2>/dev/null | awk '{print $1}')"
        printf '  %2d) %-58s %8s\n' "$((index + 1))" "${relative}" "${size:-?}"
    done
}

confirm()
{
    local prompt="$1" default_answer="${2:-y}" answer suffix="[Y/n]"
    [[ "${default_answer}" == n ]] && suffix="[y/N]"
    while true; do
        IFS= read -r -p "${prompt} ${suffix}: " answer
        answer="${answer:-${default_answer}}"
        case "${answer,,}" in
        y|yes) return 0 ;;
        n|no) return 1 ;;
        *) echo "Please enter y or n." >&2 ;;
        esac
    done
}

choose_action()
{
    local choice
    echo
    echo "TinyLiDARNet PyTorch pipeline"
    echo "  1) Preprocess recordings"
    echo "  2) Train existing dataset"
    echo "  3) Preprocess, then train (default)"
    echo "  4) Show discovered recordings/datasets/checkpoints"
    while true; do
        IFS= read -r -p "Select [1-4]: " choice
        case "${choice:-3}" in
        1|2|3|4) action="${choice:-3}"; return ;;
        *) echo "Please enter 1, 2, 3, or 4." >&2 ;;
        esac
    done
}

show_status()
{
    discover_bags || true
    print_bags
    echo
    echo "Dataset root: ${dataset_root}"
    find "${dataset_root}" -type f -name scans.npy -printf '  %h\n' 2>/dev/null | sort || true
    echo
    echo "Checkpoints: ${checkpoint_root}"
    find "${checkpoint_root}" -type f -name best_model.pth -printf '  %p\n' 2>/dev/null | sort || true
}

parse_recording_selection()
{
    local input="$1"
    local result_name="$2"
    local normalized token start end number
    local -a tokens=()
    local -A seen=()
    local -n result="${result_name}"
    result=()
    normalized="${input//,/ }"
    if [[ "${normalized,,}" == all || "${normalized,,}" == a ]]; then
        for ((number = 1; number <= ${#bag_dirs[@]}; ++number)); do
            result+=("${number}")
        done
        return 0
    fi
    [[ -z "${normalized// /}" || "${normalized,,}" == none || "${normalized,,}" == n ]] && return 0
    read -r -a tokens <<< "${normalized}"
    for token in "${tokens[@]}"; do
        if [[ "${token}" =~ ^([0-9]+)-([0-9]+)$ ]]; then
            start=$((10#${BASH_REMATCH[1]}))
            end=$((10#${BASH_REMATCH[2]}))
            if (( start < 1 || end > ${#bag_dirs[@]} || start > end )); then
                echo "[ERROR] Invalid range: ${token}" >&2
                return 1
            fi
            for ((number = start; number <= end; ++number)); do
                if [[ -z "${seen[${number}]:-}" ]]; then
                    result+=("${number}")
                    seen[${number}]=1
                fi
            done
        elif [[ "${token}" =~ ^[0-9]+$ ]]; then
            number=$((10#${token}))
            if (( number < 1 || number > ${#bag_dirs[@]} )); then
                echo "[ERROR] Recording number is out of range: ${token}" >&2
                return 1
            fi
            if [[ -z "${seen[${number}]:-}" ]]; then
                result+=("${number}")
                seen[${number}]=1
            fi
        else
            echo "[ERROR] Invalid recording number: ${token}" >&2
            return 1
        fi
    done
}

sequence_name_for_bag()
{
    local bag_dir="$1"
    local relative="${bag_dir#"${record_root%/}/"}"
    printf '%s\n' "${relative//\//_}"
}

print_preprocess_plan()
{
    local split="$1"
    shift
    local number bag_dir output_dir
    for number in "$@"; do
        bag_dir="${bag_dirs[number - 1]}"
        output_dir="${dataset_root}/${split}/$(sequence_name_for_bag "${bag_dir}")"
        printf '  %-5s %2d) %s\n' "${split}" "${number}" "${output_dir}"
    done
}

preprocess_group()
{
    local split="$1"
    local existing_policy="$2"
    shift 2
    local number bag_dir sequence_name output_dir progress=0 total=$#
    for number in "$@"; do
        progress=$((progress + 1))
        bag_dir="${bag_dirs[number - 1]}"
        sequence_name="$(sequence_name_for_bag "${bag_dir}")"
        output_dir="${dataset_root}/${split}/${sequence_name}"
        echo
        echo "[INFO] [${progress}/${total}] split=${split} recording=${number}"
        if [[ -e "${output_dir}" && "${existing_policy}" == skip ]]; then
            echo "[SKIP] Existing output: ${output_dir}"
            continue
        fi
        echo "[INFO] bag=${bag_dir}"
        echo "[INFO] output=${output_dir}"
        python3 "${preprocessor}" --bag "${bag_dir}" --output "${output_dir}"
    done
}

preprocess_interactive()
{
    local train_input val_input existing_policy_choice existing_policy
    declare -a train_numbers=() val_numbers=()
    if ! discover_bags || (( ${#bag_dirs[@]} == 0 )); then
        echo "[ERROR] No metadata.yaml found under ${record_root}" >&2
        return 1
    fi
    print_bags
    echo
    echo "Select recordings in one batch. Examples: 1 2 5-8 / 1,2,5-8 / all / none"
    while true; do
        IFS= read -r -p "Train recording numbers: " train_input
        parse_recording_selection "${train_input}" train_numbers && break
    done
    while true; do
        IFS= read -r -p "Validation recording numbers: " val_input
        parse_recording_selection "${val_input}" val_numbers && break
    done
    if (( ${#train_numbers[@]} == 0 && ${#val_numbers[@]} == 0 )); then
        echo "[INFO] No recordings selected."
        return 0
    fi

    while true; do
        IFS= read -r -p "Existing outputs [skip/overwrite] (default: skip): " existing_policy_choice
        case "${existing_policy_choice:-skip}" in
        skip|s) existing_policy=skip; break ;;
        overwrite|o) existing_policy=overwrite; break ;;
        *) echo "Please enter skip or overwrite." >&2 ;;
        esac
    done

    echo
    echo "Preprocessing plan (sequence names are generated automatically):"
    print_preprocess_plan train "${train_numbers[@]}"
    print_preprocess_plan val "${val_numbers[@]}"
    confirm "Run all selected preprocessing jobs?" y || return 0
    preprocess_group train "${existing_policy}" "${train_numbers[@]}"
    preprocess_group val "${existing_policy}" "${val_numbers[@]}"
    echo
    echo "[OK] Batch preprocessing finished: train=${#train_numbers[@]}, val=${#val_numbers[@]}"
}

count_sequences()
{
    local root="$1"
    if [[ ! -d "${root}" ]]; then
        echo 0
        return 0
    fi
    find "${root}" -type f -name scans.npy -print 2>/dev/null | wc -l
}

train_interactive()
{
    local train_count val_count architecture_choice architecture epochs batch_size workers device default_device
    local run_name output_dir
    train_count="$(count_sequences "${dataset_root}/train")"
    val_count="$(count_sequences "${dataset_root}/val")"
    if (( train_count == 0 || val_count == 0 )); then
        echo "[ERROR] Training needs at least one train and one val sequence." >&2
        echo "[ERROR] Found train=${train_count}, val=${val_count} under ${dataset_root}" >&2
        return 1
    fi
    echo
    echo "[INFO] dataset=${dataset_root} train_sequences=${train_count} val_sequences=${val_count}"
    IFS= read -r -p "Architecture [normal/small] (default: normal): " architecture_choice
    architecture="${architecture_choice:-normal}"
    [[ "${architecture}" == normal || "${architecture}" == small ]] || {
        echo "[ERROR] Architecture must be normal or small." >&2
        return 2
    }
    IFS= read -r -p "Epochs (default: 100): " epochs
    epochs="${epochs:-100}"
    IFS= read -r -p "Batch size (default: 64): " batch_size
    batch_size="${batch_size:-64}"
    IFS= read -r -p "DataLoader workers (default: 4): " workers
    workers="${workers:-4}"
    if python3 -c 'import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)' 2>/dev/null; then
        default_device=cuda
    else
        default_device=auto
    fi
    IFS= read -r -p "Device [cuda/auto/cpu] (default: ${default_device}): " device
    device="${device:-${default_device}}"

    run_name="$(date +%Y%m%d_%H%M%S)"
    output_dir="${checkpoint_root}/${run_name}"
    echo
    echo "[INFO] checkpoints=${output_dir}"
    python3 "${trainer}" \
        --train-dir "${dataset_root}/train" \
        --val-dir "${dataset_root}/val" \
        --output-dir "${output_dir}" \
        --architecture "${architecture}" \
        --epochs "${epochs}" \
        --batch-size "${batch_size}" \
        --workers "${workers}" \
        --device "${device}"
    ln -sfn "${run_name}" "${checkpoint_root}/latest"
    echo
    echo "[OK] best checkpoint: ${output_dir}/best_model.pth"
    echo "[OK] latest shortcut: ${checkpoint_root}/latest/best_model.pth"
    echo "[OK] Terminator and run_tiny_lidar_player.sh discover it automatically."
}

if [[ "${list_only}" == true ]]; then
    show_status
    exit 0
fi

choose_action
case "${action}" in
1) preprocess_interactive ;;
2) train_interactive ;;
3)
    preprocess_interactive
    train_interactive
    ;;
4) show_status ;;
esac
