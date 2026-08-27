#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
workspace_setup="${script_dir}/../../workspace/install/setup.bash"
preprocessor="${script_dir}/preprocess_bag_to_npy.py"
visualizer="${script_dir}/visualize_dataset.py"
default_record_root="${AIC_RECORD_ROOT:-${script_dir}/../../record}"
default_dataset_root="${AIC_DATASET_ROOT:-${script_dir}/datasets}"
record_root="${default_record_root}"
list_only=false
workspace_sourced=false

rsu_topics="/rsu/curve_01/scan,/rsu/curve_02/scan,/rsu/curve_03/scan,/rsu/curve_04/scan,/rsu/curve_05/scan,/rsu/curve_06/scan"

usage()
{
    cat <<EOF
Usage: $(basename "$0") [--record-root PATH] [--list]

Interactively select a Bag Manager recording, preprocess ego/RSU scans, and
validate or visualize the generated dataset.

Options:
  --record-root PATH  Directory searched recursively for rosbag metadata.yaml
                      (default: ${default_record_root})
  --list              List discovered rosbag sequences and exit
  -h, --help          Show this help

Environment overrides:
  AIC_RECORD_ROOT     Default recording search root
  AIC_DATASET_ROOT    Default output root containing train/ and val/
EOF
}

while (( $# > 0 )); do
    case "$1" in
    --record-root)
        if (( $# < 2 )); then
            echo "[ERROR] --record-root requires a path" >&2
            exit 2
        fi
        record_root="$2"
        shift 2
        ;;
    --list)
        list_only=true
        shift
        ;;
    -h|--help)
        usage
        exit 0
        ;;
    *)
        echo "[ERROR] Unknown argument: $1" >&2
        usage >&2
        exit 2
        ;;
    esac
done

declare -a bag_dirs=()

discover_bags()
{
    local root="$1"
    local metadata
    bag_dirs=()
    if [[ ! -d "${root}" ]]; then
        return 1
    fi
    while IFS= read -r -d '' metadata; do
        bag_dirs+=("${metadata%/metadata.yaml}")
    done < <(find "${root}" -type f -name metadata.yaml -print0 | sort -z)
}

print_bags()
{
    local index
    local bag_dir
    local relative
    local size
    echo
    echo "Available rosbag sequences under: ${record_root}"
    for index in "${!bag_dirs[@]}"; do
        bag_dir="${bag_dirs[index]}"
        relative="${bag_dir#"${record_root%/}/"}"
        size="$(du -sh "${bag_dir}" 2>/dev/null | awk '{print $1}')"
        printf '  %2d) %-55s %8s\n' "$((index + 1))" "${relative}" "${size:-?}"
    done
}

prompt_existing_directory()
{
    local prompt="$1"
    local initial="$2"
    local value
    while true; do
        IFS= read -e -r -i "${initial}" -p "${prompt}" value
        value="${value:-${initial}}"
        if [[ -d "${value}" ]]; then
            printf '%s\n' "${value}"
            return 0
        fi
        echo "Directory does not exist: ${value}" >&2
        initial="${value}"
    done
}

confirm()
{
    local prompt="$1"
    local default_answer="${2:-y}"
    local answer
    local suffix="[Y/n]"
    if [[ "${default_answer}" == "n" ]]; then
        suffix="[y/N]"
    fi
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

choose_split_root()
{
    local choice
    while true; do
        echo
        echo "Dataset destination:"
        echo "  1) train (default)"
        echo "  2) val"
        echo "  3) custom output root"
        IFS= read -r -p "Select [1-3]: " choice
        case "${choice:-1}" in
        1)
            output_root="${default_dataset_root}/train"
            split_name="train"
            return 0
            ;;
        2)
            output_root="${default_dataset_root}/val"
            split_name="val"
            return 0
            ;;
        3)
            IFS= read -e -r -i "${default_dataset_root}" -p "Custom output root: " output_root
            output_root="${output_root:-${default_dataset_root}}"
            split_name="custom"
            return 0
            ;;
        *) echo "Please enter 1, 2, or 3." >&2 ;;
        esac
    done
}

choose_output_name()
{
    local bag_dir="$1"
    local default_name
    local candidate
    default_name="$(basename "$(dirname "${bag_dir}")")_$(basename "${bag_dir}")"
    while true; do
        IFS= read -e -r -i "${default_name}" -p "Sequence name: " candidate
        candidate="${candidate:-${default_name}}"
        if [[ -z "${candidate}" || "${candidate}" == "." || "${candidate}" == ".." || "${candidate}" == */* ]]; then
            echo "Use a non-empty directory name without '/'." >&2
            continue
        fi
        output_dir="${output_root}/${candidate}"
        if [[ -e "${output_dir}" ]]; then
            if confirm "Output exists: ${output_dir}. Overwrite generated .npy files?" n; then
                return 0
            fi
            default_name="${candidate}_new"
            continue
        fi
        return 0
    done
}

source_workspace()
{
    if [[ "${workspace_sourced}" == true ]]; then
        return 0
    fi
    if [[ ! -f "${workspace_setup}" ]]; then
        echo "[ERROR] ROS workspace setup was not found: ${workspace_setup}" >&2
        echo "[ERROR] Build the workspace before preprocessing." >&2
        return 1
    fi
    set +u
    # shellcheck disable=SC1090
    source "${workspace_setup}"
    set -u
    workspace_sourced=true
}

preprocess_sequence()
{
    local bag_dir="$1"
    mkdir -p "${output_root}"
    echo
    echo "[INFO] Bag:       ${bag_dir}"
    echo "[INFO] Split:     ${split_name}"
    echo "[INFO] Output:    ${output_dir}"
    echo "[INFO] Ego scan:  /sensing/lidar/scan"
    echo "[INFO] RSU scans: curve_01 ... curve_06"
    echo "[INFO] Target:    acceleration + steering"
    echo

    if ! python3 "${preprocessor}" \
        --bag "${bag_dir}" \
        --output "${output_dir}" \
        --ego-scan-topic /sensing/lidar/scan \
        --rsu-scan-topics "${rsu_topics}" \
        --control-topic /control/command/control_cmd \
        --target-mode accel_steer; then
        echo "[ERROR] Preprocessing failed. The selected bag was left unchanged." >&2
        return 1
    fi

    echo
    echo "[OK] Preprocessed dataset: ${output_dir}"
    if confirm "Open the dataset visualization now?" y; then
        if [[ -n "${DISPLAY:-}" ]]; then
            python3 "${visualizer}" --dataset "${output_dir}"
        else
            preview="${output_dir}/dataset_check.png"
            echo "[INFO] DISPLAY is unavailable; saving a preview instead."
            python3 "${visualizer}" --dataset "${output_dir}" --save "${preview}" --no-show
        fi
    else
        python3 "${visualizer}" --dataset "${output_dir}" --report-only
    fi
}

if ! discover_bags "${record_root}" || (( ${#bag_dirs[@]} == 0 )); then
    if [[ "${list_only}" == true ]]; then
        echo "[ERROR] No rosbag sequences found under: ${record_root}" >&2
        exit 1
    fi
    echo "No rosbag sequences found under: ${record_root}"
    record_root="$(prompt_existing_directory "Recording search root: " "${record_root}")"
    if ! discover_bags "${record_root}" || (( ${#bag_dirs[@]} == 0 )); then
        echo "[ERROR] No metadata.yaml files found under: ${record_root}" >&2
        exit 1
    fi
fi

print_bags
if [[ "${list_only}" == true ]]; then
    exit 0
fi

while true; do
    echo
    IFS= read -r -p "Select a sequence number [1-${#bag_dirs[@]}] (q: quit): " selection
    if [[ "${selection,,}" == "q" ]]; then
        exit 0
    fi
    if ! [[ "${selection}" =~ ^[0-9]+$ ]] || (( selection < 1 || selection > ${#bag_dirs[@]} )); then
        echo "Please enter a number from 1 to ${#bag_dirs[@]}." >&2
        continue
    fi

    selected_bag="${bag_dirs[selection - 1]}"
    choose_split_root
    choose_output_name "${selected_bag}"
    source_workspace
    preprocess_sequence "${selected_bag}" || true

    if ! confirm "Process another sequence?" n; then
        break
    fi
    discover_bags "${record_root}"
    print_bags
done

echo
echo "Done."
