#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AICHALLENGE_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
REPO_ROOT="$(cd "${AICHALLENGE_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python3}"
TOOLS_DIR="${AICHALLENGE_DIR}/my_workspace/map_creation"
MAP_CLEANUP_TOOL="${TOOLS_DIR}/map_cleanup_editor.py"
CENTERLINE_TOOL="${TOOLS_DIR}/generate_centerline.py"
RACELINE_TOOL="${TOOLS_DIR}/generate_raceline.py"
OPTIMIZER_ROOT="${TOOLS_DIR}/global_racetrajectory_optimization"

MAP_PATH=""
OUTPUT_DIR=""
CLEANED_MAP_PATH=""
CENTERLINE_PATH=""
RACELINE_PATH=""
DIRECTION="forward"
EXPECTED_CENTERLINE_LENGTH_M="0.0"
ALLOW_ANY_LENGTH="false"
SKIP_CLEANUP="false"
SKIP_CENTERLINE="false"
SKIP_RACELINE="false"

usage() {
    printf '%s\n' \
        "Usage: $0 [MAP_PNG_OR_PGM] [options]" \
        "" \
        "Clean a 2D map, generate a centerline, then generate a raceline." \
        "If MAP_PNG_OR_PGM is omitted, an interactive selector is shown." \
        "" \
        "Options:" \
        "  --map PATH                         Input map image (.png/.pgm)" \
        "  --output-dir DIR                   Output directory (default: <map_dir>/lines)" \
        "  --cleaned-map PATH                 Cleaned map output/input path" \
        "  --centerline PATH                  Centerline CSV output/input path" \
        "  --raceline PATH                    Raceline CSV output path" \
        "  --direction NAME                   forward, reverse, or both for raceline (default: forward)" \
        "  --expected-centerline-length-m M   Optional lap-length filter for centerline extraction" \
        "  --allow-any-length                 Disable expected-length filtering in centerline extraction" \
        "  --skip-cleanup                     Use the selected map directly for centerline generation" \
        "  --skip-centerline                  Use --centerline or choose an existing centerline CSV" \
        "  --skip-raceline                    Stop after cleanup/centerline generation" \
        "  -h, --help                         Show this help" \
        "" \
        "Example:" \
        "  $0 /aichallenge/outputs/map/run_001/map.png" \
        "  $0 --skip-cleanup /aichallenge/outputs/map/run_001/lines/cleaned_map.png" \
        "  $0 --skip-centerline --centerline ./centerline.csv"
}

log() {
    echo "[generate_line] $*"
}

die() {
    echo "[generate_line][ERROR] $*" >&2
    exit 1
}

require_file() {
    local path="$1"
    local label="$2"
    [ -f "${path}" ] || die "${label} not found: ${path}"
}

absolute_path() {
    local path="$1"
    if [ -d "${path}" ]; then
        (cd "${path}" && pwd)
        return 0
    fi

    local dir
    local base
    dir="$(cd "$(dirname "${path}")" && pwd)"
    base="$(basename "${path}")"
    printf '%s/%s\n' "${dir}" "${base}"
}

strip_image_extension() {
    local path="$1"
    path="${path%.png}"
    path="${path%.pgm}"
    path="${path%.PNG}"
    path="${path%.PGM}"
    printf '%s\n' "${path}"
}

collect_map_candidates() {
    local root
    for root in \
        "${AICHALLENGE_DIR}/outputs/map" \
        "${REPO_ROOT}/outputs/map" \
        "${REPO_ROOT}/output"; do
        [ -d "${root}" ] || continue
        find "${root}" \
            \( -path '*/.git' -o -path '*/build' -o -path '*/install' \) -prune \
            -o -type f \( -name 'map.png' -o -name 'cleaned_map.png' -o -name '*.pgm' \) \
            -print 2>/dev/null
    done | awk '!seen[$0]++' | sort
}

collect_centerline_candidates() {
    local root
    for root in \
        "${AICHALLENGE_DIR}/outputs/map" \
        "${REPO_ROOT}/outputs/map" \
        "${REPO_ROOT}/output"; do
        [ -d "${root}" ] || continue
        find "${root}" \
            \( -path '*/.git' -o -path '*/build' -o -path '*/install' \) -prune \
            -o -type f \( -name 'centerline*.csv' -o -name '*centerline*.csv' \) \
            -print 2>/dev/null
    done | awk '!seen[$0]++' | sort
}

select_path_interactively() {
    local title="$1"
    local collector="$2"
    local candidates=()
    local candidate
    local answer
    local i
    local max_items=40

    if ! [ -r /dev/tty ]; then
        die "${title} is required when no interactive TTY is available."
    fi

    while IFS= read -r candidate; do
        [ -n "${candidate}" ] || continue
        candidates+=("${candidate}")
    done < <("${collector}")

    printf '\n[generate_line] %s を選択してください。\n' "${title}" >/dev/tty
    if [ "${#candidates[@]}" -gt 0 ]; then
        for ((i = 0; i < ${#candidates[@]} && i < max_items; i++)); do
            printf '  %2d) %s\n' "$((i + 1))" "${candidates[$i]}" >/dev/tty
        done
        if [ "${#candidates[@]}" -gt "${max_items}" ]; then
            printf '  ... (%d件中、先頭%d件だけ表示)\n' "${#candidates[@]}" "${max_items}" >/dev/tty
        fi
        printf '番号、または直接パスを入力してください [1]: ' >/dev/tty
    else
        printf '候補が見つかりませんでした。直接パスを入力してください: ' >/dev/tty
    fi

    while IFS= read -r answer </dev/tty; do
        if [ -z "${answer}" ] && [ "${#candidates[@]}" -gt 0 ]; then
            printf '%s\n' "${candidates[0]}"
            return 0
        fi
        if [[ "${answer}" =~ ^[0-9]+$ ]] && [ "${#candidates[@]}" -gt 0 ]; then
            if [ "${answer}" -ge 1 ] && [ "${answer}" -le "${#candidates[@]}" ]; then
                printf '%s\n' "${candidates[$((answer - 1))]}"
                return 0
            fi
            printf '範囲外です。番号、または直接パスを入力してください: ' >/dev/tty
            continue
        fi
        if [ -n "${answer}" ]; then
            printf '%s\n' "${answer}"
            return 0
        fi
        printf 'パスを入力してください: ' >/dev/tty
    done

    die "Failed to read ${title} from TTY."
}

infer_yaml_for_image() {
    local image_path="$1"
    local root
    local candidate
    root="$(strip_image_extension "${image_path}")"
    candidate="${root}.yaml"
    if [ -f "${candidate}" ]; then
        printf '%s\n' "${candidate}"
        return 0
    fi

    candidate="$(dirname "${image_path}")/map.yaml"
    if [ -f "${candidate}" ]; then
        printf '%s\n' "${candidate}"
        return 0
    fi

    candidate="$(dirname "$(dirname "${image_path}")")/map.yaml"
    if [ -f "${candidate}" ]; then
        printf '%s\n' "${candidate}"
        return 0
    fi

    printf '%s\n' ""
}

write_yaml_for_image() {
    local source_yaml="$1"
    local output_yaml="$2"
    local image_name="$3"

    [ -n "${source_yaml}" ] || return 0
    [ -f "${source_yaml}" ] || return 0

    "${PYTHON_BIN}" - "${source_yaml}" "${output_yaml}" "${image_name}" <<'PY'
from pathlib import Path
import re
import sys

source = Path(sys.argv[1])
output = Path(sys.argv[2])
image_name = sys.argv[3]

text = source.read_text(encoding="utf-8")
line = f"image: {image_name}"
if re.search(r"(?m)^image\s*:", text):
    text = re.sub(r"(?m)^image\s*:.*$", line, text, count=1)
else:
    text = line + "\n" + text
output.write_text(text, encoding="utf-8")
PY
}

while [ "$#" -gt 0 ]; do
    case "$1" in
    -h | --help)
        usage
        exit 0
        ;;
    --map)
        MAP_PATH="${2:?missing value for --map}"
        shift 2
        ;;
    --output-dir)
        OUTPUT_DIR="${2:?missing value for --output-dir}"
        shift 2
        ;;
    --cleaned-map)
        CLEANED_MAP_PATH="${2:?missing value for --cleaned-map}"
        shift 2
        ;;
    --centerline)
        CENTERLINE_PATH="${2:?missing value for --centerline}"
        shift 2
        ;;
    --raceline)
        RACELINE_PATH="${2:?missing value for --raceline}"
        shift 2
        ;;
    --direction)
        DIRECTION="${2:?missing value for --direction}"
        shift 2
        ;;
    --expected-centerline-length-m)
        EXPECTED_CENTERLINE_LENGTH_M="${2:?missing value for --expected-centerline-length-m}"
        shift 2
        ;;
    --allow-any-length)
        ALLOW_ANY_LENGTH="true"
        shift
        ;;
    --skip-cleanup)
        SKIP_CLEANUP="true"
        shift
        ;;
    --skip-centerline)
        SKIP_CENTERLINE="true"
        shift
        ;;
    --skip-raceline)
        SKIP_RACELINE="true"
        shift
        ;;
    --*)
        die "Unknown option: $1"
        ;;
    *)
        if [ -n "${MAP_PATH}" ]; then
            die "Only one MAP_PNG_OR_PGM can be provided"
        fi
        MAP_PATH="$1"
        shift
        ;;
    esac
done

case "${DIRECTION}" in
forward | reverse | both) ;;
*) die "Invalid --direction: ${DIRECTION}" ;;
esac

if [ "${SKIP_CENTERLINE}" = "true" ]; then
    SKIP_CLEANUP="true"
fi

require_file "${MAP_CLEANUP_TOOL}" "map cleanup tool"
require_file "${CENTERLINE_TOOL}" "centerline tool"
require_file "${RACELINE_TOOL}" "raceline tool"

if [ "${SKIP_CENTERLINE}" = "true" ] && [ -z "${CENTERLINE_PATH}" ]; then
    CENTERLINE_PATH="$(select_path_interactively "centerline CSV" collect_centerline_candidates)"
fi

if [ -z "${MAP_PATH}" ] && [ "${SKIP_CENTERLINE}" != "true" ]; then
    MAP_PATH="$(select_path_interactively "2D map image" collect_map_candidates)"
fi

if [ -n "${MAP_PATH}" ]; then
    [ -f "${MAP_PATH}" ] || die "Map image not found: ${MAP_PATH}"
    MAP_PATH="$(absolute_path "${MAP_PATH}")"
fi

if [ -n "${CENTERLINE_PATH}" ] && [ "${SKIP_CENTERLINE}" = "true" ]; then
    [ -f "${CENTERLINE_PATH}" ] || die "Centerline CSV not found: ${CENTERLINE_PATH}"
    CENTERLINE_PATH="$(absolute_path "${CENTERLINE_PATH}")"
fi

if [ -z "${OUTPUT_DIR}" ]; then
    if [ -n "${MAP_PATH}" ]; then
        if [ "$(basename "$(dirname "${MAP_PATH}")")" = "lines" ]; then
            OUTPUT_DIR="$(dirname "${MAP_PATH}")"
        else
            OUTPUT_DIR="$(dirname "${MAP_PATH}")/lines"
        fi
    elif [ -n "${CENTERLINE_PATH}" ]; then
        OUTPUT_DIR="$(dirname "${CENTERLINE_PATH}")"
    else
        OUTPUT_DIR="${AICHALLENGE_DIR}/outputs/map/lines_$(date +%Y%m%d-%H%M%S)"
    fi
fi
mkdir -p "${OUTPUT_DIR}"
OUTPUT_DIR="$(absolute_path "${OUTPUT_DIR}")"

if [ -z "${CLEANED_MAP_PATH}" ]; then
    CLEANED_MAP_PATH="${OUTPUT_DIR}/cleaned_map.png"
fi
if [ -z "${CENTERLINE_PATH}" ]; then
    CENTERLINE_PATH="${OUTPUT_DIR}/centerline.csv"
fi
if [ -z "${RACELINE_PATH}" ]; then
    RACELINE_PATH="${OUTPUT_DIR}/raceline.csv"
fi

mkdir -p "$(dirname "${CLEANED_MAP_PATH}")" "$(dirname "${CENTERLINE_PATH}")" "$(dirname "${RACELINE_PATH}")"
CLEANED_MAP_PATH="$(absolute_path "${CLEANED_MAP_PATH}")"
CENTERLINE_PATH="$(absolute_path "${CENTERLINE_PATH}")"
RACELINE_PATH="$(absolute_path "${RACELINE_PATH}")"

SOURCE_YAML=""
if [ -n "${MAP_PATH}" ]; then
    SOURCE_YAML="$(infer_yaml_for_image "${MAP_PATH}")"
fi

if [ "${SKIP_CLEANUP}" = "true" ]; then
    CENTERLINE_MAP="${MAP_PATH}"
    CENTERLINE_YAML="${SOURCE_YAML}"
else
    log "Opening cleanup editor"
    log "  input : ${MAP_PATH}"
    log "  output: ${CLEANED_MAP_PATH}"
    "${PYTHON_BIN}" "${MAP_CLEANUP_TOOL}" \
        --input "${MAP_PATH}" \
        --output "${CLEANED_MAP_PATH}"

    [ -f "${CLEANED_MAP_PATH}" ] || die "Cleaned map was not saved. In the editor, press 's' before closing."

    CLEANED_YAML="$(strip_image_extension "${CLEANED_MAP_PATH}").yaml"
    write_yaml_for_image "${SOURCE_YAML}" "${CLEANED_YAML}" "$(basename "${CLEANED_MAP_PATH}")"
    CENTERLINE_MAP="${CLEANED_MAP_PATH}"
    CENTERLINE_YAML="${CLEANED_YAML}"
fi

if [ "${SKIP_CENTERLINE}" != "true" ]; then
    CENTERLINE_DEBUG_DIR="${OUTPUT_DIR}/centerline_debug"
    mkdir -p "${CENTERLINE_DEBUG_DIR}"

    log "Generating centerline"
    log "  map   : ${CENTERLINE_MAP}"
    log "  output: ${CENTERLINE_PATH}"
    declare -a CENTERLINE_CMD=(
        "${PYTHON_BIN}" "${CENTERLINE_TOOL}"
        --map "${CENTERLINE_MAP}"
        --output "${CENTERLINE_PATH}"
        --yaml "${CENTERLINE_YAML}"
        --expected-centerline-length-m "${EXPECTED_CENTERLINE_LENGTH_M}"
        --debug-dir "${CENTERLINE_DEBUG_DIR}"
    )
    if [ "${ALLOW_ANY_LENGTH}" = "true" ]; then
        CENTERLINE_CMD+=(--allow-any-length)
    fi
    "${CENTERLINE_CMD[@]}"
fi

[ -f "${CENTERLINE_PATH}" ] || die "Centerline CSV not found: ${CENTERLINE_PATH}"

if [ "${SKIP_RACELINE}" != "true" ]; then
    log "Generating raceline"
    log "  centerline: ${CENTERLINE_PATH}"
    log "  output    : ${RACELINE_PATH}"
    "${PYTHON_BIN}" "${RACELINE_TOOL}" \
        --centerline "${CENTERLINE_PATH}" \
        --output "${RACELINE_PATH}" \
        --direction "${DIRECTION}" \
        --optimizer-root "${OPTIMIZER_ROOT}" \
        --show-progress
fi

log "Done: ${OUTPUT_DIR}"
if [ -f "${CLEANED_MAP_PATH}" ]; then
    log "Cleaned map: ${CLEANED_MAP_PATH}"
fi
log "Centerline : ${CENTERLINE_PATH}"
if [ -f "${RACELINE_PATH}" ]; then
    log "Raceline   : ${RACELINE_PATH}"
fi
