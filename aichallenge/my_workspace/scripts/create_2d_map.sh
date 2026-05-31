#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AICHALLENGE_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

BAG_PATH=""
SCAN_TOPIC="/scan"
MAP_TOPIC="/map"
OUTPUT_ROOT="${AICHALLENGE_DIR}/outputs/map"
RUN_NAME=""
CONFIG_DIR="${AICHALLENGE_DIR}/my_workspace/cartographer"
CONFIG_BASENAME="racing_kart_2d.lua"
MAP_RESOLUTION="0.05"
PLAY_RATE="1.0"
STARTUP_WAIT_SEC="2.0"
SETTLE_SEC="2.0"
SAVE_TIMEOUT_SEC="30.0"
LASER_FRAME=""
STATIC_TF_PARENT="base_link"

declare -a CHILD_PIDS=()
declare -a CHILD_NAMES=()
CLEANUP_STARTED=0
BAG_PID=""
LAST_STARTED_PID=""

usage() {
    printf '%s\n' \
        "Usage: $0 [ROSBAG_PATH] [options]" \
        "" \
        "Create a 2D occupancy grid map from a ROS 2 bag with Cartographer." \
        "If ROSBAG_PATH is omitted, an interactive selector is shown." \
        "" \
        "Options:" \
        "  --scan-topic TOPIC       LaserScan topic in the rosbag (default: /scan)" \
        "  --map-topic TOPIC        OccupancyGrid topic to save (default: /map)" \
        "  --output-root DIR        Output root dir (default: ${OUTPUT_ROOT})" \
        "  --name NAME              Output dir name (default: <bag_name>_<timestamp>)" \
        "  --config-dir DIR         Cartographer Lua config directory" \
        "  --config-basename FILE   Cartographer Lua config file (default: racing_kart_2d.lua)" \
        "  --resolution METERS      Saved map resolution (default: 0.05)" \
        "  --play-rate RATE         rosbag playback rate (default: 1.0)" \
        "  --startup-wait SEC       Wait after starting Cartographer before playback (default: 2.0)" \
        "  --settle SEC             Wait after playback before saving map (default: 2.0)" \
        "  --save-timeout SEC       map_saver timeout (default: 30.0)" \
        "  --laser-frame FRAME      Publish identity static TF: <parent> -> FRAME" \
        "  --static-tf-parent FRAME Parent frame for --laser-frame (default: base_link)" \
        "  -h, --help               Show this help" \
        "" \
        "Example:" \
        "  $0 /aichallenge/outputs/latest/d1/rosbag2_autoware --scan-topic /scan" \
        "  $0 ./bag.mcap --scan-topic /sensing/lidar/scan --laser-frame laser"
}

die() {
    echo "[create_2d_map][ERROR] $*" >&2
    exit 1
}

log() {
    echo "[create_2d_map] $*"
}

source_first_existing_setup() {
    local setup_file
    for setup_file in \
        "/aichallenge/workspace/install/setup.bash" \
        "${AICHALLENGE_DIR}/workspace/install/setup.bash" \
        "/autoware/install/setup.bash" \
        "/opt/ros/humble/setup.bash"; do
        if [ -f "${setup_file}" ]; then
            # Some ROS setup files reference unset variables.
            set +u
            # shellcheck disable=SC1090
            source "${setup_file}"
            set -u
            log "Sourced ROS setup: ${setup_file}"
            return 0
        fi
    done
    return 0
}

require_command() {
    local command_name="$1"
    if ! command -v "${command_name}" >/dev/null 2>&1; then
        die "Required command not found: ${command_name}"
    fi
}

require_ros_package() {
    local package_name="$1"
    local install_hint="$2"
    if ! ros2 pkg prefix "${package_name}" >/dev/null 2>&1; then
        die "ROS package not found: ${package_name}. Install ${install_hint} and rebuild the container."
    fi
}

repo_root() {
    (cd "${AICHALLENGE_DIR}/.." && pwd)
}

collect_rosbag_candidates() {
    local root
    local metadata_path
    local repo_dir
    repo_dir="$(repo_root)"

    for root in \
        "${AICHALLENGE_DIR}/outputs" \
        "${AICHALLENGE_DIR}/ml_workspace" \
        "${repo_dir}/outputs" \
        "${repo_dir}/output" \
        "${repo_dir}/log"; do
        [ -d "${root}" ] || continue
        find "${root}" \
            \( -path '*/.git' -o -path '*/build' -o -path '*/install' -o -path '*/log/latest' \) -prune \
            -o -type f \
            \( -name '*.mcap' -o -name '*.mcap.zstd' -o -name '*.db3' -o -name '*.db3.zstd' -o -name '*.sqlite3' -o -name '*.sqlite3.zstd' \) \
            -print 2>/dev/null
        while IFS= read -r metadata_path; do
            dirname "${metadata_path}"
        done < <(
            find "${root}" \
                \( -path '*/.git' -o -path '*/build' -o -path '*/install' -o -path '*/log/latest' \) -prune \
                -o -type f -name metadata.yaml -print 2>/dev/null
        )
    done | awk '!seen[$0]++' | sort
}

select_rosbag_interactively() {
    local candidates=()
    local candidate
    local answer
    local i
    local max_items=40

    if ! [ -r /dev/tty ]; then
        die "ROSBAG_PATH is required when no interactive TTY is available."
    fi

    while IFS= read -r candidate; do
        [ -n "${candidate}" ] || continue
        candidates+=("${candidate}")
    done < <(collect_rosbag_candidates)

    printf '\n[create_2d_map] ROS bag を選択してください。\n' >/dev/tty
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

    die "Failed to read ROSBAG_PATH from TTY."
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

strip_known_bag_extensions() {
    local name="$1"
    name="${name%.mcap.zstd}"
    name="${name%.db3.zstd}"
    name="${name%.sqlite3.zstd}"
    name="${name%.mcap}"
    name="${name%.db3}"
    name="${name%.sqlite3}"
    printf '%s\n' "${name}"
}

sanitize_name() {
    local name="$1"
    name="$(printf '%s' "${name}" | tr -c 'A-Za-z0-9._-' '_' | sed 's/^_*//; s/_*$//')"
    if [ -z "${name}" ]; then
        name="rosbag"
    fi
    printf '%s\n' "${name}"
}

process_alive() {
    local pid="$1"
    local state

    kill -0 "${pid}" 2>/dev/null || return 1
    state="$(ps -o stat= -p "${pid}" 2>/dev/null || true)"
    [ -n "${state}" ] || return 1
    case "${state}" in
    *Z*)
        return 1
        ;;
    esac
    return 0
}

stop_child_index() {
    local idx="$1"
    local pid="${CHILD_PIDS[${idx}]:-}"
    local name="${CHILD_NAMES[${idx}]:-process}"
    local i

    if [ -z "${pid}" ]; then
        return 0
    fi

    if process_alive "${pid}"; then
        log "Stopping ${name} (PID/PGID=${pid})"
        kill -INT -- "-${pid}" 2>/dev/null || kill -INT "${pid}" 2>/dev/null || true
        for ((i = 0; i < 20; i++)); do
            process_alive "${pid}" || break
            sleep 0.5
        done
    fi

    if process_alive "${pid}"; then
        log "${name} did not stop with SIGINT; sending SIGTERM"
        kill -TERM -- "-${pid}" 2>/dev/null || kill -TERM "${pid}" 2>/dev/null || true
        for ((i = 0; i < 10; i++)); do
            process_alive "${pid}" || break
            sleep 0.5
        done
    fi

    if process_alive "${pid}"; then
        log "${name} did not stop with SIGTERM; sending SIGKILL"
        kill -KILL -- "-${pid}" 2>/dev/null || kill -KILL "${pid}" 2>/dev/null || true
    fi

    wait "${pid}" 2>/dev/null || true
    CHILD_PIDS[${idx}]=""
    CHILD_NAMES[${idx}]=""
}

stop_all_children() {
    local idx
    for ((idx = ${#CHILD_PIDS[@]} - 1; idx >= 0; idx--)); do
        stop_child_index "${idx}"
    done
}

cleanup() {
    local rc=$?
    if [ "${CLEANUP_STARTED}" = "1" ]; then
        exit "${rc}"
    fi

    CLEANUP_STARTED=1
    trap - EXIT INT TERM
    stop_all_children
    exit "${rc}"
}

on_interrupt() {
    echo ""
    log "Interrupted; cleaning up child processes"
    exit 130
}

on_term() {
    log "Terminated; cleaning up child processes"
    exit 143
}

start_process() {
    local name="$1"
    local log_file="$2"
    shift 2
    local pid

    log "Starting ${name}: $*" >&2
    log "  log: ${log_file}" >&2

    if command -v setsid >/dev/null 2>&1; then
        setsid "$@" >"${log_file}" 2>&1 &
    else
        "$@" >"${log_file}" 2>&1 &
    fi

    pid=$!
    CHILD_PIDS+=("${pid}")
    CHILD_NAMES+=("${name}")
    LAST_STARTED_PID="${pid}"
    log "  pid: ${pid}" >&2
}

wait_for_startup() {
    local idx
    sleep "${STARTUP_WAIT_SEC}"
    for ((idx = 0; idx < ${#CHILD_PIDS[@]}; idx++)); do
        local pid="${CHILD_PIDS[${idx}]:-}"
        local name="${CHILD_NAMES[${idx}]:-process}"
        if [ -n "${pid}" ] && ! process_alive "${pid}"; then
            wait "${pid}" 2>/dev/null || true
            die "${name} exited during startup. See logs under ${OUTPUT_DIR}."
        fi
    done
}

wait_for_rosbag_and_monitor() {
    local idx
    local pid
    local name
    local rc

    while process_alive "${BAG_PID}"; do
        for ((idx = 0; idx < ${#CHILD_PIDS[@]}; idx++)); do
            pid="${CHILD_PIDS[${idx}]:-}"
            name="${CHILD_NAMES[${idx}]:-process}"
            [ -n "${pid}" ] || continue
            [ "${pid}" != "${BAG_PID}" ] || continue
            if ! process_alive "${pid}"; then
                set +e
                wait "${pid}"
                rc=$?
                set -e
                CHILD_PIDS[${idx}]=""
                die "${name} exited before rosbag playback finished (exit=${rc}). See logs under ${OUTPUT_DIR}."
            fi
        done
        sleep 1
    done

    set +e
    wait "${BAG_PID}"
    rc=$?
    set -e

    if [ "${rc}" -ne 0 ]; then
        die "rosbag playback failed or was interrupted (exit=${rc}). See ${OUTPUT_DIR}/rosbag_play.log."
    fi
}

storage_arg_for_bag() {
    local path="$1"
    case "${path}" in
    *.mcap | *.mcap.zstd)
        printf '%s\n' "mcap"
        ;;
    *.db3 | *.db3.zstd | *.sqlite3 | *.sqlite3.zstd)
        printf '%s\n' "sqlite3"
        ;;
    *)
        printf '%s\n' ""
        ;;
    esac
}

write_run_info() {
    {
        printf 'bag_path: %s\n' "${BAG_PATH}"
        printf 'scan_topic: %s\n' "${SCAN_TOPIC}"
        printf 'map_topic: %s\n' "${MAP_TOPIC}"
        printf 'config: %s/%s\n' "${CONFIG_DIR}" "${CONFIG_BASENAME}"
        printf 'map_resolution: %s\n' "${MAP_RESOLUTION}"
        printf 'play_rate: %s\n' "${PLAY_RATE}"
        printf 'output_dir: %s\n' "${OUTPUT_DIR}"
    } >"${OUTPUT_DIR}/run_info.txt"
}

convert_pgm_to_png_and_rewrite_yaml() {
    local pgm_path="$1"
    local png_path="$2"
    local yaml_path="$3"

    python3 - "${pgm_path}" "${png_path}" "${yaml_path}" <<'PY'
from pathlib import Path
import binascii
import re
import struct
import sys
import zlib

pgm_path = Path(sys.argv[1])
png_path = Path(sys.argv[2])
yaml_path = Path(sys.argv[3])

def read_token(fp):
    token = bytearray()
    while True:
        ch = fp.read(1)
        if not ch:
            raise ValueError("unexpected EOF while reading PGM header")
        if ch == b"#":
            fp.readline()
            continue
        if ch.isspace():
            continue
        token.extend(ch)
        break
    while True:
        ch = fp.read(1)
        if not ch or ch.isspace():
            break
        token.extend(ch)
    return bytes(token)

with pgm_path.open("rb") as fp:
    magic = read_token(fp)
    width = int(read_token(fp))
    height = int(read_token(fp))
    max_value = int(read_token(fp))
    if width <= 0 or height <= 0:
        raise ValueError(f"invalid PGM size: {width}x{height}")
    if max_value <= 0:
        raise ValueError(f"invalid PGM max value: {max_value}")

    pixel_count = width * height
    if magic == b"P5":
        if max_value < 256:
            pixels = fp.read(pixel_count)
            if len(pixels) != pixel_count:
                raise ValueError("PGM raster is shorter than expected")
        else:
            raw = fp.read(pixel_count * 2)
            if len(raw) != pixel_count * 2:
                raise ValueError("16-bit PGM raster is shorter than expected")
            pixels = bytes(round(int.from_bytes(raw[i:i + 2], "big") * 255 / max_value) for i in range(0, len(raw), 2))
    elif magic == b"P2":
        rest = fp.read().split()
        if len(rest) < pixel_count:
            raise ValueError("ASCII PGM raster is shorter than expected")
        pixels = bytes(round(int(value) * 255 / max_value) for value in rest[:pixel_count])
    else:
        raise ValueError(f"unsupported PGM magic: {magic!r}")

def png_chunk(kind, data):
    return (
        struct.pack(">I", len(data))
        + kind
        + data
        + struct.pack(">I", binascii.crc32(kind + data) & 0xFFFFFFFF)
    )

rows = bytearray()
for y in range(height):
    start = y * width
    rows.append(0)
    rows.extend(pixels[start:start + width])

png_data = bytearray(b"\x89PNG\r\n\x1a\n")
png_data.extend(png_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0)))
png_data.extend(png_chunk(b"IDAT", zlib.compress(bytes(rows), level=9)))
png_data.extend(png_chunk(b"IEND", b""))
png_path.write_bytes(bytes(png_data))

yaml_text = yaml_path.read_text(encoding="utf-8")
image_line = f"image: {png_path.name}"
if re.search(r"(?m)^image\s*:", yaml_text):
    yaml_text = re.sub(r"(?m)^image\s*:.*$", image_line, yaml_text, count=1)
else:
    yaml_text = image_line + "\n" + yaml_text
yaml_path.write_text(yaml_text, encoding="utf-8")
PY
}

while [ "$#" -gt 0 ]; do
    case "$1" in
    -h | --help)
        usage
        exit 0
        ;;
    --scan-topic)
        SCAN_TOPIC="${2:?missing value for --scan-topic}"
        shift 2
        ;;
    --map-topic)
        MAP_TOPIC="${2:?missing value for --map-topic}"
        shift 2
        ;;
    --output-root)
        OUTPUT_ROOT="${2:?missing value for --output-root}"
        shift 2
        ;;
    --name)
        RUN_NAME="${2:?missing value for --name}"
        shift 2
        ;;
    --config-dir)
        CONFIG_DIR="${2:?missing value for --config-dir}"
        shift 2
        ;;
    --config-basename)
        CONFIG_BASENAME="${2:?missing value for --config-basename}"
        shift 2
        ;;
    --resolution)
        MAP_RESOLUTION="${2:?missing value for --resolution}"
        shift 2
        ;;
    --play-rate)
        PLAY_RATE="${2:?missing value for --play-rate}"
        shift 2
        ;;
    --startup-wait)
        STARTUP_WAIT_SEC="${2:?missing value for --startup-wait}"
        shift 2
        ;;
    --settle)
        SETTLE_SEC="${2:?missing value for --settle}"
        shift 2
        ;;
    --save-timeout)
        SAVE_TIMEOUT_SEC="${2:?missing value for --save-timeout}"
        shift 2
        ;;
    --laser-frame)
        LASER_FRAME="${2:?missing value for --laser-frame}"
        shift 2
        ;;
    --static-tf-parent)
        STATIC_TF_PARENT="${2:?missing value for --static-tf-parent}"
        shift 2
        ;;
    --*)
        die "Unknown option: $1"
        ;;
    *)
        if [ -n "${BAG_PATH}" ]; then
            die "Only one ROSBAG_PATH can be provided"
        fi
        BAG_PATH="$1"
        shift
        ;;
    esac
done

if [ -z "${BAG_PATH}" ]; then
    BAG_PATH="$(select_rosbag_interactively)"
fi

[ -e "${BAG_PATH}" ] || die "ROSBAG_PATH does not exist: ${BAG_PATH}"
BAG_PATH="$(absolute_path "${BAG_PATH}")"
CONFIG_DIR="$(absolute_path "${CONFIG_DIR}")"
[ -f "${CONFIG_DIR}/${CONFIG_BASENAME}" ] || die "Cartographer config not found: ${CONFIG_DIR}/${CONFIG_BASENAME}"

mkdir -p "${OUTPUT_ROOT}"
OUTPUT_ROOT="$(absolute_path "${OUTPUT_ROOT}")"

if [ -z "${RUN_NAME}" ]; then
    bag_base="$(strip_known_bag_extensions "$(basename "${BAG_PATH}")")"
    RUN_NAME="$(sanitize_name "${bag_base}")_$(date +%Y%m%d-%H%M%S)"
fi
RUN_NAME="$(sanitize_name "${RUN_NAME}")"
OUTPUT_DIR="${OUTPUT_ROOT}/${RUN_NAME}"
mkdir -p "${OUTPUT_DIR}"

trap cleanup EXIT
trap on_interrupt INT
trap on_term TERM

source_first_existing_setup
require_command ros2
require_command python3
require_ros_package "cartographer_ros" "ros-humble-cartographer-ros"
require_ros_package "nav2_map_server" "ros-humble-nav2-map-server"
if [ -n "${LASER_FRAME}" ]; then
    require_ros_package "tf2_ros" "ros-humble-tf2-ros"
fi

cp "${CONFIG_DIR}/${CONFIG_BASENAME}" "${OUTPUT_DIR}/${CONFIG_BASENAME}"
write_run_info

log "Output directory: ${OUTPUT_DIR}"

if [ -n "${LASER_FRAME}" ]; then
    start_process \
        "static_tf" \
        "${OUTPUT_DIR}/static_tf.log" \
        ros2 run tf2_ros static_transform_publisher \
        0 0 0 0 0 0 "${STATIC_TF_PARENT}" "${LASER_FRAME}" >/dev/null
fi

start_process \
    "cartographer_node" \
    "${OUTPUT_DIR}/cartographer_node.log" \
    ros2 run cartographer_ros cartographer_node \
    -configuration_directory "${CONFIG_DIR}" \
    -configuration_basename "${CONFIG_BASENAME}" \
    --ros-args \
    -p use_sim_time:=true \
    -r "scan:=${SCAN_TOPIC}" >/dev/null

start_process \
    "cartographer_occupancy_grid_node" \
    "${OUTPUT_DIR}/cartographer_occupancy_grid_node.log" \
    ros2 run cartographer_ros cartographer_occupancy_grid_node \
    -resolution "${MAP_RESOLUTION}" \
    -publish_period_sec 1.0 \
    --ros-args \
    -p use_sim_time:=true >/dev/null

wait_for_startup

declare -a BAG_PLAY_CMD=(ros2 bag play "${BAG_PATH}" --clock --rate "${PLAY_RATE}")
storage_id="$(storage_arg_for_bag "${BAG_PATH}")"
if [ -n "${storage_id}" ]; then
    BAG_PLAY_CMD+=(-s "${storage_id}")
fi

start_process "rosbag_play" "${OUTPUT_DIR}/rosbag_play.log" "${BAG_PLAY_CMD[@]}"
BAG_PID="${LAST_STARTED_PID}"
wait_for_rosbag_and_monitor

log "Rosbag playback finished; waiting ${SETTLE_SEC}s for the last map update"
sleep "${SETTLE_SEC}"

MAP_PREFIX="${OUTPUT_DIR}/map"
log "Saving map to ${MAP_PREFIX}.{pgm,yaml}"
ros2 run nav2_map_server map_saver_cli \
    -f "${MAP_PREFIX}" \
    -t "${MAP_TOPIC}" \
    --fmt pgm \
    --mode trinary \
    --occ 0.65 \
    --free 0.25 \
    --ros-args \
    -p "save_map_timeout:=${SAVE_TIMEOUT_SEC}" \
    -p "map_subscribe_transient_local:=true"

[ -f "${MAP_PREFIX}.pgm" ] || die "Map PGM was not created: ${MAP_PREFIX}.pgm"
[ -f "${MAP_PREFIX}.yaml" ] || die "Map YAML was not created: ${MAP_PREFIX}.yaml"

log "Converting PGM to PNG and rewriting YAML image path"
convert_pgm_to_png_and_rewrite_yaml "${MAP_PREFIX}.pgm" "${MAP_PREFIX}.png" "${MAP_PREFIX}.yaml"

stop_all_children
CHILD_PIDS=()
CHILD_NAMES=()

log "Done: ${OUTPUT_DIR}"
log "Map YAML: ${MAP_PREFIX}.yaml"
log "Map PNG: ${MAP_PREFIX}.png"
