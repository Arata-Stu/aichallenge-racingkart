#!/usr/bin/env bash

set -euo pipefail

usage()
{
    cat <<'EOF'
Usage: run_data_collection_dashboard.sh [tiny|rsu]

Without an argument, select the collection dashboard by number.
  1) TinyLiDAR Studio (recommended for collection)
  2) RSU Fusion Studio
EOF
}

if (( $# > 1 )); then
    usage >&2
    exit 2
fi

dashboard="${1:-}"
if [[ -z "${dashboard}" ]]; then
    echo "Data collection dashboard:"
    echo "  1) TinyLiDAR Studio (recommended) · http://localhost:8765"
    echo "  2) RSU Fusion Studio             · http://localhost:8766"
    while true; do
        if ! IFS= read -r -p "Select [1-2] (default: 1): " selection; then
            echo >&2
            echo "[ERROR] Failed to read the dashboard selection." >&2
            exit 1
        fi
        case "${selection:-1}" in
        1) dashboard=tiny; break ;;
        2) dashboard=rsu; break ;;
        *) echo "Please enter 1 or 2." >&2 ;;
        esac
    done
fi

case "${dashboard}" in
tiny)
    relative_script="ml_workspace/tiny_lidar_net_pytorch/run_dashboard.bash"
    make_target="tiny-lidar-dashboard"
    ;;
rsu)
    relative_script="ml_workspace/rsu_fusion_net/run_dashboard.bash"
    make_target="rsu-fusion-dashboard"
    ;;
-h|--help)
    usage
    exit 0
    ;;
*)
    echo "[ERROR] Unknown dashboard: ${dashboard}" >&2
    usage >&2
    exit 2
    ;;
esac

if [[ -f "/aichallenge/${relative_script}" ]]; then
    echo "[INFO] Starting ${dashboard} collection dashboard inside Docker."
    exec bash "/aichallenge/${relative_script}"
fi

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repository_root="$(cd -- "${script_dir}/../.." && pwd)"
if [[ ! -f "${repository_root}/Makefile" ]]; then
    echo "[ERROR] Could not find the repository Makefile from ${script_dir}." >&2
    exit 3
fi

echo "[INFO] Starting ${dashboard} collection dashboard through Docker Compose."
exec make -C "${repository_root}" "${make_target}"
