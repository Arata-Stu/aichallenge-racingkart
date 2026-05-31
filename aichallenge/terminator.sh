#!/bin/bash

set -Eeuo pipefail

AICHALLENGE_DIR="${AICHALLENGE_DIR:-/aichallenge}"
SETUP_FILE="${SETUP_FILE:-${AICHALLENGE_DIR}/workspace/install/setup.bash}"
SIMULATOR_CMD="${SIMULATOR_CMD:-bash run_simulator.bash}"
AUTOWARE_CMD="${AUTOWARE_CMD:-bash run_autoware.bash awsim}"
LAYOUT_NAME="aichallenge_manual"
WORK_DIR="${TMPDIR:-/tmp}/aichallenge-terminator-${USER:-user}"

usage() {
    cat <<EOF
Usage:
  bash terminator.sh

Environment overrides:
  AICHALLENGE_DIR=${AICHALLENGE_DIR}
  SETUP_FILE=${SETUP_FILE}
  SIMULATOR_CMD=${SIMULATOR_CMD}
  AUTOWARE_CMD=${AUTOWARE_CMD}

The opened panes source SETUP_FILE, cd to AICHALLENGE_DIR, and prefill commands.
Press Enter in each pane when you want to run the prepared command.
EOF
}

if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
    usage
    exit 0
fi

if ! command -v terminator >/dev/null 2>&1; then
    echo "[ERROR] terminator command not found" >&2
    exit 1
fi

mkdir -p "${WORK_DIR}"

write_pane_script() {
    local path="$1"
    local pane_title="$2"
    local prepared_command="$3"

    {
        echo "#!/bin/bash"
        echo "set +u"
        printf 'AICHALLENGE_DIR=%q\n' "${AICHALLENGE_DIR}"
        printf 'SETUP_FILE=%q\n' "${SETUP_FILE}"
        printf 'PANE_TITLE=%q\n' "${pane_title}"
        printf 'PREPARED_COMMAND=%q\n' "${prepared_command}"
        cat <<'PANE_BODY'

if [ -f "${SETUP_FILE}" ]; then
    # shellcheck disable=SC1090
    source "${SETUP_FILE}"
    echo "[INFO] sourced ${SETUP_FILE}"
else
    echo "[WARN] setup file not found: ${SETUP_FILE}"
fi

if ! cd "${AICHALLENGE_DIR}"; then
    echo "[ERROR] failed to cd ${AICHALLENGE_DIR}"
    exec bash
fi

while true; do
    echo
    echo "[INFO] ${PANE_TITLE}: Press Enter to run, or edit the prepared command."
    if ! read -e -i "${PREPARED_COMMAND}" -p "[${PANE_TITLE}]$ " cmd; then
        echo
        exec bash
    fi
    if [ -z "${cmd//[[:space:]]/}" ]; then
        continue
    fi

    history -s "${cmd}" 2>/dev/null || true
    eval "${cmd}"
    status=$?
    echo "[INFO] command exited with status ${status}"
done
PANE_BODY
    } >"${path}"

    chmod +x "${path}"
}

simulator_script="${WORK_DIR}/simulator-pane.sh"
autoware_script="${WORK_DIR}/autoware-pane.sh"
config_file="${WORK_DIR}/terminator-config"

write_pane_script "${simulator_script}" "simulator" "${SIMULATOR_CMD}"
write_pane_script "${autoware_script}" "autoware" "${AUTOWARE_CMD}"

cat >"${config_file}" <<EOF
[global_config]
[keybindings]
[profiles]
  [[default]]
[layouts]
  [[${LAYOUT_NAME}]]
    [[[window0]]]
      type = Window
      parent = ""
      order = 0
      maximised = True
      title = AIchallenge Manual Launch
    [[[paned0]]]
      type = HPaned
      parent = window0
      order = 0
      ratio = 0.5
    [[[terminal0]]]
      type = Terminal
      parent = paned0
      order = 0
      profile = default
      title = AWSIM Simulator
      command = ${simulator_script}
    [[[terminal1]]]
      type = Terminal
      parent = paned0
      order = 1
      profile = default
      title = Autoware
      command = ${autoware_script}
[plugins]
EOF

exec terminator -g "${config_file}" -l "${LAYOUT_NAME}" "$@"
