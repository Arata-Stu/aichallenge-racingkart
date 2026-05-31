#!/bin/bash

set -Eeuo pipefail

AICHALLENGE_DIR="${AICHALLENGE_DIR:-/aichallenge}"
SETUP_FILE="${SETUP_FILE:-${AICHALLENGE_DIR}/workspace/install/setup.bash}"
SIMULATOR_DOMAIN_ID="${SIMULATOR_DOMAIN_ID:-0}"
AUTOWARE_DOMAIN_ID="${AUTOWARE_DOMAIN_ID:-1}"
SIMULATOR_CMD="${SIMULATOR_CMD:-bash run_simulator.bash dev}"
AUTOWARE_CMD="${AUTOWARE_CMD:-bash run_autoware.bash awsim ${AUTOWARE_DOMAIN_ID} /output/manual}"
LAYOUT_NAME="aichallenge_manual"
WORK_DIR="${TMPDIR:-/tmp}/aichallenge-terminator-${USER:-user}"

usage() {
    cat <<EOF
Usage:
  bash terminator.sh

Environment overrides:
  AICHALLENGE_DIR=${AICHALLENGE_DIR}
  SETUP_FILE=${SETUP_FILE}
  SIMULATOR_DOMAIN_ID=${SIMULATOR_DOMAIN_ID}
  AUTOWARE_DOMAIN_ID=${AUTOWARE_DOMAIN_ID}
  SIMULATOR_CMD=${SIMULATOR_CMD}
  AUTOWARE_CMD=${AUTOWARE_CMD}

The opened panes source SETUP_FILE, cd to AICHALLENGE_DIR, and put commands in shell history.
Press Up then Enter in each pane when you want to run the prepared command.
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

write_pane_rcfile() {
    local path="$1"
    local pane_title="$2"
    local prepared_command="$3"
    local domain_id="$4"

    {
        printf 'AICHALLENGE_DIR=%q\n' "${AICHALLENGE_DIR}"
        printf 'SETUP_FILE=%q\n' "${SETUP_FILE}"
        printf 'PANE_TITLE=%q\n' "${pane_title}"
        printf 'PREPARED_COMMAND=%q\n' "${prepared_command}"
        printf 'PANE_ROS_DOMAIN_ID=%q\n' "${domain_id}"
        cat <<'PANE_BODY'

export ROS_LOCALHOST_ONLY="${ROS_LOCALHOST_ONLY:-0}"
export RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_cyclonedds_cpp}"
export CYCLONEDDS_URI="${CYCLONEDDS_URI:-file:///opt/autoware/cyclonedds.xml}"
export ROS_DOMAIN_ID="${PANE_ROS_DOMAIN_ID}"

if [ -f "${SETUP_FILE}" ]; then
    # shellcheck disable=SC1090
    source "${SETUP_FILE}"
else
    echo "setup file not found: ${SETUP_FILE}"
fi

if ! cd "${AICHALLENGE_DIR}" 2>/dev/null; then
    echo "failed to cd ${AICHALLENGE_DIR}"
fi

history -s "${PREPARED_COMMAND}" 2>/dev/null || true
export PS1="[${PANE_TITLE}]\\$ "
PANE_BODY
    } >"${path}"
}

simulator_rcfile="${WORK_DIR}/simulator-bashrc"
autoware_rcfile="${WORK_DIR}/autoware-bashrc"
config_file="${WORK_DIR}/terminator-config"

write_pane_rcfile "${simulator_rcfile}" "simulator" "${SIMULATOR_CMD}" "${SIMULATOR_DOMAIN_ID}"
write_pane_rcfile "${autoware_rcfile}" "autoware" "${AUTOWARE_CMD}" "${AUTOWARE_DOMAIN_ID}"

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
      command = bash --rcfile ${simulator_rcfile} -i
    [[[terminal1]]]
      type = Terminal
      parent = paned0
      order = 1
      profile = default
      title = Autoware
      command = bash --rcfile ${autoware_rcfile} -i
[plugins]
EOF

exec terminator -g "${config_file}" -l "${LAYOUT_NAME}" "$@"
