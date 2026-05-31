#!/bin/bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ -x /aichallenge/terminator.sh ]; then
    exec /aichallenge/terminator.sh "$@"
fi

exec "${SCRIPT_DIR}/aichallenge/terminator.sh" "$@"
