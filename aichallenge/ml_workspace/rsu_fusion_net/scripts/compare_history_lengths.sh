#!/usr/bin/env bash
set -euo pipefail

HISTORY_LENGTHS="${HISTORY_LENGTHS:-1 3 5 8 10}"

for history_len in ${HISTORY_LENGTHS}; do
  echo "=== Training RSU fusion model with history_len=${history_len} ==="
  python3 train.py data.history_len="${history_len}" \
    train.save_dir="./checkpoints/history_${history_len}" \
    train.log_dir="./logs/history_${history_len}"
done
