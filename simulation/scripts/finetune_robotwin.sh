#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: $0 <variant-name> <checkpoint-params> [extra args...]" >&2
  exit 2
fi

RELEASE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${RELEASE_ROOT}/src:${PYTHONPATH:-}"
VARIANT="$1"
CHECKPOINT="$2"
shift 2

exec "${PYTHON:-python}" "${RELEASE_ROOT}/src/robotwin_pi05/train_robotwin_pi05.py" \
  --variant custom --custom-variant "${VARIANT}" \
  --checkpoint-path "${CHECKPOINT}" \
  --steps "${ROBOTWIN_STEPS:-5000}" \
  --save-interval "${ROBOTWIN_SAVE_INTERVAL:-1000}" \
  --overwrite \
  "$@"
