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

exec "${PYTHON:-python}" -m dexjoco4.train.train_dexjoco_pi05 \
  --variant custom --custom-variant "${VARIANT}" \
  --checkpoint-path "${CHECKPOINT}" \
  --steps "${DEXJOCO_STEPS:-10000}" \
  --save-interval "${DEXJOCO_SAVE_INTERVAL:-10000}" \
  --keep-period "${DEXJOCO_KEEP_PERIOD:-10000}" \
  "$@"
