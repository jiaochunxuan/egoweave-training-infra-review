#!/usr/bin/env bash
set -euo pipefail

: "${OPENPI_ROOT:?set OPENPI_ROOT to a public OpenPI checkout}"
: "${PI05_BASE_PARAMS:?set PI05_BASE_PARAMS to the Pi0.5 base parameter directory}"
: "${M4_MANIFEST:?set M4_MANIFEST to a prepared M4 manifest.json}"
: "${M4_NORM_STATS:?set M4_NORM_STATS to the prepared norm_stats.json}"

export PYTHONPATH="${EGO_WEAVE_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}/src:${PYTHONPATH:-}"
python -m egoweave_pretrain.train_m4 \
  --manifest "$M4_MANIFEST" \
  --norm-stats "$M4_NORM_STATS" \
  --openpi-root "$OPENPI_ROOT" \
  --base-params "$PI05_BASE_PARAMS" \
  --output "${M4_OUTPUT:-runs/m4_structured_153d}" \
  --steps "${M4_STEPS:-18400}" \
  --devices "${M4_DEVICES:-8}" \
  --batch-size "${M4_BATCH_SIZE:-256}" \
  --workers "${M4_WORKERS:-24}"
