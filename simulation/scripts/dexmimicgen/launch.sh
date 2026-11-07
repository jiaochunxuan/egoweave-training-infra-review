#!/usr/bin/env bash
set -euo pipefail
RELEASE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CODE_ROOT="${RELEASE_ROOT}/src"
PYTHON="${PYTHON:-python}"
export PYTHONUNBUFFERED=1
export PYTHONPATH="${CODE_ROOT}:${PYTHONPATH:-}"
exec "$PYTHON" -u "${RELEASE_ROOT}/scripts/dexmimicgen/run_train.py" "$@"
