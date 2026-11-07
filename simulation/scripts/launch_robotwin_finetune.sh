#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: $0 <variant> <cuda_visible_devices> [extra train args...]" >&2
  exit 2
fi

VARIANT="$1"
CUDA_IDS="$2"
shift 2

RELEASE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKSPACE="${EGO_WEAVE_ROOT:-${RELEASE_ROOT}}"
TRAIN_DIR="${RELEASE_ROOT}/src/robotwin_pi05"
PYTHON="${PYTHON:-python}"
LOG_DIR="${ROBOTWIN_LOG_ROOT:-${WORKSPACE}/logs/robotwin}"
RUN_DIR="${ROBOTWIN_RUN_ROOT:-${WORKSPACE}/runs/robotwin}"
mkdir -p "${LOG_DIR}" "${RUN_DIR}"

export PYTHONPATH="${TRAIN_DIR}:${ROBOTWIN_OPENPI_ROOT:-${WORKSPACE}/external/openpi}/src:${ROBOTWIN_OPENPI_ROOT:-${WORKSPACE}/external/openpi}/packages/openpi-client/src:${LEROBOT_ROOT:-${WORKSPACE}/external/lerobot}/src:${LEROBOT_ROOT:-${WORKSPACE}/external/lerobot}:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_IDS}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export OPENPI_JAX_COMPILATION_CACHE_DIR="${WORKSPACE}/cache/jax"
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=disabled

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
EXP_NAME="${ROBOTWIN_EXP_NAME:-robotwin_${VARIANT}_pi05_5000}"
LOG_FILE="${LOG_DIR}/${EXP_NAME}_${TIMESTAMP}.log"
PID_FILE="${RUN_DIR}/${EXP_NAME}.pid"

cd "${TRAIN_DIR}"
nohup "${PYTHON}" train_robotwin_pi05.py \
  --variant "${VARIANT}" \
  --exp-name "${EXP_NAME}" \
  --steps "${ROBOTWIN_STEPS:-5000}" \
  --batch-size "${ROBOTWIN_BATCH_SIZE:-128}" \
  --num-workers "${ROBOTWIN_NUM_WORKERS:-8}" \
  --fsdp-devices "${ROBOTWIN_FSDP_DEVICES:-4}" \
  --save-interval "${ROBOTWIN_SAVE_INTERVAL:-1000}" \
  --log-interval "${ROBOTWIN_LOG_INTERVAL:-50}" \
  --learning-rate "${ROBOTWIN_LR:-5e-5}" \
  --warmup-steps "${ROBOTWIN_WARMUP_STEPS:-100}" \
  --overwrite \
  --disable-wandb \
  "$@" > "${LOG_FILE}" 2>&1 &

PID="$!"
echo "${PID}" > "${PID_FILE}"
echo "launched variant=${VARIANT} exp=${EXP_NAME} pid=${PID} cuda=${CUDA_IDS} log=${LOG_FILE}"
