#!/usr/bin/env bash
set -euo pipefail

repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
: "${EGOWEAVE_OPENPI_ROOT:?Set the root of the compatible OpenPI snapshot}"
: "${EGOWEAVE_OPENPI_CACHE:?Set the cache containing the PaliGemma tokenizer}"
: "${EGOWEAVE_MANIFEST:?Set the cached human data manifest}"
: "${EGOWEAVE_NORM_DIR:?Set the normalization directory for the requested condition}"
: "${EGOWEAVE_BASE:?Set the converted base model.safetensors path}"
: "${EGOWEAVE_INTERFACE:?Set the seed-42 interface .npz path}"
: "${EGOWEAVE_OUTPUT:?Set a fresh output directory}"

export OPENPI_DATA_HOME="$EGOWEAVE_OPENPI_CACHE"
export JAX_PLATFORMS=cpu OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export PYTHONPATH="$repo/rebuttal:$EGOWEAVE_OPENPI_ROOT/src:$EGOWEAVE_OPENPI_ROOT/packages/openpi-client/src${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false

profile_args=()
case "${EGOWEAVE_PROFILE:-h100-matched}" in
    h100-matched) ;;
    fp32-fast) profile_args=(--compact-padding --skip-outer-checkpoint) ;;
    *) echo 'EGOWEAVE_PROFILE must be h100-matched or fp32-fast' >&2; exit 2 ;;
esac

exec "${PYTHON:-python}" -m torch.distributed.run --standalone --nproc_per_node=8 \
    "${EGOWEAVE_ENTRY:-$repo/rebuttal/train_moore_torch.py}" \
    --condition "${1:?Pass no_explicit_head, current_head, or no_scale}" \
    --manifest "$EGOWEAVE_MANIFEST" --norm-dir "$EGOWEAVE_NORM_DIR" \
    --base "$EGOWEAVE_BASE" --interface "$EGOWEAVE_INTERFACE" --output "$EGOWEAVE_OUTPUT" \
    --microbatch "${EGOWEAVE_MICROBATCH:-32}" --workers "${EGOWEAVE_WORKERS:-3}" \
    --batch-layout "${EGOWEAVE_BATCH_LAYOUT:-contiguous}" "${profile_args[@]}"
