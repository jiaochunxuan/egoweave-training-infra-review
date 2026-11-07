"""Deterministic mean/std statistics for masked M4 windows.

Quantile normalization is deliberately not used by the final M4 recipe.  The
functions here aggregate only valid channels and serialize a small, auditable
JSON contract that can be checked before training.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _check_pair(values, mask, name):
    values = np.asarray(values, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    if values.shape != mask.shape or values.ndim == 0:
        raise ValueError(f"{name} values/mask shapes differ: {values.shape} vs {mask.shape}")
    if not np.isfinite(values[mask]).all():
        raise ValueError(f"{name} has non-finite valid values")
    return values, mask


def masked_moments(values, mask):
    """Return ``count, sum, sumsq`` per channel, flattening leading axes."""
    values, mask = _check_pair(values, mask, "values")
    flat = values.reshape(-1, values.shape[-1])
    valid = mask.reshape(-1, mask.shape[-1])
    return (valid.sum(axis=0, dtype=np.int64),
            np.where(valid, flat, 0).sum(axis=0, dtype=np.float64),
            np.where(valid, flat * flat, 0).sum(axis=0, dtype=np.float64))


def merge_moments(left, right):
    if len(left) != 3 or len(right) != 3:
        raise ValueError("moments must be (count, sum, sumsq)")
    return tuple(np.asarray(a) + np.asarray(b) for a, b in zip(left, right))


def finalize_moments(moments, *, min_count=1, std_floor=1e-6):
    count, total, total_sq = (np.asarray(value) for value in moments)
    if count.ndim != 1 or total.shape != count.shape or total_sq.shape != count.shape:
        raise ValueError("moment arrays must be one-dimensional and equally shaped")
    if np.any(count < min_count):
        missing = np.flatnonzero(count < min_count).tolist()
        raise ValueError(f"insufficient valid samples in channels {missing}")
    mean = total / count
    variance = np.maximum(total_sq / count - mean * mean, 0.0)
    std = np.maximum(np.sqrt(variance), std_floor)
    return dict(mean=mean.tolist(), std=std.tolist(), count=count.astype(int).tolist())


def stats_from_batches(batches, *, min_count=1):
    """Aggregate ``(state, action, state_mask, action_mask)`` batches."""
    state_moments = None
    action_moments = None
    for states, actions, state_mask, action_mask in batches:
        current = masked_moments(states, state_mask)
        future = masked_moments(np.asarray(actions).reshape(-1, actions.shape[-1]),
                                np.asarray(action_mask, dtype=bool).reshape(-1, actions.shape[-1]))
        state_moments = current if state_moments is None else merge_moments(state_moments, current)
        action_moments = future if action_moments is None else merge_moments(action_moments, future)
    if state_moments is None:
        raise ValueError("no batches supplied")
    return dict(normalization="mean_std", state=finalize_moments(state_moments, min_count=min_count),
                actions=finalize_moments(action_moments, min_count=min_count))


def validate_stats(stats, *, state_dim=156, action_dim=153):
    if stats.get("normalization") != "mean_std":
        raise ValueError("M4 requires mean_std normalization")
    for key, width in (("state", state_dim), ("actions", action_dim)):
        item = stats.get(key)
        if not isinstance(item, dict) or len(item.get("mean", [])) != width:
            raise ValueError(f"invalid {key} normalization width")
        mean = np.asarray(item["mean"], dtype=np.float64)
        std = np.asarray(item["std"], dtype=np.float64)
        count = np.asarray(item.get("count", []), dtype=np.int64)
        if std.shape != mean.shape or count.shape != mean.shape or not np.isfinite(mean).all():
            raise ValueError(f"invalid {key} normalization arrays")
        if np.any(std <= 0) or np.any(count <= 0) or not np.isfinite(std).all():
            raise ValueError(f"invalid {key} normalization scale/count")
    return stats


def save_stats(path: str | Path, stats):
    path = Path(path)
    validate_stats(stats)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    # Match OpenPI's ``shared.normalize`` wire format while keeping the
    # in-memory helper pleasant to use.
    payload = {"norm_stats": stats}
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_stats(path: str | Path):
    path = Path(path)
    stats = json.loads(path.read_text(encoding="utf-8"))
    stats = stats.get("norm_stats", stats)
    return validate_stats(stats)
