"""Strict NumPy parameter transfer with an explicitly narrowed base action head."""

from collections.abc import Mapping
from typing import Any

import numpy as np


ACTION_LEAVES = frozenset(
    ("action_in_proj/kernel", "action_in_proj/bias", "action_out_proj/kernel", "action_out_proj/bias")
)
ACTION_CROPS = {
    "action_in_proj/kernel": (2, 0),
    "action_out_proj/kernel": (2, -1),
    "action_out_proj/bias": (1, -1),
}
BASE_ACTION_DIM = 32


def transfer_flat(
    target: Mapping[str, Any],
    source: Mapping[str, np.ndarray],
    base: Mapping[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Copy flat parameter leaves and return a JSON-serializable provenance audit.

    Non-action leaves must exist in ``source`` with the exact target shape.
    The four standard action projection leaves always come from ``base``.
    Only the action axis of its 32D interface may be narrowed, to 1..32
    dimensions; every other dimension must match. All target action axes
    present in the mapping must agree. Keys absent from the target are ignored.

    Target values may be abstract shape/dtype specifications, including JAX
    ShapeDtypeStruct: only ``.shape`` and ``.dtype`` are read. Returned arrays
    are independent copies in the requested dtype; no input is modified.
    ``source`` labels in the audit name the input mapping, not a disk path.
    """
    result: dict[str, np.ndarray] = {}
    audit: dict[str, Any] = {
        "leaves": {},
        "shared_leaves": [],
        "base_action_leaves": [],
        "ignored_source_leaves": [],
        "ignored_base_leaves": [],
        "fallback_non_action_leaves": [],
    }
    action_dim = None

    for key, expected in target.items():
        target_shape = tuple(int(size) for size in expected.shape)
        target_dtype = np.dtype(expected.dtype)
        is_action = key in ACTION_LEAVES
        origin = "base" if is_action else "source"
        selected = base if is_action else source
        if key not in selected:
            raise ValueError(f"{key}: missing from {origin}; expected shape {target_shape}")

        original = np.asarray(selected[key])
        source_shape = original.shape
        value = original
        crop = None
        if key in ACTION_CROPS:
            rank, axis = ACTION_CROPS[key]
            if original.ndim != rank or len(target_shape) != rank:
                raise ValueError(
                    f"{key}: expected rank {rank}; base shape {source_shape}, target shape {target_shape}"
                )
            width = target_shape[axis]
            if source_shape[axis] != BASE_ACTION_DIM or not 1 <= width <= BASE_ACTION_DIM:
                raise ValueError(
                    f"{key}: only a {BASE_ACTION_DIM}D base action axis can be narrowed to 1..{BASE_ACTION_DIM}; "
                    f"base shape {source_shape}, target shape {target_shape}"
                )
            allowed_shape = list(source_shape)
            allowed_shape[axis] = width
            if tuple(allowed_shape) != target_shape:
                raise ValueError(
                    f"{key}: non-action dimensions differ; base shape {source_shape}, target shape {target_shape}"
                )
            if action_dim is not None and width != action_dim:
                raise ValueError(f"{key}: target action dimension {width} disagrees with {action_dim}")
            action_dim = width
            if width < BASE_ACTION_DIM:
                slices = [slice(None)] * rank
                slices[axis] = slice(0, width)
                value = original[tuple(slices)]
                crop = {"axis": axis, "start": 0, "stop": width}
        elif source_shape != target_shape:
            raise ValueError(
                f"{key}: {origin} shape {source_shape} does not match target shape {target_shape}"
            )

        result[key] = np.array(value, dtype=target_dtype, copy=True)
        audit["leaves"][key] = {
            "source": origin,
            "source_shape": list(source_shape),
            "target_shape": list(target_shape),
            "result_shape": list(result[key].shape),
            "source_dtype": str(original.dtype),
            "target_dtype": str(target_dtype),
            "crop": crop,
        }
        audit["base_action_leaves" if is_action else "shared_leaves"].append(key)

    audit["shared_leaves"].sort()
    audit["base_action_leaves"].sort()
    audit["ignored_source_leaves"] = sorted(set(source) - set(audit["shared_leaves"]))
    audit["ignored_base_leaves"] = sorted(set(base) - set(audit["base_action_leaves"]))
    return result, audit
