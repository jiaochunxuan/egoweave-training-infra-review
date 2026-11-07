"""Strict Pi0.5-to-M4 action-interface initialization."""

from __future__ import annotations


def transfer_base_to_153(source, target):
    """Copy every compatible Pi0.5 leaf and keep only the 32-D action head random.

    ``source`` and ``target`` are flattened parameter dictionaries.  The three
    permitted shape changes are the input projection kernel, output projection
    kernel, and output bias.  Any other mismatch is an error, preventing an
    accidental partial or silent initialization.
    """
    result = {}
    changed = []
    for key, wanted in target.items():
        if key not in source:
            raise ValueError(f"missing original Pi0.5 parameter: {key}")
        value = source[key]
        old, new = tuple(value.shape), tuple(wanted.shape)
        if old == new:
            result[key] = value.astype(wanted.dtype) if value.dtype != wanted.dtype else value
            continue
        allowed = (
            key == "action_in_proj/kernel" and old[0] == 32 and new[0] == 153
        ) or (
            key == "action_out_proj/kernel" and old[1] == 32 and new[1] == 153
        ) or (key == "action_out_proj/bias" and old == (32,) and new == (153,))
        if not allowed:
            raise ValueError(f"only action-interface 32->153 changes are allowed: {key}: {old}->{new}")
        result[key] = wanted
        changed.append({"key": key, "source_shape": list(old), "target_shape": list(new)})
    if set(source) != set(target):
        raise ValueError("unused or missing parameter leaves are not allowed")
    if len(changed) != 3:
        raise ValueError(f"expected exactly three random action-interface leaves, got {len(changed)}")
    return result, {"loaded_leaves": len(result) - len(changed), "random_interface": changed}
