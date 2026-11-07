from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np


DEXJOCO_MISSING_REGEX = r".*(action_in_proj|action_out_proj).*(kernel|bias).*"
ACTION_HEAD_ROOTS = {
    "action_in_proj",
    "action_out_proj",
    "human_action_in_proj",
    "human_action_out_proj",
    "robot_action_in_proj",
    "robot_action_out_proj",
}


def _flatten(tree: Mapping[str, Any], prefix: tuple[str, ...] = ()) -> dict[tuple[str, ...], Any]:
    leaves: dict[tuple[str, ...], Any] = {}
    for key, value in tree.items():
        path = (*prefix, str(key))
        if isinstance(value, Mapping):
            leaves.update(_flatten(value, path))
        else:
            leaves[path] = value
    return leaves


def _clone(tree: Mapping[str, Any]) -> dict[str, Any]:
    return {str(key): _clone(value) if isinstance(value, Mapping) else value for key, value in tree.items()}


def _put(tree: dict[str, Any], path: tuple[str, ...], value: Any) -> None:
    cursor = tree
    for key in path[:-1]:
        cursor = cursor[key]
    cursor[path[-1]] = value


def _shape(value: Any) -> tuple[int, ...]:
    return tuple(int(item) for item in np.shape(value))


def map_clean_backbone_params(source_params: Mapping[str, Any], target_params: Mapping[str, Any]) -> dict[str, Any]:
    """Copy only shape-compatible non-action-head leaves into a target Pi0.5 tree."""

    result = _clone(target_params)
    source = _flatten(source_params)
    target = _flatten(target_params)
    for target_path, target_value in target.items():
        if target_path[0] in ACTION_HEAD_ROOTS:
            continue
        source_value = source.get(target_path)
        if source_value is not None and _shape(source_value) == _shape(target_value):
            _put(result, target_path, source_value)
    return result


def map_semiclean_basebias_params(
    source_params: Mapping[str, Any],
    base_params: Mapping[str, Any],
    target_params: Mapping[str, Any],
) -> dict[str, Any]:
    """Copy clean backbone weights and seed ``action_in_proj/bias`` from base Pi0.5.

    Base and human-pretrained controls use ``MergedCheckpointWeightLoader``
    with an action missing-regex. Shape-compatible action leaves are still
    loaded from the source checkpoint; in practice ``action_in_proj/bias`` is
    inherited from Pi0.5.
    """

    result = map_clean_backbone_params(source_params, target_params)
    base = _flatten(base_params)
    target = _flatten(target_params)
    bias_path = ("action_in_proj", "bias")
    base_bias = base.get(bias_path)
    target_bias = target.get(bias_path)
    if base_bias is None or target_bias is None:
        raise KeyError("action_in_proj/bias is missing from base or target params")
    if _shape(base_bias) != _shape(target_bias):
        raise ValueError(
            "base action_in_proj/bias shape mismatch: "
            f"base={_shape(base_bias)} target={_shape(target_bias)}"
        )
    _put(result, bias_path, base_bias)
    return result


class CleanBackboneCheckpointWeightLoader:
    """OpenPI weight loader that resets all action heads and transfers only shared weights."""

    def __init__(self, params_path: Path | str):
        self.params_path = str(params_path)

    def load(self, target_params):
        from openpi.models import model as _model

        source_params = _model.restore_params(self.params_path, restore_type=np.ndarray)
        return map_clean_backbone_params(source_params, target_params)


class SemiCleanBaseBiasCheckpointWeightLoader:
    """Reset action heads except for base Pi0.5 ``action_in_proj/bias``."""

    def __init__(self, params_path: Path | str, base_params_path: Path | str):
        self.params_path = str(params_path)
        self.base_params_path = str(base_params_path)

    def load(self, target_params):
        from openpi.models import model as _model

        source_params = _model.restore_params(self.params_path, restore_type=np.ndarray)
        base_params = _model.restore_params(self.base_params_path, restore_type=np.ndarray)
        return map_semiclean_basebias_params(source_params, base_params, target_params)


def make_dexjoco_weight_loader(params_path: Path | str):
    from openpi.training.weight_loaders import MergedCheckpointWeightLoader

    return MergedCheckpointWeightLoader(
        params_path=str(params_path),
        missing_regex=DEXJOCO_MISSING_REGEX,
    )


def make_clean_backbone_weight_loader(params_path: Path | str) -> CleanBackboneCheckpointWeightLoader:
    return CleanBackboneCheckpointWeightLoader(params_path)


def make_semiclean_basebias_weight_loader(
    params_path: Path | str,
    base_params_path: Path | str,
) -> SemiCleanBaseBiasCheckpointWeightLoader:
    return SemiCleanBaseBiasCheckpointWeightLoader(params_path, base_params_path)
