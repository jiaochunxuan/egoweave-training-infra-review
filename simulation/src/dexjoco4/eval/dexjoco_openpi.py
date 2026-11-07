"""OpenPI policy bridge for the public DexJoCo evaluation entry point.

The simulator itself is an external dependency. This module defines only the
observation repacking and checkpoint/configuration contract used by the paper.
All machine-specific locations are supplied through environment variables.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import sys
from typing import Iterable

import numpy as np


OPENPI_ROOT = Path(os.environ.get("DEXJOCO_OPENPI_ROOT", "external/openpi")).expanduser()
OPENPI_PYTHON = Path(os.environ.get("OPENPI_PYTHON", sys.executable)).expanduser()
DEXJOCO_SOURCE_ROOT = Path(os.environ.get("DEXJOCO_SOURCE_ROOT", "external/dexjoco")).expanduser()
DEXJOCO_PYTHON = Path(os.environ.get("DEXJOCO_PYTHON", sys.executable)).expanduser()
DEXJOCO_CONFIG_ROOT = Path(
    os.environ.get("DEXJOCO_CONFIG_ROOT", "external/dexjoco/configs/rand_obj")
).expanduser()
CHECKPOINT_ROOT = Path(
    os.environ.get("DEXJOCO_CHECKPOINT_ROOT", "outputs/dexjoco/checkpoints")
).expanduser()

DEFAULT_TASKS = ("fold_glasses", "hammer_nail", "pick_bucket", "pinch_tongs", "water_plant")
DEFAULT_EVAL_SEEDS = (10000, 10001, 10002, 10003, 10004)
EVAL_VARIANTS = ("base", "pretrained", "custom")


def _ensure_openpi_paths() -> None:
    for path in (
        DEXJOCO_SOURCE_ROOT,
        DEXJOCO_SOURCE_ROOT / "dexjoco",
        OPENPI_ROOT,
        OPENPI_ROOT / "src",
        OPENPI_ROOT / "packages/openpi-client/src",
    ):
        value = str(path)
        if value not in sys.path:
            sys.path.insert(0, value)


def resolve_task_config_path(task_name: str) -> Path:
    path = DEXJOCO_CONFIG_ROOT / f"{task_name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"DexJoCo task config not found: {path}")
    return path


def resolve_checkpoint_dir(variant: str, checkpoint_step: int, *, output_root: Path | None = None) -> Path:
    """Resolve a checkpoint without encoding any private experiment names."""
    root = Path(output_root or CHECKPOINT_ROOT)
    if variant == "base":
        name = os.environ.get("DEXJOCO_BASE_RUN", "base")
    elif variant == "pretrained":
        name = os.environ.get("DEXJOCO_PRETRAINED_RUN", "pretrained")
    elif variant == "custom":
        name = os.environ.get("DEXJOCO_CUSTOM_RUN", "custom")
    else:
        raise KeyError(f"Unknown variant: {variant!r}; choose from {EVAL_VARIANTS}")
    checkpoint_dir = root / name / str(checkpoint_step)
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"DexJoCo checkpoint not found: {checkpoint_dir}")
    return checkpoint_dir


@dataclass(frozen=True)
class SingleHandOpenPIRepack:
    """Convert simulator observations to canonical Pi0.5 inputs."""

    base_keys: tuple[str, ...] = ("base", "ego", "random_camera")
    wrist_keys: tuple[str, ...] = ("wrist", "wrist_right", "wrist_left")

    def __call__(self, data: dict) -> dict:
        if "image" in data and "image_mask" in data:
            return data
        base = self._first_available(data, self.base_keys)
        wrist = self._first_available(data, self.wrist_keys)
        if base is None or wrist is None:
            raise ValueError("DexJoCo observations must contain base and wrist RGB images")
        base = np.asarray(base)
        wrist = np.asarray(wrist)
        return {
            "image": {
                "base_0_rgb": base,
                "left_wrist_0_rgb": np.zeros_like(base),
                "right_wrist_0_rgb": wrist,
            },
            "image_mask": {
                "base_0_rgb": np.asarray(True),
                "left_wrist_0_rgb": np.asarray(False),
                "right_wrist_0_rgb": np.asarray(True),
            },
            "state": np.asarray(data["state"], dtype=np.float32),
            "prompt": data.get("prompt", ""),
        }

    @staticmethod
    def _first_available(data: dict, keys: Iterable[str]):
        for key in keys:
            if key in data:
                return data[key]
        return None


def make_single_hand_repack_group():
    _ensure_openpi_paths()
    import openpi.transforms as transforms

    return transforms.Group(inputs=[SingleHandOpenPIRepack()])


def build_policy(variant: str, checkpoint_step: int, *, output_root: Path | None = None):
    _ensure_openpi_paths()
    from openpi.policies import policy_config
    from dexjoco4.train.train_dexjoco_pi05 import build_run_specs, build_train_config

    specs = build_run_specs(output_root=output_root)
    if variant == "base":
        spec = specs["base_robot"]
    elif variant in {"pretrained", "custom"}:
        spec = specs["ego_robot"]
    else:
        raise KeyError(variant)
    checkpoint_dir = resolve_checkpoint_dir(variant, checkpoint_step, output_root=output_root)
    config = build_train_config(spec)
    return policy_config.create_trained_policy(
        config, checkpoint_dir, repack_transforms=make_single_hand_repack_group()
    )


def serve_policy(
    variant: str,
    checkpoint_step: int,
    *,
    port: int,
    host: str = "127.0.0.1",
    output_root: Path | None = None,
):
    _ensure_openpi_paths()
    from openpi.serving import websocket_policy_server

    policy = build_policy(variant, checkpoint_step, output_root=output_root)
    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy, host=host, port=port, metadata=policy.metadata
    )
    server.serve_forever()
