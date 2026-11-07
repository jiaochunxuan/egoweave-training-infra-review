from __future__ import annotations

import argparse
from collections.abc import Mapping
from dataclasses import dataclass
from dataclasses import replace
import importlib.util
import logging
import os
import sys
from pathlib import Path
from typing import Any

import flax.traverse_util
import jax
import numpy as np


PUBLIC_ROOT = Path(os.environ.get("EGO_WEAVE_ROOT", ".")).expanduser()
OPENPI_ROOT = Path(os.environ.get("ROBOTWIN_OPENPI_ROOT", PUBLIC_ROOT / "external/openpi")).expanduser()
OPENPI_TRAIN = OPENPI_ROOT / "scripts/train.py"
ROBOTWIN_DATA_ROOT = Path(os.environ.get("ROBOTWIN_DATA_ROOT", PUBLIC_ROOT / "data/robotwin")).expanduser()
OUTPUT_ROOT = Path(os.environ.get("ROBOTWIN_OUTPUT_ROOT", PUBLIC_ROOT / "outputs/robotwin")).expanduser()
BASE_PI05_PARAMS = Path(os.environ.get("PI05_BASE_PARAMS", PUBLIC_ROOT / "checkpoints/pi05_base/params")).expanduser()
BASE_PI05_ASSETS = Path(os.environ.get("PI05_BASE_ASSETS", PUBLIC_ROOT / "checkpoints/pi05_base/assets")).expanduser()

VARIANT_CHECKPOINTS = {"base": BASE_PI05_PARAMS}

DEFAULT_TASK_FILTERS = (
    "Place Empty Cup",
    "Move Pillbottle Pad",
    "Move Stapler Pad",
    "Adjust Bottle",
    "Grab Roller",
)


def ensure_openpi_paths() -> None:
    for path in reversed((OPENPI_ROOT, OPENPI_ROOT / "src", OPENPI_ROOT / "packages/openpi-client/src")):
        text = str(path)
        while text in sys.path:
            sys.path.remove(text)
        sys.path.insert(0, text)


ensure_openpi_paths()

from openpi import transforms as _transforms  # noqa: E402
from openpi.models import model as _model  # noqa: E402
from openpi.models import pi0_config  # noqa: E402
from openpi.training import config as _config  # noqa: E402
from openpi.training import data_loader as _data_loader  # noqa: E402
from openpi.training import optimizer as _optimizer  # noqa: E402
from openpi.training import weight_loaders  # noqa: E402

try:
    from .robotwin_v3_openpi_adapter import RoboTwinImageParquetDataset, RoboTwinV3Dataset  # type: ignore  # noqa: E402
except ImportError:  # pragma: no cover - exercised when running this file as a script.
    from robotwin_v3_openpi_adapter import RoboTwinImageParquetDataset, RoboTwinV3Dataset  # type: ignore  # noqa: E402


DEFAULT_TASK_NAMES = (
    "place_empty_cup",
    "move_pillbottle_pad",
    "move_stapler_pad",
    "adjust_bottle",
    "grab_roller",
)

ROBOTWIN_VARIANT_ALIASES = {"base_robotwin": "base"}


def _load_openpi_train_module():
    spec = importlib.util.spec_from_file_location("openpi_scripts_train_robotwin", OPENPI_TRAIN)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load OpenPI train script from {OPENPI_TRAIN}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class PromptFromTaskString(_transforms.DataTransformFn):
    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        task = data.get("task")
        if task is None:
            raise ValueError("RoboTwin sample is missing task text")
        return {**data, "prompt": task}


def _flatten(tree: Mapping[str, Any], sep: str = "/") -> dict[str, Any]:
    return flax.traverse_util.flatten_dict(tree, sep=sep)


def _unflatten(flat: Mapping[str, Any], sep: str = "/") -> dict[str, Any]:
    return flax.traverse_util.unflatten_dict(flat, sep=sep)


class RobotWinBackboneWithBaseActionBiasLoader:
    """Transfer shared non-action weights and seed the 32D ALOHA action interface from base Pi0.5.

    Human-pretrained checkpoints use human or mixed human/robot action heads
    (153-D or another non-native width), so their action projections are not directly compatible
    with RoboTwin's ALOHA/Pi0.5 32D model interface. This loader copies only
    shape-compatible non-action leaves from the source checkpoint, then copies
    the full 32D action projections from the base Pi0.5 checkpoint.
    """

    ACTION_PREFIXES = (
        "action_in_proj/",
        "action_out_proj/",
        "human_action_in_proj/",
        "human_action_out_proj/",
        "robot_action_in_proj/",
        "robot_action_out_proj/",
    )

    def __init__(self, source_params_path: Path | str, base_params_path: Path | str = BASE_PI05_PARAMS):
        self.source_params_path = str(source_params_path)
        self.base_params_path = str(base_params_path)

    @staticmethod
    def _is_action_leaf(key: str) -> bool:
        return key.startswith(RobotWinBackboneWithBaseActionBiasLoader.ACTION_PREFIXES)

    def load(self, target_params):
        ensure_openpi_paths()
        source_params = _model.restore_params(self.source_params_path, restore_type=np.ndarray)
        base_params = _model.restore_params(self.base_params_path, restore_type=np.ndarray)

        flat_target = _flatten(target_params)
        flat_source = _flatten(source_params)
        flat_base = _flatten(base_params)
        result: dict[str, Any] = {}

        for key, target_value in flat_target.items():
            source_value = flat_source.get(key)
            base_value = flat_base.get(key)
            if self._is_action_leaf(key):
                if base_value is not None and np.shape(base_value) == np.shape(target_value):
                    result[key] = base_value.astype(target_value.dtype) if base_value.dtype != target_value.dtype else base_value
                else:
                    result[key] = target_value
                continue

            if source_value is not None and np.shape(source_value) == np.shape(target_value):
                result[key] = (
                    source_value.astype(target_value.dtype) if source_value.dtype != target_value.dtype else source_value
                )
            elif base_value is not None and np.shape(base_value) == np.shape(target_value):
                result[key] = base_value.astype(target_value.dtype) if base_value.dtype != target_value.dtype else base_value
            else:
                result[key] = target_value

        return _unflatten(result)


@dataclass(frozen=True)
class RobotWinRunSpec:
    variant: str
    name: str
    checkpoint_path: Path
    data_root: Path = ROBOTWIN_DATA_ROOT
    output_root: Path = OUTPUT_ROOT
    task_filters: tuple[str, ...] = DEFAULT_TASK_FILTERS
    action_dim: int = 32
    state_dim: int = 32
    horizon: int = 50
    max_token_len: int = 640
    batch_size: int = 128
    num_workers: int = 8
    fsdp_devices: int = 4
    num_train_steps: int = 5000
    log_interval: int = 50
    save_interval: int = 1000
    keep_period: int = 1000
    learning_rate: float = 5e-5
    warmup_steps: int = 100
    lr_decay_steps: int = 1_000_000
    clip_gradient_norm: float = 1.0
    ema_decay: float | None = 0.999
    seed: int = 42
    asset_id: str = "trossen"
    adapt_to_pi: bool = True


def build_robotwin_spec(
    variant: str,
    *,
    exp_name: str | None = None,
    custom_variant: str | None = None,
    custom_checkpoint_path: Path | None = None,
    data_root: Path = ROBOTWIN_DATA_ROOT,
    output_root: Path = OUTPUT_ROOT,
    task_filters: tuple[str, ...] = DEFAULT_TASK_FILTERS,
    steps: int = 5000,
    batch_size: int = 128,
    num_workers: int = 8,
    fsdp_devices: int = 4,
    save_interval: int = 1000,
    log_interval: int = 50,
    learning_rate: float = 5e-5,
    warmup_steps: int = 100,
    seed: int = 42,
) -> RobotWinRunSpec:
    variant = ROBOTWIN_VARIANT_ALIASES.get(variant, variant)
    if variant == "custom":
        if not custom_variant:
            raise ValueError("--custom-variant is required when --variant custom is used")
        if custom_checkpoint_path is None:
            raise ValueError("--checkpoint-path is required when --variant custom is used")
        variant = custom_variant
        checkpoint_path = custom_checkpoint_path
    elif variant in VARIANT_CHECKPOINTS:
        checkpoint_path = VARIANT_CHECKPOINTS[variant]
    else:
        choices = sorted((*VARIANT_CHECKPOINTS, *ROBOTWIN_VARIANT_ALIASES))
        raise ValueError(f"unknown variant {variant!r}; choices: {choices}")
    name = exp_name or f"robotwin_{variant}_pi05_5000"
    return RobotWinRunSpec(
        variant=variant,
        name=name,
        checkpoint_path=checkpoint_path,
        data_root=data_root,
        output_root=output_root,
        task_filters=tuple(task_filters),
        batch_size=batch_size,
        num_workers=num_workers,
        fsdp_devices=fsdp_devices,
        num_train_steps=steps,
        save_interval=save_interval,
        keep_period=save_interval,
        log_interval=log_interval,
        learning_rate=learning_rate,
        warmup_steps=warmup_steps,
        seed=seed,
    )


def build_robotwin_run_specs(
    *,
    data_root: Path = ROBOTWIN_DATA_ROOT,
    output_root: Path = OUTPUT_ROOT,
    task_filters: tuple[str, ...] = DEFAULT_TASK_FILTERS,
    steps: int = 5000,
    batch_size: int = 128,
    num_workers: int = 8,
    fsdp_devices: int = 4,
    save_interval: int = 1000,
    log_interval: int = 50,
    learning_rate: float = 5e-5,
    warmup_steps: int = 100,
    seed: int = 42,
) -> dict[str, RobotWinRunSpec]:
    return {
        public_name: build_robotwin_spec(
            short_name,
            data_root=data_root,
            output_root=output_root,
            task_filters=task_filters,
            steps=steps,
            batch_size=batch_size,
            num_workers=num_workers,
            fsdp_devices=fsdp_devices,
            save_interval=save_interval,
            log_interval=log_interval,
            learning_rate=learning_rate,
            warmup_steps=warmup_steps,
            seed=seed,
        )
        for public_name, short_name in ROBOTWIN_VARIANT_ALIASES.items()
    }


def build_robotwin_train_config(
    spec: RobotWinRunSpec,
    *,
    overwrite: bool = False,
    resume: bool = False,
    disable_wandb: bool = True,
):
    return build_train_config(spec, overwrite=overwrite, resume=resume, disable_wandb=disable_wandb)


def build_train_config(
    spec: RobotWinRunSpec,
    *,
    overwrite: bool = False,
    resume: bool = False,
    disable_wandb: bool = True,
):
    ensure_openpi_paths()

    model = pi0_config.Pi0Config(
        action_dim=spec.action_dim,
        state_dim=spec.state_dim,
        action_horizon=spec.horizon,
        max_token_len=spec.max_token_len,
        pi05=True,
        discrete_state_input=True,
    )

    if spec.variant == "base":
        loader = weight_loaders.CheckpointWeightLoader(str(spec.checkpoint_path))
    else:
        loader = RobotWinBackboneWithBaseActionBiasLoader(spec.checkpoint_path, BASE_PI05_PARAMS)

    return _config.TrainConfig(
        name="robotwin_pi05",
        project_name="robot_finetune_robotwin",
        exp_name=spec.name,
        model=model,
        data=_config.LeRobotAlohaDataConfig(
            repo_id=str(spec.data_root),
            assets=_config.AssetsConfig(
                assets_dir=str(BASE_PI05_ASSETS),
                asset_id=spec.asset_id,
            ),
            base_config=_config.DataConfig(prompt_from_task=False),
            adapt_to_pi=spec.adapt_to_pi,
            repack_transforms=_transforms.Group(
                inputs=[
                    PromptFromTaskString(),
                    _transforms.RepackTransform(
                        {
                            "images": {
                                "cam_high": "observation.images.cam_high",
                                "cam_left_wrist": "observation.images.cam_left_wrist",
                                "cam_right_wrist": "observation.images.cam_right_wrist",
                            },
                            "state": "observation.state",
                            "actions": "action",
                            "prompt": "prompt",
                        }
                    ),
                ]
            ),
        ),
        weight_loader=loader,
        assets_base_dir=str(spec.output_root / "assets"),
        checkpoint_base_dir=str(spec.output_root / "checkpoints"),
        seed=spec.seed,
        batch_size=spec.batch_size,
        num_workers=spec.num_workers,
        num_train_steps=spec.num_train_steps,
        log_interval=spec.log_interval,
        save_interval=spec.save_interval,
        keep_period=spec.keep_period,
        overwrite=overwrite,
        resume=resume,
        wandb_enabled=not disable_wandb,
        fsdp_devices=spec.fsdp_devices,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=spec.warmup_steps,
            peak_lr=spec.learning_rate,
            decay_steps=spec.lr_decay_steps,
            decay_lr=spec.learning_rate,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=spec.clip_gradient_norm),
        ema_decay=spec.ema_decay,
    )


def _create_robotwin_dataset(
    data_config: _config.DataConfig,
    action_horizon: int,
    model_config: _model.BaseModelConfig,
) -> _data_loader.Dataset:
    del model_config
    data_root = Path(data_config.repo_id)
    if (data_root / "meta" / "episodes.jsonl").exists():
        return RoboTwinImageParquetDataset(
            data_root,
            action_horizon=action_horizon,
            task_names=ACTIVE_TASK_FILTERS,
        )
    return RoboTwinV3Dataset(
        data_root,
        action_horizon=action_horizon,
        video_backend="pyav",
        task_names=ACTIVE_TASK_FILTERS,
    )


ACTIVE_TASK_FILTERS = DEFAULT_TASK_FILTERS
_ORIGINAL_CREATE_TORCH_DATASET = None


def patch_robotwin_loader(task_filters: tuple[str, ...]) -> None:
    global ACTIVE_TASK_FILTERS, _ORIGINAL_CREATE_TORCH_DATASET
    ACTIVE_TASK_FILTERS = tuple(task_filters)
    if _ORIGINAL_CREATE_TORCH_DATASET is None:
        _ORIGINAL_CREATE_TORCH_DATASET = _data_loader.create_torch_dataset
    _data_loader.create_torch_dataset = _create_robotwin_dataset


def unpatch_robotwin_loader() -> None:
    global _ORIGINAL_CREATE_TORCH_DATASET
    if _ORIGINAL_CREATE_TORCH_DATASET is not None:
        _data_loader.create_torch_dataset = _ORIGINAL_CREATE_TORCH_DATASET
        _ORIGINAL_CREATE_TORCH_DATASET = None


def probe(config: _config.TrainConfig, spec: RobotWinRunSpec) -> None:
    patch_robotwin_loader(spec.task_filters)
    data_config = config.data.create(config.assets_dirs, config.model)
    dataset = _create_robotwin_dataset(data_config, config.model.action_horizon, config.model)
    print("variant", spec.variant, flush=True)
    print("checkpoint", spec.checkpoint_path, flush=True)
    print("dataset length", len(dataset), flush=True)
    sample = dataset[0]
    print("task filters", ACTIVE_TASK_FILTERS, flush=True)
    print("sample keys", sorted(sample.keys()), flush=True)
    print("sample task", sample["task"], flush=True)
    transformed = _data_loader.transform_dataset(dataset, data_config)
    item = transformed[0]
    print("transformed image keys", sorted(item["image"].keys()), flush=True)
    print("transformed state shape", item["state"].shape, flush=True)
    print("transformed actions shape", item["actions"].shape, flush=True)
    loader = _data_loader.create_data_loader(
        config,
        shuffle=False,
        num_batches=1,
        skip_norm_stats=False,
    )
    obs, actions = next(iter(loader))
    print("batch image keys", sorted(obs.images.keys()), flush=True)
    print("batch state shape", obs.state.shape, flush=True)
    print("batch actions shape", actions.shape, flush=True)


def train(config: _config.TrainConfig, spec: RobotWinRunSpec) -> None:
    patch_robotwin_loader(spec.task_filters)
    openpi_train = _load_openpi_train_module()
    try:
        openpi_train.main(config)
    finally:
        unpatch_robotwin_loader()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=tuple((*VARIANT_CHECKPOINTS, *ROBOTWIN_VARIANT_ALIASES, "custom")), default="base")
    parser.add_argument("--custom-variant", default=None)
    parser.add_argument("--checkpoint-path", type=Path, default=None)
    parser.add_argument("--exp-name", default=None)
    parser.add_argument("--data-root", type=Path, default=ROBOTWIN_DATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--save-interval", type=int, default=1000)
    parser.add_argument("--fsdp-devices", type=int, default=4)
    parser.add_argument("--task-filter", action="append", default=None)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--disable-wandb", action="store_true", default=True)
    parser.add_argument("--enable-wandb", action="store_false", dest="disable_wandb")
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.resume and args.overwrite:
        raise ValueError("Cannot use --resume and --overwrite together.")

    task_filters = tuple(args.task_filter or DEFAULT_TASK_FILTERS)
    spec = build_robotwin_spec(
        args.variant,
        exp_name=args.exp_name,
        custom_variant=args.custom_variant,
        custom_checkpoint_path=args.checkpoint_path,
        data_root=args.data_root,
        output_root=args.output_root,
        task_filters=task_filters,
        steps=args.steps,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        fsdp_devices=args.fsdp_devices,
        save_interval=args.save_interval,
        log_interval=args.log_interval,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        seed=args.seed,
    )
    config = build_train_config(
        spec,
        overwrite=args.overwrite,
        resume=args.resume,
        disable_wandb=args.disable_wandb,
    )
    print(spec, flush=True)
    if args.dry_run:
        print(config, flush=True)
        return 0
    if args.probe:
        probe(config, spec)
        return 0
    logging.info("Starting RoboTwin Pi0.5 finetune: %s", spec)
    train(config, spec)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
