from __future__ import annotations

import argparse
from dataclasses import dataclass
from dataclasses import replace
import os
from pathlib import Path
import sys

# Make direct script execution independent of the caller's PYTHONPATH.
PACKAGE_ROOT = Path(__file__).resolve().parents[2]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from dexjoco4.manifest import DEFAULT_DATA_ROOT, DEFAULT_TASKS
from dexjoco4.parquet_video_dataset import DexJoCoDataset, compute_norm_stats
from dexjoco4.model_adapters.checkpoint_loader import make_dexjoco_weight_loader


BASE_PI05_PARAMS = Path(
    os.environ.get("PI05_BASE_PARAMS", "checkpoints/pi05_base/params")
).expanduser()
# This is intentionally a user-supplied path.  The released code does not
# contain or name the private human-pretraining runs.
EGO_PI05_PARAMS = Path(
    os.environ.get("EGO_PRETRAIN_PARAMS", "checkpoints/ego_pretrain/params")
).expanduser()
OUTPUT_ROOT = Path(
    os.environ.get("DEXJOCO_OUTPUT_ROOT", "outputs/dexjoco")
).expanduser()
OPENPI_ROOT = Path(
    os.environ.get("DEXJOCO_OPENPI_ROOT", "external/openpi")
).expanduser()


@dataclass(frozen=True)
class DexJoCoRunSpec:
    name: str
    checkpoint_path: Path
    output_root: Path
    data_root: Path = DEFAULT_DATA_ROOT
    task_names: tuple[str, ...] = DEFAULT_TASKS
    action_dim: int = 22
    state_dim: int = 23
    horizon: int = 50
    batch_size: int = 32
    num_workers: int = 16
    fsdp_devices: int = 4
    # The public Chapter 4 protocol uses 10000 optimizer updates; OpenPI
    # writes the resulting final checkpoint with the zero-based label 9999.
    num_train_steps: int = 10000
    save_interval: int = 10000
    keep_period: int = 10000
    learning_rate: float = 5e-5
    warmup_steps: int = 100
    seed: int = 42
    action_head_seed: int = 20260807
    norm_stats_asset_id: str = "dexjoco4_singlehand_v1"


@dataclass(frozen=True)
class DexJoCoOpenPIPatch:
    create_torch_dataset: object
    create_torch_data_loader: object


@dataclass(frozen=True)
class DexJoCoDataConfigFactory:
    data_root: Path = DEFAULT_DATA_ROOT
    task_names: tuple[str, ...] = DEFAULT_TASKS
    asset_id: str = "dexjoco4_singlehand_v1"
    repo_id: str = "dexjoco4"

    def create(self, assets_dirs, model_config):
        _ensure_openpi_paths()
        import openpi.training.config as _config
        from openpi.shared import normalize as _normalize

        norm_stats = _normalize.load(Path(assets_dirs) / self.asset_id)
        return _config.DataConfig(
            repo_id=self.repo_id,
            asset_id=self.asset_id,
            norm_stats=norm_stats,
            model_transforms=_config.ModelTransformFactory(default_prompt=None)(model_config),
            use_quantile_norm=False,
        )


def _ensure_openpi_paths(openpi_root: Path = OPENPI_ROOT) -> None:
    for path in reversed((openpi_root, openpi_root / "src", openpi_root / "packages/openpi-client/src")):
        text = str(path)
        while text in sys.path:
            sys.path.remove(text)
        sys.path.insert(0, text)


def build_run_specs(
    *,
    output_root: Path = OUTPUT_ROOT,
    data_root: Path = DEFAULT_DATA_ROOT,
    task_names: tuple[str, ...] = DEFAULT_TASKS,
    num_workers: int = 16,
    seed: int = 42,
    action_head_seed: int = 20260807,
    custom_variant: str | None = None,
    custom_checkpoint_path: Path | None = None,
) -> dict[str, DexJoCoRunSpec]:
    common = {
        "output_root": output_root,
        "data_root": data_root,
        "task_names": task_names,
        "num_workers": num_workers,
        "seed": seed,
        "action_head_seed": action_head_seed,
    }
    specs = {
        "base_robot": DexJoCoRunSpec(name="base_robot", checkpoint_path=BASE_PI05_PARAMS, **common),
        "ego_robot": DexJoCoRunSpec(name="ego_robot", checkpoint_path=EGO_PI05_PARAMS, **common),
    }
    if custom_variant is not None or custom_checkpoint_path is not None:
        if not custom_variant:
            raise ValueError("--custom-variant is required when using a custom checkpoint")
        if custom_checkpoint_path is None:
            raise ValueError("--checkpoint-path is required when using a custom variant")
        specs[custom_variant] = DexJoCoRunSpec(
            name=custom_variant,
            checkpoint_path=custom_checkpoint_path,
            **common,
        )
    return specs


def build_train_config(
    spec: DexJoCoRunSpec,
    *,
    overwrite: bool = False,
    resume: bool = False,
):
    _ensure_openpi_paths()
    import openpi.models.pi0_config as pi0_config
    import openpi.training.config as _config
    import openpi.training.optimizer as _optimizer
    import flax.nnx as _nnx

    model = pi0_config.Pi0Config(
        action_dim=spec.action_dim,
        state_dim=spec.state_dim,
        action_horizon=spec.horizon,
        max_token_len=640,
        pi05=True,
        discrete_state_input=True,
        )
    # Full-parameter fine-tuning is the public protocol.
    _freeze_filter = _nnx.Nothing

    return _config.TrainConfig(
        name="dexjoco4",
        project_name="robot_finetune",
        exp_name=spec.name,
        model=model,
        weight_loader=make_dexjoco_weight_loader(spec.checkpoint_path),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=spec.warmup_steps,
            peak_lr=spec.learning_rate,
            decay_steps=spec.num_train_steps,
            decay_lr=spec.learning_rate,
        ),
        data=DexJoCoDataConfigFactory(
            data_root=spec.data_root,
            task_names=spec.task_names,
            asset_id=spec.norm_stats_asset_id,
        ),
        assets_base_dir=str(spec.output_root / "assets"),
        checkpoint_base_dir=str(spec.output_root / "checkpoints"),
        seed=spec.action_head_seed,
        batch_size=spec.batch_size,
        num_workers=spec.num_workers,
        num_train_steps=spec.num_train_steps,
        log_interval=10,
        save_interval=spec.save_interval,
        keep_period=spec.keep_period,
        overwrite=overwrite,
        resume=resume,
        wandb_enabled=False,
        fsdp_devices=spec.fsdp_devices,
        freeze_filter=_freeze_filter,
    )


def ensure_norm_stats_assets(spec: DexJoCoRunSpec, train_config) -> Path:
    _ensure_openpi_paths()
    from openpi.shared import normalize as _normalize

    stats_dir = Path(train_config.assets_dirs) / spec.norm_stats_asset_id
    stats_file = stats_dir / "norm_stats.json"
    if not stats_file.exists():
        stats = compute_norm_stats(spec.data_root, task_names=spec.task_names)
        _normalize.save(stats_dir, stats)
    return stats_file


def install_dexjoco_dataset_patch(spec: DexJoCoRunSpec):
    _ensure_openpi_paths()
    import openpi.training.data_loader as _data_loader

    original_dataset = _data_loader.create_torch_dataset
    original_loader = _data_loader.create_torch_data_loader

    def create_torch_dataset(data_config, action_horizon, model_config):
        if getattr(data_config, "repo_id", None) == "dexjoco4":
            return DexJoCoDataset.from_root(
                spec.data_root,
                task_names=spec.task_names,
                horizon=action_horizon,
            )
        return original_dataset(data_config, action_horizon, model_config)

    def create_torch_data_loader(data_config, *args, **kwargs):
        # Keep OpenPI default shuffle=True to match the original 4750-step
        # ego_robot run from 2026-08-10.
        return original_loader(data_config, *args, **kwargs)

    _data_loader.create_torch_dataset = create_torch_dataset
    _data_loader.create_torch_data_loader = create_torch_data_loader
    return DexJoCoOpenPIPatch(
        create_torch_dataset=original_dataset,
        create_torch_data_loader=original_loader,
    )


def uninstall_dexjoco_dataset_patch(patch: DexJoCoOpenPIPatch) -> None:
    _ensure_openpi_paths()
    import openpi.training.data_loader as _data_loader

    _data_loader.create_torch_dataset = patch.create_torch_dataset
    _data_loader.create_torch_data_loader = patch.create_torch_data_loader


def run_training(spec: DexJoCoRunSpec, *, overwrite: bool = False, resume: bool = False) -> None:
    _ensure_openpi_paths()
    from scripts import train as openpi_train

    train_config = build_train_config(spec, overwrite=overwrite, resume=resume)
    ensure_norm_stats_assets(spec, train_config)
    patch = install_dexjoco_dataset_patch(spec)
    try:
        openpi_train.main(train_config)
    finally:
        uninstall_dexjoco_dataset_patch(patch)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=("base_robot", "ego_robot", "custom"), required=True)
    parser.add_argument(
        "--custom-variant",
        default=None,
        help="Name for an explicit non-built-in checkpoint variant.",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=None,
        help="Explicit params checkpoint path for --variant custom.",
    )
    parser.add_argument(
        "--exp-name",
        default=None,
        help="Optional run/checkpoint directory name override.",
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--fsdp-devices", type=int, default=None)
    parser.add_argument("--save-interval", type=int, default=None,
        help="Checkpoint interval in training steps.")
    parser.add_argument("--keep-period", type=int, default=None,
        help="Checkpoint retention period; checkpoints at multiples are kept.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    specs = build_run_specs(
        output_root=args.output_root,
        data_root=args.data_root,
        custom_variant=args.custom_variant if args.variant == "custom" else None,
        custom_checkpoint_path=args.checkpoint_path if args.variant == "custom" else None,
    )
    spec_key = args.custom_variant if args.variant == "custom" else args.variant
    spec = specs[spec_key]
    if (
        args.steps is not None
        or args.batch_size is not None
        or args.num_workers is not None
        or args.fsdp_devices is not None
        or args.save_interval is not None
        or args.keep_period is not None
    ):
        spec = replace(
            spec,
            num_train_steps=args.steps if args.steps is not None else spec.num_train_steps,
            batch_size=args.batch_size if args.batch_size is not None else spec.batch_size,
            num_workers=args.num_workers if args.num_workers is not None else spec.num_workers,
            fsdp_devices=args.fsdp_devices if args.fsdp_devices is not None else spec.fsdp_devices,
            save_interval=args.save_interval if args.save_interval is not None else spec.save_interval,
            keep_period=args.keep_period if args.keep_period is not None else spec.keep_period,
        )
    if args.exp_name is not None:
        spec = replace(spec, name=args.exp_name)
    if args.dry_run:
        print(spec)
        return 0
    run_training(spec, overwrite=args.overwrite, resume=args.resume)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
