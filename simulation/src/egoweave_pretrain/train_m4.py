"""Public M4 pretraining entry point.

The script owns the data/model contract and delegates the optimizer/runtime to
an external OpenPI checkout.  It has a dependency-free preflight mode, so a
manifest and normalization file can be audited before reserving eight GPUs.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
from pathlib import Path
import sys

import numpy as np

from .checkpoint import transfer_base_to_153
from .dataset import MaskAfterNormalize, MaskedM4Dataset, load_manifest
from .normalization import load_stats, sha256_file
from .recipe import M4_RECIPE, recipe_for_public_config


def _json_dump(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def preflight(manifest_path: str | Path, norm_stats_path: str | Path, *, steps: int,
              batch_size: int, devices: int):
    manifest_path = Path(manifest_path).resolve()
    norm_stats_path = Path(norm_stats_path).resolve()
    manifest = load_manifest(manifest_path)
    stats = load_stats(norm_stats_path)
    if int(manifest.get("horizon", M4_RECIPE["action_horizon"])) != M4_RECIPE["action_horizon"]:
        raise ValueError("manifest horizon does not match final M4 recipe")
    if steps < 1 or batch_size < 1 or devices < 1:
        raise ValueError("steps, batch_size and devices must be positive")
    if int(manifest["total_windows"]) < batch_size:
        raise ValueError("manifest is smaller than one global batch")
    contract = {
        "recipe": recipe_for_public_config(),
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "normalization": str(norm_stats_path),
        "normalization_sha256": sha256_file(norm_stats_path),
        "episodes": int(manifest["total_episodes"]),
        "windows": int(manifest["total_windows"]),
        "steps": int(steps),
        "batch_size": int(batch_size),
        "devices": int(devices),
        "final_checkpoint_label": int(steps - 1),
        "input_contract": {"state_dim": 156, "action_dim": 153,
                           "action_horizon": 32, "normalization": stats["normalization"]},
    }
    return contract


def _bootstrap(openpi_root: Path):
    if not (openpi_root / "src" / "openpi").is_dir():
        raise FileNotFoundError(f"OpenPI source tree not found under {openpi_root}")
    for candidate in (openpi_root, openpi_root / "src", openpi_root / "packages" / "openpi-client" / "src"):
        if str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))
    os.environ.setdefault("WANDB_MODE", "disabled")
    os.chdir(openpi_root)


def _require_m4_overlay(pi0_config, model_lib):
    config_fields = {field.name for field in dataclasses.fields(pi0_config.Pi0Config)}
    required = {"image_keys", "preserve_image_token_layout", "action_loss_weights",
                "flow_loss_type", "m4_local21_recon_loss_weight",
                "m4_local21_recon_state_mean", "m4_local21_recon_action_mean"}
    missing = sorted(required - config_fields)
    observation_fields = set(getattr(model_lib.Observation, "__annotations__", {}))
    if "reliability" not in observation_fields:
        missing.append("Observation.reliability")
    if missing:
        raise RuntimeError(
            "The external OpenPI checkout lacks the M4 model overlay fields: "
            + ", ".join(missing)
            + ". Apply the M4 model-side overlay described in the root README before running distributed training."
        )


@dataclasses.dataclass(frozen=True)
class M4DataFactory:
    manifest_path: Path
    norm_dir: Path
    max_token_len: int

    def create(self, assets_dirs, model_config):
        from openpi import transforms
        from openpi.shared import normalize
        from openpi.training.config import DataConfig

        stats = normalize.load(self.norm_dir)
        from .strict_tokenizer import build_strict_tokenizer

        tokenizer = build_strict_tokenizer(self.max_token_len)(self.max_token_len)
        model_transforms = transforms.Group(inputs=[
            MaskAfterNormalize(),
            transforms.ResizeImages(224, 224),
            transforms.TokenizePrompt(tokenizer, discrete_state_input=True),
            transforms.PadStatesAndActions(153, model_state_dim=156),
        ])
        # repo_id is metadata only: the training loader below replaces the
        # upstream dataset factory with MaskedM4Dataset.
        return DataConfig(repo_id="public_m4_manifest", asset_id="m4",
                          norm_stats=stats, use_quantile_norm=False,
                          model_transforms=model_transforms)


def _model_config(norm_dir: Path, max_token_len: int):
    from openpi.models import model as model_lib
    from openpi.models import pi0_config

    _require_m4_overlay(pi0_config, model_lib)
    stats = load_stats(norm_dir / "norm_stats.json")
    geometry = M4_RECIPE["geometry_loss"]
    return pi0_config.Pi0Config(
        pi05=True, action_dim=153, state_dim=156, action_horizon=32,
        max_token_len=max_token_len, paligemma_variant="gemma_2b",
        action_expert_variant="gemma_300m", image_keys=("base_0_rgb",),
        preserve_image_token_layout=True,
        action_loss_weights=tuple(M4_RECIPE["action_loss_weights"]),
        flow_loss_type="mse",
        m4_local21_recon_loss_weight=geometry["weight"],
        m4_local21_recon_left_weight=geometry["left_weight"],
        m4_local21_recon_right_weight=geometry["right_weight"],
        m4_local21_recon_left_scale=geometry["left_scale"],
        m4_local21_recon_right_scale=geometry["right_scale"],
        m4_local21_recon_wrist_pose_relative_to_head=geometry["wrist_pose_relative_to_head"],
        m4_local21_recon_wrist_position_absolute=geometry["wrist_position_absolute"],
        m4_local21_recon_wrist_rotation_absolute=geometry["wrist_rotation_absolute"],
        m4_local21_recon_state_mean=tuple(stats["state"]["mean"]),
        m4_local21_recon_state_std=tuple(stats["state"]["std"]),
        m4_local21_recon_action_mean=tuple(stats["actions"]["mean"]),
        m4_local21_recon_action_std=tuple(stats["actions"]["std"]),
    )


def run(args):
    contract = preflight(args.manifest, args.norm_stats, steps=args.steps,
                         batch_size=args.batch_size, devices=args.devices)
    if args.preflight_only:
        if args.contract:
            _json_dump(args.contract, contract)
        print("M4_PREFLIGHT_PASS", json.dumps(contract, sort_keys=True))
        return contract
    if args.openpi_root is None or args.base_params is None:
        raise ValueError("distributed training requires --openpi-root and --base-params")
    _bootstrap(args.openpi_root.resolve())
    import flax.nnx as nnx
    import flax.traverse_util as traverse
    import jax
    from openpi.models import model as model_lib
    from openpi.training import config as configs, data_loader, optimizer
    from openpi.training import config as training_config
    from scripts import train as base_train

    model_config = _model_config(args.norm_stats.resolve().parent, args.max_token_len)
    data_factory = M4DataFactory(args.manifest.resolve(), args.norm_stats.resolve().parent, args.max_token_len)
    data_config = data_factory.create(None, model_config)
    dataset = MaskedM4Dataset(args.manifest, horizon=M4_RECIPE["action_horizon"])
    transformed = data_loader.transform_dataset(dataset, data_config)

    class BaseLoader:
        def load(self, target):
            source = traverse.flatten_dict(model_lib.restore_params(str(args.base_params), restore_type=np.ndarray), sep="/")
            merged, audit = transfer_base_to_153(source, traverse.flatten_dict(target, sep="/"))
            if args.contract:
                _json_dump(args.contract.with_name("weight_init.json"), audit)
            return traverse.unflatten_dict(merged, sep="/")

    class DataFactory:
        def create(self, assets_dirs, model_config):
            return data_config

    cfg = training_config.TrainConfig(
        name="m4_structured_153d", project_name="egoweave_m4", exp_name=args.run_name,
        model=model_config, data=DataFactory(), weight_loader=BaseLoader(),
        assets_base_dir=str(args.output / "assets"), checkpoint_base_dir=str(args.output / "checkpoints"),
        batch_size=args.batch_size, num_workers=args.workers, num_train_steps=args.steps,
        seed=42, fsdp_devices=args.devices, log_interval=20, save_interval=max(1, args.steps // 5),
        checkpoints_per_run=5, keep_period=1, overwrite=False, resume=False,
        ema_decay=None, wandb_enabled=False, freeze_filter=nnx.Nothing,
        lr_schedule=optimizer.CosineDecaySchedule(warmup_steps=20, peak_lr=2.5e-5,
                                                   decay_steps=args.steps, decay_lr=2.5e-5),
        optimizer=optimizer.AdamW(),
    )
    if jax.device_count() != args.devices or jax.process_count() != 1:
        raise RuntimeError(f"expected one process with {args.devices} visible devices")

    class EpochSampler:
        def __init__(self, n, batch, seed):
            self.n, self.batch, self.seed = n, batch, seed
            self.steps = (n + batch - 1) // batch

        def __iter__(self):
            permutation = np.random.default_rng(self.seed).permutation(self.n)
            total = self.steps * self.batch
            for position in range(total):
                yield int(permutation[position % self.n])

        def __len__(self):
            return self.steps * self.batch

    def create_loader(config, *, sharding=None, shuffle=False, **kwargs):
        if not shuffle:
            raise ValueError("M4 requires deterministic shuffled sampling")
        sampler = EpochSampler(len(dataset), args.batch_size, 42)
        runtime = data_loader.TorchDataLoader(
            transformed, args.batch_size, sharding=sharding, sampler=sampler,
            shuffle=False, num_batches=args.steps, num_workers=args.workers, seed=42,
        )
        return data_loader.DataLoaderImpl(data_config, runtime)

    data_loader.create_data_loader = create_loader
    args.output.mkdir(parents=True, exist_ok=True)
    if args.contract:
        _json_dump(args.contract, contract)
    try:
        base_train.main(cfg)
    finally:
        dataset.close()
    final = args.output / "checkpoints" / args.run_name / str(args.steps - 1)
    result = {"complete": True, "final_checkpoint_label": args.steps - 1,
              "checkpoint": str(final), "contract": contract}
    if args.contract:
        _json_dump(args.contract.with_name("complete.json"), result)
    print("M4_TRAIN_COMPLETE", json.dumps(result, sort_keys=True))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--norm-stats", type=Path, required=True,
                        help="norm_stats.json produced by egoweave_pretrain.prepare")
    parser.add_argument("--output", type=Path, default=Path("runs/m4"))
    parser.add_argument("--contract", type=Path, default=Path("runs/m4/contract.json"))
    parser.add_argument("--openpi-root", type=Path)
    parser.add_argument("--base-params", type=Path)
    parser.add_argument("--steps", type=int, default=18400)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--devices", type=int, default=8)
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--max-token-len", type=int, default=768)
    parser.add_argument("--run-name", default="m4_structured_153d")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args(argv)
    run(args)


if __name__ == "__main__":
    main()
