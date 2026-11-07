"""Serializable specifications for the isolated DexMimicGen A/B study."""

import hashlib
import json
import os
from numbers import Integral
from pathlib import Path


ROOT = Path(os.environ.get("EGO_WEAVE_ROOT", ".")).expanduser()
STUDY_ID = "dexmimicgen_ab"
STUDY_ROOT = ROOT / "experiments" / STUDY_ID
OUTPUT_ROOT = ROOT / "outputs/robot_finetune" / STUDY_ID
DATA_ROOT = ROOT / "data/dexmimicgen"
BASE_CHECKPOINT = Path(
    os.environ.get("PI05_BASE_PARAMS", str(ROOT / "checkpoints/pi05_base/params"))
).expanduser()
OUR_PRETRAIN = Path(
    os.environ.get("EGO_OUR_PARAMS", str(ROOT / "checkpoints/ego_our/params"))
).expanduser()
E48_PRETRAIN = Path(
    os.environ.get("EGO_STYLE_PARAMS", str(ROOT / "checkpoints/ego_style/params"))
).expanduser()
SOURCES = {
    "base": str(BASE_CHECKPOINT),
    "ego_our": str(OUR_PRETRAIN),
    "egodexstyle": str(E48_PRETRAIN),
}
GROUPS = {
    "A": {
        "action_dim": 24,
        "state_dim": 38,
        "task_names": ("TwoArmBoxCleanup", "TwoArmDrawerCleanup", "TwoArmLiftTray"),
    },
    "B": {
        "action_dim": 30,
        "state_dim": 36,
        "task_names": ("TwoArmCoffee", "TwoArmPouring", "TwoArmCanSortRandom"),
    },
}


def make_spec(group, variant, steps, smoke=False):
    """Return a deterministic specification without accessing or writing files.

    The launcher must verify manifests, sources and asset provenance, then
    refuse an existing output before execution. A deterministic identity is
    intentional: repeating the same request must not silently create another
    run or overwrite the previous one. Total steps are always explicit.
    """
    if group not in GROUPS:
        raise ValueError(f"group must be A or B, got {group!r}")
    if variant not in SOURCES:
        raise ValueError(f"unknown pretrained variant {variant!r}")
    if isinstance(steps, bool) or not isinstance(steps, Integral) or steps <= 0:
        raise ValueError(f"steps must be an explicit positive integer, got {steps!r}")
    if not isinstance(smoke, bool):
        raise ValueError("smoke must be a boolean")
    steps = int(steps)
    mode = "smoke" if smoke else "formal"
    dimensions = GROUPS[group]
    run_id = f"{group.lower()}_{variant}_{mode}_{steps}steps"
    name = "dexmimicgen_pi05"
    assets_dir = STUDY_ROOT / "assets" / mode / group
    asset_id = f"dexmimicgen_{group.lower()}_{mode}_v2"
    checkpoint_base = OUTPUT_ROOT / mode / group / variant
    return {
        "study_id": STUDY_ID,
        "group": group,
        "variant": variant,
        "mode": mode,
        "run_id": run_id,
        "task_names": list(dimensions["task_names"]),
        "checkpoint_path": SOURCES[variant],
        "base_checkpoint_path": str(BASE_CHECKPOINT),
        "study_root": str(STUDY_ROOT),
        "data_root": str(DATA_ROOT),
        "output_root": str(OUTPUT_ROOT),
        "manifest_path": str(STUDY_ROOT / "manifests" / mode / f"{group}.json"),
        "assets_dir": str(assets_dir),
        "asset_id": asset_id,
        "norm_stats_path": str(assets_dir / asset_id / "norm_stats.json"),
        "run_dir": str(STUDY_ROOT / "runs" / mode / group / variant / run_id),
        "checkpoint_base_dir": str(checkpoint_base),
        "checkpoint_dir": str(checkpoint_base / name / run_id),
        "name": name,
        "project_name": "dexmimicgen_ab",
        "exp_name": run_id,
        "action_dim": dimensions["action_dim"],
        "state_dim": dimensions["state_dim"],
        "horizon": 50,
        "max_token_len": 640,
        "batch_size": 4 if smoke else 64,
        "num_workers": 0 if smoke else 4,
        "fsdp_devices": 2 if smoke else 4,
        "num_train_steps": steps,
        "seed": 42,
        "learning_rate": 5e-5,
        "warmup_steps": 100,
        "lr_decay_steps": 1000000,
        "decay_lr": 5e-5,
        "ema_decay": .999,
        "clip_gradient_norm": 1.0,
        "save_interval": 10000,
        "keep_period": 10000,
        "checkpoint_mode": "synchronous_host_snapshot",
        "shuffle": True,
        "overwrite": False,
        "resume": False,
        "dtype": "bfloat16",
        "pi05": True,
        "discrete_state_input": True,
    }


def config_fingerprint(spec):
    """Hash every serialized field; key order has no effect on the digest."""
    encoded = json.dumps(spec, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()
