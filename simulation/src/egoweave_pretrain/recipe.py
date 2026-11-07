"""Machine-independent final-M4 pretraining recipe."""

from __future__ import annotations


ACTION_LOSS_WEIGHTS = tuple(
    [3.0] * 3 + [1.5] * 6 + ([20.0] * 3 + [4.0] * 6) * 2 + [0.3] * 126
)

M4_RECIPE = {
    "name": "m4_structured_153d",
    "representation": "m4_153d_camera_head_progress_local21_v1",
    "state_dim": 156,
    "action_dim": 153,
    "action_horizon": 32,
    "sample_stride": 2,
    "max_token_len": 768,
    "normalization": "mean_std",
    "action_loss": "availability_masked_flow_mse",
    "action_loss_weights": ACTION_LOSS_WEIGHTS,
    "geometry_loss": {
        "enabled": True,
        "kind": "local21_reconstruction",
        "weight": 100.0,
        "left_weight": 1.0,
        "right_weight": 1.0,
        "left_scale": 0.09564778953790665,
        "right_scale": 0.09564778953790665,
        "wrist_pose_relative_to_head": True,
        "wrist_position_absolute": False,
        "wrist_rotation_absolute": False,
    },
    "optimizer": {
        "type": "AdamW",
        "learning_rate": 2.5e-5,
        "warmup_steps": 20,
        "weight_decay": 0.0,
        "seed": 42,
    },
    "distributed": {
        "devices": 8,
        "per_device_batch_size": 32,
        "global_batch_size": 256,
        "workers": 24,
    },
    "checkpoint_label": "updates_minus_one",
    "default_updates": 18400,
    "initialization": "pi05_base",
}


def recipe_for_public_config():
    """Return JSON-serializable recipe data without mutating the constants."""
    recipe = dict(M4_RECIPE)
    recipe["action_loss_weights"] = list(ACTION_LOSS_WEIGHTS)
    recipe["geometry_loss"] = dict(M4_RECIPE["geometry_loss"])
    recipe["optimizer"] = dict(M4_RECIPE["optimizer"])
    recipe["distributed"] = dict(M4_RECIPE["distributed"])
    return recipe
