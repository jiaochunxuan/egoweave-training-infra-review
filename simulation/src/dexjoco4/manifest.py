from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path


# The release never assumes a workstation-specific path.  Set
# ``DEXJOCO_DATA_ROOT`` (or pass ``--data-root`` to the training script).
DEFAULT_DATA_ROOT = Path(
    os.environ.get("DEXJOCO_DATA_ROOT", "data/dexjoco_lerobot_datasets")
).expanduser()
DEFAULT_TASKS = ("fold_glasses", "hammer_nail", "pick_bucket", "pinch_tongs", "water_plant")
EXCLUDED_TASKS = ("click_mouse",)
FRONT_VIDEO_KEY = "observation.images.front"
WRIST_VIDEO_KEY = "observation.images.wrist"

DEFAULT_PROMPTS = {
    "fold_glasses": "fold the glasses",
    "hammer_nail": "hammer the nail",
    "pick_bucket": "pick up the bucket",
    "pinch_tongs": "pinch with the tongs",
    "water_plant": "water the plant",
}


@dataclass(frozen=True)
class DexJoCoTaskManifest:
    task_name: str
    task_root: Path
    parquet_paths: tuple[Path, ...]
    front_video_paths: tuple[Path, ...]
    wrist_video_paths: tuple[Path, ...]
    total_frames: int
    total_episodes: int
    fps: int
    action_dim: int
    state_dim: int
    front_video_key: str = FRONT_VIDEO_KEY
    wrist_video_key: str = WRIST_VIDEO_KEY
    prompt: str = ""

    @property
    def parquet_path(self) -> Path:
        if len(self.parquet_paths) != 1:
            raise ValueError(f"{self.task_name} has {len(self.parquet_paths)} parquet files")
        return self.parquet_paths[0]


def _shape_width(info: dict, feature_key: str) -> int:
    try:
        shape = info["features"][feature_key]["shape"]
    except KeyError as exc:
        raise ValueError(f"Missing required feature {feature_key!r}") from exc
    if not shape:
        raise ValueError(f"Feature {feature_key!r} has no shape")
    return int(shape[0])


def _video_paths(task_root: Path, video_key: str) -> tuple[Path, ...]:
    video_root = task_root / "videos" / video_key
    paths = tuple(sorted(video_root.glob("chunk-*/*.mp4")))
    if not paths:
        raise FileNotFoundError(f"No videos found for {video_key}: {video_root}")
    return paths


def _parquet_paths(task_root: Path) -> tuple[Path, ...]:
    paths = tuple(sorted((task_root / "data").glob("chunk-*/*.parquet")))
    if not paths:
        raise FileNotFoundError(f"No parquet files found under {task_root / 'data'}")
    return paths


def build_task_manifest(
    data_root: Path | str = DEFAULT_DATA_ROOT,
    *,
    task_names: tuple[str, ...] | None = None,
) -> list[DexJoCoTaskManifest]:
    root = Path(data_root)
    requested = task_names or DEFAULT_TASKS
    manifest: list[DexJoCoTaskManifest] = []
    for task_name in requested:
        if task_name in EXCLUDED_TASKS:
            continue
        task_root = root / task_name
        info_path = task_root / "meta" / "info.json"
        if not info_path.exists():
            raise FileNotFoundError(f"DexJoCo metadata not found: {info_path}")
        info = json.loads(info_path.read_text(encoding="utf-8"))
        features = info.get("features", {})
        if FRONT_VIDEO_KEY not in features or WRIST_VIDEO_KEY not in features:
            raise ValueError(
                f"{task_name} must expose {FRONT_VIDEO_KEY!r} and {WRIST_VIDEO_KEY!r}; "
                f"got {sorted(k for k in features if k.startswith('observation.images'))}"
            )
        manifest.append(
            DexJoCoTaskManifest(
                task_name=task_name,
                task_root=task_root,
                parquet_paths=_parquet_paths(task_root),
                front_video_paths=_video_paths(task_root, FRONT_VIDEO_KEY),
                wrist_video_paths=_video_paths(task_root, WRIST_VIDEO_KEY),
                total_frames=int(info["total_frames"]),
                total_episodes=int(info["total_episodes"]),
                fps=int(info["fps"]),
                action_dim=_shape_width(info, "action"),
                state_dim=_shape_width(info, "observation.state"),
                prompt=DEFAULT_PROMPTS.get(task_name, task_name.replace("_", " ")),
            )
        )
    return manifest
