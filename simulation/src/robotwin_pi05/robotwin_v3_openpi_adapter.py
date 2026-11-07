from __future__ import annotations

from collections import OrderedDict
import io
import json
from pathlib import Path
import re
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import torch
from PIL import Image


CAMERA_KEYS = (
    "observation.images.cam_high",
    "observation.images.cam_left_wrist",
    "observation.images.cam_right_wrist",
)

ROBOTWIN_EPISODES_PER_TASK = 550
ROBOTWIN_TASK_ORDER = (
    "adjust_bottle",
    "beat_block_hammer",
    "blocks_ranking_rgb",
    "blocks_ranking_size",
    "click_alarmclock",
    "click_bell",
    "dump_bin_bigbin",
    "grab_roller",
    "handover_block",
    "handover_mic",
    "hanging_mug",
    "lift_pot",
    "move_can_pot",
    "move_pillbottle_pad",
    "move_playingcard_away",
    "move_stapler_pad",
    "open_laptop",
    "open_microwave",
    "pick_diverse_bottles",
    "pick_dual_bottles",
    "place_a2b_left",
    "place_a2b_right",
    "place_bread_basket",
    "place_bread_skillet",
    "place_burger_fries",
    "place_can_basket",
    "place_cans_plasticbox",
    "place_container_plate",
    "place_dual_shoes",
    "place_empty_cup",
    "place_fan",
    "place_mouse_pad",
    "place_object_basket",
    "place_object_scale",
    "place_object_stand",
    "place_phone_stand",
    "place_shoe",
    "press_stapler",
    "put_bottles_dustbin",
    "put_object_cabinet",
    "rotate_qrcode",
    "scan_object",
    "shake_bottle",
    "shake_bottle_horizontally",
    "stack_blocks_three",
    "stack_blocks_two",
    "stack_bowls_three",
    "stack_bowls_two",
    "stamp_seal",
    "turn_switch",
)
ROBOTWIN_TASK_BLOCKS = {task_name: index for index, task_name in enumerate(ROBOTWIN_TASK_ORDER)}


def _normalize_task_text(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def canonicalize_task_name(task_name: str) -> str:
    normalized = _normalize_task_text(task_name)
    normalized = normalized.replace("pill bottle", "pillbottle")
    normalized = normalized.replace("alarm clock", "alarmclock")
    normalized = normalized.replace("playing card", "playingcard")
    normalized = normalized.replace("qr code", "qrcode")
    return normalized.replace(" ", "_")


def resolve_task_names(task_names: list[str] | tuple[str, ...] | None) -> tuple[str, ...]:
    if not task_names:
        return ()
    resolved = []
    unknown = []
    for task_name in task_names:
        canonical = canonicalize_task_name(task_name)
        if canonical in ROBOTWIN_TASK_BLOCKS:
            resolved.append(canonical)
        else:
            unknown.append(task_name)
    if unknown:
        known = ", ".join(ROBOTWIN_TASK_ORDER)
        raise ValueError(f"Unknown RoboTwin task names {unknown}. Known task names: {known}")
    return tuple(resolved)


def compile_task_patterns(task_filters: list[str] | tuple[str, ...] | None) -> tuple[tuple[str, ...], ...]:
    if not task_filters:
        return ()
    patterns = []
    for task_filter in task_filters:
        normalized = _normalize_task_text(task_filter)
        if normalized:
            patterns.append(tuple(normalized.split()))
    return tuple(patterns)


def task_matches(task: str, patterns: tuple[tuple[str, ...], ...]) -> bool:
    if not patterns:
        return True
    normalized_task = _normalize_task_text(task)
    return any(all(token in normalized_task for token in pattern) for pattern in patterns)


def _episode_tasks(episode: dict[str, Any]) -> list[str]:
    tasks = episode.get("tasks", [])
    if isinstance(tasks, np.ndarray):
        tasks = tasks.tolist()
    if isinstance(tasks, (list, tuple)):
        return [str(task) for task in tasks]
    if tasks is None:
        return []
    return [str(tasks)]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if text:
                rows.append(json.loads(text))
    return rows


def _read_jsonl_tasks(path: Path) -> dict[int, str]:
    return {int(row["task_index"]): str(row["task"]) for row in _read_jsonl(path)}


def _image_bytes_to_chw(image_value: Any) -> np.ndarray:
    """Decode a LeRobot image cell into uint8 CHW, matching OpenPI AlohaInputs."""

    if isinstance(image_value, dict):
        data = image_value.get("bytes")
        if data is None:
            data = image_value.get("path")
        if isinstance(data, bytes):
            with Image.open(io.BytesIO(data)) as image:
                array = np.asarray(image.convert("RGB"), dtype=np.uint8)
        else:
            array = np.asarray(data)
    else:
        array = np.asarray(image_value)

    if array.ndim != 3:
        raise ValueError(f"Expected image with 3 dims, got {array.shape}")
    if array.shape[0] in (1, 3) and array.shape[-1] not in (1, 3):
        chw = array
    else:
        chw = np.transpose(array, (2, 0, 1))
    if chw.dtype != np.uint8:
        if np.issubdtype(chw.dtype, np.floating) and float(np.nanmax(chw)) <= 1.5:
            chw = chw * 255.0
        chw = np.clip(chw, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(chw)


class RoboTwinImageParquetDataset(torch.utils.data.Dataset):
    """Reader for RoboTwin v2.1 image-in-parquet LeRobot datasets.

    The historical RoboTwin Pi0.5 training path used this format rather than
    random-seeking AV1 videos.  It stores one parquet per episode and embeds
    PNG bytes for each camera frame.
    """

    def __init__(
        self,
        root: str | Path,
        *,
        action_horizon: int,
        parquet_cache_size: int = 2,
        task_names: list[str] | tuple[str, ...] | None = None,
        task_filters: list[str] | tuple[str, ...] | None = None,
    ) -> None:
        self.root = Path(root)
        self.action_horizon = action_horizon
        self.parquet_cache_size = parquet_cache_size
        self.task_names = resolve_task_names(task_names)
        self.task_patterns = compile_task_patterns(task_filters)
        self.info = json.loads((self.root / "meta" / "info.json").read_text())
        self.episodes = _read_jsonl(self.root / "meta" / "episodes.jsonl")
        self.tasks = _read_jsonl_tasks(self.root / "meta" / "tasks.jsonl")
        self.chunks_size = int(self.info.get("chunks_size", 1000))
        self.episode_rows = self._build_episode_rows(self.episodes)
        self.total_frames = int(sum(row["length"] for row in self.episode_rows))
        self.starts = np.asarray([row["start"] for row in self.episode_rows], dtype=np.int64)
        self._episode_cache: OrderedDict[int, dict[str, Any]] = OrderedDict()

    def __len__(self) -> int:
        return self.total_frames

    def __getitem__(self, idx: int) -> dict[str, Any]:
        if idx < 0:
            idx += self.total_frames
        if idx < 0 or idx >= self.total_frames:
            raise IndexError(idx)

        row_index = int(np.searchsorted(self.starts, idx, side="right") - 1)
        episode_row = self.episode_rows[row_index]
        episode = episode_row["episode"]
        episode_index = int(episode["episode_index"])
        local_frame = int(idx - int(episode_row["start"]))
        data = self._load_episode(episode_index)
        length = int(episode["length"])

        state = np.asarray(data["observation.state"][local_frame], dtype=np.float32)
        action_indices = np.minimum(np.arange(local_frame, local_frame + self.action_horizon), length - 1)
        actions = np.asarray([data["action"][int(i)] for i in action_indices], dtype=np.float32)
        task_index = int(data["task_index"][local_frame])

        result = {
            "observation.state": state,
            "action": actions,
            "timestamp": np.asarray(float(data["timestamp"][local_frame]), dtype=np.float32),
            "frame_index": np.asarray(local_frame, dtype=np.int64),
            "episode_index": np.asarray(episode_index, dtype=np.int64),
            "index": np.asarray(int(data["index"][local_frame]), dtype=np.int64),
            "task_index": np.asarray(task_index, dtype=np.int64),
            "task": self.tasks[task_index],
        }
        for camera_key in CAMERA_KEYS:
            result[camera_key] = _image_bytes_to_chw(data[camera_key][local_frame])
        return result

    def _build_episode_rows(self, episodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
        rows = []
        cursor = 0
        selected_blocks = frozenset(ROBOTWIN_TASK_BLOCKS[name] for name in self.task_names)
        for episode in episodes:
            episode_index = int(episode["episode_index"])
            # v2.1 official_raw was already generated as a five-task subset and
            # episode indices are compacted. If a caller requests task names, use
            # the historical 550 episodes/task block convention when possible.
            if selected_blocks:
                episode_block = episode_index // ROBOTWIN_EPISODES_PER_TASK
                if episode_block not in selected_blocks and len(episodes) > len(selected_blocks) * ROBOTWIN_EPISODES_PER_TASK:
                    continue
            episode_tasks = _episode_tasks(episode)
            if self.task_patterns and not any(task_matches(task, self.task_patterns) for task in episode_tasks):
                continue
            length = int(episode["length"])
            rows.append({"episode": episode, "start": cursor, "length": length})
            cursor += length
        if not rows:
            raise ValueError(f"No RoboTwin v2.1 episodes matched filters: {self.task_patterns}")
        return rows

    def _episode_path(self, episode_index: int) -> Path:
        chunk = episode_index // self.chunks_size
        return self.root / "data" / f"chunk-{chunk:03d}" / f"episode_{episode_index:06d}.parquet"

    def _load_episode(self, episode_index: int) -> dict[str, Any]:
        cached = self._episode_cache.get(episode_index)
        if cached is not None:
            self._episode_cache.move_to_end(episode_index)
            return cached

        table = pq.read_table(
            self._episode_path(episode_index),
            columns=[
                "observation.state",
                "action",
                *CAMERA_KEYS,
                "timestamp",
                "frame_index",
                "episode_index",
                "index",
                "task_index",
            ],
        )
        data = table.to_pydict()
        self._episode_cache[episode_index] = data
        while len(self._episode_cache) > self.parquet_cache_size:
            self._episode_cache.popitem(last=False)
        return data


def _decode_video_frames(*args: Any, **kwargs: Any) -> torch.Tensor:
    from lerobot.common.datasets.video_utils import decode_video_frames

    return decode_video_frames(*args, **kwargs)


class RoboTwinV3Dataset(torch.utils.data.Dataset):
    """Minimal LeRobot v3 reader for the RoboTwin chunked dataset layout."""

    def __init__(
        self,
        root: str | Path,
        *,
        action_horizon: int,
        video_backend: str = "pyav",
        tolerance_s: float = 1e-4,
        parquet_cache_size: int = 2,
        task_names: list[str] | tuple[str, ...] | None = None,
        task_filters: list[str] | tuple[str, ...] | None = None,
    ) -> None:
        self.root = Path(root)
        self.action_horizon = action_horizon
        self.video_backend = video_backend
        self.tolerance_s = tolerance_s
        self.parquet_cache_size = parquet_cache_size
        self.task_names = resolve_task_names(task_names)
        self.task_blocks = frozenset(ROBOTWIN_TASK_BLOCKS[task_name] for task_name in self.task_names)
        self.task_patterns = compile_task_patterns(task_filters)

        self.info = json.loads((self.root / "meta" / "info.json").read_text())
        self.tasks = _read_tasks(self.root / "meta" / "tasks.parquet")
        self.episodes = _read_episodes(self.root / "meta" / "episodes")
        self.episode_rows = self._build_episode_rows(self.episodes)
        self.total_frames = int(sum(row["length"] for row in self.episode_rows))
        self._parquet_cache: OrderedDict[int, dict[str, Any]] = OrderedDict()

    def __len__(self) -> int:
        return self.total_frames

    def __getitem__(self, idx: int) -> dict[str, Any]:
        if idx < 0:
            idx += self.total_frames
        if idx < 0 or idx >= self.total_frames:
            raise IndexError(idx)

        row_index = int(np.searchsorted(self.starts, idx, side="right") - 1)
        episode_row = self.episode_rows[row_index]
        episode = episode_row["episode"]
        local_frame = idx - int(episode_row["start"])
        file_index = int(episode["data/file_index"])
        global_index = int(episode["dataset_from_index"]) + local_frame
        file_offset = global_index - self._data_file_start(file_index)
        file_data = self._load_data_file(file_index)

        state = np.asarray(file_data["observation.state"][file_offset], dtype=np.float32)
        action_indices = np.minimum(
            np.arange(local_frame, local_frame + self.action_horizon),
            int(episode["length"]) - 1,
        )
        global_action_indices = int(episode["dataset_from_index"]) + action_indices
        file_action_indices = global_action_indices - self._data_file_start(file_index)
        actions = np.asarray([file_data["action"][i] for i in file_action_indices], dtype=np.float32)

        timestamp = float(file_data["timestamp"][file_offset])
        images = {}
        for camera_key in CAMERA_KEYS:
            video_path = self._video_path(episode, camera_key)
            video_timestamp = float(episode[f"videos/{camera_key}/from_timestamp"]) + timestamp
            frame = _decode_video_frames(
                video_path,
                [video_timestamp],
                self.tolerance_s,
                self.video_backend,
            ).squeeze(0)
            images[camera_key] = np.asarray(frame, dtype=np.uint8)

        task_index = int(file_data["task_index"][file_offset])
        return {
            "observation.state": state,
            "action": actions,
            "timestamp": np.asarray(timestamp, dtype=np.float32),
            "frame_index": np.asarray(local_frame, dtype=np.int64),
            "episode_index": np.asarray(int(episode["episode_index"]), dtype=np.int64),
            "index": np.asarray(global_index, dtype=np.int64),
            "task_index": np.asarray(task_index, dtype=np.int64),
            "task": self.tasks[task_index],
            **images,
        }

    def _build_episode_rows(self, episodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
        rows = []
        cursor = 0
        for episode in episodes:
            episode_block = int(episode["episode_index"]) // ROBOTWIN_EPISODES_PER_TASK
            if self.task_blocks and episode_block not in self.task_blocks:
                continue
            episode_tasks = _episode_tasks(episode)
            if self.task_patterns and not any(task_matches(task, self.task_patterns) for task in episode_tasks):
                continue
            length = int(episode["length"])
            rows.append({"episode": episode, "start": cursor, "length": length})
            cursor += length
        if not rows:
            raise ValueError(f"No RoboTwin episodes matched task filters: {self.task_patterns}")
        self.starts = np.asarray([row["start"] for row in rows], dtype=np.int64)
        return rows

    def _data_file_start(self, file_index: int) -> int:
        if file_index == 0:
            return 0
        # First episode in each data file carries the global start index.
        for episode in self.episodes:
            if int(episode["data/file_index"]) == file_index:
                return int(episode["dataset_from_index"])
        raise KeyError(file_index)

    def _load_data_file(self, file_index: int) -> dict[str, Any]:
        cached = self._parquet_cache.get(file_index)
        if cached is not None:
            self._parquet_cache.move_to_end(file_index)
            return cached

        path = self.root / "data" / "chunk-000" / f"file-{file_index:03d}.parquet"
        table = pq.read_table(
            path,
            columns=[
                "observation.state",
                "action",
                "timestamp",
                "frame_index",
                "episode_index",
                "index",
                "task_index",
            ],
        )
        data = table.to_pydict()
        self._parquet_cache[file_index] = data
        while len(self._parquet_cache) > self.parquet_cache_size:
            self._parquet_cache.popitem(last=False)
        return data

    def _video_path(self, episode: dict[str, Any], camera_key: str) -> Path:
        chunk_index = int(episode[f"videos/{camera_key}/chunk_index"])
        file_index = int(episode[f"videos/{camera_key}/file_index"])
        return self.root / "videos" / camera_key / f"chunk-{chunk_index:03d}" / f"file-{file_index:03d}.mp4"


def _read_tasks(path: Path) -> dict[int, str]:
    table = pq.read_table(path, columns=["task_index", "task"]).to_pydict()
    return {int(index): task for index, task in zip(table["task_index"], table["task"], strict=True)}


def _read_episodes(path: Path) -> list[dict[str, Any]]:
    files = sorted(path.glob("**/*.parquet"))
    if not files:
        raise FileNotFoundError(f"No episode metadata parquet files found under {path}")

    columns = [
        "episode_index",
        "data/chunk_index",
        "data/file_index",
        "dataset_from_index",
        "dataset_to_index",
        "tasks",
        "length",
    ]
    for camera_key in CAMERA_KEYS:
        columns.extend(
            [
                f"videos/{camera_key}/chunk_index",
                f"videos/{camera_key}/file_index",
                f"videos/{camera_key}/from_timestamp",
                f"videos/{camera_key}/to_timestamp",
            ]
        )

    rows: list[dict[str, Any]] = []
    for file in files:
        data = pq.read_table(file, columns=columns).to_pydict()
        keys = list(data)
        for values in zip(*(data[key] for key in keys), strict=True):
            rows.append(dict(zip(keys, values, strict=True)))
    rows.sort(key=lambda row: int(row["episode_index"]))
    return rows
