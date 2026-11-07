from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from dexjoco4.manifest import DEFAULT_DATA_ROOT, DEFAULT_TASKS, DexJoCoTaskManifest, build_task_manifest
from dexjoco4.video_reader import ConcatenatedVideoReader


ACTION_KEY = "action"
STATE_KEY = "observation.state"


def episode_window_starts(episode_lengths: list[int] | tuple[int, ...], *, horizon: int) -> list[int]:
    if horizon <= 0:
        raise ValueError(f"horizon must be positive, got {horizon}")
    starts: list[int] = []
    offset = 0
    for length in episode_lengths:
        length = int(length)
        if length >= horizon:
            starts.extend(range(offset, offset + length - horizon + 1))
        offset += length
    return starts


def _fixed_list_column(table, name: str, width: int) -> np.ndarray:
    values = np.asarray(table[name].to_pylist(), dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != width:
        raise ValueError(f"{name} expected shape (N, {width}), got {values.shape}")
    return values


def _scalar_column(table, name: str, dtype) -> np.ndarray:
    return np.asarray(table[name].to_pylist(), dtype=dtype)


def _episode_lengths(episode_indices: np.ndarray) -> list[int]:
    if episode_indices.size == 0:
        return []
    changes = np.nonzero(episode_indices[1:] != episode_indices[:-1])[0] + 1
    starts = np.concatenate(([0], changes))
    stops = np.concatenate((changes, [episode_indices.size]))
    return [int(stop - start) for start, stop in zip(starts, stops, strict=True)]


@dataclass
class _TaskTable:
    manifest: DexJoCoTaskManifest
    actions: np.ndarray
    states: np.ndarray
    global_indices: np.ndarray
    frame_indices: np.ndarray
    episode_indices: np.ndarray
    front_reader: ConcatenatedVideoReader
    wrist_reader: ConcatenatedVideoReader


def _load_task_table(task: DexJoCoTaskManifest) -> _TaskTable:
    tables = [pq.read_table(path, columns=[ACTION_KEY, STATE_KEY, "index", "frame_index", "episode_index"]) for path in task.parquet_paths]
    table = pq.concat_tables(tables) if len(tables) > 1 else tables[0]
    actions = _fixed_list_column(table, ACTION_KEY, task.action_dim)
    states = _fixed_list_column(table, STATE_KEY, task.state_dim)
    # Videos are NOT 1:1 with parquet episodes (e.g. fold_glasses has 100
    # episodes packed into 4 mp4s), so the segment map must come from counting
    # video frames.  Do it once here in the main process so that pickled
    # workers do not each recount on first access.
    front_reader = ConcatenatedVideoReader(task.front_video_paths, use_seek=True)
    front_reader.ensure_segments()
    wrist_reader = ConcatenatedVideoReader(task.wrist_video_paths, use_seek=True)
    wrist_reader.ensure_segments()
    return _TaskTable(
        manifest=task,
        actions=actions,
        states=states,
        global_indices=_scalar_column(table, "index", np.int64),
        frame_indices=_scalar_column(table, "frame_index", np.int64),
        episode_indices=_scalar_column(table, "episode_index", np.int64),
        front_reader=front_reader,
        wrist_reader=wrist_reader,
    )


class DexJoCoDataset:
    def __init__(
        self,
        tasks: list[DexJoCoTaskManifest],
        *,
        horizon: int,
    ):
        if horizon <= 0:
            raise ValueError(f"horizon must be positive, got {horizon}")
        self.horizon = int(horizon)
        self.tables = [_load_task_table(task) for task in tasks]
        self.refs: list[tuple[int, int]] = []
        for task_index, table in enumerate(self.tables):
            starts = episode_window_starts(_episode_lengths(table.episode_indices), horizon=self.horizon)
            self.refs.extend((task_index, start) for start in starts)
        if not self.refs:
            raise ValueError("DexJoCoDataset has no valid action windows")

    @classmethod
    def from_root(
        cls,
        data_root: Path | str = DEFAULT_DATA_ROOT,
        *,
        task_names: tuple[str, ...] = DEFAULT_TASKS,
        horizon: int = 50,
    ) -> "DexJoCoDataset":
        return cls(
            build_task_manifest(data_root, task_names=task_names),
            horizon=horizon,
        )

    def __len__(self) -> int:
        return len(self.refs)

    def __getitem__(self, index: int) -> dict:
        task_index, row = self.refs[int(index)]
        table = self.tables[task_index]
        task = table.manifest
        global_index = int(table.global_indices[row])
        front = table.front_reader.frame_at(global_index)
        wrist = table.wrist_reader.frame_at(global_index)
        blank_left = np.zeros_like(front)
        actions = table.actions[row : row + self.horizon].astype(np.float32, copy=False)
        if actions.shape != (self.horizon, task.action_dim):
            raise ValueError(f"Invalid action window at row {row}: {actions.shape}")
        result = {
            "image": {
                "base_0_rgb": front,
                "left_wrist_0_rgb": blank_left,
                "right_wrist_0_rgb": wrist,
            },
            "image_mask": {
                "base_0_rgb": np.asarray(True),
                "left_wrist_0_rgb": np.asarray(False),
                "right_wrist_0_rgb": np.asarray(True),
            },
            "state": table.states[row].astype(np.float32, copy=False),
            "actions": actions,
            "prompt": task.prompt,
            "episode_index": np.asarray(table.episode_indices[row], dtype=np.int64),
            "frame_index": np.asarray(table.frame_indices[row], dtype=np.int64),
        }
        return result

    def close(self) -> None:
        for table in self.tables:
            table.front_reader.close()
            table.wrist_reader.close()


def compute_norm_stats(
    data_root: Path | str = DEFAULT_DATA_ROOT,
    *,
    task_names: tuple[str, ...] = DEFAULT_TASKS,
) -> dict:
    from openpi.shared.normalize import NormStats

    state_chunks = []
    action_chunks = []
    for task in build_task_manifest(data_root, task_names=task_names):
        for parquet_path in task.parquet_paths:
            table = pq.read_table(parquet_path, columns=[ACTION_KEY, STATE_KEY])
            action_chunks.append(_fixed_list_column(table, ACTION_KEY, task.action_dim))
            state_chunks.append(_fixed_list_column(table, STATE_KEY, task.state_dim))
    actions = np.concatenate(action_chunks, axis=0)
    states = np.concatenate(state_chunks, axis=0)

    def stats(array: np.ndarray) -> NormStats:
        return NormStats(
            mean=np.mean(array, axis=0),
            std=np.std(array, axis=0),
            q01=np.quantile(array, 0.01, axis=0),
            q99=np.quantile(array, 0.99, axis=0),
        )

    return {"state": stats(states), "actions": stats(actions)}
