"""Immutable manifests and a lazy, worker-safe native HDF5 dataset.

SHA-256 fingerprints are computed once while building a manifest. Readers
validate the manifest hash and source size/mtime on each new HDF5 open; this
source stat check is deliberately not a repeated cryptographic file audit.
Images retain the stored robomimic orientation without flipping or resizing.
"""

from collections import OrderedDict
import hashlib
import json
import operator
import os
from pathlib import Path
import re
from typing import Any, Iterator

import h5py
import numpy as np

from .contracts import (
    ACTION_FIELDS, GROUP_TASKS, STATE_FIELDS, action_dim, action_keys, camera_map,
    state_dim, state_keys, task_contract, task_group, validate_env_args, validate_group,
)


def _canonical_hash(value: dict[str, Any]) -> str:
    payload = {key: item for key, item in value.items() if key != "manifest_sha256"}
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _plain(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.ndarray):
        return [_plain(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _source_stat(path: Path) -> dict[str, int]:
    stat = path.stat()
    return {"size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def _check_source_stat(source: dict[str, Any]) -> None:
    current = _source_stat(Path(source["path"]))
    if any(source["fingerprint"].get(key) != value for key, value in current.items()):
        raise ValueError(f"Source fingerprint changed since manifest creation: {source['path']}")


def _array(group: h5py.Group, key: str, width: int, length: int, selection: Any) -> np.ndarray:
    if key not in group or not isinstance(group[key], h5py.Dataset):
        raise ValueError(f"Missing required dataset {group.name}/{key}")
    dataset = group[key]
    if dataset.shape != (length, width):
        raise ValueError(f"{dataset.name}: expected shape {(length, width)}, got {dataset.shape}")
    if dataset.dtype.kind not in "fiu":
        raise ValueError(f"{dataset.name}: expected real numeric dtype, got {dataset.dtype}")
    with np.errstate(over="ignore", invalid="ignore"):
        result = np.asarray(dataset[selection], dtype=np.float32)
    if not np.isfinite(result).all():
        raise ValueError(f"{dataset.name}: nonfinite state/action values")
    return result


def _vector(ep: h5py.Group, folder: str, fields: tuple[tuple[str, int], ...],
            length: int, selection: Any) -> np.ndarray:
    if folder not in ep or not isinstance(ep[folder], h5py.Group):
        raise ValueError(f"{ep.name}: missing required group {folder}")
    return np.concatenate([_array(ep[folder], key, width, length, selection)
                           for key, width in fields], axis=-1)


def _validate_episode(ep: h5py.Group, task: str) -> int:
    raw_length = ep.attrs.get("num_samples")
    if (not isinstance(raw_length, (int, np.integer))
            or isinstance(raw_length, (bool, np.bool_)) or int(raw_length) < 1):
        raise ValueError(f"{ep.name}: num_samples must be a positive integer")
    length = int(raw_length)
    contract = task_contract(task)
    # Validate every selected state/action frame in bounded chunks; never decode images here.
    for begin in range(0, length, 4096):
        selection = slice(begin, min(begin + 4096, length))
        _vector(ep, "obs", contract.state_fields, length, selection)
        _vector(ep, "action_dict", contract.action_fields, length, selection)
    for raw in camera_map(task).values():
        if raw not in ep["obs"] or not isinstance(ep["obs"][raw], h5py.Dataset):
            raise ValueError(f"{ep.name}: missing required camera {raw}")
        image = ep["obs"][raw]
        if image.shape != (length, 84, 84, 3) or image.dtype != np.uint8:
            raise ValueError(f"{image.name}: expected uint8 image shape {(length, 84, 84, 3)}, "
                             f"got {image.dtype} {image.shape}")
    return length


def build_manifest(task_paths: dict[str, str], group: str, output: str | Path,
                   max_episodes: int | None = None, smoke: bool = False) -> dict[str, Any]:
    """Audit selected episodes and exclusively create a provenance-bearing manifest.

    max_episodes limits episodes per task, in numeric demo order. Formal
    manifests require all three tasks and reject any truncated/probe source.
    """
    validate_group(group)
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite manifest: {output}")
    if not task_paths:
        raise ValueError("task_paths must not be empty")
    if not isinstance(smoke, bool):
        raise ValueError("smoke must be a boolean")
    if max_episodes is not None and (isinstance(max_episodes, bool)
                                    or not isinstance(max_episodes, int) or max_episodes < 1):
        raise ValueError("max_episodes must be a positive integer or None")
    for task in task_paths:
        if task_group(task) != group:
            raise ValueError(f"Task {task} does not belong to group {group}")
    if not smoke and set(task_paths) != set(GROUP_TASKS[group]):
        raise ValueError(f"Formal group {group} requires all three tasks: {GROUP_TASKS[group]}")
    sources = []
    for task in GROUP_TASKS[group]:
        if task not in task_paths:
            continue
        path = Path(task_paths[task]).resolve(strict=True)
        initial_stat = _source_stat(path)
        with h5py.File(path, "r") as handle:
            if "data" not in handle or not isinstance(handle["data"], h5py.Group):
                raise ValueError(f"{path}: missing data group")
            data = handle["data"]
            try:
                env_args = json.loads(_plain(data.attrs["env_args"]))
            except (KeyError, TypeError, json.JSONDecodeError) as exc:
                raise ValueError(f"{path}: valid env_args metadata is required") from exc
            validate_env_args(task, env_args)
            provenance = {}
            marker_names = ("smoke_only", "artificial_episode_truncation",
                            "probe_only_first_episode_slice", "probe_source_url",
                            "source_url", "source_sha256")
            for attrs in (handle.attrs, data.attrs):
                for key in marker_names:
                    if key in attrs:
                        value = _plain(attrs[key])
                        # A true root marker must survive a false data marker.
                        provenance[key] = (bool(provenance.get(key, False)) or bool(value)
                                           if key in marker_names[:3] else value)
            smoke_only = any(bool(provenance.get(key, False)) for key in marker_names[:3])
            if not smoke and smoke_only:
                raise ValueError(f"Formal manifest cannot use smoke_only/probe data: {path}")
            keys = sorted((key for key in data if re.fullmatch(r"demo_\d+", key)),
                          key=lambda key: int(key.split("_")[-1]))
            if max_episodes is not None:
                keys = keys[:max_episodes]
            if not keys:
                raise ValueError(f"{path}: no selected demo episodes")
            episodes = [{"key": key, "length": _validate_episode(data[key], task)} for key in keys]
            source = {"task": task, "path": str(path), "env_args": env_args,
                      "fingerprint": initial_stat, "smoke_only": smoke_only,
                      "provenance": provenance, "episodes": episodes,
                      "camera_map": camera_map(task), "prompt": task_contract(task).prompt}
        source["fingerprint"]["sha256"] = _sha256(path)
        _check_source_stat(source)
        sources.append(source)
    manifest = {
        "schema_version": 1, "group": group, "smoke": smoke, "split": "train",
        "state_dim": state_dim(group), "action_dim": action_dim(group),
        "state_keys": list(state_keys(group)), "action_keys": list(action_keys(group)),
        "total_episodes": sum(len(source["episodes"]) for source in sources),
        "total_frames": sum(ep["length"] for source in sources for ep in source["episodes"]),
        "sources": sources,
        "source_validation": "sha256_at_creation; size_and_mtime_on_open",
        "image_orientation": "stored_robomimic_no_flip",
    }
    manifest["manifest_sha256"] = _canonical_hash(manifest)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(manifest, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    return manifest


def load_manifest(path: str | Path) -> dict[str, Any]:
    """Check immutable content hash and revalidate its declared data contract."""
    with Path(path).open(encoding="utf-8") as stream:
        manifest = json.load(stream)
    if manifest.get("manifest_sha256") != _canonical_hash(manifest):
        raise ValueError("Manifest hash mismatch")
    if manifest.get("schema_version") != 1 or manifest.get("split") != "train":
        raise ValueError("Only schema_version=1 training manifests are supported")
    group = validate_group(manifest.get("group"))
    if (manifest.get("state_dim") != state_dim(group) or manifest.get("action_dim") != action_dim(group)
            or manifest.get("state_keys") != list(state_keys(group))
            or manifest.get("action_keys") != list(action_keys(group))):
        raise ValueError("Manifest dimensions or field order do not match the native contract")
    sources = manifest.get("sources")
    if not isinstance(sources, list) or not sources or not isinstance(manifest.get("smoke"), bool):
        raise ValueError("Manifest must declare sources and boolean smoke mode")
    tasks = [source["task"] for source in sources]
    if len(tasks) != len(set(tasks)):
        raise ValueError("Manifest has duplicate task sources")
    if not manifest["smoke"] and set(tasks) != set(GROUP_TASKS[group]):
        raise ValueError("Formal manifest requires all three tasks")
    total_frames, total_episodes = 0, 0
    for source in sources:
        task = source["task"]
        if task_group(task) != group:
            raise ValueError(f"{task}: mixed group manifest")
        validate_env_args(task, source["env_args"])
        if source.get("camera_map") != camera_map(task) or source.get("prompt") != task_contract(task).prompt:
            raise ValueError(f"{task}: camera/prompt contract mismatch")
        if not manifest["smoke"] and source.get("smoke_only"):
            raise ValueError("Formal manifest cannot use smoke_only source")
        if not Path(source["path"]).is_absolute():
            raise ValueError("Manifest source paths must be absolute")
        episodes = source.get("episodes")
        if not isinstance(episodes, list) or not episodes:
            raise ValueError("Manifest source has no selected episodes")
        seen = set()
        for ep in episodes:
            key, length = ep.get("key"), ep.get("length")
            if (not isinstance(key, str) or not re.fullmatch(r"demo_\d+", key) or key in seen
                    or not isinstance(length, int) or isinstance(length, bool) or length < 1):
                raise ValueError("Invalid or duplicate manifest episode")
            seen.add(key)
            total_frames += length
            total_episodes += 1
    if manifest.get("total_frames") != total_frames or manifest.get("total_episodes") != total_episodes:
        raise ValueError("Manifest episode totals are inconsistent")
    return manifest


class HDF5Dataset:
    """Map-style samples, one per selected frame, with episode-local action chunks."""

    def __init__(self, manifest_path: str | Path, horizon: int = 50, max_open_files: int = 4):
        if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon < 1:
            raise ValueError("horizon must be a positive integer")
        if isinstance(max_open_files, bool) or not isinstance(max_open_files, int) or not 1 <= max_open_files <= 4:
            raise ValueError("max_open_files must be between 1 and 4")
        self.manifest_path = str(Path(manifest_path).resolve())
        self.manifest = load_manifest(self.manifest_path)
        self.group = self.manifest["group"]
        self.state_dim = state_dim(self.group)
        self.action_dim = action_dim(self.group)
        self.horizon = horizon
        self.max_open_files = max_open_files
        self._episodes = [(source_index, ep["key"], ep["length"])
                          for source_index, source in enumerate(self.manifest["sources"])
                          for ep in source["episodes"]]
        self._ends = np.cumsum([ep[2] for ep in self._episodes], dtype=np.int64)
        self._handles: OrderedDict[int, h5py.File] = OrderedDict()
        self._pid = os.getpid()

    def __len__(self) -> int:
        return int(self._ends[-1])

    def _handle(self, source_index: int) -> h5py.File:
        if self._pid != os.getpid():
            self.close()
            self._pid = os.getpid()
        if source_index in self._handles:
            self._handles.move_to_end(source_index)
            return self._handles[source_index]
        source = self.manifest["sources"][source_index]
        _check_source_stat(source)
        if len(self._handles) >= self.max_open_files:
            _, oldest = self._handles.popitem(last=False)
            oldest.close()
        handle = h5py.File(source["path"], "r")
        self._handles[source_index] = handle
        return handle

    def __getitem__(self, index: int) -> dict[str, Any]:
        index = operator.index(index)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        ep_index = int(np.searchsorted(self._ends, index, side="right"))
        start = 0 if ep_index == 0 else int(self._ends[ep_index - 1])
        frame = index - start
        source_index, key, length = self._episodes[ep_index]
        source = self.manifest["sources"][source_index]
        ep = self._handle(source_index)["data"][key]
        state = _vector(ep, "obs", STATE_FIELDS[self.group], length, frame)
        end = min(frame + self.horizon, length)
        actions = _vector(ep, "action_dict", ACTION_FIELDS[self.group], length, slice(frame, end))
        if len(actions) < self.horizon:
            actions = np.concatenate((actions, np.repeat(actions[-1:], self.horizon - len(actions), axis=0)), axis=0)
        images = {}
        for canonical, raw in source["camera_map"].items():
            if raw not in ep["obs"]:
                raise ValueError(f"{ep.name}: missing required camera {raw}")
            image = np.asarray(ep["obs"][raw][frame])
            if image.shape != (84, 84, 3) or image.dtype != np.uint8:
                raise ValueError(f"{ep.name}/{raw}: expected uint8 HWC 84x84 image")
            images[canonical] = image
        return {"image": images, "image_mask": {key: np.bool_(True) for key in images},
                "state": state, "actions": actions, "prompt": source["prompt"]}

    def iter_episode_arrays(self) -> Iterator[dict[str, np.ndarray]]:
        """Yield unpadded selected training arrays without reading camera pixels."""
        for source_index, key, length in self._episodes:
            ep = self._handle(source_index)["data"][key]
            yield {"state": _vector(ep, "obs", STATE_FIELDS[self.group], length, slice(None)),
                   "actions": _vector(ep, "action_dict", ACTION_FIELDS[self.group], length, slice(None))}

    def close(self) -> None:
        for handle in getattr(self, "_handles", {}).values():
            handle.close()
        self._handles = OrderedDict()

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        state["_handles"] = OrderedDict()
        state["_pid"] = None
        return state

    def __del__(self):
        self.close()


def iter_training_arrays(manifest_path: str | Path) -> Iterator[dict[str, np.ndarray]]:
    """Read only states and native actions from selected manifest training episodes."""
    dataset = HDF5Dataset(manifest_path)
    try:
        yield from dataset.iter_episode_arrays()
    finally:
        dataset.close()
