"""Indexed masked-window reader for the public M4 manifest contract."""

from __future__ import annotations

from collections import OrderedDict
import hashlib
import io
import json
import os
from pathlib import Path

import numpy as np

from .representation import REPRESENTATION_VERSION, encode_many_masked


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _path_from_manifest(root: Path, value) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (root / path).resolve()


def load_manifest(path: str | Path):
    path = Path(path).resolve()
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("representation") != REPRESENTATION_VERSION:
        raise ValueError("manifest representation does not match public M4 code")
    episodes = manifest.get("episodes")
    if not isinstance(episodes, list) or not episodes:
        raise ValueError("manifest must contain non-empty episodes")
    for item in episodes:
        for key in ("packet_path", "image_cache_path", "frames", "windows", "prompt"):
            if key not in item:
                raise ValueError(f"episode is missing {key}")
    manifest["_path"] = str(path)
    return manifest


class MaskedM4Dataset:
    """Random-access RGB + masked M4 windows.

    The packet is an ``npz`` with ``pose153``, ``anchors``, ``camera_valid``,
    ``head_valid``, ``hand_valid`` and ``alignment_version`` arrays.  The RGB
    cache is an HDF5 file containing ``anchors``, ``jpeg_offsets`` and
    ``jpeg_bytes`` plus a JSON ``header_json`` attribute.  Both files are
    content-addressed in the manifest before a worker opens them.
    """

    def __init__(self, manifest_path: str | Path, *, horizon=32, max_open=8):
        if horizon < 1:
            raise ValueError("horizon must be positive")
        self.manifest_path = Path(manifest_path).resolve()
        self.root = self.manifest_path.parent
        self.manifest = load_manifest(self.manifest_path)
        manifest_root = Path(self.manifest.get("data_root", self.root)).expanduser()
        self.data_root = (manifest_root if manifest_root.is_absolute() else self.root / manifest_root).resolve()
        self.horizon = int(horizon)
        self.max_open = int(max_open)
        if self.max_open < 1:
            raise ValueError("max_open must be positive")
        self.ends = np.cumsum([int(e["windows"]) for e in self.manifest["episodes"]])
        if int(self.manifest.get("total_windows", self.ends[-1])) != int(self.ends[-1]):
            raise ValueError("manifest total_windows does not match episodes")
        self.handles = OrderedDict()
        self.pid = os.getpid()

    def __len__(self):
        return int(self.ends[-1])

    def __getstate__(self):
        return {**self.__dict__, "handles": OrderedDict(), "pid": None}

    def close(self):
        for handle, _ in self.handles.values():
            handle.close()
        self.handles.clear()

    def _episode(self, index):
        import h5py

        if self.pid != os.getpid():
            self.close()
            self.pid = os.getpid()
        if index in self.handles:
            self.handles.move_to_end(index)
            return self.handles[index]
        entry = self.manifest["episodes"][index]
        packet_path = _path_from_manifest(self.data_root, entry["packet_path"])
        if entry.get("packet_sha256") and sha256_file(packet_path) != entry["packet_sha256"]:
            raise ValueError("pose packet hash changed")
        with np.load(packet_path, allow_pickle=False) as values:
            packet = {key: values[key] for key in
                      ("pose153", "anchors", "camera_valid", "head_valid", "hand_valid", "alignment_version")}
        if str(np.asarray(packet["alignment_version"])) != REPRESENTATION_VERSION:
            raise ValueError("packet alignment version differs from public M4 version")
        image_path = _path_from_manifest(self.data_root, entry["image_cache_path"])
        if entry.get("image_cache_sha256") and sha256_file(image_path) != entry["image_cache_sha256"]:
            raise ValueError("RGB cache hash changed")
        handle = h5py.File(image_path, "r")
        try:
            header = json.loads(handle.attrs["header_json"])
            if (not header.get("committed") or header.get("key") != entry["key"]
                    or int(header.get("frames", -1)) != int(entry["frames"])
                    or (self.manifest.get("image_config_sha256", "unspecified") != "unspecified"
                        and header.get("config_sha256") != self.manifest["image_config_sha256"])):
                raise ValueError("RGB cache header does not match manifest")
            image_anchors = handle["anchors"][:]
            offsets = handle["jpeg_offsets"][:]
            locations = np.searchsorted(image_anchors, packet["anchors"])
            if np.any(locations >= len(image_anchors)):
                raise ValueError("RGB cache is missing a packet anchor")
            if not np.array_equal(image_anchors[locations], packet["anchors"]):
                raise ValueError("RGB cache is missing a packet anchor")
            while len(self.handles) >= self.max_open:
                self.handles.popitem(last=False)[1][0].close()
            value = handle, packet, offsets, locations
            self.handles[index] = value
            return value
        except Exception:
            handle.close()
            raise

    def __getitem__(self, index):
        from PIL import Image

        if not 0 <= index < len(self):
            raise IndexError(index)
        episode = int(np.searchsorted(self.ends, index, side="right"))
        offset = index - (int(self.ends[episode - 1]) if episode else 0)
        handle, packet, offsets, locations = self._episode(episode)
        anchor = int(packet["anchors"][offset])
        state, actions, state_valid, reliability = encode_many_masked(
            packet, np.asarray([anchor]), horizon=self.horizon)
        row = int(locations[offset])
        payload = handle["jpeg_bytes"][int(offsets[row]):int(offsets[row + 1])].tobytes()
        with Image.open(io.BytesIO(payload)) as image_file:
            image = np.asarray(image_file.convert("RGB"), dtype=np.uint8).copy()
        if image.shape != (224, 224, 3):
            raise ValueError("M4 expects cached RGB images at 224x224")
        prompt = str(self.manifest["episodes"][episode]["prompt"])
        missing = [side for side, valid in zip(("left", "right"), packet["hand_valid"][anchor]) if not valid]
        if missing:
            prompt += " Current hand tracking unavailable: " + ", ".join(missing) + "."
        return {
            "state": state[0],
            "actions": actions[0],
            "state_valid": state_valid[0],
            "reliability": reliability[0],
            "prompt": prompt,
            "image": {"base_0_rgb": image},
            "image_mask": {"base_0_rgb": True},
        }


class MaskAfterNormalize:
    """Apply the explicit zero-placeholder policy after standardization."""

    def __call__(self, data):
        data = dict(data)
        data["state"] = np.where(data.pop("state_valid"), data["state"], 0).astype(np.float32)
        data["actions"] = np.where(data["reliability"] > 0, data["actions"], 0).astype(np.float32)
        return data
