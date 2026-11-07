"""Build and validate a portable M4 packet/RGB manifest."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .dataset import sha256_file
from .representation import REPRESENTATION_VERSION


def _portable_path(path: Path, root: Path | None):
    if root is None:
        return str(path)
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def validate_packet(path: str | Path, horizon=32):
    path = Path(path).resolve()
    with np.load(path, allow_pickle=False) as values:
        required = ("pose153", "anchors", "camera_valid", "head_valid", "hand_valid", "alignment_version")
        missing = [key for key in required if key not in values]
        if missing:
            raise ValueError(f"packet missing {missing}: {path}")
        pose = np.asarray(values["pose153"])
        anchors = np.asarray(values["anchors"])
        camera = np.asarray(values["camera_valid"], dtype=bool)
        head = np.asarray(values["head_valid"], dtype=bool)
        hands = np.asarray(values["hand_valid"], dtype=bool)
        version = str(np.asarray(values["alignment_version"]))
    if pose.ndim != 2 or pose.shape[1] != 162 or not np.isfinite(pose).all():
        raise ValueError(f"packet pose153 has invalid shape/values: {path}")
    if (anchors.ndim != 1 or anchors.dtype.kind not in "iu" or len(anchors) == 0
            or np.any(anchors < 0) or np.any(anchors + horizon >= len(pose))):
        raise ValueError(f"packet anchors are invalid: {path}")
    if (camera.shape != (len(pose),) or head.shape != camera.shape
            or hands.shape != (len(pose), 2)):
        raise ValueError(f"packet validity arrays have invalid shapes: {path}")
    if version != REPRESENTATION_VERSION:
        raise ValueError(f"packet version {version!r} is not the public M4 version")
    return dict(frames=len(pose), windows=len(anchors), packet_sha256=sha256_file(path))


def build_manifest(index_path: str | Path, output_dir: str | Path, *, horizon=32,
                   data_root: str | Path | None = None, image_config_sha256="unspecified"):
    """Create ``manifest.json`` from a JSONL index produced by preprocessing.

    Each line contains ``key``, ``packet_path``, ``image_cache_path``,
    ``prompt``, and optional ``source_mp4_sha256``.  Paths are made relative to
    ``data_root`` when possible, so generated manifests do not depend on a
    particular workstation layout.
    """
    index_path = Path(index_path).resolve()
    output_dir = Path(output_dir).resolve()
    root = Path(data_root).resolve() if data_root else None
    entries = []
    for line_number, line in enumerate(index_path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        try:
            packet = Path(row["packet_path"]).expanduser().resolve()
            image = Path(row["image_cache_path"]).expanduser().resolve()
            key = str(row["key"])
            prompt = str(row["prompt"])
        except KeyError as exc:
            raise ValueError(f"index line {line_number} is missing {exc.args[0]}") from exc
        info = validate_packet(packet, horizon=horizon)
        if not image.is_file():
            raise FileNotFoundError(image)
        entries.append({
            "key": key,
            "frames": info["frames"],
            "windows": info["windows"],
            "prompt": prompt,
            "packet_path": _portable_path(packet, root),
            "packet_sha256": info["packet_sha256"],
            "image_cache_path": _portable_path(image, root),
            "image_cache_sha256": sha256_file(image),
            "source_mp4_sha256": str(row.get("source_mp4_sha256", "unspecified")),
        })
    if not entries:
        raise ValueError("index is empty")
    entries.sort(key=lambda item: item["key"])
    manifest = {
        "schema_version": 1,
        "representation": REPRESENTATION_VERSION,
        "data_root": str(root or output_dir),
        "horizon": int(horizon),
        "sample_stride": 2,
        "image_config_sha256": str(image_config_sha256),
        "episodes": entries,
        "total_episodes": len(entries),
        "total_frames": sum(item["frames"] for item in entries),
        "total_windows": sum(item["windows"] for item in entries),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    temporary = manifest_path.with_suffix(".tmp")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(manifest_path)
    ready = {"status": "complete", "manifest_sha256": sha256_file(manifest_path),
             "episodes": manifest["total_episodes"], "windows": manifest["total_windows"]}
    (output_dir / "READY.json").write_text(json.dumps(ready, indent=2) + "\n", encoding="utf-8")
    return manifest
