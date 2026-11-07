"""CPU preprocessing entry point for M4 packets, manifest, and normalization."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .dataset import load_manifest
from .manifest import build_manifest
from .normalization import masked_moments, finalize_moments, save_stats
from .representation import encode_many_masked


def _moments_for_manifest(manifest_path: Path, horizon: int):
    manifest = load_manifest(manifest_path)
    data_root = Path(manifest.get("data_root", manifest_path.parent))
    if not data_root.is_absolute():
        data_root = manifest_path.parent / data_root
    state_total = action_total = None
    for item in manifest["episodes"]:
        packet_path = Path(item["packet_path"])
        if not packet_path.is_absolute():
            packet_path = data_root / packet_path
        with np.load(packet_path, allow_pickle=False) as values:
            packet = {key: values[key] for key in
                      ("pose153", "anchors", "camera_valid", "head_valid", "hand_valid", "alignment_version")}
        for first in range(0, len(packet["anchors"]), 256):
            anchors = packet["anchors"][first:first + 256]
            states, actions, state_mask, action_mask = encode_many_masked(packet, anchors, horizon)
            current = masked_moments(states, state_mask)
            future = masked_moments(actions.reshape(-1, actions.shape[-1]),
                                    action_mask.reshape(-1, action_mask.shape[-1]))
            state_total = current if state_total is None else tuple(a + b for a, b in zip(state_total, current))
            action_total = future if action_total is None else tuple(a + b for a, b in zip(action_total, future))
    if state_total is None:
        raise ValueError("manifest contains no windows")
    return {"normalization": "mean_std",
            "state": finalize_moments(state_total),
            "actions": finalize_moments(action_total)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True,
                        help="JSONL packet/RGB index from preprocessing")
    parser.add_argument("--output", type=Path, required=True,
                        help="portable manifest directory")
    parser.add_argument("--data-root", type=Path,
                        help="optional root used to make manifest paths relative")
    parser.add_argument("--horizon", type=int, default=32)
    parser.add_argument("--image-config-sha256", default="unspecified")
    args = parser.parse_args(argv)
    build_manifest(args.index, args.output, horizon=args.horizon,
                   data_root=args.data_root, image_config_sha256=args.image_config_sha256)
    stats = _moments_for_manifest(args.output / "manifest.json", args.horizon)
    save_stats(args.output / "norm_stats.json", stats)
    print(json.dumps({"manifest": str(args.output / "manifest.json"),
                      "norm_stats": str(args.output / "norm_stats.json"),
                      "episodes": stats["state"]["count"][:1]}, indent=2))


if __name__ == "__main__":
    main()
