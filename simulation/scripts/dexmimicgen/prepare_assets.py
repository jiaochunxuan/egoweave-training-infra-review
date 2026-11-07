"""Compute exact native state/action normalization for manifest-selected episodes.

Only selected state/action arrays are read. Images, padded action chunks, and
privileged simulator state are not used. --output-dir is the complete asset-ID
directory, which must not already exist.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))


def prepare_assets(manifest, output_dir):
    destination = Path(output_dir).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite asset directory {destination}")
    manifest_path = Path(manifest).expanduser().resolve()
    manifest_bytes = manifest_path.read_bytes()
    metadata = json.loads(manifest_bytes)
    if metadata.get("split") != "train":
        raise ValueError("Normalization requires a training manifest")
    group = metadata.get("group")
    if group not in ("A", "B") or not isinstance(metadata.get("smoke"), bool):
        raise ValueError("Manifest must explicitly declare group A/B and boolean smoke")
    total_frames = int(metadata["total_frames"])
    total_episodes = int(metadata["total_episodes"])
    state_dim, action_dim = int(metadata["state_dim"]), int(metadata["action_dim"])
    if total_frames <= 0 or total_episodes <= 0:
        raise ValueError("Cannot prepare assets from an empty manifest")
    expected_dims = (38, 24) if group == "A" else (36, 30)
    if (state_dim, action_dim) != expected_dims:
        raise ValueError(f"Manifest has incompatible {group} native dimensions")
    from dexmg.dataset import iter_training_arrays
    from dexmg.normalization import compute_norm_stats, normalization_diagnostics

    # Preallocation avoids retaining every episode plus a second concatenation.
    # float64 is also compute_norm_stats' exact arithmetic input dtype.
    states = np.empty((total_frames, state_dim), dtype=np.float64)
    actions = np.empty((total_frames, action_dim), dtype=np.float64)
    frames = episodes = 0
    for arrays in iter_training_arrays(manifest_path):
        episode_state = arrays["state"]
        episode_actions = arrays["actions"]
        n = len(episode_state)
        if episode_state.shape != (n, state_dim) or episode_actions.shape != (n, action_dim):
            raise ValueError("Selected episode has incompatible state/action shapes")
        if n <= 0 or frames + n > total_frames:
            raise ValueError("Selected episode frame count disagrees with manifest")
        states[frames : frames + n] = episode_state
        actions[frames : frames + n] = episode_actions
        frames += n
        episodes += 1
    if frames != total_frames or episodes != total_episodes:
        raise ValueError("Selected episode counts disagree with manifest")
    norm_stats = compute_norm_stats(states, actions)
    diagnostics = normalization_diagnostics(states, actions, norm_stats)
    if diagnostics['max_abs_normalized']['actions'] > 100:
        raise ValueError('Normalized action magnitude exceeds 100; audit the source before training')
    if manifest_path.read_bytes() != manifest_bytes:
        raise ValueError("Manifest changed while preparing normalization assets")
    norm_bytes = (json.dumps(norm_stats, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    provenance = {
        "manifest_path": str(manifest_path),
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "manifest_declared_sha256": metadata.get("manifest_sha256"),
        "norm_sha256": hashlib.sha256(norm_bytes).hexdigest(),
        "smoke": metadata["smoke"],
        "group": group,
        "frames": frames,
        "episodes": episodes,
        "state_dim": state_dim,
        "action_dim": action_dim,
        "method": "exact_all_selected_frames",
        "normalization": diagnostics,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    destination.mkdir(parents=True, exist_ok=False)
    with (destination / "norm_stats.json").open("xb") as stream:
        stream.write(norm_bytes)
    with (destination / "provenance.json").open("x", encoding="utf-8") as stream:
        json.dump(provenance, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(f"Prepared group={group} smoke={metadata['smoke']} frames={frames} episodes={episodes} assets={destination}", flush=True)
    return provenance


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True, help="Exact new asset-ID directory containing norm_stats.json")
    args = parser.parse_args(argv)
    prepare_assets(args.manifest, args.output_dir)


if __name__ == "__main__":
    main()
