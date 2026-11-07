"""Build an immutable A/B training manifest from exact local task files."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from dexmg.contracts import GROUP_TASKS


def resolve_task_paths(data_root, group, *, smoke=False):
    if group not in GROUP_TASKS:
        raise ValueError("group must be A or B")
    root = Path(data_root).expanduser().resolve()
    suffix = ".first64.hdf5" if smoke else ".hdf5"
    paths = {}
    for task in GROUP_TASKS[group]:
        candidates = [root / f"{task}{suffix}"]
        if not smoke:
            candidates.append(root / "generated" / f"{task}{suffix}")
        present = [path for path in candidates if path.is_file()]
        if len(present) > 1:
            raise ValueError(f"Ambiguous source files for {task}: {present}")
        if not present:
            raise FileNotFoundError(f"Missing {'smoke' if smoke else 'formal'} task {task}; expected one of {candidates}")
        paths[task] = present[0]
    return paths


def prepare_data(group, data_root, output, *, smoke=False, max_episodes=None):
    destination = Path(output).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite manifest {destination}")
    if max_episodes is not None and max_episodes <= 0:
        raise ValueError("max_episodes must be positive")
    paths = resolve_task_paths(data_root, group, smoke=smoke)
    from dexmg.dataset import build_manifest

    manifest = build_manifest(paths, group, destination, max_episodes=max_episodes, smoke=smoke)
    print(f"Prepared group={group} smoke={smoke} manifest={destination}", flush=True)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group", required=True, choices=("A", "B"))
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output", required=True, help="Exact new manifest JSON path")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--max-episodes", type=int, help="Maximum selected episodes per task")
    args = parser.parse_args(argv)
    prepare_data(args.group, args.data_root, args.output, smoke=args.smoke, max_episodes=args.max_episodes)


if __name__ == "__main__":
    main()
