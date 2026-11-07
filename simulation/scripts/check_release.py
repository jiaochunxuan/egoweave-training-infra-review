"""Small dependency-free release gate used before publishing the directory."""

from __future__ import annotations

from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
TEXT_SUFFIXES = {
    ".cfg", ".env", ".ini", ".json", ".md", ".patch", ".ps1", ".py",
    ".sh", ".toml", ".txt", ".yaml", ".yml",
}

# Match private infrastructure values, not ordinary words such as ``host`` or
# ``token``. Public loopback addresses and placeholder paths remain allowed.
PRIVATE_VALUE = re.compile(
    r"""(?ix)
    (?:
        [a-z]:[\\/](?:users|data|home|mnt|workspace|root)[\\/] |
        (?:^|[\\/])(?:data|home|mnt|workspace|root)[\\/] |
        [\\/]users[\\/] |
        \b(?:h100|b300)-[a-z0-9_.-]+\b |
        \b(?:ssh|scp|rsync|srun|sbatch)\s+ |
        \b(?:slurm|systemctl)\b |
        \b(?:quest_local21|r6lite|cib_|future_latent|v-jepa)\b |
        \b(?!(?:127\.0\.0\.1)\b)(?:\d{1,3}\.){3}\d{1,3}\b |
        \b(?:api[_-]?key|access[_-]?token|secret|password)\s*[:=]\s*[^\s<\"']+
    )
    """
)


def main() -> int:
    required = [
        ROOT / "README.md",
        ROOT / "configs/experiment_protocol.yaml",
        ROOT / "src/dexjoco4/train/train_dexjoco_pi05.py",
        ROOT / "src/dexmg/train.py",
        ROOT / "src/robotwin_pi05/train_robotwin_pi05.py",
        ROOT / "src/egoweave_pretrain/representation.py",
        ROOT / "src/egoweave_pretrain/train_m4.py",
        ROOT / "configs/m4_pretraining.yaml",
    ]
    missing = [str(path.relative_to(ROOT)) for path in required if not path.exists()]
    if missing:
        raise SystemExit(f"missing release files: {missing}")

    violations = []
    for path in ROOT.rglob("*"):
        if ".git" in path.parts:
            continue
        if path.resolve() == Path(__file__).resolve():
            continue
        if path.is_dir() and path.name == "__pycache__":
            violations.append(f"{path.relative_to(ROOT)} (generated cache directory)")
            continue
        if not path.is_file():
            continue
        if path.suffix.lower() in {".pyc", ".pyo"}:
            violations.append(f"{path.relative_to(ROOT)} (compiled artifact)")
            continue
        if path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        if PRIVATE_VALUE.search(text):
            violations.append(str(path.relative_to(ROOT)))
    if violations:
        raise SystemExit(f"private-path or generated-artifact scan failed: {violations}")
    print("release-check: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
