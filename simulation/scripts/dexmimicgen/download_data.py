"""Download only the six scoped official generated DexMimicGen files.

Each run resolves a single immutable HF revision, resumes through the official
Hub cache, and verifies the byte length and LFS SHA256 before marking a file done.
No download occurs on import. The CLI must be launched explicitly.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re


REPO_ID = "MimicGen/dexmimicgen_datasets"
TASK_GROUPS = {
    "A": ("two_arm_box_cleanup", "two_arm_drawer_cleanup", "two_arm_lift_tray"),
    "B": ("two_arm_coffee", "two_arm_pouring", "two_arm_can_sort_random"),
}


def _now():
    return datetime.now(timezone.utc).isoformat()


def _write_receipt(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download_data(group, output_root, *, revision="main", receipt_path=None, workers=3, source_metadata=None, endpoint="https://huggingface.co"):
    if group not in (*TASK_GROUPS, "all"):
        raise ValueError("group must be A, B, or all")
    if not 1 <= workers <= 3:
        raise ValueError("workers must be between 1 and 3")
    if endpoint not in ("https://huggingface.co", "https://hf-mirror.com"):
        raise ValueError("endpoint must be the official HF service or the supported public mirror")
    if endpoint != "https://huggingface.co" and source_metadata is None:
        raise ValueError("Mirror downloads require previously verified official source metadata")
    root = Path(output_root).expanduser().resolve()
    receipt_path = Path(receipt_path).expanduser().resolve() if receipt_path else root / "download_receipt.json"
    if receipt_path.exists():
        previous = json.loads(receipt_path.read_text(encoding="utf-8"))
        if previous.get("repo_id") != REPO_ID or previous.get("group") != group:
            raise ValueError("Existing receipt belongs to a different dataset or group")
        if revision == "main":
            revision = previous["revision"]
        elif revision != previous["revision"]:
            raise ValueError("Existing receipt uses another pinned revision")
    # Prevent the optional Xet backend from opening its own large connection pool.
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
    os.environ["HF_ENDPOINT"] = endpoint
    from huggingface_hub import HfApi, hf_hub_download

    metadata_sha = None
    if source_metadata is not None:
        source_bytes = Path(source_metadata).expanduser().resolve().read_bytes()
        metadata_sha = hashlib.sha256(source_bytes).hexdigest()
        metadata = json.loads(source_bytes)
        if metadata.get("repo_id") != REPO_ID or not metadata.get("source_api_url", "").startswith("https://huggingface.co/api/datasets/MimicGen/dexmimicgen_datasets/"):
            raise ValueError("Pinned source metadata must identify the official DexMimicGen dataset")
        pinned_revision = metadata["revision"]
        if revision != "main" and revision != pinned_revision:
            raise ValueError("Requested revision disagrees with pinned source metadata")
        remote_files = {item["filename"]: item for item in metadata["files"]}
    else:
        metadata = HfApi().repo_info(repo_id=REPO_ID, repo_type="dataset", revision=revision, files_metadata=True, token=False)
        pinned_revision = metadata.sha
        remote_files = {}
        for item in metadata.siblings:
            lfs = item.lfs
            digest = lfs.get("sha256") if isinstance(lfs, dict) else getattr(lfs, "sha256", None)
            remote_files[item.rfilename] = {"expected_bytes": item.size, "expected_sha256": digest}
    if not re.fullmatch(r"[0-9a-f]{40}", pinned_revision):
        raise ValueError("HF did not resolve an immutable commit revision")
    tasks = TASK_GROUPS["A"] + TASK_GROUPS["B"] if group == "all" else TASK_GROUPS[group]
    entries = []
    for task in tasks:
        filename = f"generated/{task}.hdf5"
        item = remote_files[filename]
        expected_sha = item["expected_sha256"]
        if not expected_sha or not re.fullmatch(r"[0-9a-f]{64}", expected_sha):
            raise ValueError(f"Missing official LFS SHA256 for {filename}")
        entries.append({"task": task, "filename": filename, "expected_bytes": int(item["expected_bytes"]), "expected_sha256": expected_sha, "verified": False, "status": "pending"})
    root.mkdir(parents=True, exist_ok=True)
    receipt = {"repo_id": REPO_ID, "revision": pinned_revision, "source_metadata_sha256": metadata_sha, "download_endpoint": endpoint, "group": group, "output_root": str(root), "workers": workers, "started_at": _now(), "status": "downloading", "files": entries}
    _write_receipt(receipt_path, receipt)
    print(f"Pinned {REPO_ID}@{pinned_revision}; files={len(entries)} bytes={sum(item['expected_bytes'] for item in entries)} workers={workers}", flush=True)

    def fetch(entry):
        print(f"START {entry['filename']} expected_bytes={entry['expected_bytes']}", flush=True)
        path = Path(hf_hub_download(repo_id=REPO_ID, repo_type="dataset", filename=entry["filename"], revision=pinned_revision, local_dir=str(root), token=False))
        actual_bytes = path.stat().st_size
        if actual_bytes != entry["expected_bytes"]:
            raise ValueError(f"Size mismatch for {entry['filename']}: {actual_bytes} != {entry['expected_bytes']}")
        actual_sha = _sha256(path)
        if actual_sha != entry["expected_sha256"]:
            raise ValueError(f"SHA256 mismatch for {entry['filename']}")
        return {"path": str(path.resolve()), "actual_bytes": actual_bytes, "actual_sha256": actual_sha, "verified": True, "status": "complete", "completed_at": _now()}

    failed = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(fetch, entry): entry for entry in entries}
        for future in as_completed(futures):
            entry = futures[future]
            try:
                entry.update(future.result())
                print(f"VERIFIED {entry['filename']} bytes={entry['actual_bytes']} sha256={entry['actual_sha256']}", flush=True)
            except Exception as exc:
                entry.update(status="failed", error=f"{type(exc).__name__}: {exc}")
                failed.append(entry["filename"])
                print(f"FAILED {entry['filename']} {type(exc).__name__}", flush=True)
            receipt["updated_at"] = _now()
            _write_receipt(receipt_path, receipt)
    receipt["status"] = "failed" if failed else "complete"
    receipt["completed_at"] = _now()
    receipt["verified_bytes"] = sum(entry.get("actual_bytes", 0) for entry in entries if entry["verified"])
    _write_receipt(receipt_path, receipt)
    if failed:
        raise RuntimeError(f"Download or verification failed: {', '.join(failed)}; see {receipt_path}")
    print(f"COMPLETE receipt={receipt_path} verified_bytes={receipt['verified_bytes']}", flush=True)
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group", choices=("A", "B", "all"), required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--revision", default="main", help="Resolved once to a pinned commit; an existing receipt preserves its revision")
    parser.add_argument("--receipt")
    parser.add_argument("--source-metadata", help="Pinned revision/sizes/SHA256 JSON previously fetched from official HF")
    parser.add_argument("--endpoint", default="https://huggingface.co", choices=("https://huggingface.co", "https://hf-mirror.com"))
    parser.add_argument("--workers", type=int, default=3, choices=(1, 2, 3))
    args = parser.parse_args(argv)
    download_data(args.group, args.output_root, revision=args.revision, receipt_path=args.receipt, workers=args.workers, source_metadata=args.source_metadata, endpoint=args.endpoint)


if __name__ == "__main__":
    main()
