from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

PACKAGE_PARENT = Path(__file__).resolve().parents[1]
if str(PACKAGE_PARENT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_PARENT))

from dexjoco4.eval.dexjoco_openpi import (
    DEFAULT_TASKS,
    DEXJOCO_PYTHON,
    DEXJOCO_SOURCE_ROOT,
    EVAL_VARIANTS,
    OPENPI_ROOT,
    OPENPI_PYTHON,
    resolve_task_config_path,
    serve_policy,
)

DEFAULT_OUTPUT_ROOT = Path(os.environ.get("DEXJOCO_EVAL_OUTPUT_ROOT", "outputs/dexjoco_eval"))


def _ensure_openpi_paths() -> None:
    for path in (
        str(DEXJOCO_SOURCE_ROOT),
        str(DEXJOCO_SOURCE_ROOT / "dexjoco"),
        str(OPENPI_ROOT),
        str(OPENPI_ROOT / "src"),
        str(OPENPI_ROOT / "packages/openpi-client/src"),
        str(PACKAGE_PARENT),
    ):
        if path not in sys.path:
            sys.path.insert(0, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=EVAL_VARIANTS, required=True)
    parser.add_argument("--checkpoint-step", type=int, default=4000)
    parser.add_argument("--task", choices=tuple(DEFAULT_TASKS), default="fold_glasses")
    parser.add_argument("--seed", type=int, default=10000)
    parser.add_argument("--port", type=int, default=8011)
    parser.add_argument("--server-gpu", type=str, default="0")
    parser.add_argument("--server-only", action="store_true")
    parser.add_argument("--client-only", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--diffusion-steps", type=int, default=10)
    parser.add_argument("--sample-seed", type=int, default=0)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--rand-full", action="store_true")
    parser.add_argument("--randomize-dynamics", action="store_true")
    return parser.parse_args()


def _server_env(gpu: str) -> dict[str, str]:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu
    env.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    env.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.80")
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("MUJOCO_GL", "egl")
    env["PYTHONPATH"] = ":".join(
        [
            str(PACKAGE_PARENT),
            str(OPENPI_ROOT),
            str(OPENPI_ROOT / "src"),
            str(OPENPI_ROOT / "packages/openpi-client/src"),
            str(DEXJOCO_SOURCE_ROOT),
            str(DEXJOCO_PACKAGE_PARENT),
            env.get("PYTHONPATH", ""),
        ]
    )
    return env


def _launch_server(args: argparse.Namespace) -> subprocess.Popen[str]:
    cmd = [
        str(OPENPI_PYTHON),
        str(Path(__file__).resolve()),
        "--variant",
        args.variant,
        "--checkpoint-step",
        str(args.checkpoint_step),
        "--port",
        str(args.port),
        "--server-only",
    ]
    env = _server_env(args.server_gpu)
    return subprocess.Popen(cmd, env=env)


def _wait_for_port(host: str, port: int, timeout_s: float = 60.0) -> None:
    deadline = time.time() + timeout_s
    import socket

    while time.time() < deadline:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(1.0)
            if sock.connect_ex((host, port)) == 0:
                return
        time.sleep(1.0)
    raise TimeoutError(f"Policy server did not start on {host}:{port} within {timeout_s}s")


def _run_eval(args: argparse.Namespace) -> int:
    _ensure_openpi_paths()
    from dexjoco_openpi_client import evaluate_dexjoco_openpi

    config_path = resolve_task_config_path(args.task)
    output_dir = args.output_dir or Path(
        DEFAULT_OUTPUT_ROOT / f"step_{args.checkpoint_step}" / args.variant / f"{args.task}_seed{args.seed:06d}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    evaluate_dexjoco_openpi(
        config=config_path,
        seed=args.seed,
        rand_full=args.rand_full,
        randomize_dynamics=args.randomize_dynamics,
        port=args.port,
        host=args.host,
        output=output_dir,
        episodes=args.episodes,
        render_mode="rgb_array",
    )
    return 0


def _client_env() -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("MUJOCO_GL", "egl")
    env["PYTHONPATH"] = ":".join(
        [
            str(DEXJOCO_SOURCE_ROOT),
            str(DEXJOCO_SOURCE_ROOT / "dexjoco"),
            str(OPENPI_ROOT / "packages/openpi-client/src"),
            env.get("PYTHONPATH", ""),
        ]
    )
    return env


def _reexec_in_dexjoco_python_if_needed() -> None:
    if Path(sys.executable).resolve() == DEXJOCO_PYTHON.resolve():
        return
    if os.environ.get("DEXJOCO4_CLIENT_REEXEC") == "1":
        return
    env = _client_env()
    env["DEXJOCO4_CLIENT_REEXEC"] = "1"
    completed = subprocess.run([str(DEXJOCO_PYTHON), str(Path(__file__).resolve()), *sys.argv[1:]], env=env)
    raise SystemExit(completed.returncode)


def main() -> int:
    args = parse_args()
    if args.server_only:
        serve_policy(args.variant, args.checkpoint_step, port=args.port)
        return 0

    _reexec_in_dexjoco_python_if_needed()
    if args.client_only:
        return _run_eval(args)

    proc = _launch_server(args)
    try:
        _wait_for_port(args.host, args.port)
        return _run_eval(args)
    finally:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=20)


if __name__ == "__main__":
    raise SystemExit(main())
