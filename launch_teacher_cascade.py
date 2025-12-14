#!/usr/bin/env python
"""Simple multi-process / multi-GPU launcher for run_teacher_cascade.py.

This script spawns multiple shard processes in parallel and assigns each
shard to a GPU in a round-robin fashion.

Usage examples (from the repo root):

    # 8 shards spread across 2 GPUs (0 and 1)
    python launch_teacher_cascade.py \
        --config config.yaml \
        --lang ru \
        --direction ruen \
        --num-shards 8 \
        --gpus 0,1

    # Same but limit each shard to first 1,000 samples for a quick dev run
    python launch_teacher_cascade.py \
        --config config.yaml \
        --lang ru \
        --direction ruen \
        --num-shards 8 \
        --gpus 0,1 \
        --max-samples 1000

The launcher will:
  * Spawn one process per shard (0..num_shards-1)
  * Pass --shard-idx / --num-shards / --device / --max-samples to run_teacher_cascade.py
  * Assign GPUs round-robin from the provided --gpus list
  * Wait for all children and exit non-zero if any shard fails
"""

import argparse
import os
import subprocess
import sys
from typing import List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch multiple run_teacher_cascade.py shards in parallel across GPUs.",
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config passed through to run_teacher_cascade.py.",
    )
    parser.add_argument(
        "--lang",
        type=str,
        choices=["ru", "en"],
        required=True,
        help="Source language (passed to run_teacher_cascade.py).",
    )
    parser.add_argument(
        "--direction",
        type=str,
        choices=["ruen", "enru"],
        required=True,
        help="Translation direction (passed to run_teacher_cascade.py).",
    )
    parser.add_argument(
        "--num-shards",
        type=int,
        required=True,
        help="Total number of shards to split the manifest into.",
    )
    parser.add_argument(
        "--gpus",
        type=str,
        default="0",
        help=(
            "Comma-separated list of GPU indices to use, e.g. '0,1'. "
            "Shards are assigned to GPUs round-robin."
        ),
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help=(
            "Optional per-shard max-samples passed through to run_teacher_cascade.py "
            "(0 = no limit)."
        ),
    )
    parser.add_argument(
        "--python-exe",
        type=str,
        default=sys.executable,
        help="Python executable to use for child processes (default: current interpreter).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the commands that would be executed, but do not launch any processes.",
    )
    return parser.parse_args()


def parse_gpu_list(gpus_str: str) -> List[int]:
    parts = [p.strip() for p in gpus_str.split(",") if p.strip()]
    if not parts:
        raise ValueError("--gpus must contain at least one GPU index, e.g. '0' or '0,1'")
    try:
        return [int(p) for p in parts]
    except ValueError as e:
        raise ValueError(f"Invalid --gpus value {gpus_str!r}; expected comma-separated integers") from e


def build_shard_command(
    shard_idx: int,
    num_shards: int,
    gpu_id: int,
    args: argparse.Namespace,
) -> List[str]:
    """Build the command-line for a single shard process."""
    cmd = [
        args.python_exe,
        "run_teacher_cascade.py",
        "--config",
        args.config,
        "--lang",
        args.lang,
        "--direction",
        args.direction,
        "--shard-idx",
        str(shard_idx),
        "--num-shards",
        str(num_shards),
        "--device",
        f"cuda:{gpu_id}",
    ]
    if args.max_samples and args.max_samples > 0:
        cmd.extend(["--max-samples", str(args.max_samples)])
    return cmd


def main() -> None:
    args = parse_args()
    gpu_ids = parse_gpu_list(args.gpus)

    print(f"Launching {args.num_shards} shard(s) across GPUs {gpu_ids}...")

    # Build all shard commands first
    shard_cmds = []
    for shard_idx in range(args.num_shards):
        gpu_id = gpu_ids[shard_idx % len(gpu_ids)]
        cmd = build_shard_command(shard_idx, args.num_shards, gpu_id, args)
        shard_cmds.append((shard_idx, gpu_id, cmd))

    # Dry-run: just print commands and exit
    if args.dry_run:
        print("[DRY RUN] The following commands would be executed:")
        for shard_idx, gpu_id, cmd in shard_cmds:
            print(f"  [shard {shard_idx}] GPU {gpu_id}:", " ".join(cmd))
        return

    # Launch all shards in parallel
    procs = []
    for shard_idx, gpu_id, cmd in shard_cmds:
        env = os.environ.copy()
        # We pass the GPU explicitly via --device, so CUDA_VISIBLE_DEVICES
        # is optional. If you prefer hard isolation per shard, uncomment:
        # env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

        print(f"[launch] Starting shard {shard_idx} on GPU {gpu_id}:", " ".join(cmd))
        p = subprocess.Popen(cmd, env=env)
        procs.append((shard_idx, gpu_id, p))

    # Wait for all shards to complete and collect exit codes
    overall_rc = 0
    for shard_idx, gpu_id, p in procs:
        rc = p.wait()
        if rc == 0:
            print(f"[launch] Shard {shard_idx} on GPU {gpu_id} finished successfully.")
        else:
            print(f"[launch] Shard {shard_idx} on GPU {gpu_id} FAILED with exit code {rc}.")
            overall_rc = overall_rc or rc

    if overall_rc == 0:
        print("[launch] All shards finished successfully.")
    else:
        print(f"[launch] Some shards failed. Overall exit code: {overall_rc}")

    sys.exit(overall_rc)


if __name__ == "__main__":
    main()
