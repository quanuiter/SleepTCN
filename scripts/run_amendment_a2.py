"""Amendment A2 on a rented GPU: reproduction check, training, locked Sleep-EDF test, export.

Commands (run from the repository root, see docs/AMENDMENT_A2_RUNBOOK_VI.md):
  plan                      list A2 runs and their state (works before the config is locked)
  repro                     re-train E4 fold 0 seed 123 and compare with the archived run
  train --workers N         validation-only training of every A2 run, N processes on one GPU
  test --confirm PHRASE     open the Sleep-EDF test folds once for every trained A2 run
  status                    progress summary
  export                    pack runs/a2 (checkpoints, predictions, logs) into one archive
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tarfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sleeptcn.amendment_a2 import (  # noqa: E402
    A2_ROOT,
    TEST_CONFIRMATION,
    execute_test,
    load_a2_config,
    repro_target,
    reproduction_summary,
    test_preflight,
    train_target,
    training_state,
    training_targets,
)
from sleeptcn.io.serialization import atomic_write_json  # noqa: E402


def require_cuda_settings(device: str) -> None:
    if device.startswith("cuda") and os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise SystemExit("export CUBLAS_WORKSPACE_CONFIG=:4096:8 before any CUDA run (runbook step 4)")


def cmd_plan(args: argparse.Namespace) -> int:
    config, sha = load_a2_config(ROOT, require_locked=False)
    targets = training_targets(config)
    print(f"A2 config status: {config['status']}  sha256={sha[:12]}")
    counts: dict[str, int] = {}
    for target in targets:
        state = training_state(ROOT, target)
        counts[state] = counts.get(state, 0) + 1
    by_arm: dict[str, list[int]] = {}
    for target in targets:
        by_arm.setdefault(target.arm, [])
        if target.seed not in by_arm[target.arm]:
            by_arm[target.arm].append(target.seed)
    for arm, seeds in by_arm.items():
        print(f"  {arm}: seeds {seeds} x 10 folds")
    print(f"{len(targets)} runs; states: {counts}")
    print(f"reproduction check ({repro_target().key}): {training_state(ROOT, repro_target())}")
    return 0


def cmd_repro(args: argparse.Namespace) -> int:
    require_cuda_settings(args.device)
    config, sha = load_a2_config(ROOT)
    target = repro_target()
    result = train_target(ROOT, config, sha, target, device=args.device, num_workers=args.num_workers)
    print(json.dumps(result))
    summary = reproduction_summary(ROOT)
    path = ROOT / A2_ROOT / "reproduction_check.json"
    atomic_write_json(path, summary)
    print(json.dumps(summary, indent=2))
    print(f"REPRODUCTION CHECK: {'IDENTICAL' if summary['identical'] else 'DIFFERS (see file)'} -> {path}")
    return 0


def cmd_train_worker(args: argparse.Namespace) -> int:
    require_cuda_settings(args.device)
    config, sha = load_a2_config(ROOT)
    targets = training_targets(config)[args.index::args.count]
    for number, target in enumerate(targets, start=1):
        print(f"[worker {args.index}] {number}/{len(targets)} {target.key} START {time.strftime('%H:%M:%S')}",
              flush=True)
        result = train_target(ROOT, config, sha, target, device=args.device, num_workers=args.num_workers)
        print(f"[worker {args.index}] {target.key} {json.dumps(result)}", flush=True)
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    require_cuda_settings(args.device)
    load_a2_config(ROOT)  # refuse early if not locked
    logs = ROOT / A2_ROOT / "monitoring"
    logs.mkdir(parents=True, exist_ok=True)
    processes = []
    for index in range(args.workers):
        log = (logs / f"train_worker_{index}.log").open("a", encoding="utf-8")
        command = [sys.executable, str(Path(__file__).resolve()), "train-worker", "--index", str(index),
                   "--count", str(args.workers), "--device", args.device, "--num-workers", str(args.num_workers)]
        processes.append((index, subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT), log))
        print(f"worker {index} started (log: {logs / f'train_worker_{index}.log'})", flush=True)
    failed = []
    for index, process, log in processes:
        if process.wait() != 0:
            failed.append(index)
        log.close()
    if failed:
        print(f"FAILED workers: {failed}; fix the cause and rerun 'train' (finished runs are skipped)")
        return 1
    print("ALL A2 TRAINING RUNS FINISHED")
    return cmd_status(args)


def cmd_test(args: argparse.Namespace) -> int:
    config, _ = load_a2_config(ROOT)
    targets = training_targets(config)
    if not args.confirm:
        journal = test_preflight(ROOT, targets)
        print(f"Preflight passed for {len(journal['targets'])} runs. To open the test folds run:\n"
              f"  python scripts/run_amendment_a2.py test --confirm {TEST_CONFIRMATION}")
        return 0
    require_cuda_settings(args.device)
    journal = execute_test(ROOT, config, targets, device=args.device, num_workers=args.num_workers,
                           confirmation=args.confirm)
    print(f"A2 TEST CAMPAIGN: {journal['status']} ({len(journal['targets'])} runs)")
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    config, _ = load_a2_config(ROOT, require_locked=False)
    rows: dict[str, dict[str, int]] = {}
    for target in training_targets(config):
        key = f"{target.arm} seed {target.seed}"
        state = training_state(ROOT, target)
        rows.setdefault(key, {})
        rows[key][state] = rows[key].get(state, 0) + 1
    for key, counts in rows.items():
        print(f"  {key:<16} {counts}")
    times = ROOT / A2_ROOT / "monitoring" / "training_times.jsonl"
    if times.is_file():
        values = [json.loads(line)["wall_seconds"] for line in times.read_text().splitlines() if line.strip()]
        print(f"  finished runs: {len(values)}, total {sum(values) / 3600:.1f} GPU-process hours")
    return 0


def cmd_export(args: argparse.Namespace) -> int:
    source = ROOT / A2_ROOT
    destination = Path(args.output).resolve()
    with tarfile.open(destination, "w:gz") as archive:
        archive.add(source, arcname="runs/a2")
        for extra in ("runs/a2_step_time_benchmark.json",):
            if (ROOT / extra).is_file():
                archive.add(ROOT / extra, arcname=extra)
    print(f"EXPORTED {destination} ({destination.stat().st_size / 1e6:.1f} MB)")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("plan", "status"):
        sub.add_parser(name)
    for name in ("repro", "train", "train-worker", "test"):
        p = sub.add_parser(name)
        p.add_argument("--device", default="cuda")
        p.add_argument("--num-workers", type=int, default=2, help="DataLoader workers per process")
        if name == "train":
            p.add_argument("--workers", type=int, default=3, help="training processes sharing the GPU")
        if name == "train-worker":
            p.add_argument("--index", type=int, required=True)
            p.add_argument("--count", type=int, required=True)
        if name == "test":
            p.add_argument("--confirm", default="")
    p = sub.add_parser("export")
    p.add_argument("--output", default="a2_results.tar.gz")
    args = parser.parse_args()
    handler = {"plan": cmd_plan, "repro": cmd_repro, "train": cmd_train, "train-worker": cmd_train_worker,
               "test": cmd_test, "status": cmd_status, "export": cmd_export}[args.command]
    return handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
