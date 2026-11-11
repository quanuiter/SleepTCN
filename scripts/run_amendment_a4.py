"""Amendment A4: input preparation, GPU training, locked Sleep-EDF test and export.

Commands (run from the repository root, see docs/AMENDMENT_A4_RUNBOOK_VI.md):
  plan                      list A4 runs and their state (works before the config is locked)
  prepare                   build and check the A4 inputs for Sleep-EDF, the holdout and telemetry (local)
  bundle / verify-bundle    pack the Sleep-EDF inputs for the GPU machine / check them after unpacking
  env-check                 interpreter, locked torch/CUDA, split hash, lock, clean Git and history
  repro                     re-train E4 fold 0 seed 123 and compare with the archived run
  train --workers N         validation-only training of every A4 run, N processes on one GPU
  test --confirm PHRASE     open the Sleep-EDF test folds once for every trained A4 run
  status                    progress summary
  export                    pack runs/a4/full (checkpoints, predictions, logs) into one archive
Zero-shot inference and analysis run locally afterwards: scripts/a4_infer.py, scripts/a4_analyze.py.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import subprocess
import sys
import tarfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sleeptcn.amendment_a4 import (  # noqa: E402
    A4_CONFIG,
    A4_ROOT,
    TEST_CONFIRMATION,
    execute_test,
    load_a4_config,
    repro_target,
    reproduction_summary,
    test_preflight,
    train_target,
    training_state,
    training_targets,
)
from sleeptcn.io.serialization import atomic_write_json  # noqa: E402

EXPECTED_TORCH = "2.5.1"
EXPECTED_CUDA = "12.1"
LOCK_SUBJECT = "Lock Amendment A4"
BUNDLE_MANIFEST = A4_ROOT / "inputs_bundle_sleepedf.json"


def require_cuda_settings(device: str) -> None:
    if device.startswith("cuda") and os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise SystemExit("export CUBLAS_WORKSPACE_CONFIG=:4096:8 before any CUDA run (runbook)")


def cmd_plan(args: argparse.Namespace) -> int:
    config, sha = load_a4_config(ROOT, require_locked=False)
    targets = training_targets(config)
    counts: dict[str, int] = {}
    for target in targets:
        state = training_state(ROOT, target)
        counts[state] = counts.get(state, 0) + 1
    print(f"A4 config status: {config['status']}  sha256={sha[:12]}")
    for arm, spec in config["training"]["arms"].items():
        print(f"  {arm}: {spec['data_variant']}, seeds {spec['seeds']} x {len(config['training']['folds'])} folds")
    print(f"{len(targets)} runs; states: {counts}")
    print(f"reproduction check ({repro_target().key}): {training_state(ROOT, repro_target())}")
    return 0


def cmd_prepare(args: argparse.Namespace) -> int:
    from sleeptcn.amendment_a4_data import prepare

    manifest = prepare(ROOT, args.cohorts)
    print(json.dumps({c: {k: v for k, v in m.items() if k != "rows"} for c, m in manifest["cohorts"].items()},
                     indent=2))
    print("A4 INPUTS WRITTEN AND CHECKED")
    return 0


def _bundle_files() -> list[tuple[Path, str]]:
    from sleeptcn.amendment_a4_data import MANIFEST, VARIANTS

    files = [(ROOT / MANIFEST, MANIFEST.as_posix())]
    for variant in ("bandpass_v2", *VARIANTS):  # bandpass_v2 for the reproduction check
        files += [(p, f"data/processed/{variant}/{p.name}")
                  for p in sorted((ROOT / "data" / "processed" / variant).glob("SC*.npz"))]
    return files


def cmd_bundle(args: argparse.Namespace) -> int:
    from sleeptcn.io.hashing import sha256_file

    load_a4_config(ROOT)
    files = _bundle_files()
    missing = [str(src) for src, _ in files if not src.is_file()]
    if missing or len(files) != 1 + 3 * 153:
        raise SystemExit(f"bundle needs the manifest and 3 x 153 Sleep-EDF inputs; missing e.g. {missing[:3]}")
    manifest = {"schema_version": 1, "amendment": "A4",
                "files": [{"path": arc, "sha256": sha256_file(src)} for src, arc in files]}
    destination = Path(args.output).resolve()
    with tarfile.open(destination, "w") as archive:
        for src, arc in files:
            archive.add(src, arcname=arc)
        payload = json.dumps(manifest, indent=1).encode("utf-8")
        info = tarfile.TarInfo(BUNDLE_MANIFEST.as_posix())
        info.size = len(payload)
        archive.addfile(info, io.BytesIO(payload))
    print(f"BUNDLE {destination} ({destination.stat().st_size / 1e9:.2f} GB, {len(files)} files)")
    return 0


def cmd_verify_bundle(args: argparse.Namespace) -> int:
    from sleeptcn.io.hashing import sha256_file

    path = ROOT / BUNDLE_MANIFEST
    if not path.is_file():
        raise SystemExit(f"{path} not found; unpack the bundle at the repository root")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    bad = [f["path"] for f in manifest["files"]
           if not (ROOT / f["path"]).is_file() or sha256_file(ROOT / f["path"]) != f["sha256"]]
    if bad:
        raise SystemExit(f"{len(bad)} bundle files missing or changed, e.g. {bad[:3]}")
    print(f"BUNDLE VERIFIED: {len(manifest['files'])} files")
    return 0


def environment_report(device: str) -> dict:
    import platform

    import torch

    from sleeptcn.io.hashing import sha256_file

    errors = []
    if not (3, 10) <= sys.version_info[:2] < (3, 12):
        errors.append("python_must_be_3.10_or_3.11")
    if torch.__version__.split("+")[0] != EXPECTED_TORCH:
        errors.append(f"torch_{torch.__version__}_expected_{EXPECTED_TORCH}")
    cuda = torch.cuda.is_available()
    if device.startswith("cuda"):
        if not cuda:
            errors.append("cuda_unavailable")
        elif torch.version.cuda != EXPECTED_CUDA:
            errors.append(f"cuda_{torch.version.cuda}_expected_{EXPECTED_CUDA}")
    split = ROOT / "data/splits/sleepedf_sc_10fold_seed42_v2.json"
    if sha256_file(split) != split.with_name(split.name + ".sha256").read_text(encoding="ascii").split()[0]:
        errors.append("split_sha256_differs_from_sidecar")
    try:
        load_a4_config(ROOT)
    except ValueError:
        errors.append("a4_config_not_locked")

    def git(*command: str) -> subprocess.CompletedProcess:
        return subprocess.run(["git", "-C", str(ROOT), *command], capture_output=True, text=True)

    if git("status", "--porcelain").stdout.strip():
        errors.append("git_worktree_not_clean")
    if git("cat-file", "-t", "8af1d12").stdout.strip() != "commit":
        errors.append("git_history_without_8af1d12_use_a_full_clone")
    config_path = A4_CONFIG.as_posix()
    lock_commit, lock_subject = (git("log", "-1", "--format=%H%x00%s", "--", config_path).stdout.strip().split("\x00")
                                 + [""])[:2]
    if lock_subject != LOCK_SUBJECT:
        errors.append("last_commit_touching_a4_config_is_not_the_lock_commit")
    elif git("diff", "--quiet", lock_commit, "--", config_path).returncode != 0:
        errors.append("a4_config_differs_from_lock_commit")
    return {"schema_version": 1, "amendment": "A4", "passed": not errors, "errors": errors,
            "python": sys.version, "platform": platform.platform(), "torch": torch.__version__,
            "cuda_available": cuda, "cuda_version": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if cuda else None,
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "git_commit": git("rev-parse", "HEAD").stdout.strip(), "a4_lock_commit": lock_commit,
            "experiments_config_sha256": sha256_file(ROOT / "configs/experiments_v2.json"),
            "split_sha256": sha256_file(split)}


def cmd_env_check(args: argparse.Namespace) -> int:
    report = environment_report(args.device)
    path = ROOT / A4_ROOT / "monitoring" / "environment_check.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(path, report)
    print(json.dumps(report, indent=2))
    print("ENVIRONMENT OK" if report["passed"] else "ENVIRONMENT NOT READY")
    return 0 if report["passed"] else 1


def cmd_repro(args: argparse.Namespace) -> int:
    require_cuda_settings(args.device)
    config, sha = load_a4_config(ROOT)
    print(json.dumps(train_target(ROOT, config, sha, repro_target(), device=args.device,
                                  num_workers=args.num_workers)))
    summary = reproduction_summary(ROOT)
    path = ROOT / A4_ROOT / "reproduction_check.json"
    atomic_write_json(path, summary)
    print(json.dumps(summary, indent=2))
    print(f"REPRODUCTION CHECK: {'IDENTICAL' if summary['identical'] else 'DIFFERS (see file)'} -> {path}")
    return 0


def cmd_train_worker(args: argparse.Namespace) -> int:
    require_cuda_settings(args.device)
    config, sha = load_a4_config(ROOT)
    targets = training_targets(config)[args.index::args.count]
    for number, target in enumerate(targets, start=1):
        print(f"[worker {args.index}] {number}/{len(targets)} {target.key} START {time.strftime('%H:%M:%S')}",
              flush=True)
        result = train_target(ROOT, config, sha, target, device=args.device, num_workers=args.num_workers)
        print(f"[worker {args.index}] {target.key} {json.dumps(result)}", flush=True)
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    require_cuda_settings(args.device)
    load_a4_config(ROOT)
    logs = ROOT / A4_ROOT / "monitoring"
    logs.mkdir(parents=True, exist_ok=True)
    processes = []
    for index in range(args.workers):
        log = (logs / f"train_worker_{index}.log").open("a", encoding="utf-8")
        command = [sys.executable, str(Path(__file__).resolve()), "train-worker", "--index", str(index),
                   "--count", str(args.workers), "--device", args.device, "--num-workers", str(args.num_workers)]
        processes.append((index, subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT), log))
        print(f"worker {index} started (log: {logs / f'train_worker_{index}.log'})", flush=True)
    failed = [index for index, process, _ in processes if process.wait() != 0]
    for _, _, log in processes:
        log.close()
    if failed:
        print(f"FAILED workers: {failed}; fix the cause and rerun 'train' (finished runs are skipped)")
        return 1
    print("ALL A4 TRAINING RUNS FINISHED")
    return cmd_status(args)


def cmd_test(args: argparse.Namespace) -> int:
    config, _ = load_a4_config(ROOT)
    targets = training_targets(config)
    if not args.confirm:
        journal = test_preflight(ROOT, targets)
        print(f"Preflight passed for {len(journal['targets'])} runs. To open the test folds run:\n"
              f"  python scripts/run_amendment_a4.py test --confirm {TEST_CONFIRMATION}")
        return 0
    require_cuda_settings(args.device)
    journal = execute_test(ROOT, config, targets, device=args.device, num_workers=args.num_workers,
                           confirmation=args.confirm)
    print(f"A4 TEST CAMPAIGN: {journal['status']} ({len(journal['targets'])} runs)")
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    config, _ = load_a4_config(ROOT, require_locked=False)
    rows: dict[str, dict[str, int]] = {}
    for target in training_targets(config):
        key = f"{target.arm} seed {target.seed}"
        rows.setdefault(key, {})
        state = training_state(ROOT, target)
        rows[key][state] = rows[key].get(state, 0) + 1
    for key, counts in rows.items():
        print(f"  {key:<16} {counts}")
    times = ROOT / A4_ROOT / "monitoring" / "training_times.jsonl"
    if times.is_file():
        values = [json.loads(line)["wall_seconds"] for line in times.read_text().splitlines() if line.strip()]
        print(f"  finished runs: {len(values)}, total {sum(values) / 3600:.1f} GPU-process hours")
    return 0


def cmd_export(args: argparse.Namespace) -> int:
    destination = Path(args.output).resolve()
    with tarfile.open(destination, "w:gz") as archive:
        for name in ("full", "monitoring", "test_campaign_a4.json", "reproduction_check.json"):
            if (ROOT / A4_ROOT / name).exists():
                archive.add(ROOT / A4_ROOT / name, arcname=(A4_ROOT / name).as_posix())
    print(f"EXPORTED {destination} ({destination.stat().st_size / 1e6:.1f} MB)")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("plan", "status", "verify-bundle"):
        sub.add_parser(name)
    p = sub.add_parser("prepare")
    p.add_argument("--cohorts", nargs="+", default=["sleep_edf", "shhs1_holdout", "telemetry"])
    p = sub.add_parser("bundle")
    # Outside the repository: a file in the worktree would block the clean-tree runner.
    p.add_argument("--output", default=str(ROOT.parent / "a4_inputs_sleepedf.tar"))
    p = sub.add_parser("env-check")
    p.add_argument("--device", default="cuda")
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
    p.add_argument("--output", default=str(ROOT.parent / "a4_results.tar.gz"))
    args = parser.parse_args()
    handler = {"plan": cmd_plan, "prepare": cmd_prepare, "bundle": cmd_bundle, "verify-bundle": cmd_verify_bundle,
               "env-check": cmd_env_check, "repro": cmd_repro, "train": cmd_train,
               "train-worker": cmd_train_worker, "test": cmd_test, "status": cmd_status,
               "export": cmd_export}[args.command]
    return handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
