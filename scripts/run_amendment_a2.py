"""Amendment A2: data preparation, GPU training, locked Sleep-EDF test, holdout inference, export.

Commands (run from the repository root, see docs/AMENDMENT_A2_RUNBOOK_VI.md):
  plan                      list A2 runs and their state (works before the config is locked)
  prepare-sleepedf          write and check the A2 Sleep-EDF variant (local)
  prepare-shhs              write and check the A2 holdout variant (local)
  bundle / verify-bundle    pack the inputs for the GPU machine / check them after unpacking
  repro                     re-train E4 fold 0 seed 123 and compare with the archived run
  train --workers N         validation-only training of every A2 run, N processes on one GPU
  test --confirm PHRASE     open the Sleep-EDF test folds once for every trained A2 run
  step-benchmark            time per optimiser step of each stage kind (analysis E)
  holdout-check             reproduce stored parent CPU probabilities with the A2 code on a device
  holdout --workers N       zero-shot inference of the A2 arms on the 500 holdout participants
  status                    progress summary
  export                    pack runs/a2 (checkpoints, predictions, logs) into one archive
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


PARENT_RESTORE = ROOT / "runs" / "holdout_ckpt_restore_20261010"
PARENT_PREDICTIONS = ROOT / "runs" / "holdout_inference_v1" / "fold_predictions"
# Inside runs/a2 (ignored by Git): a file at the repository root would block the clean-tree runner.
BUNDLE_PARTS = ("sleepedf", "holdout")
PARENT_MANIFEST_COPY = A2_ROOT / "manifests" / "parent_preprocess_holdout_v1.json"


def bundle_manifest_path(part: str) -> str:
    return (A2_ROOT / f"inputs_bundle_{part}.json").as_posix()


def cmd_prepare_sleepedf(args: argparse.Namespace) -> int:
    from sleeptcn.amendment_a2_data import prepare_sleepedf

    manifest = prepare_sleepedf(ROOT, args.data_dir, args.raw_manifest, workers=args.workers)
    print(json.dumps({"records": len(manifest["records"]), "pipeline_reproduction": manifest["pipeline_reproduction"],
                      "input_scale_in_window": manifest["input_scale_in_window"]}, indent=2))
    print("SLEEP-EDF A2 VARIANT WRITTEN AND CHECKED")
    return 0


def cmd_prepare_shhs(args: argparse.Namespace) -> int:
    from sleeptcn.amendment_a2_data import prepare_shhs

    manifest = prepare_shhs(ROOT, args.parent_manifest, args.technical_audit, args.preprocess_config, args.edf_dir,
                            args.xml_dir, args.parent_processed_root, workers=args.workers)
    # Keep a byte-identical copy next to the A2 manifests; later commands default to it.
    copy = ROOT / PARENT_MANIFEST_COPY
    if not copy.exists():
        copy.write_bytes(args.parent_manifest.read_bytes())
    print(json.dumps({"records": len(manifest["records"]), "input_scale_in_window": manifest["input_scale_in_window"]},
                     indent=2))
    print("HOLDOUT A2 VARIANT WRITTEN AND CHECKED")
    return 0


def _bundle_files(args: argparse.Namespace) -> list[tuple[Path, str]]:
    """(source file, path inside the bundle relative to the repository root) for one bundle part.

    ``sleepedf`` is all that training, the reproduction check and the Sleep-EDF test need;
    ``holdout`` adds the holdout inputs and the parent E4_s42 files of the device check, and can be
    uploaded while training runs.
    """
    from sleeptcn.amendment_a2_data import SHHS_MANIFEST, SHHS_PROCESSED, SLEEPEDF_MANIFEST, VARIANT

    if args.part == "sleepedf":
        files = [(ROOT / SLEEPEDF_MANIFEST, SLEEPEDF_MANIFEST.as_posix())]
        for variant in ("bandpass_v2", VARIANT):
            files += [(p, f"data/processed/{variant}/{p.name}")
                      for p in sorted((ROOT / "data" / "processed" / variant).glob("SC*.npz"))]
        return files
    files = [(ROOT / PARENT_MANIFEST_COPY, PARENT_MANIFEST_COPY.as_posix()),
             (ROOT / SHHS_MANIFEST, SHHS_MANIFEST.as_posix())]
    files += [(p, f"data/processed/shhs_holdout_v1/bandpass_v2/{p.name}")
              for p in sorted((args.parent_processed_root / "bandpass_v2").glob("shhs1-*.npz"))]
    files += [(p, f"{SHHS_PROCESSED.as_posix()}/{VARIANT}/{p.name}")
              for p in sorted((ROOT / SHHS_PROCESSED / VARIANT).glob("shhs1-*.npz"))]
    restore = [PARENT_RESTORE / "configs" / "experiments_v2.json",
               PARENT_RESTORE / "data" / "splits" / "sleepedf_sc_10fold_seed42_v2.json"]
    keys = sorted(p.stem for p in (args.parent_processed_root / "bandpass_v2").glob("shhs1-*.npz"))[:2]
    for fold in (0, 9):
        restore += sorted(p for p in (PARENT_RESTORE / "runs/v2/full/E4" / f"fold_{fold:02d}" / "seed_42").rglob("*")
                          if p.is_file())
        restore += [PARENT_PREDICTIONS / "E4_s42" / f"fold_{fold:02d}" / f"{key}.npz" for key in keys]
    files += [(p, p.relative_to(ROOT).as_posix()) for p in restore]
    return files


def cmd_bundle(args: argparse.Namespace) -> int:
    from sleeptcn.io.hashing import sha256_file

    load_a2_config(ROOT)
    files = _bundle_files(args)
    missing = [str(src) for src, _ in files if not src.is_file()]
    if missing:
        raise SystemExit(f"{len(missing)} bundle inputs missing, e.g. {missing[:3]}")
    manifest = {"schema_version": 1, "amendment": "A2", "part": args.part,
                "files": [{"path": arc, "sha256": sha256_file(src)} for src, arc in files]}
    destination = Path(args.output or ROOT.parent / f"a2_inputs_{args.part}.tar").resolve()
    with tarfile.open(destination, "w") as archive:
        for src, arc in files:
            archive.add(src, arcname=arc)
        payload = json.dumps(manifest, indent=1).encode("utf-8")
        info = tarfile.TarInfo(bundle_manifest_path(args.part))
        info.size = len(payload)
        archive.addfile(info, io.BytesIO(payload))
    print(f"BUNDLE {destination} ({destination.stat().st_size / 1e9:.2f} GB, {len(files)} files)")
    return 0


def cmd_verify_bundle(args: argparse.Namespace) -> int:
    from sleeptcn.io.hashing import sha256_file

    manifest_path = ROOT / bundle_manifest_path(args.part)
    if not manifest_path.is_file():
        raise SystemExit(f"{manifest_path} not found; unpack the {args.part} bundle at the repository root")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    bad = [f["path"] for f in manifest["files"] if not (ROOT / f["path"]).is_file()
           or sha256_file(ROOT / f["path"]) != f["sha256"]]
    if bad:
        raise SystemExit(f"{len(bad)} bundle files missing or changed, e.g. {bad[:3]}")
    print(f"BUNDLE VERIFIED ({args.part}): {len(manifest['files'])} files")
    return 0


def cmd_step_benchmark(args: argparse.Namespace) -> int:
    from sleeptcn.amendment_a2_cost import BENCHMARK_FILE, benchmark_step_times

    require_cuda_settings(args.device)
    load_a2_config(ROOT)
    path = ROOT / BENCHMARK_FILE
    if path.exists():
        raise SystemExit(f"{path} exists; the step-time benchmark is write-once")
    report = benchmark_step_times(ROOT, args.device)
    atomic_write_json(path, report)
    print(json.dumps(report["kinds"], indent=2))
    return 0


def _holdout_setup(args: argparse.Namespace):
    import torch

    from sleeptcn.amendment_a2_holdout import holdout_inputs

    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    torch.set_flush_denormal(True)
    config, _ = load_a2_config(ROOT)
    return config, holdout_inputs(ROOT, config, args.parent_manifest, args.parent_processed_root)


def cmd_holdout_check(args: argparse.Namespace) -> int:
    from sleeptcn.amendment_a2_holdout import device_check

    require_cuda_settings(args.device)
    _, inputs = _holdout_setup(args)
    report = device_check(ROOT, inputs, args.restore_workspace, args.parent_predictions, args.device, args.batch_size)
    print(json.dumps(report, indent=2))
    print(f"DEVICE CHECK {'PASSED' if report['passed'] else 'FAILED'} ({args.device})")
    return 0 if report["passed"] else 1


def cmd_holdout_worker(args: argparse.Namespace) -> int:
    from sleeptcn.amendment_a2_holdout import infer

    require_cuda_settings(args.device)
    _, inputs = _holdout_setup(args)
    done = infer(ROOT, inputs, device=args.device, batch_size=args.batch_size, worker_index=args.index,
                 worker_count=args.count)
    print(f"[holdout worker {args.index}] {done} new fold artifacts", flush=True)
    return 0


def cmd_holdout(args: argparse.Namespace) -> int:
    from sleeptcn.amendment_a2_holdout import HOLDOUT_ROOT, finalize

    require_cuda_settings(args.device)
    logs = ROOT / HOLDOUT_ROOT / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    processes = []
    for index in range(args.workers):
        log = (logs / f"holdout_worker_{index}.log").open("a", encoding="utf-8")
        command = [sys.executable, str(Path(__file__).resolve()), "holdout-worker", "--index", str(index),
                   "--count", str(args.workers), "--device", args.device, "--threads", str(args.threads),
                   "--batch-size", str(args.batch_size), "--parent-manifest", str(args.parent_manifest),
                   "--parent-processed-root", str(args.parent_processed_root)]
        processes.append((index, subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT), log))
        print(f"holdout worker {index} started (log: {logs / f'holdout_worker_{index}.log'})", flush=True)
    failed = [index for index, process, log in processes if process.wait() != 0]
    for _, _, log in processes:
        log.close()
    if failed:
        print(f"FAILED holdout workers: {failed}; fix the cause and rerun 'holdout' (finished files are kept)")
        return 1
    _, inputs = _holdout_setup(args)
    manifest = finalize(ROOT, inputs)
    print(json.dumps(manifest, indent=2))
    print("A2 HOLDOUT INFERENCE COMPLETE (no metrics computed)")
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
    # Outside the repository: a file in the worktree would block the clean-tree test campaign.
    p.add_argument("--output", default=str(ROOT.parent / "a2_results.tar.gz"))

    p = sub.add_parser("prepare-sleepedf")
    p.add_argument("--data-dir", type=Path, required=True, help="Sleep-EDF sleep-cassette directory")
    p.add_argument("--raw-manifest", type=Path, default=ROOT / "data" / "manifests" / "raw_inventory.json")
    p.add_argument("--workers", type=int, default=4)

    p = sub.add_parser("prepare-shhs")
    p.add_argument("--parent-manifest", type=Path, required=True, help="preprocess_holdout_v1.json")
    p.add_argument("--technical-audit", type=Path, required=True, help="holdout_audit_v1.json")
    p.add_argument("--preprocess-config", type=Path, default=ROOT / "configs" / "shhs_preprocessing_v1.json")
    p.add_argument("--edf-dir", type=Path, required=True)
    p.add_argument("--xml-dir", type=Path, required=True)
    p.add_argument("--parent-processed-root", type=Path, default=ROOT / "data" / "processed" / "shhs_holdout_v1")
    p.add_argument("--workers", type=int, default=4)

    holdout_defaults = {"parent_manifest": ROOT / PARENT_MANIFEST_COPY,
                        "parent_processed_root": ROOT / "data" / "processed" / "shhs_holdout_v1"}
    for name in ("bundle", "holdout-check", "holdout", "holdout-worker"):
        p = sub.add_parser(name)
        p.add_argument("--parent-manifest", type=Path, default=holdout_defaults["parent_manifest"])
        p.add_argument("--parent-processed-root", type=Path, default=holdout_defaults["parent_processed_root"])
        if name == "bundle":
            p.add_argument("--part", choices=BUNDLE_PARTS, default="sleepedf")
            p.add_argument("--output", default=None, help="default: ../a2_inputs_<part>.tar")
            continue
        p.add_argument("--device", default="cpu")
        p.add_argument("--threads", type=int, default=2, help="torch CPU threads per process")
        p.add_argument("--batch-size", type=int, default=256)
        if name == "holdout-check":
            p.add_argument("--restore-workspace", type=Path, default=PARENT_RESTORE)
            p.add_argument("--parent-predictions", type=Path, default=PARENT_PREDICTIONS)
        if name == "holdout":
            p.add_argument("--workers", type=int, default=4, help="inference processes")
        if name == "holdout-worker":
            p.add_argument("--index", type=int, required=True)
            p.add_argument("--count", type=int, required=True)
    p = sub.add_parser("verify-bundle")
    p.add_argument("--part", choices=BUNDLE_PARTS, default="sleepedf")
    p = sub.add_parser("step-benchmark")
    p.add_argument("--device", default="cuda")

    args = parser.parse_args()
    handler = {"plan": cmd_plan, "repro": cmd_repro, "train": cmd_train, "train-worker": cmd_train_worker,
               "test": cmd_test, "status": cmd_status, "export": cmd_export,
               "prepare-sleepedf": cmd_prepare_sleepedf, "prepare-shhs": cmd_prepare_shhs, "bundle": cmd_bundle,
               "verify-bundle": cmd_verify_bundle, "step-benchmark": cmd_step_benchmark,
               "holdout-check": cmd_holdout_check, "holdout": cmd_holdout,
               "holdout-worker": cmd_holdout_worker}[args.command]
    return handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
