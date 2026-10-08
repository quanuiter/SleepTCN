"""CUDA source-only continuation campaign, preserving the frozen training recipe."""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
import torch
from sleeptcn.revision_weighted import train_arm, evaluate_source
from sleeptcn.revision_campaign import write_once_json
from sleeptcn.revision_weighted_campaign import atomic_json


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(args):
    require_cuda = torch.cuda.is_available()
    if not require_cuda:
        raise RuntimeError("CUDA unavailable; no silent CPU fallback")
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    manifest = json.loads((ROOT / "manifest.json").read_bytes())
    for filename, digest in manifest["files"].items():
        if sha(ROOT / filename) != digest:
            raise ValueError("Bundle source/data hash differs")
    cfg = json.loads((ROOT / "configs/teacher_revision_weighted_10fold_v1.json").read_bytes())
    roles, counts = {}, np.zeros(5, dtype=np.int64)
    for role in ["train", "validation", "test"]:
        roles[role] = []
        for entry in manifest["records"]:
            if entry["role"] != role:
                continue
            with np.load(ROOT / entry["path"], allow_pickle=False) as z:
                x, y = z["features"].copy(), z["labels"].astype(np.int64)
            if x.shape != (entry["epochs"], 128) or int((y >= 0).sum()) != entry["valid_epochs"]:
                raise ValueError("Role support differs")
            roles[role].append((torch.from_numpy(x), torch.from_numpy(y)))
            if role == "train":
                counts += np.bincount(y[y >= 0], minlength=5)
    np.testing.assert_array_equal(counts, manifest["train_class_counts"])
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    identity_base = {"fold": manifest["fold"], "seed": cfg["seed"],
        "protocol_sha256": sha(ROOT / "configs/teacher_revision_weighted_10fold_v1.json"),
        "encoder_sha256": manifest["source_specification"]["encoder_sha256"],
        "bundle_manifest_sha256": sha(ROOT / "manifest.json"),
        "runner_sha256": sha(__file__), "torch_version": str(torch.__version__), "backend": "cuda"}
    write_once_json(output / "execution_specification.json", {"identity": identity_base, "protocol": cfg,
        "gpu": torch.cuda.get_device_name(0), "tf32": False, "float32_only": True,
        "source_feature_backend": "CPU historical cache", "target_access": False,
        "cpu_partial_checkpoint_imported": False, "attempt_max_seconds": args.max_seconds})
    started = time.monotonic()
    context = {"status": "running", "fold": manifest["fold"], "phase": "initialization"}
    def guard(event=None):
        elapsed = time.monotonic() - started
        if event and event["phase"] == "epoch_saved":
            row = event["history"][-1]
            context.update(phase="training", arm=event["arm"], epoch=row["epoch"],
                           best_epoch=row["best_epoch"], stale_epochs=row["stale_epochs"],
                           epoch_seconds=row["seconds"], validation_macro_f1=row["validation_macro_f1"])
            atomic_json(output / "progress.json", {**context, "elapsed_attempt_seconds": elapsed})
        if elapsed >= args.max_seconds:
            raise TimeoutError("five-hour attempt limit; saved epochs retained")
    atomic_json(output / "progress.json", context)
    try:
        selections, models = {}, {}
        for arm, weights in [("unweighted", np.ones(5)), ("weighted", counts.sum() / (5 * counts))]:
            guard()
            print(f"START fold {manifest['fold']} {arm} CUDA", flush=True)
            models[arm], selections[arm] = train_arm(roles["train"], roles["validation"], weights, cfg,
                output / arm, {**identity_base, "arm": arm}, device="cuda", runtime_guard=guard)
        if selections["unweighted"]["initial_state_sha256"] != selections["weighted"]["initial_state_sha256"]:
            raise ValueError("Matched initialization differs")
        common = min(s["epochs_completed"] for s in selections.values())
        left = [r["batch_order_sha256"] for r in selections["unweighted"]["history"][:common]]
        right = [r["batch_order_sha256"] for r in selections["weighted"]["history"][:common]]
        if left != right:
            raise ValueError("Matched batch order differs")
        source = {}
        for arm, model in models.items():
            guard()
            val = evaluate_source(model, roles["validation"], cfg["batch_size_records"], "cuda")
            if val["macro_f1"] != selections[arm]["selected_validation_macro_f1"]:
                raise ValueError("Selected validation score does not replay")
            source[arm] = {"validation": val,
                           "outer_test": evaluate_source(model, roles["test"], cfg["batch_size_records"], "cuda")}
        result = {"status": "complete_source_only_cuda_pair", "fold": manifest["fold"],
                  "provenance": identity_base, "selections": selections, "source_results": source,
                  "train_class_weights": (counts.sum() / (5 * counts)).tolist(), "target_access": False}
        write_once_json(output / "aggregate_results.json", result)
        write_once_json(output / "verification.json", {"status": "passed",
            "aggregate_results_sha256": sha(output / "aggregate_results.json"),
            "source_selected_checkpoint_count": 2, "validation_replayed": True,
            "matched_initialization_and_batch_orders": True, "target_access": False})
        context.update(status="complete_source_only_cuda_pair", phase="complete")
    except Exception as error:
        context.update(status="stopped_resource_budget" if isinstance(error, TimeoutError) else "failed",
                       reason=f"{type(error).__name__}: {error}", saved_checkpoints_retained=True)
        raise
    finally:
        atomic_json(output / "progress.json", {**context, "elapsed_attempt_seconds": time.monotonic()-started})
        archive = output.parent / f"SleepTCN_Fold{manifest['fold']:02d}_CUDA_Checkpoints.zip"
        temp = archive.with_suffix(".tmp.zip")
        with zipfile.ZipFile(temp, "w", compression=zipfile.ZIP_DEFLATED) as z:
            for path in sorted(output.rglob("*")):
                if path.is_file() and path.suffix in [".json", ".pt", ".log"]:
                    z.write(path, path.relative_to(output).as_posix())
        temp.replace(archive)
        print("RESULT_ARCHIVE", str(archive), "SHA256", sha(archive), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-seconds", type=float, default=18000)
    args = parser.parse_args()
    if not 0 < args.max_seconds <= 18000:
        raise ValueError("Positive attempt budget at most five hours required")
    run(args)
