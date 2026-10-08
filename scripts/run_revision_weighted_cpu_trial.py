"""Run one next-fold matched TCN pair on CPU, with a five-hour resource guard.

No SHHS predictions or scores are generated: the ten-fold campaign requires all
twenty source-selected sequence checkpoints before a new target ensemble is scored.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from sleeptcn.demo import load_processed_demo_record
from sleeptcn.calibration import source_class_weights
from sleeptcn.models import SleepTCN
from sleeptcn.metrics import confusion_matrix_5, metrics_from_confusion
from sleeptcn.preprocessing import sha256_file
from sleeptcn.revision_campaign import load_restored_pair, extract_features, write_once_json
from sleeptcn.revision_weighted import train_arm, evaluate_source, state_digest

PROTOCOL = ROOT / "configs/teacher_revision_weighted_10fold_v1.json"


class ResourceStop(RuntimeError):
    pass


class RuntimeGuard:
    def __init__(self, max_seconds, campaign_hours_limit=5., remaining_folds=9):
        if not 0 < max_seconds <= 18000 or campaign_hours_limit <= 0:
            raise ValueError("resource limits must be positive and at most five hours per trial")
        self.started = time.perf_counter()
        self.max_seconds = max_seconds
        self.campaign_hours_limit = campaign_hours_limit
        self.remaining_folds = remaining_folds
        self.feature_seconds = 0.
        self.last_projection = None

    def __call__(self, progress=None):
        elapsed = time.perf_counter() - self.started
        if elapsed >= self.max_seconds:
            raise ResourceStop("five-hour wall-clock budget reached; last completed epoch is retained")
        if progress and progress["phase"] == "epoch_saved" and len(progress["history"]) >= 5:
            seconds = float(np.median([r["seconds"] for r in progress["history"][-5:]]))
            # Planning projection uses the completed fold-0 pair's 42+53 epochs.
            # It is a runtime estimate, never a scientific stopping/selection metric.
            estimate = self.remaining_folds * (self.feature_seconds + 95 * seconds) / 3600
            self.last_projection = {"epoch_seconds_median": seconds,
                "assumed_epochs_per_pair_from_fold0": 95, "remaining_fold_pairs": self.remaining_folds,
                "source_cache_seconds_per_fold_measured": self.feature_seconds,
                "projected_remaining_feature_train_validation_hours": estimate,
                "excludes_target_inference_and_final_verification": True,
                "elapsed_trial_seconds": elapsed}
            if estimate > self.campaign_hours_limit:
                raise ResourceStop("measured throughput projects remaining weighted campaign above five hours")


def prepare_cache(encoder, encoder_hash, split_path, fold, cache, guard):
    split = json.loads(split_path.read_bytes())["outer_runs"][fold]
    split_hash = sha256_file(split_path)
    specification = {"fold": fold, "seed": 123, "encoder_sha256": encoder_hash,
                     "split_sha256": split_hash, "variant": "filtered_v2", "batch_size": 64}
    write_once_json(cache / "execution_specification.json", specification)
    entries, roles, counts = [], {}, np.zeros(5, dtype=np.int64)
    tick = time.perf_counter()
    for role in ["train", "validation", "test"]:
        records, support = [], 0
        for ordinal, key in enumerate(sorted(split[role]["record_keys"]), 1):
            guard()
            path = ROOT / "data/processed/filtered_v2" / (key + ".npz")
            record = load_processed_demo_record(path)
            with np.load(path, allow_pickle=False) as source:
                if str(source["record_key"]) != key or str(source["subject_id"]) != key[:5]:
                    raise ValueError("source record identity differs")
            metadata = {**specification, "record_key": key, "subject_id": key[:5], "role": role,
                        "input_sha256": sha256_file(path)}
            destination = cache / role / (key + ".npz")
            if not destination.exists():
                features = extract_features(encoder, record.x, 64).astype(np.float32, copy=False)
                destination.parent.mkdir(parents=True, exist_ok=True)
                temporary = destination.with_suffix(".tmp.npz")
                with temporary.open("wb") as stream:
                    np.savez_compressed(stream, features=features, labels=record.labels,
                        original_epoch_index=record.original_epoch_index,
                        metadata_json=np.array(json.dumps(metadata)))
                temporary.replace(destination)
            with np.load(destination, allow_pickle=False) as z:
                if json.loads(str(z["metadata_json"])) != metadata:
                    raise ValueError("resumed cache provenance differs")
                np.testing.assert_array_equal(z["labels"], record.labels)
                np.testing.assert_array_equal(z["original_epoch_index"], record.original_epoch_index)
                features = z["features"]
                if features.shape != (len(record.x), 128) or features.dtype != np.float32 or not np.isfinite(features).all():
                    raise ValueError("invalid cached features")
                records.append((torch.from_numpy(features.copy()), torch.from_numpy(z["labels"].astype(np.int64))))
            valid = record.labels >= 0
            support += int(valid.sum())
            if role == "train":
                counts += np.bincount(record.labels[valid], minlength=5)
            entries.append({**metadata, "path": str(destination.resolve()), "sha256": sha256_file(destination),
                            "epochs": len(record.labels), "valid_epochs": int(valid.sum())})
            if ordinal % 20 == 0 or ordinal == len(split[role]["record_keys"]):
                print(f"fold {fold} source features {role}: {ordinal}/{len(split[role]['record_keys'])}", flush=True)
        if support != split[role]["valid_epochs"]:
            raise ValueError("source role support differs")
        if {e["subject_id"] for e in entries if e["role"] == role} != set(split[role]["subject_ids"]):
            raise ValueError("source subject membership differs")
        roles[role] = records
    if not np.array_equal(counts, [split["train"]["label_counts"][str(i)] for i in range(5)]):
        raise ValueError("train-only class counts differ")
    guard.feature_seconds = time.perf_counter() - tick
    manifest = {"specification": specification, "records": entries, "train_class_counts": counts.tolist()}
    write_once_json(cache / "private_manifest.json", manifest)
    labels = np.concatenate([y.numpy() for _, y in roles["train"]])
    weights = source_class_weights(labels)
    np.testing.assert_allclose(weights, counts.sum() / (5 * counts))
    return roles, weights, manifest


def verify_pair(output, cache, roles, selections, cfg, encoder, manifest, guard):
    source_results, batch_orders = {}, {}
    for arm, selected in selections.items():
        guard()
        if sha256_file(Path(selected["checkpoint_path"])) != selected["checkpoint_sha256"]:
            raise ValueError("checkpoint hash changed")
        payload = torch.load(selected["checkpoint_path"], map_location="cpu", weights_only=True)
        model = SleepTCN(input_dim=128).eval()
        model.load_state_dict(payload["model_state"], strict=True)
        validation = evaluate_source(model, roles["validation"], cfg["batch_size_records"])
        if validation["macro_f1"] != selected["selected_validation_macro_f1"]:
            raise ValueError("selected validation score does not replay")
        # Source outer test is evaluated only after both arms have been selected.
        source_results[arm] = {"validation": validation,
            "outer_test": evaluate_source(model, roles["test"], cfg["batch_size_records"])}
        batch_orders[arm] = [r["batch_order_sha256"] for r in selected["history"]]
    common = min(map(len, batch_orders.values()))
    if batch_orders["weighted"][:common] != batch_orders["unweighted"][:common]:
        raise ValueError("matched source batch orders differ")
    replayed = []
    for role in ["train", "validation", "test"]:
        entry = next(e for e in manifest["records"] if e["role"] == role)
        guard()
        source = ROOT / "data/processed/filtered_v2" / (entry["record_key"] + ".npz")
        record = load_processed_demo_record(source)
        actual = extract_features(encoder, record.x, 64)
        with np.load(entry["path"], allow_pickle=False) as z:
            np.testing.assert_array_equal(actual, z["features"])
        replayed.append({"role": role, "max_feature_residual": 0.})
    for entry in manifest["records"]:
        if sha256_file(Path(entry["path"])) != entry["sha256"]:
            raise ValueError("feature cache changed")
    return source_results, {"status": "passed", "source_selected_checkpoints_verified": 2,
        "validation_selection_scores_replayed": True, "matching_initialization_and_batch_orders": True,
        "source_feature_cache_hashes_verified": len(manifest["records"]),
        "feature_replays": replayed, "target_inference_performed": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--restored", required=True, type=Path)
    parser.add_argument("--fold", type=int, default=1, choices=range(1, 10))
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-seconds", type=float, default=18000)
    args = parser.parse_args()
    output, cache = args.output.resolve(), args.cache.resolve()
    if not output.is_relative_to((ROOT / "runs").resolve()) or not cache.is_relative_to((ROOT / "data/cache").resolve()):
        raise ValueError("require private output/cache folders")
    if (output / "aggregate_results.json").exists():
        raise FileExistsError("completed trial will not be overwritten")
    output.mkdir(parents=True, exist_ok=True)
    cfg = json.loads(PROTOCOL.read_bytes())
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    guard = RuntimeGuard(args.max_seconds)
    encoder, _, hashes = load_restored_pair(args.restored, "E3", args.fold, cfg["seed"])
    split_path = args.restored / "data/splits/sleepedf_sc_10fold_seed42_v2.json"
    code_paths = [Path(__file__).resolve(), PROTOCOL, ROOT / "src/sleeptcn/revision_weighted.py",
                  ROOT / "src/sleeptcn/revision_campaign.py", ROOT / "src/sleeptcn/models.py",
                  ROOT / "src/sleeptcn/training.py", ROOT / "src/sleeptcn/metrics.py",
                  ROOT / "src/sleeptcn/demo.py", ROOT / "src/sleeptcn/calibration.py"]
    spec = {"fold": args.fold, "seed": cfg["seed"], "protocol": cfg,
        "protocol_sha256": sha256_file(PROTOCOL), "split_sha256": sha256_file(split_path),
        "encoder_sha256": hashes[0], "restoration_sha256": sha256_file(args.restored / "restoration_manifest.json"),
        "resource_limits": {"cpu_threads": 4, "trial_max_seconds": args.max_seconds,
            "remaining_campaign_projection_stop_hours": 5., "remaining_pairs_for_projection": 9,
            "projection_reference_epochs": 95},
        "code_sha256": {str(p.relative_to(ROOT)): sha256_file(p) for p in code_paths},
        "torch_version": str(torch.__version__), "target_inference_performed": False}
    write_once_json(output / "execution_specification.json", spec)
    for path in code_paths:
        destination = output / "code_snapshot" / path.relative_to(ROOT)
        if not destination.exists():
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, destination)
    try:
        roles, weights, manifest = prepare_cache(encoder, hashes[0], split_path, args.fold, cache, guard)
        selections = {}
        for arm, values in [("unweighted", np.ones(5)), ("weighted", weights)]:
            guard()
            identity = {"fold": args.fold, "seed": cfg["seed"], "arm": arm,
                "protocol_sha256": spec["protocol_sha256"], "encoder_sha256": hashes[0],
                "cache_manifest_sha256": sha256_file(cache / "private_manifest.json"),
                "execution_specification_sha256": sha256_file(output / "execution_specification.json")}
            _, selections[arm] = train_arm(roles["train"], roles["validation"], values, cfg,
                output / arm, identity, runtime_guard=guard)
        if selections["unweighted"]["initial_state_sha256"] != selections["weighted"]["initial_state_sha256"]:
            raise ValueError("paired initial weights differ")
        write_once_json(output / "both_arms_selected.json", selections)
        source_results, verification = verify_pair(output, cache, roles, selections, cfg, encoder, manifest, guard)
        result = {"status": "complete_source_only_next_fold_cpu_trial", "fold": args.fold,
            "seed": cfg["seed"], "provenance": spec, "selections": selections,
            "source_results": source_results, "train_class_weights": weights.tolist(),
            "source_cache_seconds": guard.feature_seconds,
            "trial_elapsed_seconds": time.perf_counter() - guard.started,
            "throughput_projection": guard.last_projection,
            "target_inference_performed": False}
        write_once_json(output / "aggregate_results.json", result)
        verification["aggregate_results_sha256"] = sha256_file(output / "aggregate_results.json")
        write_once_json(output / "verification.json", verification)
        print(json.dumps({"status": result["status"], "fold": args.fold,
            "elapsed_seconds": result["trial_elapsed_seconds"],
            "train_validation_seconds": {arm: value["training_and_validation_seconds"] for arm,value in selections.items()},
            "projection": guard.last_projection}, indent=2), flush=True)
    except (ResourceStop, KeyboardInterrupt) as error:
        stopped = {"status": "stopped_resource_budget" if isinstance(error, ResourceStop) else "interrupted",
                   "reason": str(error), "elapsed_seconds": time.perf_counter() - guard.started,
                   "throughput_projection": guard.last_projection, "completed_epoch_checkpoints_retained": True}
        # Separate attempt reports preserve earlier interruptions when resumed.
        ordinal = len(list(output.glob("stop_attempt_*.json"))) + 1
        write_once_json(output / f"stop_attempt_{ordinal:02d}.json", stopped)
        print(json.dumps(stopped, indent=2), flush=True)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
