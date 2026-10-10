"""Locked inference for the SHHS1 holdout protocol (configs/shhs_holdout_confirmatory_v1.json).

Stage 1 (checkpoint gate) verifies every checkpoint of every arm before any inference and
writes ``checkpoint_gate.json``. Stage 2 runs the arms in protocol order and writes per-fold
probabilities and ten-fold ensembles. **No metric is computed and no label is written to the
prediction artifacts**; scoring is done afterwards by the analysis script, once all arms exist.

Zero-shot arms reuse the v1 code path (``build_context`` + ``test_gate`` loaders +
``shhs_zero_shot.infer_probabilities`` + ``ensemble_probabilities``). ADAST arms reuse the
locked ADAST code (``run_colab_adast_training.build_models/predict``): maximum dual-head logits,
softmax per fold, float64 mean cast to float32; target attention for ADAST and source attention
for source-only. ADAST is evaluated on the holdout ``filtered_v2`` epochs: these are the same
values as the full-record ``label_free_signal_variants`` output restricted to the scoring window,
because the ADAST network classifies each 30-s epoch independently.

Modes: ``gate`` (verify checkpoints once), ``worker`` (one of N parallel processes, each
running a disjoint slice of the 120 arm x fold units and streaming records from disk),
``finalize`` (ensembles + manifests once every fold artifact exists), ``progress``, or ``all``.
The run is resumable: existing fold artifacts are kept and never overwritten.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from sleeptcn.evaluation.shhs_zero_shot import ensemble_probabilities  # noqa: E402
from sleeptcn.io.hashing import sha256_file  # noqa: E402
from sleeptcn.io.serialization import atomic_savez, atomic_write_json  # noqa: E402

ZERO_SHOT_ARMS = [f"{e}_s{s}" for s in (42, 123) for e in ("E3", "E0", "E2", "E6", "E4")]
ADAST_ARMS = ["ADAST_full_e30", "SourceOnly_full_e30"]
ARM_ORDER = ZERO_SHOT_ARMS[:5] + ADAST_ARMS + ZERO_SHOT_ARMS[5:]
FOLDS = tuple(range(10))
ROLE = "holdout_test"
ADAST_PAYLOAD = ROOT / "runs" / "colab_adast_20261004" / "payload"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_bytes())


# --------------------------------------------------------------------------- inputs
def load_inputs(args: argparse.Namespace) -> dict[str, Any]:
    protocol_raw = args.protocol.read_bytes()
    protocol = json.loads(protocol_raw)
    protocol_sha = sha256_bytes(protocol_raw)
    if protocol.get("status") != "locked_before_holdout_preprocessing":
        raise ValueError("Holdout protocol is not locked")
    pre_raw = args.preprocess_manifest.read_bytes()
    pre = json.loads(pre_raw)
    if pre.get("status") != "complete" or pre.get("protocol_sha256") != protocol_sha:
        raise ValueError("Preprocess manifest is incomplete or was produced under another protocol")
    if len(pre["analysis_cohort"]) != protocol["cohort"]["n_target"]:
        raise ValueError("Preprocessed cohort size differs from protocol")
    entries: dict[str, list[dict[str, Any]]] = {}
    for record in pre["records"]:
        entries.setdefault(record["variant"], []).append(record)
    for variant, rows in entries.items():
        rows.sort(key=lambda r: r["record_key"])
        if len(rows) != len(pre["analysis_cohort"]):
            raise ValueError(f"{variant}: record count differs from cohort")
    return {
        "protocol": protocol,
        "protocol_sha256": protocol_sha,
        "preprocess_manifest_sha256": sha256_bytes(pre_raw),
        "entries": entries,
    }


def arm_spec(protocol: dict[str, Any], arm: str) -> dict[str, Any]:
    spec = protocol["arms"][arm]
    if arm in ADAST_ARMS:
        return {"kind": "adast", "variant": "filtered_v2", "seed": 123,
                "attention": "target" if arm.startswith("ADAST") else "source",
                "checkpoint_arm": "adast" if arm.startswith("ADAST") else "source_only"}
    return {"kind": "zero_shot", "experiment": spec["experiment"], "seed": int(spec["seed"]),
            "variant": spec["data_variant"]}


# --------------------------------------------------------------------------- gate
def checkpoint_gate(args: argparse.Namespace, ctx: dict[str, Any], arms: list[str]) -> dict[str, Any]:
    gate_path = args.output_root / "checkpoint_gate.json"
    protocol = ctx["protocol"]
    restore_manifest = args.restore_workspace / "restoration_manifest.json"
    restore = read_json(restore_manifest)
    if restore.get("status") != "complete" or restore.get("protocol_sha256") != ctx["protocol_sha256"]:
        raise ValueError("Checkpoint restoration is incomplete or used another protocol")
    if restore.get("audit_sha256") != protocol["checkpoint_gate"]["reference_hash_sources"]["git_history_audit_sha256"]:
        raise ValueError("Restoration used another checkpoint audit")
    results: dict[str, Any] = {}
    for arm in arms:
        spec = arm_spec(protocol, arm)
        if spec["kind"] == "zero_shot":
            prefix = f"runs/v2/full/{spec['experiment']}/"
            rows = [r for r in restore["records"] if r["git_path"].startswith(prefix)
                    and f"/seed_{spec['seed']}/" in r["git_path"]]
            bad = [r["git_path"] for r in rows
                   if sha256_file(args.restore_workspace / r["git_path"]) != r["sha256"]]
            n_best = sum(r["audited_checkpoint"] for r in rows)
            results[arm] = {"files_checked": len(rows), "best_checkpoints": n_best, "mismatches": bad,
                            "passed": not bad and n_best == restore["arms"][arm]["best_checkpoints"]}
        else:
            backup = read_json(args.adast_backup / "backup_verification.json")
            if backup.get("frozen_list_sha256") != protocol["checkpoint_gate"]["reference_hash_sources"]["adast_final_frozen_list_sha256"]:
                raise ValueError("ADAST backup refers to another frozen checkpoint list")
            rows = [r for r in backup["records"] if r["arm"] == spec["checkpoint_arm"]]
            bad = [r["backup_path"] for r in rows if sha256_file(ROOT / r["backup_path"]) != r["sha256"]
                   or not r["matches_frozen"]]
            results[arm] = {"files_checked": len(rows), "mismatches": bad,
                            "passed": not bad and sorted(r["fold"] for r in rows) == list(FOLDS)}
    report = {
        "schema_version": 1,
        "gate": "SHHS1_HOLDOUT_CHECKPOINT_GATE",
        "status": "passed" if all(v["passed"] for v in results.values()) else "failed",
        "protocol_sha256": ctx["protocol_sha256"],
        "preprocess_manifest_sha256": ctx["preprocess_manifest_sha256"],
        "restoration_manifest_sha256": sha256_file(restore_manifest),
        "arms": results,
        "dropped_arms": [arm for arm, v in results.items() if not v["passed"]],
        "inference_started": False,
    }
    if gate_path.exists():
        previous = read_json(gate_path)
        if previous["status"] != "passed" or previous["arms"].keys() != results.keys() or report["status"] != "passed":
            raise ValueError("Checkpoint gate differs from the gate recorded before inference")
    else:
        args.output_root.mkdir(parents=True, exist_ok=True)
        atomic_write_json(gate_path, report, ensure_ascii=False, sort_keys=False)
    if report["status"] != "passed":
        raise ValueError(f"Checkpoint gate failed: {report['dropped_arms']}")
    return report


# --------------------------------------------------------------------------- records
def load_record(entry: dict[str, Any], variant: str, processed_root: Path) -> dict[str, Any]:
    path = Path(entry["output_path"]).resolve()
    if not path.is_relative_to(processed_root.resolve()):
        raise ValueError(f"{path}: outside processed root")
    if sha256_file(path) != entry["output_sha256"]:
        raise ValueError(f"{path.name}: input hash differs from preprocess manifest")
    with np.load(path, allow_pickle=False) as z:
        checks = {
            "record_key": str(z["record_key"].item()) == entry["record_key"] == path.stem,
            "subject_id": str(z["subject_id"].item()) == entry["subject_id"],
            "role": str(z["role"].item()) == ROLE,
            "variant": str(z["preprocess_version"].item()) == variant,
        }
        x = z["x"].copy()
        indices = z["original_epoch_index"].copy()
    if not all(checks.values()) or x.dtype != np.float32 or x.ndim != 2 or x.shape[1] != 3000 or not np.isfinite(x).all():
        raise ValueError(f"{path.name}: invalid holdout input {checks}")
    return {"record_key": entry["record_key"], "subject_id": entry["subject_id"], "input_sha256": entry["output_sha256"],
            "x": x, "original_epoch_index": indices}


def fold_path(root: Path, arm: str, fold: int, key: str) -> Path:
    return root / "fold_predictions" / arm / f"fold_{fold:02d}" / f"{key}.npz"


def ensemble_path(root: Path, arm: str, key: str) -> Path:
    return root / "ensemble" / arm / f"{key}.npz"


def save_probabilities(path: Path, probabilities: np.ndarray, record: dict[str, Any], metadata: dict[str, Any]) -> None:
    atomic_savez(path, {
        "metadata_json": np.array(json.dumps(metadata, sort_keys=True)),
        "probabilities": probabilities.astype(np.float32),
        "original_epoch_index": record["original_epoch_index"],
    })


def load_probabilities(path: Path, metadata: dict[str, Any], n_epochs: int) -> np.ndarray:
    with np.load(path, allow_pickle=False) as z:
        stored = json.loads(str(z["metadata_json"].item()))
        p = z["probabilities"].copy()
    if stored != metadata:
        raise ValueError(f"{path}: existing artifact has different provenance")
    if p.shape != (n_epochs, 5) or not np.isfinite(p).all() or not np.allclose(p.sum(1), 1.0, atol=1e-5):
        raise ValueError(f"{path}: invalid stored probabilities")
    return p


# --------------------------------------------------------------------------- models
def zero_shot_predictor(args: argparse.Namespace, spec: dict[str, Any], fold: int):
    from sleeptcn.experiment import build_context
    from sleeptcn.shhs_zero_shot import ExternalRecord, infer_probabilities
    from sleeptcn.test_gate import _load_extractor, _load_sequence_model

    context = build_context(args.restore_workspace.resolve(), spec["experiment"], fold, spec["seed"], "cpu",
                            smoke=False, allow_test_evaluation=True, num_workers=0, resume=True)
    extractor_kind, extractor, extractor_hash = _load_extractor(context)
    sequence_kind, sequence, sequence_path = _load_sequence_model(context)
    model_hashes = {"extractor_sha256": extractor_hash, "sequence_checkpoint_sha256": sha256_file(sequence_path)}

    def predict(record: dict[str, Any]) -> np.ndarray:
        n = len(record["x"])
        external = ExternalRecord(path=Path(record["record_key"]), input_sha256=record["input_sha256"],
                                  record_key=record["record_key"], subject_id=record["subject_id"], role=ROLE,
                                  variant=spec["variant"], x=record["x"], y=np.zeros(n, dtype=np.int8),
                                  valid_mask=np.ones(n, dtype=bool), original_epoch_index=record["original_epoch_index"])
        return infer_probabilities(external, extractor_kind, extractor, sequence, sequence_kind, args.batch_size)

    return predict, model_hashes


def adast_predictor(args: argparse.Namespace, spec: dict[str, Any], fold: int):
    from run_colab_adast_training import build_models, load_module, predict as adast_predict

    backup = read_json(args.adast_backup / "backup_verification.json")
    row = next(r for r in backup["records"] if r["fold"] == fold and r["arm"] == spec["checkpoint_arm"])
    path = ROOT / row["backup_path"]
    if sha256_file(path) != row["sha256"]:
        raise ValueError(f"{path}: ADAST checkpoint hash changed")
    module = load_module("locked_ADAST_model", ADAST_PAYLOAD / "upstream" / "models.py")
    cfg = load_module("locked_ADAST_config", ADAST_PAYLOAD / "upstream" / "configs.py").Config()
    payload = torch.load(path, map_location="cpu", weights_only=True)
    models = build_models(module, cfg, 123, "cpu")
    for name, model in models.items():
        model.load_state_dict(payload["models"][name], strict=True)
        model.eval()

    def predict(record: dict[str, Any]) -> np.ndarray:
        logits = adast_predict(models, record["x"], spec["attention"], "cpu", batch_size=args.adast_batch_size)
        values = torch.as_tensor(logits, dtype=torch.float32)
        if values.ndim != 2 or values.shape[1] != 5 or not torch.isfinite(values).all():
            raise ValueError("ADAST returned invalid logits")
        return torch.softmax(values, -1).numpy()

    return predict, {"checkpoint_sha256": row["sha256"], "attention": spec["attention"]}


# --------------------------------------------------------------------------- work units
def all_units() -> list[tuple[str, int]]:
    """(arm, fold) units in protocol priority order: confirmatory seed-42 arms first."""
    return [(arm, fold) for arm in ARM_ORDER for fold in FOLDS]


def worker_units(index: int, count: int) -> list[tuple[str, int]]:
    if count <= 0 or not 0 <= index < count:
        raise ValueError("invalid worker index/count")
    return all_units()[index::count]


def run_unit(args: argparse.Namespace, ctx: dict[str, Any], arm: str, fold: int) -> int:
    """Predict one fold of one arm for every record, streaming records from disk."""
    spec = arm_spec(ctx["protocol"], arm)
    entries = ctx["entries"][spec["variant"]]
    base_meta = {"schema_version": 1, "protocol_sha256": ctx["protocol_sha256"],
                 "preprocess_manifest_sha256": ctx["preprocess_manifest_sha256"], "arm": arm,
                 "data_variant": spec["variant"], "seed": spec["seed"], "role": ROLE,
                 "device": "cpu", "torch_version": torch.__version__}
    pending = [e for e in entries if not fold_path(args.output_root, arm, fold, e["record_key"]).exists()]
    if not pending:
        return 0
    maker = zero_shot_predictor if spec["kind"] == "zero_shot" else adast_predictor
    predictor, model_meta = maker(args, spec, fold)
    started = time.perf_counter()
    for index, entry in enumerate(pending, start=1):
        record = load_record(entry, spec["variant"], args.processed_root)
        metadata = {**base_meta, **model_meta, "artifact_type": "holdout_fold_prediction",
                    "outer_fold": fold, "record_key": record["record_key"],
                    "subject_id": record["subject_id"], "input_npz_sha256": record["input_sha256"]}
        probabilities = predictor(record)
        if probabilities.shape != (len(record["x"]), 5) or not np.isfinite(probabilities).all():
            raise ValueError(f"{arm}/{record['record_key']}: invalid probabilities")
        save_probabilities(fold_path(args.output_root, arm, fold, record["record_key"]), probabilities, record, metadata)
        if index % 50 == 0 or index == len(pending):
            rate = (time.perf_counter() - started) / index
            print(f"[{arm}] fold {fold:02d} {index}/{len(pending)} new records ({rate:.2f}s/record)", flush=True)
    return len(pending)


def finalize_arm(args: argparse.Namespace, ctx: dict[str, Any], arm: str) -> dict[str, Any]:
    """Verify all ten fold artifacts of an arm and write its ensembles and arm manifest."""
    spec = arm_spec(ctx["protocol"], arm)
    entries = ctx["entries"][spec["variant"]]
    base_meta = {"schema_version": 1, "protocol_sha256": ctx["protocol_sha256"],
                 "preprocess_manifest_sha256": ctx["preprocess_manifest_sha256"], "arm": arm,
                 "data_variant": spec["variant"], "seed": spec["seed"], "role": ROLE,
                 "device": "cpu", "torch_version": torch.__version__}
    fold_hashes, ensemble_hashes = [], []
    for entry in entries:
        parts, reference_index = [], None
        for fold in FOLDS:
            path = fold_path(args.output_root, arm, fold, entry["record_key"])
            if not path.exists():
                raise FileNotFoundError(f"{path}: fold artifact missing")
            with np.load(path, allow_pickle=False) as z:
                meta = json.loads(str(z["metadata_json"].item()))
                if (meta["input_npz_sha256"] != entry["output_sha256"] or meta["arm"] != arm
                        or meta["outer_fold"] != fold or meta["protocol_sha256"] != ctx["protocol_sha256"]):
                    raise ValueError(f"{path}: provenance differs")
                index = z["original_epoch_index"].copy()
                if reference_index is None:
                    reference_index = index
                np.testing.assert_array_equal(index, reference_index)
                parts.append(z["probabilities"].copy())
            fold_hashes.append({"fold": fold, "record_key": entry["record_key"], "sha256": sha256_file(path)})
        mean = ensemble_probabilities(parts)  # float64 accumulator, cast to float32 (v1 and ADAST spec)
        prediction = np.argmax(mean, axis=1).astype(np.int8)  # ties -> lowest class index
        path = ensemble_path(args.output_root, arm, entry["record_key"])
        metadata = {**base_meta, "artifact_type": "holdout_ensemble_prediction", "folds": list(FOLDS),
                    "aggregation": "arithmetic_mean_probability_float64_accumulator",
                    "record_key": entry["record_key"], "subject_id": entry["subject_id"],
                    "input_npz_sha256": entry["output_sha256"]}
        atomic_savez(path, {"metadata_json": np.array(json.dumps(metadata, sort_keys=True)),
                            "probabilities": mean, "prediction": prediction,
                            "original_epoch_index": reference_index})
        ensemble_hashes.append({"record_key": entry["record_key"], "subject_id": entry["subject_id"],
                                "sha256": sha256_file(path)})
    summary = {"arm": arm, "status": "complete", "records": len(entries), "folds": len(FOLDS),
               "fold_artifacts": len(fold_hashes), "ensemble_artifacts": len(ensemble_hashes),
               "fold_records": fold_hashes, "ensemble_records": ensemble_hashes}
    atomic_write_json(args.output_root / "arm_manifests" / f"{arm}.json", summary, ensure_ascii=False, sort_keys=False)
    return summary


def run_arm(args: argparse.Namespace, ctx: dict[str, Any], arm: str) -> dict[str, Any]:
    for fold in FOLDS:
        run_unit(args, ctx, arm, fold)
    return finalize_arm(args, ctx, arm)


def progress(args: argparse.Namespace, ctx: dict[str, Any]) -> tuple[int, int]:
    root = args.output_root / "fold_predictions"
    done = sum(1 for _ in root.glob("*/fold_*/*.npz")) if root.exists() else 0
    total = sum(len(ctx["entries"][arm_spec(ctx["protocol"], a)["variant"]]) for a in ARM_ORDER) * len(FOLDS)
    return done, total


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=ROOT / "configs" / "shhs_holdout_confirmatory_v1.json")
    parser.add_argument("--preprocess-manifest", type=Path, required=True)
    parser.add_argument("--processed-root", type=Path, default=ROOT / "data" / "processed" / "shhs_holdout_v1")
    parser.add_argument("--restore-workspace", type=Path, default=ROOT / "runs" / "holdout_ckpt_restore_20261010")
    parser.add_argument("--adast-backup", type=Path, default=ROOT / "runs" / "adast_fullsource_checkpoints_backup_20261010")
    parser.add_argument("--output-root", type=Path, default=ROOT / "runs" / "holdout_inference_v1")
    parser.add_argument("--mode", choices=("all", "gate", "worker", "finalize", "progress"), default="all")
    parser.add_argument("--worker-index", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--adast-batch-size", type=int, default=128)
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    torch.set_flush_denormal(True)
    ctx = load_inputs(args)

    if args.mode == "progress":
        done, total = progress(args, ctx)
        print(f"{done}/{total}")
        return 0
    if args.mode in ("all", "gate"):
        gate = checkpoint_gate(args, ctx, ARM_ORDER)
        print(f"Checkpoint gate: {gate['status']} ({len(gate['arms'])} arms)", flush=True)
        if args.mode == "gate":
            return 0
    gate_path = args.output_root / "checkpoint_gate.json"
    if not gate_path.exists() or read_json(gate_path).get("status") != "passed":
        raise ValueError("Run --mode gate first; no inference before a passed checkpoint gate")
    if read_json(gate_path)["protocol_sha256"] != ctx["protocol_sha256"]:
        raise ValueError("Checkpoint gate refers to another protocol")

    if args.mode in ("all", "worker"):
        units = all_units() if args.mode == "all" else worker_units(args.worker_index, args.num_workers)
        started = time.perf_counter()
        for arm, fold in units:
            run_unit(args, ctx, arm, fold)
        print(f"Worker {args.worker_index}/{args.num_workers}: {len(units)} units done in "
              f"{time.perf_counter() - started:.0f}s", flush=True)
        if args.mode == "worker":
            return 0

    # finalize: ensembles and manifests, only when every fold artifact exists.
    done, total = progress(args, ctx)
    if done != total:
        print(f"Not finalized: {done}/{total} fold artifacts present")
        return 2
    for arm in ARM_ORDER:
        finalize_arm(args, ctx, arm)
        print(f"[{arm}] ensembles written", flush=True)
    manifest = {"schema_version": 1, "status": "complete_all_arms_no_metrics_computed",
                "protocol_sha256": ctx["protocol_sha256"],
                "preprocess_manifest_sha256": ctx["preprocess_manifest_sha256"],
                "checkpoint_gate_sha256": sha256_file(gate_path),
                "arm_manifest_sha256": {a: sha256_file(args.output_root / "arm_manifests" / f"{a}.json") for a in ARM_ORDER},
                "execution": {"device": "cpu", "torch_version": torch.__version__,
                              "deterministic_algorithms": True, "flush_denormal": True}}
    atomic_write_json(args.output_root / "run_manifest.json", manifest, ensure_ascii=False, sort_keys=False)
    print("STATUS: ALL ARMS COMPLETE (no metrics computed)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
