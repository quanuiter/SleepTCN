"""Amendment A2: zero-shot inference of the new A2 arms on the 500 SHHS1 holdout participants.

The model path is the one of the parent holdout inference (``test_gate`` loaders, ResNet-1D
features in batches, TCN over the whole record without padding, softmax); only the run roots
(``runs/a2``) and the device differ. Before any A2 arm is run on a device, the same code must
reproduce the stored parent CPU fold probabilities of ``E4_s42`` for two records at folds 0
and 9 within 1e-5 with identical argmax (``device_check``). Fold artifacts are write-once;
ensembles use the parent rule (float64 mean of ten fold probabilities, argmax).
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .amendment_a2 import A2_ROOT, A2Target, build_a2_context, load_a2_config, test_journal_path
from .amendment_a2_data import (SHHS_MANIFEST, SHHS_PROCESSED, SHHS_REFERENCE_VARIANT, VARIANT,
                                load_parent_manifest, parent_entries)
from .io.hashing import sha256_file
from .io.serialization import atomic_savez, atomic_write_json, read_json

ROLE = "holdout_test"
FOLDS = tuple(range(10))
HOLDOUT_ROOT = A2_ROOT / "holdout"
CHECK_ARM = "E4_s42"
CHECK_FOLDS = (0, 9)
CHECK_RECORDS = 2
CHECK_TOLERANCE = 1e-5


@dataclass(frozen=True)
class HoldoutArm:
    name: str          # e.g. "E6R_s42"
    arm: str           # A2 arm key in the config
    seed: int
    variant: str


def holdout_arms(config: dict[str, Any]) -> list[HoldoutArm]:
    arms = config["training"]["arms"]
    return [HoldoutArm(f"{arm}_s{int(seed)}", arm, int(seed), arms[arm]["data_variant"])
            for arm, seed in config["training"]["run_order"]]


# --------------------------------------------------------------------------- inputs
def holdout_inputs(workspace: Path, config: dict[str, Any], parent_manifest: Path,
                   parent_processed_root: Path) -> dict[str, Any]:
    """Entries per variant with the local input path and the manifest hash it must have."""
    parent = load_parent_manifest(parent_manifest, config)
    a2_path = workspace / SHHS_MANIFEST
    a2 = read_json(a2_path)
    if a2.get("status") != "complete" or a2.get("parent_preprocess_manifest_sha256") != sha256_file(parent_manifest):
        raise ValueError("A2 holdout preprocess manifest is incomplete or belongs to another parent")
    entries: dict[str, list[dict[str, Any]]] = {}
    for variant, rows, root in ((SHHS_REFERENCE_VARIANT, parent_entries(parent, SHHS_REFERENCE_VARIANT).values(),
                                 parent_processed_root),
                                (VARIANT, [r for r in a2["records"] if r["variant"] == VARIANT],
                                 workspace / SHHS_PROCESSED)):
        entries[variant] = sorted(({"record_key": r["record_key"], "subject_id": str(r["subject_id"]),
                                    "sha256": r["output_sha256"], "path": root / variant / f"{r['record_key']}.npz"}
                                   for r in rows), key=lambda e: e["record_key"])
        if len(entries[variant]) != len(parent["analysis_cohort"]):
            raise ValueError(f"{variant}: holdout record count differs from the cohort")
    return {"entries": entries, "parent_manifest_sha256": sha256_file(parent_manifest),
            "a2_manifest_sha256": sha256_file(a2_path)}


def load_input(entry: dict[str, Any], variant: str) -> dict[str, Any]:
    path = Path(entry["path"])
    if sha256_file(path) != entry["sha256"]:
        raise ValueError(f"{path.name}: input hash differs from its preprocess manifest")
    with np.load(path, allow_pickle=False) as z:
        checks = {"record_key": str(z["record_key"].item()) == entry["record_key"] == path.stem,
                  "subject_id": str(z["subject_id"].item()) == entry["subject_id"],
                  "role": str(z["role"].item()) == ROLE,
                  "variant": str(z["preprocess_version"].item()) == variant}
        x, index = z["x"].copy(), z["original_epoch_index"].copy()
    if not all(checks.values()) or x.dtype != np.float32 or x.ndim != 2 or x.shape[1] != 3000 \
            or not np.isfinite(x).all():
        raise ValueError(f"{path.name}: invalid holdout input {checks}")
    return {"record_key": entry["record_key"], "subject_id": entry["subject_id"], "input_sha256": entry["sha256"],
            "x": x, "original_epoch_index": index}


# --------------------------------------------------------------------------- models
def make_predictor(context: Any, batch_size: int) -> tuple[Callable[[np.ndarray], np.ndarray], dict[str, str]]:
    """ResNet-1D features in batches, then the TCN over the whole record (parent inference path)."""
    import torch

    from .test_gate import _load_extractor, _load_sequence_model

    extractor_kind, extractor, extractor_hash = _load_extractor(context)
    sequence_kind, sequence, sequence_path = _load_sequence_model(context)
    if extractor_kind != "resnet1d" or sequence_kind != "tcn":
        raise ValueError("A2 holdout inference supports ResNet-1D + TCN arms only")
    device = context.device
    extractor, sequence = extractor.to(device).eval(), sequence.to(device).eval()

    def predict(x: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            parts = [extractor.extract_features(torch.from_numpy(x[s:s + batch_size]).unsqueeze(1).to(device)).cpu()
                     for s in range(0, len(x), batch_size)]
            features = torch.cat(parts).to(torch.float32)
            logits = sequence(features.unsqueeze(0).to(device), padding_mask=None)
            p = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy().astype(np.float32)
        if p.shape != (len(x), 5) or not np.isfinite(p).all() or not np.allclose(p.sum(1), 1.0, rtol=1e-5, atol=1e-6):
            raise ValueError("invalid probability matrix")
        return p

    return predict, {"extractor_sha256": extractor_hash, "sequence_checkpoint_sha256": sha256_file(sequence_path)}


def _runtime(device: str) -> dict[str, Any]:
    import torch

    info = {"device": device, "torch_version": torch.__version__}
    if device.startswith("cuda"):
        info["gpu"] = torch.cuda.get_device_name(0)
    return info


def check_path(workspace: Path, device: str) -> Path:
    return workspace / HOLDOUT_ROOT / f"device_check_{device.replace(':', '_')}.json"


def device_check(workspace: Path, inputs: dict[str, Any], restore_workspace: Path, parent_predictions: Path,
                 device: str, batch_size: int) -> dict[str, Any]:
    """Reproduce stored parent CPU fold probabilities of E4_s42 with the A2 code on ``device``."""
    from .experiment import build_context

    rows = []
    for fold in CHECK_FOLDS:
        context = build_context(restore_workspace.resolve(), "E4", fold, 42, device, smoke=False,
                                allow_test_evaluation=True, num_workers=0, resume=True)
        predict, hashes = make_predictor(context, batch_size)
        for entry in inputs["entries"][SHHS_REFERENCE_VARIANT][:CHECK_RECORDS]:
            record = load_input(entry, SHHS_REFERENCE_VARIANT)
            stored_path = parent_predictions / CHECK_ARM / f"fold_{fold:02d}" / f"{entry['record_key']}.npz"
            with np.load(stored_path, allow_pickle=False) as z:
                stored = z["probabilities"].astype(np.float64)
                np.testing.assert_array_equal(z["original_epoch_index"], record["original_epoch_index"])
            new = predict(record["x"]).astype(np.float64)
            rows.append({"fold": fold, "record_key": entry["record_key"], **hashes,
                         "stored_sha256": sha256_file(stored_path),
                         "max_abs_difference": float(np.abs(new - stored).max()),
                         "argmax_agreement": float((new.argmax(1) == stored.argmax(1)).mean())})
    report = {"schema_version": 1, "amendment": "A2", "check": "reproduce parent CPU fold probabilities of E4_s42",
              "tolerance": CHECK_TOLERANCE, **_runtime(device), "rows": rows,
              "passed": all(r["max_abs_difference"] <= CHECK_TOLERANCE and r["argmax_agreement"] == 1.0
                            for r in rows)}
    path = check_path(workspace, device)
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(path, report, ensure_ascii=False, sort_keys=False)
    return report


# --------------------------------------------------------------------------- inference
def fold_path(workspace: Path, arm: str, fold: int, key: str) -> Path:
    return workspace / HOLDOUT_ROOT / "fold_predictions" / arm / f"fold_{fold:02d}" / f"{key}.npz"


def ensemble_path(workspace: Path, arm: str, key: str) -> Path:
    return workspace / HOLDOUT_ROOT / "ensemble" / arm / f"{key}.npz"


def _require_ready(workspace: Path, device: str) -> dict[str, Any]:
    journal_path = test_journal_path(workspace)
    if not journal_path.is_file() or read_json(journal_path).get("status") != "complete":
        raise ValueError("Holdout inference starts only after the A2 Sleep-EDF test campaign is complete")
    path = check_path(workspace, device)
    if not path.is_file() or not read_json(path).get("passed"):
        raise ValueError(f"Run the device check for {device} first; no inference before it passes")
    check = read_json(path)
    runtime = _runtime(device)
    if any(check.get(k) != v for k, v in runtime.items()):
        raise ValueError("Device check was run in another runtime (torch version or GPU differs)")
    return check


def units(config: dict[str, Any]) -> list[tuple[HoldoutArm, int]]:
    return [(arm, fold) for arm in holdout_arms(config) for fold in FOLDS]


def run_unit(workspace: Path, config: dict[str, Any], config_sha: str, inputs: dict[str, Any], arm: HoldoutArm,
             fold: int, *, device: str, batch_size: int) -> int:
    entries = inputs["entries"][arm.variant]
    pending = [e for e in entries if not fold_path(workspace, arm.name, fold, e["record_key"]).exists()]
    if not pending:
        return 0
    context = build_a2_context(workspace, config, A2Target(arm.arm, arm.seed, fold), device, num_workers=0,
                               allow_test_evaluation=False, resume=True)
    predict, model_meta = make_predictor(context, batch_size)
    base = {"schema_version": 1, "amendment": "A2", "a2_config_sha256": config_sha,
            "parent_preprocess_manifest_sha256": inputs["parent_manifest_sha256"],
            "a2_preprocess_manifest_sha256": inputs["a2_manifest_sha256"], "arm": arm.name,
            "data_variant": arm.variant, "seed": arm.seed, "role": ROLE, **_runtime(device), **model_meta,
            "artifact_type": "holdout_fold_prediction", "outer_fold": fold}
    started = time.perf_counter()
    for index, entry in enumerate(pending, start=1):
        record = load_input(entry, arm.variant)
        probabilities = predict(record["x"])
        metadata = {**base, "record_key": record["record_key"], "subject_id": record["subject_id"],
                    "input_npz_sha256": record["input_sha256"]}
        atomic_savez(fold_path(workspace, arm.name, fold, record["record_key"]), {
            "metadata_json": np.array(json.dumps(metadata, sort_keys=True)),
            "probabilities": probabilities, "original_epoch_index": record["original_epoch_index"]})
        if index % 100 == 0 or index == len(pending):
            rate = (time.perf_counter() - started) / index
            print(f"[{arm.name}] fold {fold:02d} {index}/{len(pending)} records ({rate:.2f}s/record)", flush=True)
    return len(pending)


def infer(workspace: Path, inputs: dict[str, Any], *, device: str, batch_size: int, worker_index: int = 0,
          worker_count: int = 1) -> int:
    workspace = workspace.resolve()
    config, config_sha = load_a2_config(workspace)
    _require_ready(workspace, device)
    if worker_count <= 0 or not 0 <= worker_index < worker_count:
        raise ValueError("invalid worker index/count")
    done = 0
    for arm, fold in units(config)[worker_index::worker_count]:
        done += run_unit(workspace, config, config_sha, inputs, arm, fold, device=device, batch_size=batch_size)
    return done


def finalize(workspace: Path, inputs: dict[str, Any]) -> dict[str, Any]:
    """Verify all fold artifacts, write ensembles, arm manifests and the run manifest."""
    from .evaluation.shhs_zero_shot import ensemble_probabilities

    workspace = workspace.resolve()
    config, config_sha = load_a2_config(workspace)
    arm_hashes = {}
    for arm in holdout_arms(config):
        fold_records, ensemble_records = [], []
        for entry in inputs["entries"][arm.variant]:
            parts, reference_index, devices = [], None, set()
            for fold in FOLDS:
                path = fold_path(workspace, arm.name, fold, entry["record_key"])
                with np.load(path, allow_pickle=False) as z:
                    meta = json.loads(str(z["metadata_json"].item()))
                    if (meta["input_npz_sha256"] != entry["sha256"] or meta["arm"] != arm.name
                            or meta["outer_fold"] != fold or meta["a2_config_sha256"] != config_sha):
                        raise ValueError(f"{path}: provenance differs")
                    index = z["original_epoch_index"].copy()
                    reference_index = index if reference_index is None else reference_index
                    np.testing.assert_array_equal(index, reference_index)
                    parts.append(z["probabilities"].copy())
                devices.add(meta["device"])
                fold_records.append({"fold": fold, "record_key": entry["record_key"], "sha256": sha256_file(path)})
            mean = ensemble_probabilities(parts)
            metadata = {"schema_version": 1, "amendment": "A2", "a2_config_sha256": config_sha, "arm": arm.name,
                        "data_variant": arm.variant, "seed": arm.seed, "role": ROLE,
                        "artifact_type": "holdout_ensemble_prediction", "folds": list(FOLDS),
                        "aggregation": "arithmetic_mean_probability_float64_accumulator",
                        "fold_devices": sorted(devices), "record_key": entry["record_key"],
                        "subject_id": entry["subject_id"], "input_npz_sha256": entry["sha256"]}
            path = ensemble_path(workspace, arm.name, entry["record_key"])
            atomic_savez(path, {"metadata_json": np.array(json.dumps(metadata, sort_keys=True)),
                                "probabilities": mean, "prediction": np.argmax(mean, axis=1).astype(np.int8),
                                "original_epoch_index": reference_index})
            ensemble_records.append({"record_key": entry["record_key"], "subject_id": entry["subject_id"],
                                     "sha256": sha256_file(path)})
        manifest_path = workspace / HOLDOUT_ROOT / "arm_manifests" / f"{arm.name}.json"
        atomic_write_json(manifest_path, {"arm": arm.name, "status": "complete", "data_variant": arm.variant,
                                          "records": len(ensemble_records), "fold_records": fold_records,
                                          "ensemble_records": ensemble_records}, ensure_ascii=False, sort_keys=False)
        arm_hashes[arm.name] = sha256_file(manifest_path)
    manifest = {"schema_version": 1, "amendment": "A2", "status": "complete_no_metrics_computed",
                "a2_config_sha256": config_sha, "parent_preprocess_manifest_sha256": inputs["parent_manifest_sha256"],
                "a2_preprocess_manifest_sha256": inputs["a2_manifest_sha256"], "arm_manifest_sha256": arm_hashes}
    atomic_write_json(workspace / HOLDOUT_ROOT / "run_manifest.json", manifest, ensure_ascii=False, sort_keys=False)
    return manifest
