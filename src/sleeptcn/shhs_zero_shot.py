"""CPU zero-shot inference for locked Sleep-EDF checkpoints on SHHS1."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

from .artifacts import sha256_file
from .experiment import build_context
from .features import extract_15cnn_features
from .test_gate import _load_extractor, _load_sequence_model


EXPERIMENT_VARIANTS = {
    "E0": "paper_raw_v1",
    "E3": "filtered_v2",
    "E6": "filtered_zscore_v2",
}
FOLDS = tuple(range(10))
TEST_CONFIRMATION = "OPEN-SHHS-ZERO-SHOT-TEST-ONCE"


@dataclass(frozen=True)
class ExternalRecord:
    path: Path
    input_sha256: str
    record_key: str
    subject_id: str
    role: str
    variant: str
    x: np.ndarray
    y: np.ndarray
    valid_mask: np.ndarray
    original_epoch_index: np.ndarray


def atomic_json(path: Path, document: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.",
        suffix=".tmp", delete=False
    ) as handle:
        temporary = Path(handle.name)
        json.dump(document, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_npz(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="wb", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
    ) as handle:
        temporary = Path(handle.name)
        np.savez_compressed(handle, **arrays)
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def load_locked_protocol(
    path: Path,
    *,
    experiment_variants: dict[str, str] | None = None,
    expected_status: str = "locked_before_validation_inference",
) -> tuple[dict[str, Any], str]:
    selected_variants = EXPERIMENT_VARIANTS if experiment_variants is None else experiment_variants
    raw = path.read_bytes()
    protocol = json.loads(raw.decode("utf-8"))
    if protocol.get("status") != expected_status:
        raise ValueError("SHHS zero-shot protocol is not locked")
    if tuple(protocol.get("experiments", {})) != tuple(selected_variants):
        raise ValueError("SHHS zero-shot experiment order differs")
    observed_variants = {
        experiment: details.get("data_variant")
        for experiment, details in protocol["experiments"].items()
    }
    if observed_variants != selected_variants:
        raise ValueError("SHHS zero-shot data variants differ")
    policy = protocol.get("checkpoint_policy", {})
    ensemble = protocol.get("ensemble", {})
    checks = {
        "folds": policy.get("outer_folds") == list(FOLDS),
        "all_folds": policy.get("use_all_folds") is True,
        "no_ranking": policy.get("rank_or_select_fold_by_validation_metric") is False,
        "best_only": policy.get("checkpoint_filename") == "best.pt",
        "probability_mean": ensemble.get("aggregation") == "arithmetic_mean_probability",
        "fold_order": ensemble.get("fold_order") == list(FOLDS),
    }
    failed = sorted(key for key, value in checks.items() if not value)
    if failed:
        raise ValueError(f"Invalid locked zero-shot policy: {failed}")
    return protocol, hashlib.sha256(raw).hexdigest()


def load_inventory(
    path: Path,
    protocol_sha256: str,
    *,
    expected_best_checkpoints: int = 200,
) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    inventory = json.loads(raw.decode("utf-8"))
    if inventory.get("status") != "passed":
        raise ValueError("Checkpoint inventory has not passed")
    if inventory.get("protocol_sha256") != protocol_sha256:
        raise ValueError("Checkpoint inventory points to another protocol")
    if inventory.get("summary", {}).get("best_checkpoints") != expected_best_checkpoints:
        raise ValueError(
            "Checkpoint inventory does not contain "
            f"{expected_best_checkpoints} best checkpoint references"
        )
    return inventory, hashlib.sha256(raw).hexdigest()


def load_preprocess_manifest(
    path: Path, protocol: dict[str, Any]
) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    expected = protocol["preprocessing_provenance"]["manifest_sha256"]
    if digest != expected:
        raise ValueError("Preprocess manifest SHA-256 differs from zero-shot protocol")
    manifest = json.loads(raw.decode("utf-8"))
    if manifest.get("status") != "complete" or manifest.get("scope") != "primary":
        raise ValueError("Primary preprocessing manifest is not complete")
    return manifest, digest


def input_entries(
    preprocess_manifest: dict[str, Any], role: str, variant: str
) -> list[dict[str, Any]]:
    if role not in {"validation", "test"}:
        raise ValueError("Zero-shot inference role must be validation or test")
    expected = 15 if role == "validation" else 180
    entries = sorted(
        (
            entry for entry in preprocess_manifest["records"]
            if entry["role"] == role and entry["variant"] == variant
        ),
        key=lambda entry: entry["record_key"],
    )
    if len(entries) != expected:
        raise ValueError(f"Expected {expected} {role}/{variant} records, found {len(entries)}")
    keys = [entry["record_key"] for entry in entries]
    if len(keys) != len(set(keys)):
        raise ValueError(f"Duplicate record in {role}/{variant}")
    return entries


def load_external_record(
    entry: dict[str, Any], expected_role: str, expected_variant: str,
    processed_root: Path | None = None,
) -> ExternalRecord:
    path = Path(entry["output_path"]).resolve()
    if processed_root is not None and not path.is_relative_to(processed_root.resolve()):
        raise ValueError(f"{path.name}: input escaped the locked processed root")
    if sha256_file(path) != entry["output_sha256"]:
        raise ValueError(f"{path.name}: input NPZ SHA-256 differs from preprocess manifest")
    with np.load(path, allow_pickle=False) as npz:
        record_key = str(npz["record_key"].item())
        subject_id = str(npz["subject_id"].item())
        role = str(npz["role"].item())
        variant = str(npz["preprocess_version"].item())
        x = npz["x"].copy()
        y = npz["y"].copy()
        valid = npz["valid_mask"].copy()
        indices = npz["original_epoch_index"].copy()
    checks = {
        "record_key": record_key == entry["record_key"] == path.stem,
        "subject_id": subject_id == entry["subject_id"],
        "role": role == expected_role,
        "variant": variant == expected_variant,
        "x": x.dtype == np.float32 and x.ndim == 2 and x.shape[1] == 3000,
        "y": y.dtype == np.int8 and y.shape == (len(x),),
        "valid": valid.dtype == np.bool_ and np.array_equal(valid, y >= 0),
        "indices": indices.dtype == np.int32 and indices.shape == y.shape,
        "finite": np.isfinite(x).all(),
    }
    failed = sorted(key for key, value in checks.items() if not value)
    if failed:
        raise ValueError(f"{path.name}: invalid zero-shot input {failed}")
    return ExternalRecord(
        path=path,
        input_sha256=entry["output_sha256"],
        record_key=record_key,
        subject_id=subject_id,
        role=role,
        variant=variant,
        x=x,
        y=y,
        valid_mask=valid,
        original_epoch_index=indices,
    )


def inventory_fold(inventory: dict[str, Any], experiment: str, fold: int) -> dict[str, Any]:
    matches = [
        item for item in inventory["folds"]
        if item["experiment"] == experiment and int(item["outer_fold"]) == fold
    ]
    if len(matches) != 1:
        raise ValueError(f"Inventory must contain one {experiment}/fold_{fold:02d}")
    return matches[0]


def load_fold_models(
    workspace: Path,
    experiment: str,
    fold: int,
    inventory: dict[str, Any],
    checkpoint_seed: int = 42,
) -> tuple[str, Any, str, torch.nn.Module, str, str]:
    context = build_context(
        workspace, experiment, fold, checkpoint_seed, "cpu", smoke=False,
        allow_test_evaluation=True, num_workers=0, resume=True
    )
    extractor_kind, extractor, extractor_hash = _load_extractor(context)
    sequence_kind, sequence, sequence_path = _load_sequence_model(context)
    sequence_hash = sha256_file(sequence_path)
    expected = inventory_fold(inventory, experiment, fold)
    if extractor_hash != expected["extractor_sha256"]:
        raise ValueError(f"{experiment}/fold_{fold:02d}: extractor hash changed")
    if sequence_hash != expected["sequence_checkpoint_sha256"]:
        raise ValueError(f"{experiment}/fold_{fold:02d}: sequence hash changed")
    return extractor_kind, extractor, extractor_hash, sequence, sequence_kind, sequence_hash


@torch.no_grad()
def infer_probabilities(
    record: ExternalRecord,
    extractor_kind: str,
    extractor: Any,
    sequence: torch.nn.Module,
    sequence_kind: str,
    batch_size: int,
) -> np.ndarray:
    if extractor_kind == "cnn15":
        features = extract_15cnn_features(
            record.x, extractor, device="cpu", batch_size=batch_size
        )
    elif extractor_kind == "resnet1d":
        extractor = extractor.to("cpu").eval()
        parts = []
        for start in range(0, len(record.x), batch_size):
            signals = torch.from_numpy(record.x[start : start + batch_size]).unsqueeze(1)
            parts.append(extractor.extract_features(signals).cpu().numpy())
        features = np.concatenate(parts).astype(np.float32, copy=False)
    else:
        raise ValueError(f"Unknown extractor kind: {extractor_kind}")
    tensor = torch.from_numpy(features).unsqueeze(0)
    sequence = sequence.to("cpu").eval()
    if sequence_kind == "bilstm":
        logits = sequence(tensor, torch.tensor([len(record.y)], dtype=torch.long))
    elif sequence_kind == "tcn":
        logits = sequence(tensor, padding_mask=None)
    else:
        raise ValueError(f"Unknown sequence kind: {sequence_kind}")
    probabilities = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy().astype(np.float32)
    if probabilities.shape != (len(record.y), 5) or not np.isfinite(probabilities).all():
        raise ValueError(f"{record.record_key}: invalid probability matrix")
    if not np.allclose(probabilities.sum(axis=1), 1.0, rtol=1e-5, atol=1e-6):
        raise ValueError(f"{record.record_key}: probabilities do not sum to one")
    return probabilities


def fold_artifact_path(output_root: Path, role: str, experiment: str, fold: int, key: str) -> Path:
    return output_root / role / "fold_predictions" / experiment / f"fold_{fold:02d}" / f"{key}.npz"


def ensemble_artifact_path(output_root: Path, role: str, experiment: str, key: str) -> Path:
    return output_root / role / "ensemble" / experiment / f"{key}.npz"


def save_fold_artifact(
    path: Path,
    record: ExternalRecord,
    probabilities: np.ndarray,
    metadata: dict[str, Any],
) -> str:
    atomic_npz(
        path,
        metadata_json=np.array(json.dumps(metadata, sort_keys=True)),
        probabilities=probabilities,
        y=record.y,
        valid_mask=record.valid_mask,
        original_epoch_index=record.original_epoch_index,
    )
    return sha256_file(path)


def load_prediction_artifact(
    path: Path, expected: dict[str, Any]
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as npz:
        metadata = json.loads(str(npz["metadata_json"].item()))
        mismatches = {
            key: (metadata.get(key), value)
            for key, value in expected.items()
            if metadata.get(key) != value
        }
        if mismatches:
            raise ValueError(f"{path}: metadata mismatch {mismatches}")
        probabilities = npz["probabilities"].copy()
        y = npz["y"].copy()
        valid = npz["valid_mask"].copy()
        indices = npz["original_epoch_index"].copy()
    if probabilities.shape != (len(y), 5) or not np.isfinite(probabilities).all():
        raise ValueError(f"{path}: invalid probabilities")
    if (
        y.dtype != np.int8
        or valid.dtype != np.bool_
        or indices.dtype != np.int32
        or valid.shape != y.shape
        or indices.shape != y.shape
        or not np.array_equal(valid, y >= 0)
    ):
        raise ValueError(f"{path}: invalid valid mask")
    return metadata, probabilities, y, valid, indices


def ensemble_probabilities(parts: Iterable[np.ndarray]) -> np.ndarray:
    matrices = list(parts)
    if len(matrices) != 10:
        raise ValueError("Locked ensemble requires exactly ten folds")
    shape = matrices[0].shape
    if any(matrix.shape != shape for matrix in matrices):
        raise ValueError("Fold probability shapes differ")
    accumulator = np.zeros(shape, dtype=np.float64)
    for matrix in matrices:
        accumulator += matrix.astype(np.float64, copy=False)
    mean = (accumulator / 10.0).astype(np.float32)
    if not np.allclose(mean.sum(axis=1), 1.0, rtol=1e-5, atol=1e-6):
        raise ValueError("Ensemble probabilities do not sum to one")
    return mean


def confusion_matrix(y: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    matrix = np.zeros((5, 5), dtype=np.int64)
    for truth, predicted in zip(y, prediction, strict=True):
        matrix[int(truth), int(predicted)] += 1
    return matrix


def metrics_from_confusion(matrix: np.ndarray) -> dict[str, Any]:
    total = int(matrix.sum())
    per_class_f1 = []
    per_class_recall = []
    for index in range(5):
        tp = int(matrix[index, index])
        fp = int(matrix[:, index].sum() - tp)
        fn = int(matrix[index, :].sum() - tp)
        f1 = 0.0 if 2 * tp + fp + fn == 0 else 2 * tp / (2 * tp + fp + fn)
        recall = 0.0 if tp + fn == 0 else tp / (tp + fn)
        per_class_f1.append(float(f1))
        per_class_recall.append(float(recall))
    accuracy = float(np.trace(matrix) / total) if total else 0.0
    expected = float(
        np.dot(matrix.sum(axis=1), matrix.sum(axis=0)) / (total * total)
    ) if total else 0.0
    kappa = 0.0 if math.isclose(expected, 1.0) else (accuracy - expected) / (1.0 - expected)
    return {
        "macro_f1": float(np.mean(per_class_f1)),
        "accuracy": accuracy,
        "cohen_kappa": float(kappa),
        "per_class_f1": per_class_f1,
        "per_class_recall": per_class_recall,
        "confusion_matrix": matrix.tolist(),
        "valid_epochs": total,
    }


def run_role(
    *,
    workspace: Path,
    processed_root: Path,
    preprocess_manifest_path: Path,
    protocol_path: Path,
    inventory_path: Path,
    output_root: Path,
    role: str,
    threads: int,
    batch_size: int,
    resume: bool,
    validation_gate_path: Path | None,
    confirmation: str | None,
    experiment_variants: dict[str, str] | None = None,
    expected_protocol_status: str = "locked_before_validation_inference",
    expected_inventory_checkpoints: int = 200,
    test_confirmation: str = TEST_CONFIRMATION,
    checkpoint_seed: int = 42,
) -> dict[str, Any]:
    if threads <= 0 or batch_size <= 0:
        raise ValueError("threads and batch_size must be positive")
    selected_variants = (
        EXPERIMENT_VARIANTS if experiment_variants is None else experiment_variants
    )
    protocol, protocol_sha256 = load_locked_protocol(
        protocol_path,
        experiment_variants=selected_variants,
        expected_status=expected_protocol_status,
    )
    inventory, inventory_sha256 = load_inventory(
        inventory_path,
        protocol_sha256,
        expected_best_checkpoints=expected_inventory_checkpoints,
    )
    preprocess_manifest, preprocess_manifest_sha256 = load_preprocess_manifest(
        preprocess_manifest_path, protocol
    )
    if role == "test":
        if confirmation != test_confirmation:
            raise ValueError(f"test confirmation must equal {test_confirmation!r}")
        if validation_gate_path is None or not validation_gate_path.is_file():
            raise FileNotFoundError("Passed validation gate report is required for test")
        gate = json.loads(validation_gate_path.read_text(encoding="utf-8"))
        if gate.get("status") != "passed" or gate.get("role") != "validation":
            raise ValueError("Validation gate has not passed")
        if gate.get("protocol_sha256") != protocol_sha256:
            raise ValueError("Validation gate points to another protocol")
        if gate.get("checkpoint_inventory_sha256") != inventory_sha256:
            raise ValueError("Validation gate points to another checkpoint inventory")
    elif role != "validation":
        raise ValueError("role must be validation or test")

    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)
    role_root = output_root / role
    final_manifest_path = role_root / "run_manifest.json"
    if final_manifest_path.exists() and not resume:
        raise FileExistsError(final_manifest_path)
    fold_records = []
    ensemble_records = []
    started = time.perf_counter()
    for experiment, variant in selected_variants.items():
        entries = input_entries(preprocess_manifest, role, variant)
        loaded_records = [
            load_external_record(entry, role, variant, processed_root)
            for entry in entries
        ]
        for fold in FOLDS:
            fold_info = inventory_fold(inventory, experiment, fold)
            extractor_kind, extractor, extractor_hash, sequence, sequence_kind, sequence_hash = load_fold_models(
                workspace, experiment, fold, inventory, checkpoint_seed
            )
            for index, record in enumerate(loaded_records, start=1):
                path = fold_artifact_path(output_root, role, experiment, fold, record.record_key)
                metadata = {
                    "schema_version": 1,
                    "artifact_type": "shhs_zero_shot_fold_prediction",
                    "protocol_sha256": protocol_sha256,
                    "checkpoint_inventory_sha256": inventory_sha256,
                    "preprocess_manifest_sha256": preprocess_manifest_sha256,
                    "input_npz_sha256": record.input_sha256,
                    "experiment": experiment,
                    "data_variant": variant,
                    "outer_fold": fold,
                    "seed": checkpoint_seed,
                    "role": role,
                    "record_key": record.record_key,
                    "subject_id": record.subject_id,
                    "extractor_sha256": extractor_hash,
                    "sequence_checkpoint_sha256": sequence_hash,
                    "fold_checkpoint_set_sha256": fold_info["fold_checkpoint_set_sha256"],
                    "device": "cpu",
                    "torch_version": torch.__version__,
                    "cpu_threads": threads,
                    "gradient_enabled": False,
                }
                if path.exists():
                    if not resume:
                        raise FileExistsError(path)
                    load_prediction_artifact(path, metadata)
                else:
                    probabilities = infer_probabilities(
                        record, extractor_kind, extractor, sequence, sequence_kind, batch_size
                    )
                    save_fold_artifact(path, record, probabilities, metadata)
                fold_records.append(
                    {
                        "experiment": experiment,
                        "outer_fold": fold,
                        "record_key": record.record_key,
                        "path": str(path.resolve()),
                        "sha256": sha256_file(path),
                    }
                )
                print(
                    f"[{role}] {experiment} fold={fold:02d} record={index:03d}/{len(loaded_records):03d}",
                    flush=True,
                )

        for index, record in enumerate(loaded_records, start=1):
            parts = []
            reference = None
            for fold in FOLDS:
                path = fold_artifact_path(output_root, role, experiment, fold, record.record_key)
                _, probabilities, y, valid, indices = load_prediction_artifact(
                    path,
                    {
                        "protocol_sha256": protocol_sha256,
                        "checkpoint_inventory_sha256": inventory_sha256,
                        "experiment": experiment,
                        "outer_fold": fold,
                        "role": role,
                        "record_key": record.record_key,
                    },
                )
                if reference is None:
                    reference = (y, valid, indices)
                elif not all(
                    np.array_equal(left, right)
                    for left, right in zip(reference, (y, valid, indices), strict=True)
                ):
                    raise ValueError(f"{experiment}/{record.record_key}: folds are not aligned")
                parts.append(probabilities)
            mean_probability = ensemble_probabilities(parts)
            prediction = np.argmax(mean_probability, axis=1).astype(np.int8)
            path = ensemble_artifact_path(output_root, role, experiment, record.record_key)
            metadata = {
                "schema_version": 1,
                "artifact_type": "shhs_zero_shot_ensemble_prediction",
                "protocol_sha256": protocol_sha256,
                "checkpoint_inventory_sha256": inventory_sha256,
                "preprocess_manifest_sha256": preprocess_manifest_sha256,
                "input_npz_sha256": record.input_sha256,
                "experiment": experiment,
                "data_variant": variant,
                "folds": list(FOLDS),
                "aggregation": "arithmetic_mean_probability_float64_accumulator",
                "role": role,
                "record_key": record.record_key,
                "subject_id": record.subject_id,
            }
            atomic_npz(
                path,
                metadata_json=np.array(json.dumps(metadata, sort_keys=True)),
                probabilities=mean_probability,
                prediction=prediction,
                y=record.y,
                valid_mask=record.valid_mask,
                original_epoch_index=record.original_epoch_index,
            )
            valid = record.valid_mask
            subject_metrics = metrics_from_confusion(
                confusion_matrix(record.y[valid], prediction[valid])
            )
            ensemble_records.append(
                {
                    "experiment": experiment,
                    "record_key": record.record_key,
                    "subject_id": record.subject_id,
                    "role": role,
                    "path": str(path.resolve()),
                    "sha256": sha256_file(path),
                    "metrics": subject_metrics,
                }
            )
            print(
                f"[{role}] {experiment} ensemble={index:03d}/{len(loaded_records):03d}",
                flush=True,
            )

    metrics = {}
    for experiment in selected_variants:
        subject_entries = [item for item in ensemble_records if item["experiment"] == experiment]
        pooled = np.zeros((5, 5), dtype=np.int64)
        for item in subject_entries:
            pooled += np.asarray(item["metrics"]["confusion_matrix"], dtype=np.int64)
        pooled_metrics = metrics_from_confusion(pooled)
        metrics[experiment] = {
            "subject_macro_f1_mean": float(
                np.mean([item["metrics"]["macro_f1"] for item in subject_entries])
            ),
            "subject_macro_f1": {
                item["subject_id"]: item["metrics"]["macro_f1"] for item in subject_entries
            },
            "pooled": pooled_metrics,
        }
    report = {
        "schema_version": 1,
        "status": "complete",
        "role": role,
        "protocol_sha256": protocol_sha256,
        "checkpoint_inventory_sha256": inventory_sha256,
        "preprocess_manifest_sha256": preprocess_manifest_sha256,
        "execution": {
            "device": "cpu",
            "torch_version": torch.__version__,
            "cpu_threads": threads,
            "batch_size": batch_size,
            "deterministic_algorithms": True,
            "elapsed_seconds": time.perf_counter() - started,
        },
        "summary": {
            "experiments": len(selected_variants),
            "folds_per_experiment": 10,
            "subjects": 15 if role == "validation" else 180,
            "fold_prediction_artifacts": len(fold_records),
            "ensemble_artifacts": len(ensemble_records),
        },
        "metrics": metrics,
        "fold_records": fold_records,
        "ensemble_records": ensemble_records,
    }
    atomic_json(final_manifest_path, report)
    final_hash = sha256_file(final_manifest_path)
    final_manifest_path.with_suffix(final_manifest_path.suffix + ".sha256").write_text(
        f"{final_hash}  {final_manifest_path.name}\n", encoding="ascii"
    )
    return report
