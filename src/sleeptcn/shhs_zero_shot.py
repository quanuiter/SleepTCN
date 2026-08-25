"""CPU zero-shot inference for locked Sleep-EDF checkpoints on SHHS1."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .artifacts import sha256_file
from .experiment import build_context
from .features import extract_15cnn_features
from .io.serialization import atomic_savez, atomic_write_json
from .evaluation.shhs_zero_shot import (
    confusion_matrix,
    ensemble_probabilities,
    load_prediction_artifact,
    metrics_from_confusion,
)
from .test_gate import _load_extractor, _load_sequence_model
from .workflows.shhs_protocol import (
    EXPERIMENT_VARIANTS,
    FOLDS,
    TEST_CONFIRMATION,
    input_entries,
    load_inventory,
    load_preprocess_manifest,
    load_protocol as load_locked_protocol,
)


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
    atomic_write_json(path, document, ensure_ascii=False, sort_keys=False)


def atomic_npz(path: Path, **arrays: Any) -> None:
    atomic_savez(path, arrays)


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
