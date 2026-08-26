"""Validation-only ResNet-1D tuning runner.

This module deliberately has no test-data API.  It trains one ResNet candidate
per fold and seed, records validation artifacts, and writes checkpoints below
an explicitly supplied output root outside the source checkout.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from .dataset import SleepRecord
from .engine import (
    epoch_forward,
    fit_model,
    load_model_checkpoint,
    run_loader,
    seed_everything,
)
from .io.hashing import sha256_file
from .io.serialization import atomic_write_json, read_json
from .metrics import compute_metrics
from .models import EEGResNet1D
from .training import masked_cross_entropy
from .training_data import (
    RecordEpochDataset,
    class_counts_from_records,
    load_partition_records,
    resolve_fold_partitions,
)
from .workflows.provenance import resnet_tuning_code_sha256


TUNING_SCHEMA_VERSION = 1
TUNING_EXPERIMENT_ID = "RESNET_TUNING_V3"


def _git_state(workspace: Path) -> tuple[str | None, bool]:
    commit = subprocess.run(
        ["git", "-C", str(workspace), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    status = subprocess.run(
        ["git", "-C", str(workspace), "status", "--porcelain"],
        capture_output=True,
        text=True,
        check=False,
    )
    return (
        commit.stdout.strip() if commit.returncode == 0 else None,
        status.returncode != 0 or bool(status.stdout.strip()),
    )


def _loader(
    records: tuple[SleepRecord, ...],
    *,
    batch_size: int,
    shuffle: bool,
    generator: torch.Generator | None,
    device: torch.device,
    num_workers: int,
) -> DataLoader:
    return DataLoader(
        RecordEpochDataset(records),
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
    )


def _validate_search_document(document: dict[str, Any]) -> None:
    if document.get("schema_version") != TUNING_SCHEMA_VERSION:
        raise ValueError("unsupported ResNet tuning schema")
    if document.get("status") != "development_only":
        raise ValueError("tuning search document must remain development_only")
    if document.get("selection_role") != "validation_only":
        raise ValueError("tuning search must use validation_only selection")
    if document.get("test_policy") != "test_loader_is_not_constructed":
        raise ValueError("tuning search must prohibit test-loader construction")
    if document.get("selection_metric") != "validation_macro_f1":
        raise ValueError("tuning runner requires validation_macro_f1")
    if document.get("selection_policy") != "per_outer_fold_validation":
        raise ValueError(
            "tuning search must select candidates per outer fold; "
            "a pooled ranking is descriptive only"
        )
    n_folds = document.get("n_folds")
    if not isinstance(n_folds, int) or n_folds <= 1:
        raise ValueError("tuning search must define n_folds > 1")
    candidates = document.get("candidates")
    if not isinstance(candidates, dict) or not candidates:
        raise ValueError("tuning search must define candidates")


def _validate_candidate(candidate_id: str, candidate: dict[str, Any]) -> None:
    required = {
        "resnet1d",
        "learning_rate",
        "weight_decay",
        "batch_size_epochs",
        "max_epochs",
        "early_stopping_patience_epochs",
    }
    missing = sorted(required - set(candidate))
    if missing:
        raise ValueError(f"candidate {candidate_id} is missing {missing}")
    if float(candidate["learning_rate"]) <= 0:
        raise ValueError("learning_rate must be positive")
    if float(candidate["weight_decay"]) < 0:
        raise ValueError("weight_decay must be nonnegative")
    if int(candidate["batch_size_epochs"]) <= 0:
        raise ValueError("batch_size_epochs must be positive")
    if (
        int(candidate["max_epochs"]) <= 0
        or int(candidate["early_stopping_patience_epochs"]) <= 0
    ):
        raise ValueError("epoch and patience values must be positive")
    if candidate.get("optimizer", "adamw") != "adamw":
        raise ValueError("only AdamW is supported by the v3 tuning runner")


@torch.no_grad()
def _subject_validation_metrics(
    model: EEGResNet1D,
    records: tuple[SleepRecord, ...],
    *,
    batch_size: int,
    device: torch.device,
) -> dict[str, Any]:
    """Evaluate validation predictions with one Macro-F1 value per subject."""

    model.eval()
    subject_parts: dict[str, dict[str, list[np.ndarray]]] = {}
    for record in records:
        loader = _loader(
            (record,),
            batch_size=batch_size,
            shuffle=False,
            generator=None,
            device=device,
            num_workers=0,
        )
        true_parts: list[np.ndarray] = []
        predicted_parts: list[np.ndarray] = []
        for signals, targets in loader:
            logits = model(signals.to(device))
            valid = (targets >= 0) & (targets < 5)
            true_parts.append(targets[valid].numpy())
            predictions = logits.argmax(dim=-1).cpu().numpy()
            predicted_parts.append(predictions[valid.numpy()])
        if not true_parts:
            raise ValueError(
                f"validation record has no valid epochs: {record.info.record_key}"
            )
        subject = subject_parts.setdefault(
            record.info.subject_id, {"true": [], "predicted": []}
        )
        subject["true"].append(np.concatenate(true_parts))
        subject["predicted"].append(np.concatenate(predicted_parts))

    per_subject: dict[str, dict[str, Any]] = {}
    for subject, parts in subject_parts.items():
        metrics = compute_metrics(
            np.concatenate(parts["true"]), np.concatenate(parts["predicted"])
        )
        per_subject[subject] = metrics
    values = np.asarray(
        [float(metrics["macro_f1"]) for metrics in per_subject.values()],
        dtype=np.float64,
    )
    per_class_mean = {
        stage: float(
            np.mean(
                [
                    float(metrics["per_class"][stage]["f1"])
                    for metrics in per_subject.values()
                ]
            )
        )
        for stage in ("W", "N1", "N2", "N3", "REM")
    }
    return {
        "n_subjects": int(len(values)),
        "mean_macro_f1": float(values.mean()),
        "std_macro_f1": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
        "per_class_mean_f1": per_class_mean,
        "subjects": per_subject,
    }


def run_resnet_tuning(
    workspace: Path,
    search_config_path: Path,
    candidate_id: str,
    outer_fold: int,
    seed: int,
    device: str,
    output_root: Path,
    *,
    num_workers: int = 0,
    smoke: bool = False,
    resume: bool = False,
) -> dict[str, Any]:
    """Train one candidate using train/validation records only."""

    workspace = workspace.resolve()
    search_config_path = search_config_path.resolve()
    output_root = output_root.resolve()
    search = read_json(search_config_path)
    _validate_search_document(search)
    n_folds = int(search["n_folds"])
    if outer_fold not in range(n_folds) or seed < 0:
        raise ValueError("invalid outer_fold or seed")
    if num_workers < 0:
        raise ValueError("num_workers must be nonnegative")

    candidates = search["candidates"]
    if candidate_id not in candidates:
        raise ValueError(f"unknown candidate: {candidate_id}")
    candidate = candidates[candidate_id]
    if not isinstance(candidate, dict):
        raise ValueError(f"candidate {candidate_id} must be a mapping")
    _validate_candidate(candidate_id, candidate)

    split_path = workspace / search["split_path"]
    variant = str(search["data_variant"])
    processed_root = workspace / "data" / "processed"
    search_config_sha256 = sha256_file(search_config_path)
    split_sha256 = sha256_file(split_path)
    torch_device = torch.device(device)
    if torch_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    run_root = output_root / candidate_id / f"fold_{outer_fold:02d}" / f"seed_{seed}"
    checkpoint_dir = run_root / "checkpoints" / "resnet1d"
    resolved_config_path = run_root / "resolved_config.json"
    resolved_config = {
        "schema_version": TUNING_SCHEMA_VERSION,
        "campaign_id": search["campaign_id"],
        "n_folds": n_folds,
        "candidate_id": candidate_id,
        "candidate": candidate,
        "dataset": search["dataset"],
        "data_variant": variant,
        "split_path": str(search["split_path"]),
        "search_config_sha256": search_config_sha256,
        "split_sha256": split_sha256,
        "selection_role": "validation_only",
        "selection_policy": "per_outer_fold_validation",
        "test_policy": "test_loader_is_not_constructed",
    }
    atomic_write_json(resolved_config_path, resolved_config)
    config_sha256 = sha256_file(resolved_config_path)

    partitions = resolve_fold_partitions(
        processed_root, split_path, outer_fold, variant
    )
    # Intentionally load only train and validation.  The test partition is
    # resolved for split-integrity checks but its records are never opened.
    train_records = load_partition_records(partitions.train, variant)
    validation_records = load_partition_records(partitions.validation, variant)

    seed_everything(seed)
    generator = torch.Generator().manual_seed(seed)
    batch_size = int(candidate["batch_size_epochs"])
    train_loader = _loader(
        train_records,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        device=torch_device,
        num_workers=num_workers,
    )
    validation_loader = _loader(
        validation_records,
        batch_size=batch_size,
        shuffle=False,
        generator=None,
        device=torch_device,
        num_workers=num_workers,
    )

    counts = class_counts_from_records(train_records).astype(np.float64)
    weights = torch.from_numpy((counts.sum() / (5.0 * counts)).astype(np.float32))

    def loss_function(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return masked_cross_entropy(logits, targets, weights)

    model = EEGResNet1D.from_config(candidate["resnet1d"])
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(candidate["learning_rate"]),
        weight_decay=float(candidate["weight_decay"]),
    )
    latest_path = checkpoint_dir / "latest.pt"
    resume_from = latest_path if resume and latest_path.is_file() else None
    fit_result = fit_model(
        model,
        train_loader,
        validation_loader,
        optimizer,
        loss_function,
        epoch_forward,
        device=torch_device,
        max_epochs=1 if smoke else int(candidate["max_epochs"]),
        patience=int(candidate["early_stopping_patience_epochs"]),
        checkpoint_dir=checkpoint_dir,
        experiment_id=TUNING_EXPERIMENT_ID,
        stage=f"resnet1d/{candidate_id}",
        outer_fold=outer_fold,
        seed=seed,
        config_sha256=config_sha256,
        split_sha256=split_sha256,
        data_variant=variant,
        selection_metric="validation_macro_f1",
        loader_generator=generator,
        resume_from=resume_from,
        max_train_batches=1 if smoke else None,
        max_validation_batches=1 if smoke else None,
    )

    best_path = fit_result.best_checkpoint
    expected_metadata = {
        "experiment_id": TUNING_EXPERIMENT_ID,
        "stage": f"resnet1d/{candidate_id}",
        "outer_fold": outer_fold,
        "seed": seed,
        "config_sha256": config_sha256,
        "split_sha256": split_sha256,
        "data_variant": variant,
        "model_class": "EEGResNet1D",
        "selection_metric": "validation_macro_f1",
    }
    load_model_checkpoint(
        best_path,
        model,
        expected_metadata=expected_metadata,
        device=torch_device,
    )
    validation_summary = run_loader(
        model,
        validation_loader,
        torch_device,
        epoch_forward,
        loss_function,
    )
    subject_metrics = _subject_validation_metrics(
        model,
        validation_records,
        batch_size=batch_size,
        device=torch_device,
    )

    atomic_write_json(
        run_root / "training_history.json",
        {
            "schema_version": TUNING_SCHEMA_VERSION,
            "candidate_id": candidate_id,
            "best_epoch": fit_result.progress.best_epoch,
            "completed_epochs": fit_result.progress.completed_epochs,
            "stopped_early": fit_result.stopped_early,
            "history": list(fit_result.history),
        },
    )
    atomic_write_json(
        run_root / "validation_metrics.json",
        {
            "schema_version": TUNING_SCHEMA_VERSION,
            "role": "validation",
            "selection_metric": "validation_macro_f1",
            "loss": validation_summary.loss,
            "metrics": validation_summary.metrics,
            "subject_level": subject_metrics,
        },
    )
    git_commit, git_dirty = _git_state(workspace)
    manifest = {
        "schema_version": TUNING_SCHEMA_VERSION,
        "status": "complete",
        "experiment_id": TUNING_EXPERIMENT_ID,
        "campaign_id": search["campaign_id"],
        "candidate_id": candidate_id,
        "outer_fold": outer_fold,
        "seed": seed,
        "device": str(torch_device),
        "data_variant": variant,
        "selection_role": "validation_only",
        "selection_policy": "per_outer_fold_validation",
        "test_policy": "test_loader_is_not_constructed",
        "test_records_loaded": False,
        "config_path": str(search_config_path),
        "search_config_sha256": search_config_sha256,
        "resolved_config": str(resolved_config_path),
        "config_sha256": config_sha256,
        "split_path": str(split_path),
        "split_sha256": split_sha256,
        "git_commit": git_commit,
        "git_dirty": git_dirty,
        "runner_code_sha256": resnet_tuning_code_sha256(workspace),
        "train_subjects": list(partitions.train.subject_ids),
        "validation_subjects": list(partitions.validation.subject_ids),
        "test_subjects_not_loaded": list(partitions.test.subject_ids),
        "best_checkpoint": str(best_path),
        "best_checkpoint_sha256": sha256_file(best_path),
        "best_epoch": fit_result.progress.best_epoch,
        "completed_epochs": fit_result.progress.completed_epochs,
        "validation_macro_f1": float(validation_summary.metrics["macro_f1"]),
        "validation_subject_macro_f1": subject_metrics["mean_macro_f1"],
    }
    atomic_write_json(run_root / "run_manifest.json", manifest)
    return manifest
