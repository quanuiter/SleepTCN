"""Dieu phoi E0-E3 theo tung giai doan co the kiem toan."""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .artifacts import (
    combined_sha256,
    extract_15cnn_sequence,
    extract_resnet_sequence,
    load_feature_sequence,
    prediction_table_from_parts,
    save_feature_sequence,
    save_prediction_table,
    sha256_file,
)
from .dataset import SleepRecord, load_record
from .engine import (
    epoch_forward,
    fit_model,
    load_model_checkpoint,
    seed_everything,
    sequence_forward,
)
from .features import (
    MANIPULATIONS,
    MANIPULATION_PREFIX,
    STAGE_NAMES,
    class_specific_weights,
    expected_15cnn_keys,
)
from .models import BiLSTMSleepNet, EEGResNet1D, SleepCNN, SleepTCN
from .training import collate_feature_sequences, masked_cross_entropy
from .training_data import (
    FeatureSequence,
    FeatureSequenceDataset,
    RecordEpochDataset,
    RolePartition,
    class_counts_from_records,
    resolve_fold_partitions,
)


EXPERIMENT_IDS = ("E0", "E1", "E2", "E3")


@dataclass(frozen=True)
class RunContext:
    workspace: Path
    experiment_id: str
    outer_fold: int
    seed: int
    device: torch.device
    smoke: bool
    allow_test_evaluation: bool
    resume: bool
    num_workers: int
    config: dict[str, Any]
    config_sha256: str
    split_sha256: str
    data_variant: str
    run_root: Path
    cache_root: Path


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def _git_state(workspace: Path) -> tuple[str | None, bool]:
    commit_result = subprocess.run(
        ["git", "-C", str(workspace), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    status_result = subprocess.run(
        ["git", "-C", str(workspace), "status", "--porcelain"],
        capture_output=True,
        text=True,
        check=False,
    )
    commit = (
        commit_result.stdout.strip() if commit_result.returncode == 0 else None
    )
    dirty = status_result.returncode != 0 or bool(status_result.stdout.strip())
    return commit, dirty


def runner_code_sha256(workspace: Path) -> str:
    relative_paths = (
        "src/sleeptcn/artifacts.py",
        "src/sleeptcn/dataset.py",
        "src/sleeptcn/engine.py",
        "src/sleeptcn/experiment.py",
        "src/sleeptcn/features.py",
        "src/sleeptcn/metrics.py",
        "src/sleeptcn/models.py",
        "src/sleeptcn/training.py",
        "src/sleeptcn/training_data.py",
    )
    return combined_sha256(
        {path: sha256_file(workspace / path) for path in relative_paths}
    )


def build_context(
    workspace: Path,
    experiment_id: str,
    outer_fold: int,
    seed: int,
    device: str,
    *,
    smoke: bool,
    allow_test_evaluation: bool,
    num_workers: int,
    resume: bool = False,
) -> RunContext:
    workspace = workspace.resolve()
    if experiment_id not in EXPERIMENT_IDS:
        raise ValueError(f"experiment_id must be one of {EXPERIMENT_IDS}")
    if outer_fold not in range(10) or seed < 0 or num_workers < 0:
        raise ValueError("invalid fold, seed or num_workers")
    if smoke and allow_test_evaluation:
        raise ValueError("smoke mode must not evaluate the test role")
    torch_device = torch.device(device)
    if torch_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    config_path = workspace / "configs" / "experiments_v1.json"
    split_path = workspace / "data" / "splits" / "sleepedf_sc_10fold_seed42_v1.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config["protocol"].get("validation_schedule") != "end_of_epoch":
        raise ValueError("runner only accepts the frozen end_of_epoch schedule")
    for component in ("cnn15", "bilstm", "resnet1d", "common_tcn"):
        if config["components"][component].get("validation_schedule") != "end_of_epoch":
            raise ValueError(f"unexpected validation schedule for {component}")
    experiment = config["experiments"][experiment_id]
    mode = "smoke" if smoke else "full"
    run_root = (
        workspace
        / "runs"
        / mode
        / experiment_id
        / f"fold_{outer_fold:02d}"
        / f"seed_{seed}"
    )
    cache_root = (
        workspace
        / "data"
        / "cache"
        / "features"
        / mode
        / f"fold_{outer_fold:02d}"
        / f"seed_{seed}"
    )
    return RunContext(
        workspace=workspace,
        experiment_id=experiment_id,
        outer_fold=outer_fold,
        seed=seed,
        device=torch_device,
        smoke=smoke,
        allow_test_evaluation=allow_test_evaluation,
        resume=resume,
        num_workers=num_workers,
        config=config,
        config_sha256=sha256_file(config_path),
        split_sha256=sha256_file(split_path),
        data_variant=experiment["data_variant"],
        run_root=run_root,
        cache_root=cache_root,
    )


def _checkpoint_metadata(
    context: RunContext, model: nn.Module, stage: str, component_seed: int
) -> dict[str, Any]:
    return {
        "experiment_id": context.experiment_id,
        "stage": stage,
        "outer_fold": context.outer_fold,
        "seed": component_seed,
        "config_sha256": context.config_sha256,
        "split_sha256": context.split_sha256,
        "data_variant": context.data_variant,
        "model_class": type(model).__name__,
    }


def _selected_records(
    partition: RolePartition, variant: str, limit: int | None
) -> tuple[SleepRecord, ...]:
    paths = partition.paths if limit is None else partition.paths[:limit]
    records = tuple(load_record(path, variant) for path in paths)
    if tuple(record.info.record_key for record in records) != tuple(
        path.stem for path in paths
    ):
        raise AssertionError("loaded record order differs from selected paths")
    if not {record.info.subject_id for record in records}.issubset(
        set(partition.subject_ids)
    ):
        raise AssertionError("loaded record escaped its role partition")
    return records


def _loader(
    dataset: Any,
    *,
    batch_size: int,
    shuffle: bool,
    generator: torch.Generator | None,
    context: RunContext,
    collate_fn: Any = None,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        num_workers=context.num_workers,
        pin_memory=context.device.type == "cuda",
        persistent_workers=context.num_workers > 0,
        collate_fn=collate_fn,
    )


def _fit_kwargs(
    context: RunContext,
    *,
    stage: str,
    component_seed: int,
    checkpoint_dir: Path,
    max_epochs: int,
    patience: int,
    generator: torch.Generator,
    selection_metric: str,
) -> dict[str, Any]:
    latest_path = checkpoint_dir / "latest.pt"
    return {
        "device": context.device,
        "max_epochs": 1 if context.smoke else max_epochs,
        "patience": patience,
        "checkpoint_dir": checkpoint_dir,
        "experiment_id": context.experiment_id,
        "stage": stage,
        "outer_fold": context.outer_fold,
        "seed": component_seed,
        "config_sha256": context.config_sha256,
        "split_sha256": context.split_sha256,
        "data_variant": context.data_variant,
        "selection_metric": selection_metric,
        "loader_generator": generator,
        "max_train_batches": 1 if context.smoke else None,
        "max_validation_batches": 1 if context.smoke else None,
        "resume_from": (
            latest_path if context.resume and latest_path.is_file() else None
        ),
    }


def _stage_marker_path(checkpoint_dir: Path) -> Path:
    return checkpoint_dir / "complete.json"


def _stage_is_complete(
    context: RunContext,
    checkpoint_dir: Path,
    stage: str,
    component_seed: int,
) -> bool:
    marker_path = _stage_marker_path(checkpoint_dir)
    best_path = checkpoint_dir / "best.pt"
    if not marker_path.is_file():
        return False
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    expected = {
        "stage": stage,
        "outer_fold": context.outer_fold,
        "component_seed": component_seed,
        "config_sha256": context.config_sha256,
        "split_sha256": context.split_sha256,
        "data_variant": context.data_variant,
        "smoke": context.smoke,
    }
    mismatches = {
        key: (marker.get(key), value)
        for key, value in expected.items()
        if marker.get(key) != value
    }
    if mismatches:
        raise ValueError(f"stage completion marker mismatch: {mismatches}")
    if not best_path.is_file() or sha256_file(best_path) != marker.get(
        "best_checkpoint_sha256"
    ):
        raise ValueError(f"completed stage checkpoint mismatch: {stage}")
    return True


def _mark_stage_complete(
    context: RunContext,
    checkpoint_dir: Path,
    stage: str,
    component_seed: int,
) -> None:
    best_path = checkpoint_dir / "best.pt"
    _write_json_atomic(
        _stage_marker_path(checkpoint_dir),
        {
            "schema_version": 1,
            "stage": stage,
            "outer_fold": context.outer_fold,
            "component_seed": component_seed,
            "config_sha256": context.config_sha256,
            "split_sha256": context.split_sha256,
            "data_variant": context.data_variant,
            "smoke": context.smoke,
            "best_checkpoint_sha256": sha256_file(best_path),
        },
    )


def train_cnn15(
    context: RunContext,
    train_records: Sequence[SleepRecord],
    validation_records: Sequence[SleepRecord],
) -> tuple[dict[str, SleepCNN], str]:
    cfg = context.config["components"]["cnn15"]
    counts = class_counts_from_records(train_records)
    models: dict[str, SleepCNN] = {}
    hashes: dict[str, str] = {}
    component_index = 0
    for manipulation in MANIPULATIONS:
        train_dataset = RecordEpochDataset(train_records, manipulation)
        validation_dataset = RecordEpochDataset(validation_records, manipulation)
        for class_index, stage_name in enumerate(STAGE_NAMES):
            key = f"{MANIPULATION_PREFIX[manipulation]}_{stage_name}"
            component_seed = context.seed + component_index
            component_index += 1
            seed_everything(component_seed)
            generator = torch.Generator().manual_seed(component_seed)
            train_loader = _loader(
                train_dataset,
                batch_size=cfg["batch_size"],
                shuffle=True,
                generator=generator,
                context=context,
            )
            validation_loader = _loader(
                validation_dataset,
                batch_size=cfg["batch_size"],
                shuffle=False,
                generator=None,
                context=context,
            )
            model = SleepCNN()
            optimizer = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"])
            weights = torch.from_numpy(class_specific_weights(class_index, counts))

            def loss_function(
                logits: torch.Tensor,
                targets: torch.Tensor,
                class_weights: torch.Tensor = weights,
            ) -> torch.Tensor:
                return masked_cross_entropy(logits, targets, class_weights)

            stage = f"cnn15/{key}"
            checkpoint_dir = context.run_root / "checkpoints" / "cnn15" / key
            if not (
                context.resume
                and _stage_is_complete(
                    context, checkpoint_dir, stage, component_seed
                )
            ):
                fit_model(
                    model,
                    train_loader,
                    validation_loader,
                    optimizer,
                    loss_function,
                    epoch_forward,
                    **_fit_kwargs(
                        context,
                        stage=stage,
                        component_seed=component_seed,
                        checkpoint_dir=checkpoint_dir,
                        max_epochs=cfg["max_epochs"],
                        patience=cfg["early_stopping_patience_validations"],
                        generator=generator,
                        selection_metric="validation_loss",
                    ),
                )
                _mark_stage_complete(
                    context, checkpoint_dir, stage, component_seed
                )
            best_path = checkpoint_dir / "best.pt"
            load_model_checkpoint(
                best_path,
                model,
                expected_metadata={
                    **_checkpoint_metadata(context, model, stage, component_seed),
                    "selection_metric": "validation_loss",
                },
                device=context.device,
            )
            models[key] = model.eval()
            hashes[key] = sha256_file(best_path)
    if tuple(models) != expected_15cnn_keys():
        raise AssertionError("15CNN models were not produced in canonical order")
    return models, combined_sha256(hashes)


def load_cnn15_from_e0(context: RunContext) -> tuple[dict[str, SleepCNN], str]:
    if context.experiment_id != "E1":
        raise ValueError("E0 extractor reuse is only valid for E1")
    source_context = build_context(
        context.workspace,
        "E0",
        context.outer_fold,
        context.seed,
        str(context.device),
        smoke=context.smoke,
        allow_test_evaluation=False,
        num_workers=context.num_workers,
        resume=False,
    )
    models: dict[str, SleepCNN] = {}
    hashes: dict[str, str] = {}
    for component_index, key in enumerate(expected_15cnn_keys()):
        component_seed = context.seed + component_index
        model = SleepCNN()
        path = source_context.run_root / "checkpoints" / "cnn15" / key / "best.pt"
        if not path.is_file():
            raise FileNotFoundError(
                f"E1 requires the verified E0 CNN checkpoint: {path}"
            )
        stage = f"cnn15/{key}"
        marker_dir = path.parent
        if not _stage_is_complete(
            source_context, marker_dir, stage, component_seed
        ):
            raise ValueError(f"E0 CNN stage is not marked complete: {stage}")
        load_model_checkpoint(
            path,
            model,
            expected_metadata={
                **_checkpoint_metadata(source_context, model, stage, component_seed),
                "selection_metric": "validation_loss",
            },
            device=context.device,
        )
        models[key] = model.eval()
        hashes[key] = sha256_file(path)
    return models, combined_sha256(hashes)


def train_resnet(
    context: RunContext,
    train_records: Sequence[SleepRecord],
    validation_records: Sequence[SleepRecord],
) -> tuple[EEGResNet1D, str]:
    cfg = context.config["components"]["resnet1d"]
    seed_everything(context.seed)
    generator = torch.Generator().manual_seed(context.seed)
    train_loader = _loader(
        RecordEpochDataset(train_records),
        batch_size=cfg["batch_size_epochs"],
        shuffle=True,
        generator=generator,
        context=context,
    )
    validation_loader = _loader(
        RecordEpochDataset(validation_records),
        batch_size=cfg["batch_size_epochs"],
        shuffle=False,
        generator=None,
        context=context,
    )
    counts = class_counts_from_records(train_records).astype(np.float64)
    weights = torch.from_numpy(
        (counts.sum() / (5.0 * counts)).astype(np.float32)
    )

    def loss_function(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return masked_cross_entropy(logits, targets, weights)

    model = EEGResNet1D()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["learning_rate"],
        weight_decay=cfg["weight_decay"],
    )
    stage = "resnet1d"
    checkpoint_dir = context.run_root / "checkpoints" / stage
    if not (
        context.resume
        and _stage_is_complete(context, checkpoint_dir, stage, context.seed)
    ):
        fit_model(
            model,
            train_loader,
            validation_loader,
            optimizer,
            loss_function,
            epoch_forward,
            **_fit_kwargs(
                context,
                stage=stage,
                component_seed=context.seed,
                checkpoint_dir=checkpoint_dir,
                max_epochs=cfg["max_epochs"],
                patience=cfg["early_stopping_patience_epochs"],
                generator=generator,
                selection_metric="validation_macro_f1",
            ),
        )
        _mark_stage_complete(context, checkpoint_dir, stage, context.seed)
    best_path = checkpoint_dir / "best.pt"
    load_model_checkpoint(
        best_path,
        model,
        expected_metadata={
            **_checkpoint_metadata(context, model, stage, context.seed),
            "selection_metric": "validation_macro_f1",
        },
        device=context.device,
    )
    return model.eval(), sha256_file(best_path)


def feature_sequences(
    context: RunContext,
    records: Sequence[SleepRecord],
    role: str,
    *,
    extractor_kind: str,
    extractor: Any,
    extractor_sha256: str,
) -> tuple[FeatureSequence, ...]:
    if role not in {"train", "validation", "test"}:
        raise ValueError("invalid feature role")
    extractor_id = (
        f"{extractor_kind}_fold{context.outer_fold}_seed{context.seed}"
    )
    output: list[FeatureSequence] = []
    for record in records:
        path = (
            context.cache_root
            / extractor_kind
            / extractor_sha256[:16]
            / role
            / f"{record.info.record_key}.npz"
        )
        if path.is_file():
            sequence = load_feature_sequence(
                path,
                expected_extractor_sha256=extractor_sha256,
                expected_split_sha256=context.split_sha256,
                expected_outer_fold=context.outer_fold,
                expected_seed=context.seed,
            )
        elif extractor_kind == "cnn15":
            sequence = extract_15cnn_sequence(
                record,
                extractor,
                extractor_id=extractor_id,
                device=context.device,
                batch_size=64 if context.smoke else 256,
            )
            save_feature_sequence(
                path,
                sequence,
                extractor_sha256=extractor_sha256,
                split_sha256=context.split_sha256,
                outer_fold=context.outer_fold,
                seed=context.seed,
            )
        elif extractor_kind == "resnet1d":
            sequence = extract_resnet_sequence(
                record,
                extractor,
                extractor_id=extractor_id,
                device=context.device,
                batch_size=32 if context.smoke else 256,
            )
            save_feature_sequence(
                path,
                sequence,
                extractor_sha256=extractor_sha256,
                split_sha256=context.split_sha256,
                outer_fold=context.outer_fold,
                seed=context.seed,
            )
        else:
            raise ValueError(f"unknown extractor kind: {extractor_kind}")
        if (
            sequence.record_key != record.info.record_key
            or sequence.subject_id != record.info.subject_id
            or sequence.preprocess_version != record.info.preprocess_version
            or not np.array_equal(sequence.labels, record.y)
            or not np.array_equal(
                sequence.original_epoch_index, record.original_epoch_index
            )
        ):
            raise ValueError(f"cached feature/source mismatch: {record.info.record_key}")
        output.append(sequence)
    return tuple(output)


def train_sequence_model(
    context: RunContext,
    train_sequences: Sequence[FeatureSequence],
    validation_sequences: Sequence[FeatureSequence],
    kind: str,
) -> tuple[nn.Module, Path]:
    if kind == "bilstm":
        cfg = context.config["components"]["bilstm"]
        model: nn.Module = BiLSTMSleepNet(input_dim=75, hidden_dim=cfg["hidden_dim"])
        learning_rate = cfg["learning_rate"]
        batch_size = cfg["batch_size_records"]
        max_epochs = cfg["max_epochs"]
        patience = cfg["early_stopping_patience_validations"]
    elif kind == "tcn":
        cfg = context.config["components"]["common_tcn"]
        feature_dim = train_sequences[0].features.shape[1]
        model = SleepTCN(
            input_dim=feature_dim,
            hidden_dim=cfg["hidden_dim"],
            kernel_size=cfg["kernel_size"],
            n_blocks=cfg["residual_blocks"],
            dropout=cfg["dropout"],
        )
        learning_rate = cfg["learning_rate"]
        batch_size = cfg["batch_size_records"]
        max_epochs = cfg["max_epochs"]
        patience = cfg["early_stopping_patience_validations"]
    else:
        raise ValueError("sequence kind must be bilstm or tcn")
    seed_everything(context.seed)
    generator = torch.Generator().manual_seed(context.seed)
    train_loader = _loader(
        FeatureSequenceDataset(train_sequences),
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        context=context,
        collate_fn=collate_feature_sequences,
    )
    validation_loader = _loader(
        FeatureSequenceDataset(validation_sequences),
        batch_size=batch_size,
        shuffle=False,
        generator=None,
        context=context,
        collate_fn=collate_feature_sequences,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    stage = f"sequence/{kind}"
    checkpoint_dir = context.run_root / "checkpoints" / "sequence" / kind
    if not (
        context.resume
        and _stage_is_complete(context, checkpoint_dir, stage, context.seed)
    ):
        fit_model(
            model,
            train_loader,
            validation_loader,
            optimizer,
            masked_cross_entropy,
            sequence_forward(kind),
            gradient_clip_norm=1.0,
            **_fit_kwargs(
                context,
                stage=stage,
                component_seed=context.seed,
                checkpoint_dir=checkpoint_dir,
                max_epochs=max_epochs,
                patience=patience,
                generator=generator,
                selection_metric="validation_macro_f1",
            ),
        )
        _mark_stage_complete(context, checkpoint_dir, stage, context.seed)
    best_path = checkpoint_dir / "best.pt"
    load_model_checkpoint(
        best_path,
        model,
        expected_metadata={
            **_checkpoint_metadata(context, model, stage, context.seed),
            "selection_metric": "validation_macro_f1",
        },
        device=context.device,
    )
    return model.eval(), best_path


@torch.no_grad()
def predict_sequences(
    model: nn.Module,
    sequences: Sequence[FeatureSequence],
    kind: str,
    device: torch.device,
) -> Any:
    model = model.to(device).eval()
    parts: list[tuple[FeatureSequence, np.ndarray]] = []
    for sequence in sequences:
        features = torch.from_numpy(sequence.features).unsqueeze(0).to(device)
        if kind == "bilstm":
            logits = model(
                features, torch.tensor([len(sequence.labels)], dtype=torch.long)
            )
        else:
            logits = model(features, padding_mask=None)
        parts.append((sequence, logits.squeeze(0).cpu().numpy().astype(np.float32)))
    return prediction_table_from_parts(parts)


def _save_role_predictions(
    context: RunContext,
    model: nn.Module,
    sequences: Sequence[FeatureSequence],
    kind: str,
    role: str,
    checkpoint_path: Path,
) -> dict[str, object]:
    table = predict_sequences(model, sequences, kind, context.device)
    checkpoint_hash = sha256_file(checkpoint_path)
    save_prediction_table(
        context.run_root / "predictions" / f"{role}.npz",
        table,
        {
            "experiment_id": context.experiment_id,
            "outer_fold": context.outer_fold,
            "seed": context.seed,
            "split_sha256": context.split_sha256,
            "checkpoint_sha256": checkpoint_hash,
            "data_variant": context.data_variant,
            "role": role,
            "smoke": context.smoke,
        },
    )
    metrics = table.metrics()
    _write_json_atomic(
        context.run_root / "metrics" / f"{role}.json",
        {
            "metadata": {
                "experiment_id": context.experiment_id,
                "outer_fold": context.outer_fold,
                "seed": context.seed,
                "role": role,
                "checkpoint_sha256": checkpoint_hash,
            },
            "metrics": metrics,
        },
    )
    return metrics


def run_experiment(context: RunContext) -> dict[str, Any]:
    git_commit, git_dirty = _git_state(context.workspace)
    if not context.smoke and git_dirty:
        raise RuntimeError("full experiment requires a clean Git worktree")
    split_path = (
        context.workspace
        / "data"
        / "splits"
        / "sleepedf_sc_10fold_seed42_v1.json"
    )
    partitions = resolve_fold_partitions(
        context.workspace / "data" / "processed",
        split_path,
        context.outer_fold,
        context.data_variant,
    )
    limit_train = 2 if context.smoke else None
    limit_validation = 1 if context.smoke else None
    train_records = _selected_records(
        partitions.train, context.data_variant, limit_train
    )
    validation_records = _selected_records(
        partitions.validation, context.data_variant, limit_validation
    )
    run_manifest: dict[str, Any] = {
        "schema_version": 1,
        "status": "running",
        "experiment_id": context.experiment_id,
        "outer_fold": context.outer_fold,
        "seed": context.seed,
        "device": str(context.device),
        "smoke": context.smoke,
        "allow_test_evaluation": context.allow_test_evaluation,
        "resume": context.resume,
        "config_sha256": context.config_sha256,
        "split_sha256": context.split_sha256,
        "git_commit": git_commit,
        "git_dirty": git_dirty,
        "runner_code_sha256": runner_code_sha256(context.workspace),
        "data_variant": context.data_variant,
        "role_records": {
            "train": [record.info.record_key for record in train_records],
            "validation": [
                record.info.record_key for record in validation_records
            ],
            "test": "locked_until_best_checkpoint",
        },
    }
    _write_json_atomic(context.run_root / "run_manifest.json", run_manifest)

    if context.experiment_id == "E0":
        extractor, extractor_hash = train_cnn15(
            context, train_records, validation_records
        )
        extractor_kind = "cnn15"
        sequence_kind = "bilstm"
    elif context.experiment_id == "E1":
        extractor, extractor_hash = load_cnn15_from_e0(context)
        extractor_kind = "cnn15"
        sequence_kind = "tcn"
    else:
        extractor, extractor_hash = train_resnet(
            context, train_records, validation_records
        )
        extractor_kind = "resnet1d"
        sequence_kind = "tcn"

    train_sequences = feature_sequences(
        context,
        train_records,
        "train",
        extractor_kind=extractor_kind,
        extractor=extractor,
        extractor_sha256=extractor_hash,
    )
    validation_sequences = feature_sequences(
        context,
        validation_records,
        "validation",
        extractor_kind=extractor_kind,
        extractor=extractor,
        extractor_sha256=extractor_hash,
    )
    sequence_model, best_sequence_checkpoint = train_sequence_model(
        context, train_sequences, validation_sequences, sequence_kind
    )
    validation_metrics = _save_role_predictions(
        context,
        sequence_model,
        validation_sequences,
        sequence_kind,
        "validation",
        best_sequence_checkpoint,
    )
    metrics: dict[str, Any] = {"validation": validation_metrics}
    if context.allow_test_evaluation:
        if not best_sequence_checkpoint.is_file():
            raise AssertionError("test role unlocked before best checkpoint exists")
        test_records = _selected_records(partitions.test, context.data_variant, None)
        test_sequences = feature_sequences(
            context,
            test_records,
            "test",
            extractor_kind=extractor_kind,
            extractor=extractor,
            extractor_sha256=extractor_hash,
        )
        metrics["test"] = _save_role_predictions(
            context,
            sequence_model,
            test_sequences,
            sequence_kind,
            "test",
            best_sequence_checkpoint,
        )
        run_manifest["role_records"]["test"] = [
            record.info.record_key for record in test_records
        ]
    run_manifest["status"] = "complete"
    run_manifest["extractor_sha256"] = extractor_hash
    run_manifest["sequence_checkpoint_sha256"] = sha256_file(
        best_sequence_checkpoint
    )
    run_manifest["metrics_roles"] = list(metrics)
    _write_json_atomic(context.run_root / "run_manifest.json", run_manifest)
    return {"run_manifest": run_manifest, "metrics": metrics}
