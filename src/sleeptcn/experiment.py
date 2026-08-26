"""Dieu phoi E0-E3 theo tung giai doan co the kiem toan."""

from __future__ import annotations

import subprocess
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
    sha256_file,
)
from .dataset import SleepRecord, load_record
from .engine import (
    epoch_forward,
    fit_model,
    seed_everything,
    sequence_forward,
)
from .evaluation import save_role_artifacts
from .features import (
    MANIPULATIONS,
    MANIPULATION_PREFIX,
    STAGE_NAMES,
    class_specific_weights,
    expected_15cnn_keys,
)
from .io.serialization import atomic_write_json, read_json
from .models import EEGResNet1D, SleepCNN
from .training import collate_feature_sequences, masked_cross_entropy
from .training_data import (
    FeatureSequence,
    FeatureSequenceDataset,
    RecordEpochDataset,
    RolePartition,
    class_counts_from_records,
    resolve_fold_partitions,
)
from .workflows.checkpoints import load_verified_checkpoint
from .workflows.layout import EXPERIMENT_IDS, build_experiment_layout
from .workflows.model_factory import (
    build_sequence_model,
    load_sequence_checkpoint,
    sequence_component_config,
)
from .workflows.provenance import runner_code_sha256
from .workflows.stages import (
    mark_stage_complete as _mark_stage_complete,
    stage_is_complete as _stage_is_complete,
)


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
    config_path: Path
    split_path: Path
    artifact_root: Path | None
    data_variant: str
    run_root: Path
    cache_root: Path


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


def _manifest_path(workspace: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(workspace.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


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
    config_path: Path | None = None,
    artifact_root: Path | None = None,
) -> RunContext:
    if num_workers < 0:
        raise ValueError("invalid fold, seed or num_workers")
    if smoke and allow_test_evaluation:
        raise ValueError("smoke mode must not evaluate the test role")
    layout = build_experiment_layout(
        workspace,
        experiment_id,
        outer_fold,
        seed,
        smoke=smoke,
        artifact_root=artifact_root,
    )
    workspace = layout.workspace
    torch_device = torch.device(device)
    if torch_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    config_path = (
        config_path.resolve()
        if config_path is not None
        else workspace / "configs" / "experiments_v2.json"
    )
    config = read_json(config_path)
    tuning_metadata = config.get("resnet_tuning")
    if isinstance(tuning_metadata, dict):
        selection_policy = tuning_metadata.get("selection_policy")
        if selection_policy == "per_outer_fold_validation":
            selected_fold = tuning_metadata.get("selected_outer_fold")
            selected_seed = tuning_metadata.get("selected_seed")
            if selected_fold is None or selected_seed is None:
                raise ValueError(
                    "per-fold ResNet tuning config must record selected fold and seed"
                )
            if int(selected_fold) != outer_fold or int(selected_seed) != seed:
                raise ValueError(
                    "locked ResNet tuning config was selected for a different "
                    f"fold/seed ({selected_fold}, {selected_seed})"
                )
    split_path = workspace / config["dataset"]["split_manifest"]
    if config["protocol"].get("validation_schedule") != "end_of_epoch":
        raise ValueError("runner only accepts the frozen end_of_epoch schedule")
    for component in ("cnn15", "bilstm", "resnet1d", "common_tcn"):
        if config["components"][component].get("validation_schedule") != "end_of_epoch":
            raise ValueError(f"unexpected validation schedule for {component}")
    experiment = config["experiments"][experiment_id]
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
        config_path=config_path,
        split_path=split_path,
        artifact_root=artifact_root.resolve() if artifact_root is not None else None,
        data_variant=experiment["data_variant"],
        run_root=layout.run_root,
        cache_root=layout.cache_root,
    )


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
            best_path = load_verified_checkpoint(
                context,
                model,
                checkpoint_dir,
                stage,
                component_seed,
                selection_metric="validation_loss",
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
        config_path=context.config_path,
        artifact_root=context.artifact_root,
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
        load_verified_checkpoint(
            source_context,
            model,
            marker_dir,
            stage,
            component_seed,
            selection_metric="validation_loss",
            incomplete_message=f"E0 CNN stage is not marked complete: {stage}",
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

    architecture_keys = (
        "input_channels",
        "stem",
        "residual_blocks",
        "feature_dim",
        "classifier_dropout",
    )
    architecture_config = {key: cfg[key] for key in architecture_keys}
    model = EEGResNet1D.from_config(architecture_config)
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
        fit_result = fit_model(
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
        atomic_write_json(
            checkpoint_dir / "training_history.json",
            {
                "schema_version": 1,
                "stage": stage,
                "best_checkpoint": str(checkpoint_dir / "best.pt"),
                "best_epoch": fit_result.progress.best_epoch,
                "completed_epochs": fit_result.progress.completed_epochs,
                "stopped_early": fit_result.stopped_early,
                "history": list(fit_result.history),
            },
        )
        _mark_stage_complete(context, checkpoint_dir, stage, context.seed)
    best_path = load_verified_checkpoint(
        context,
        model,
        checkpoint_dir,
        stage,
        context.seed,
        selection_metric="validation_macro_f1",
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
    if not train_sequences:
        raise ValueError("train_sequences must not be empty")
    cfg = sequence_component_config(context.config, kind)
    feature_dim = train_sequences[0].features.shape[1]
    model = build_sequence_model(context.config, kind, input_dim=feature_dim)
    learning_rate = cfg["learning_rate"]
    batch_size = cfg["batch_size_records"]
    max_epochs = cfg["max_epochs"]
    patience = cfg["early_stopping_patience_validations"]
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
    best_path = load_sequence_checkpoint(
        context,
        model,
        kind,
        checkpoint_dir,
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
    return save_role_artifacts(
        context.run_root,
        table,
        role,
        prediction_metadata={
            "experiment_id": context.experiment_id,
            "outer_fold": context.outer_fold,
            "seed": context.seed,
            "split_sha256": context.split_sha256,
            "checkpoint_sha256": checkpoint_hash,
            "data_variant": context.data_variant,
            "role": role,
            "smoke": context.smoke,
        },
        metrics_metadata={
            "experiment_id": context.experiment_id,
            "outer_fold": context.outer_fold,
            "seed": context.seed,
            "role": role,
            "checkpoint_sha256": checkpoint_hash,
        },
    )


def run_experiment(context: RunContext) -> dict[str, Any]:
    git_commit, git_dirty = _git_state(context.workspace)
    if not context.smoke and git_dirty:
        raise RuntimeError("full experiment requires a clean Git worktree")
    partitions = resolve_fold_partitions(
        context.workspace / "data" / "processed",
        context.split_path,
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
        "config_path": _manifest_path(context.workspace, context.config_path),
        "split_path": _manifest_path(context.workspace, context.split_path),
        "data_variant": context.data_variant,
        "role_records": {
            "train": [record.info.record_key for record in train_records],
            "validation": [
                record.info.record_key for record in validation_records
            ],
            "test": "locked_until_best_checkpoint",
        },
    }
    atomic_write_json(context.run_root / "run_manifest.json", run_manifest)

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
    atomic_write_json(context.run_root / "run_manifest.json", run_manifest)
    return {"run_manifest": run_manifest, "metrics": metrics}
