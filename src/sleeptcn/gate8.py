"""Gate 8: ablation nhom dac trung C/P/N voi test bi khoa."""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

from .artifacts import sha256_file
from .dataset import SleepRecord, load_record
from .engine import fit_model, load_model_checkpoint, seed_everything, sequence_forward
from .evaluation import PredictionTable, load_prediction_table, save_role_artifacts
from .experiment import (
    RunContext,
    build_context,
    feature_sequences,
    load_cnn15_from_e0,
    predict_sequences,
)
from .metrics import compute_metrics
from .models import SleepTCN
from .run_validation import validate_run
from .io.serialization import atomic_savez, atomic_write_json, read_json
from .training import collate_feature_sequences, masked_cross_entropy
from .training_data import (
    FeatureSequence,
    FeatureSequenceDataset,
    resolve_fold_partitions,
)
from .workflows.context_ablation import (
    CONDITIONS,
    GROUP_SLICES,
    context_groups,
    mask_feature_sequences,
    train_replacement_mean,
)
from .workflows.gate8_protocol import (
    GATE8_CONFIG,
    SPLIT_PATH,
    UNLOCK_CONFIRMATION,
    load_protocol,
)
from .workflows.model_factory import build_sequence_model
from .workflows.provenance import runner_code_sha256


@dataclass(frozen=True)
class Gate8Context:
    workspace: Path
    condition: str
    outer_fold: int
    seed: int
    device: torch.device
    num_workers: int
    smoke: bool
    resume: bool
    protocol: dict[str, Any]
    protocol_sha256: str
    split_sha256: str
    source_context: RunContext
    run_root: Path

    @property
    def experiment_id(self) -> str:
        return f"G8_{self.condition}"


def _git(workspace: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(workspace), *arguments],
        capture_output=True,
        text=True,
        check=False,
    )


def clean_git_commit(workspace: Path) -> str:
    commit = _git(workspace, "rev-parse", "HEAD")
    status = _git(workspace, "status", "--porcelain")
    if commit.returncode or status.returncode:
        raise RuntimeError("Gate 8 requires a readable Git repository")
    if status.stdout.strip():
        raise RuntimeError("full Gate 8 work requires a clean Git worktree")
    return commit.stdout.strip()


def _assert_reference_ancestor(workspace: Path, reference: str) -> None:
    result = _git(workspace, "merge-base", "--is-ancestor", reference, "HEAD")
    if result.returncode != 0:
        raise RuntimeError(
            f"Gate 8 source reference {reference} is not an ancestor of HEAD"
        )


def gate8_runner_sha256(workspace: Path) -> str:
    return runner_code_sha256(workspace, include_gate8=True)


def build_gate8_context(
    workspace: Path,
    condition: str,
    outer_fold: int,
    seed: int,
    device: str,
    *,
    num_workers: int,
    smoke: bool = False,
    resume: bool = False,
) -> Gate8Context:
    workspace = workspace.resolve()
    protocol, protocol_hash = load_protocol(workspace)
    if condition not in CONDITIONS:
        raise ValueError(f"condition must be one of {CONDITIONS}")
    if outer_fold not in range(10) or seed != protocol["seed"]:
        raise ValueError("Gate 8 fold/seed differs from the frozen protocol")
    if num_workers < 0:
        raise ValueError("num_workers must be non-negative")
    torch_device = torch.device(device)
    if torch_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    source = build_context(
        workspace,
        "E1",
        outer_fold,
        seed,
        str(torch_device),
        smoke=smoke,
        allow_test_evaluation=False,
        num_workers=num_workers,
        resume=False,
    )
    mode = "smoke" if smoke else "full"
    return Gate8Context(
        workspace=workspace,
        condition=condition,
        outer_fold=outer_fold,
        seed=seed,
        device=torch_device,
        num_workers=num_workers,
        smoke=smoke,
        resume=resume,
        protocol=protocol,
        protocol_sha256=protocol_hash,
        split_sha256=sha256_file(workspace / SPLIT_PATH),
        source_context=source,
        run_root=(
            workspace
            / "runs"
            / "v2"
            / "gate8"
            / mode
            / condition
            / f"fold_{outer_fold:02d}"
            / f"seed_{seed}"
        ),
    )


def _load_role_records(
    context: Gate8Context, role: str, *, permit_test: bool
) -> tuple[SleepRecord, ...]:
    if role == "test" and not permit_test:
        raise RuntimeError("Gate 8 test role is locked")
    partitions = resolve_fold_partitions(
        context.workspace / "data" / "processed",
        context.workspace / SPLIT_PATH,
        context.outer_fold,
        "paper_raw_v1",
    )
    partition = partitions.for_role(role)
    paths = partition.paths
    if context.smoke:
        paths = paths[:2] if role == "train" else paths[:1]
    records = tuple(load_record(path, "paper_raw_v1") for path in paths)
    if tuple(record.info.record_key for record in records) != tuple(
        path.stem for path in paths
    ):
        raise AssertionError("Gate 8 record order mismatch")
    return records


def _feature_sequences_for_role(
    context: Gate8Context,
    role: str,
    records: Sequence[SleepRecord],
    extractor: Any,
    extractor_hash: str,
) -> tuple[FeatureSequence, ...]:
    return feature_sequences(
        context.source_context,
        records,
        role,
        extractor_kind="cnn15",
        extractor=extractor,
        extractor_sha256=extractor_hash,
    )


def _replacement_path(context: Gate8Context) -> Path:
    return context.run_root / "feature_mask" / "train_replacement_mean.npz"


def _save_replacement(
    context: Gate8Context,
    mean: np.ndarray,
    valid_train_epochs: int,
    extractor_hash: str,
) -> str:
    metadata = {
        "schema_version": 1,
        "artifact_type": "gate8_train_replacement_mean",
        "condition": context.condition,
        "outer_fold": context.outer_fold,
        "seed": context.seed,
        "fit_role": "train_valid_labels_only",
        "valid_train_epochs": valid_train_epochs,
        "masked_groups": list(
            context.protocol["conditions"][context.condition]["masked_groups"]
        ),
        "gate8_config_sha256": context.protocol_sha256,
        "split_sha256": context.split_sha256,
        "extractor_sha256": extractor_hash,
    }
    path = _replacement_path(context)
    atomic_savez(
        path,
        {
            "metadata_json": np.asarray(json.dumps(metadata, sort_keys=True)),
            "replacement_mean": mean,
        },
    )
    return sha256_file(path)


def _load_replacement(context: Gate8Context, extractor_hash: str) -> np.ndarray:
    path = _replacement_path(context)
    with np.load(path, allow_pickle=False) as npz:
        metadata = json.loads(str(npz["metadata_json"].item()))
        mean = npz["replacement_mean"].copy()
    expected = {
        "schema_version": 1,
        "artifact_type": "gate8_train_replacement_mean",
        "condition": context.condition,
        "outer_fold": context.outer_fold,
        "seed": context.seed,
        "fit_role": "train_valid_labels_only",
        "gate8_config_sha256": context.protocol_sha256,
        "split_sha256": context.split_sha256,
        "extractor_sha256": extractor_hash,
    }
    mismatches = {
        key: (metadata.get(key), value)
        for key, value in expected.items()
        if metadata.get(key) != value
    }
    if mismatches or mean.shape != (75,) or not np.isfinite(mean).all():
        raise ValueError(f"invalid Gate 8 replacement artifact: {mismatches}")
    return mean.astype(np.float32, copy=False)


def _make_tcn(context: Gate8Context) -> SleepTCN:
    return build_sequence_model(
        context.source_context.config,
        "common_tcn",
        input_dim=75,
    )


def _checkpoint_metadata(context: Gate8Context, model: SleepTCN) -> dict[str, Any]:
    return {
        "experiment_id": context.experiment_id,
        "stage": f"gate8/{context.condition}/sequence/tcn",
        "outer_fold": context.outer_fold,
        "seed": context.seed,
        "config_sha256": context.protocol_sha256,
        "split_sha256": context.split_sha256,
        "data_variant": "paper_raw_v1",
        "model_class": type(model).__name__,
        "selection_metric": "validation_macro_f1",
    }


def _train_tcn(
    context: Gate8Context,
    train_sequences: Sequence[FeatureSequence],
    validation_sequences: Sequence[FeatureSequence],
) -> tuple[SleepTCN, Path]:
    cfg = context.source_context.config["components"]["common_tcn"]
    seed_everything(context.seed)
    generator = torch.Generator().manual_seed(context.seed)
    loader_kwargs = {
        "num_workers": context.num_workers,
        "pin_memory": context.device.type == "cuda",
        "persistent_workers": context.num_workers > 0,
        "collate_fn": collate_feature_sequences,
    }
    train_loader = DataLoader(
        FeatureSequenceDataset(train_sequences),
        batch_size=cfg["batch_size_records"],
        shuffle=True,
        generator=generator,
        **loader_kwargs,
    )
    validation_loader = DataLoader(
        FeatureSequenceDataset(validation_sequences),
        batch_size=cfg["batch_size_records"],
        shuffle=False,
        **loader_kwargs,
    )
    model = _make_tcn(context)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"])
    checkpoint_dir = context.run_root / "checkpoints" / "sequence" / "tcn"
    latest = checkpoint_dir / "latest.pt"
    fit_model(
        model,
        train_loader,
        validation_loader,
        optimizer,
        masked_cross_entropy,
        sequence_forward("tcn"),
        device=context.device,
        max_epochs=1 if context.smoke else cfg["max_epochs"],
        patience=cfg["early_stopping_patience_validations"],
        checkpoint_dir=checkpoint_dir,
        experiment_id=context.experiment_id,
        stage=f"gate8/{context.condition}/sequence/tcn",
        outer_fold=context.outer_fold,
        seed=context.seed,
        config_sha256=context.protocol_sha256,
        split_sha256=context.split_sha256,
        data_variant="paper_raw_v1",
        selection_metric="validation_macro_f1",
        gradient_clip_norm=1.0,
        loader_generator=generator,
        resume_from=latest if context.resume and latest.is_file() else None,
        max_train_batches=1 if context.smoke else None,
        max_validation_batches=1 if context.smoke else None,
    )
    best = checkpoint_dir / "best.pt"
    load_model_checkpoint(
        best,
        model,
        expected_metadata=_checkpoint_metadata(context, model),
        device=context.device,
    )
    return model.eval(), best


def _save_role(
    context: Gate8Context,
    model: SleepTCN,
    sequences: Sequence[FeatureSequence],
    role: str,
    checkpoint: Path,
) -> dict[str, Any]:
    table = predict_sequences(model, sequences, "tcn", context.device)
    checkpoint_hash = sha256_file(checkpoint)
    return save_role_artifacts(
        context.run_root,
        table,
        role,
        prediction_metadata={
            "experiment_id": context.experiment_id,
            "condition": context.condition,
            "outer_fold": context.outer_fold,
            "seed": context.seed,
            "split_sha256": context.split_sha256,
            "checkpoint_sha256": checkpoint_hash,
            "gate8_config_sha256": context.protocol_sha256,
            "data_variant": "paper_raw_v1",
            "role": role,
            "smoke": context.smoke,
        },
        metrics_metadata={
            "experiment_id": context.experiment_id,
            "condition": context.condition,
            "outer_fold": context.outer_fold,
            "seed": context.seed,
            "role": role,
            "checkpoint_sha256": checkpoint_hash,
            "gate8_config_sha256": context.protocol_sha256,
        },
    )


def _parameter_count(model: torch.nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters()))


def run_validation_condition(context: Gate8Context) -> dict[str, Any]:
    if not context.smoke:
        commit = clean_git_commit(context.workspace)
        _assert_reference_ancestor(
            context.workspace, context.protocol["source_reference_commit"]
        )
    else:
        result = _git(context.workspace, "rev-parse", "HEAD")
        commit = result.stdout.strip() if result.returncode == 0 else None
    manifest_path = context.run_root / "run_manifest.json"
    if manifest_path.exists():
        if not context.resume:
            raise FileExistsError(f"Gate 8 run already exists; use --resume: {context.run_root}")
        existing = read_json(manifest_path)
        if existing.get("status") in {"validation_complete", "complete"}:
            return validate_gate8_run(context.workspace, context.run_root)

    train_records = _load_role_records(context, "train", permit_test=False)
    validation_records = _load_role_records(context, "validation", permit_test=False)
    source_e1_root = context.source_context.run_root
    source_e1_report = validate_run(context.workspace, source_e1_root)
    extractor, extractor_hash = load_cnn15_from_e0(context.source_context)
    train_features = _feature_sequences_for_role(
        context, "train", train_records, extractor, extractor_hash
    )
    validation_features = _feature_sequences_for_role(
        context, "validation", validation_records, extractor, extractor_hash
    )
    mean, valid_train_epochs = train_replacement_mean(train_features)
    replacement_hash = _save_replacement(
        context, mean, valid_train_epochs, extractor_hash
    )
    masked_train = mask_feature_sequences(train_features, context.condition, mean)
    masked_validation = mask_feature_sequences(
        validation_features, context.condition, mean
    )
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "status": "training",
        "gate": "GATE_8_CONTEXT_GROUP_ABLATION",
        "experiment_id": context.experiment_id,
        "condition": context.condition,
        "retained_groups": list(context_groups(context.condition)),
        "masked_groups": list(
            context.protocol["conditions"][context.condition]["masked_groups"]
        ),
        "outer_fold": context.outer_fold,
        "seed": context.seed,
        "smoke": context.smoke,
        "git_commit": commit,
        "gate8_config_path": str(GATE8_CONFIG).replace("\\", "/"),
        "gate8_config_sha256": context.protocol_sha256,
        "split_path": str(SPLIT_PATH).replace("\\", "/"),
        "split_sha256": context.split_sha256,
        "runner_code_sha256": gate8_runner_sha256(context.workspace),
        "data_variant": "paper_raw_v1",
        "source_experiment": "E1",
        "source_e1_manifest_sha256": source_e1_report["manifest_sha256"],
        "source_e1_sequence_checkpoint_sha256": source_e1_report[
            "sequence_checkpoint_sha256"
        ],
        "source_e0_extractor_sha256": extractor_hash,
        "replacement_mean_sha256": replacement_hash,
        "replacement_fit_valid_train_epochs": valid_train_epochs,
        "input_dimension": 75,
        "role_records": {
            "train": [record.info.record_key for record in train_records],
            "validation": [record.info.record_key for record in validation_records],
            "test": "locked_until_30_validation_runs_complete",
        },
        "metrics_roles": [],
    }
    atomic_write_json(manifest_path, manifest)
    model, best = _train_tcn(context, masked_train, masked_validation)
    validation_metrics = _save_role(
        context, model, masked_validation, "validation", best
    )
    manifest.update(
        {
            "status": "validation_complete",
            "sequence_checkpoint_sha256": sha256_file(best),
            "trainable_parameters": _parameter_count(model),
            "metrics_roles": ["validation"],
        }
    )
    atomic_write_json(manifest_path, manifest)
    report = validate_gate8_run(context.workspace, context.run_root)
    atomic_write_json(context.run_root / "validation_report.json", report)
    return {"validation": validation_metrics, "validation_report": report}


def _load_prediction(path: Path) -> tuple[PredictionTable, dict[str, Any]]:
    return load_prediction_table(path)


def _compare_nested(left: Any, right: Any, path: str = "metrics") -> None:
    if isinstance(left, dict) and isinstance(right, dict):
        if left.keys() != right.keys():
            raise ValueError(f"{path} keys differ")
        for key in left:
            _compare_nested(left[key], right[key], f"{path}.{key}")
    elif isinstance(left, list) and isinstance(right, list):
        if left != right:
            raise ValueError(f"{path} differs")
    elif isinstance(left, float) or isinstance(right, float):
        if not np.isclose(left, right, rtol=0.0, atol=1e-12):
            raise ValueError(f"{path} differs")
    elif left != right:
        raise ValueError(f"{path} differs")


def validate_gate8_run(workspace: Path, run_root: Path) -> dict[str, Any]:
    workspace, run_root = workspace.resolve(), run_root.resolve()
    manifest_path = run_root / "run_manifest.json"
    manifest = read_json(manifest_path)
    if manifest.get("status") not in {
        "validation_complete",
        "test_running",
        "complete",
    }:
        raise ValueError("Gate 8 run is not complete for its current stage")
    condition = manifest.get("condition")
    context = build_gate8_context(
        workspace,
        condition,
        int(manifest["outer_fold"]),
        int(manifest["seed"]),
        "cpu",
        num_workers=0,
        smoke=bool(manifest["smoke"]),
        resume=True,
    )
    current = {
        "gate8_config_sha256": context.protocol_sha256,
        "split_sha256": context.split_sha256,
        "runner_code_sha256": gate8_runner_sha256(workspace),
    }
    mismatches = {
        key: (manifest.get(key), value)
        for key, value in current.items()
        if manifest.get(key) != value
    }
    if mismatches:
        raise ValueError(f"Gate 8 provenance mismatch: {mismatches}")
    checkpoint = run_root / "checkpoints" / "sequence" / "tcn" / "best.pt"
    replacement = run_root / "feature_mask" / "train_replacement_mean.npz"
    if sha256_file(checkpoint) != manifest.get("sequence_checkpoint_sha256"):
        raise ValueError("Gate 8 sequence checkpoint hash mismatch")
    if sha256_file(replacement) != manifest.get("replacement_mean_sha256"):
        raise ValueError("Gate 8 replacement mean hash mismatch")
    replacement_mean = _load_replacement(
        context, manifest["source_e0_extractor_sha256"]
    )
    if replacement_mean.dtype != np.float32:
        raise ValueError("Gate 8 replacement mean must be float32")
    expected_parameters = _parameter_count(_make_tcn(context))
    if manifest.get("trainable_parameters") != expected_parameters:
        raise ValueError("Gate 8 TCN parameter count differs from the locked architecture")
    roles: dict[str, Any] = {}
    for role in manifest["metrics_roles"]:
        prediction_path = run_root / "predictions" / f"{role}.npz"
        metrics_path = run_root / "metrics" / f"{role}.json"
        table, metadata = _load_prediction(prediction_path)
        expected_metadata = {
            "experiment_id": manifest["experiment_id"],
            "condition": condition,
            "outer_fold": manifest["outer_fold"],
            "seed": manifest["seed"],
            "split_sha256": manifest["split_sha256"],
            "checkpoint_sha256": manifest["sequence_checkpoint_sha256"],
            "gate8_config_sha256": manifest["gate8_config_sha256"],
            "data_variant": "paper_raw_v1",
            "role": role,
            "smoke": manifest["smoke"],
        }
        meta_mismatch = {
            key: (metadata.get(key), value)
            for key, value in expected_metadata.items()
            if metadata.get(key) != value
        }
        if meta_mismatch:
            raise ValueError(f"Gate 8 {role} metadata mismatch: {meta_mismatch}")
        records = manifest["role_records"][role]
        if list(dict.fromkeys(table.record_key.tolist())) != records:
            raise ValueError(f"Gate 8 {role} record order mismatch")
        for record_key in records:
            record = load_record(
                workspace
                / "data"
                / "processed"
                / "paper_raw_v1"
                / f"{record_key}.npz",
                "paper_raw_v1",
            )
            selected = table.record_key == record_key
            valid = record.valid_mask
            if not np.array_equal(
                table.original_epoch_index[selected], record.original_epoch_index[valid]
            ):
                raise ValueError(f"Gate 8 {record_key} epoch indices mismatch")
            if not np.array_equal(table.true_label[selected], record.y[valid]):
                raise ValueError(f"Gate 8 {record_key} true labels mismatch")
            if not np.all(table.subject_id[selected] == record.info.subject_id):
                raise ValueError(f"Gate 8 {record_key} subject identity mismatch")
        metric_payload = read_json(metrics_path)
        _compare_nested(metric_payload["metrics"], table.metrics())
        roles[role] = {
            "records": len(records),
            "valid_epochs": len(table.true_label),
            "prediction_sha256": sha256_file(prediction_path),
            "metrics_sha256": sha256_file(metrics_path),
        }
    if manifest["status"] == "validation_complete":
        if manifest["role_records"]["test"] != "locked_until_30_validation_runs_complete":
            raise ValueError("Gate 8 test role was exposed before unlock")
        if (run_root / "predictions" / "test.npz").exists():
            raise ValueError("Gate 8 test prediction exists before unlock")
    return {
        "schema_version": 1,
        "passed": True,
        "condition": condition,
        "outer_fold": manifest["outer_fold"],
        "seed": manifest["seed"],
        "smoke": manifest["smoke"],
        "status": manifest["status"],
        "manifest_sha256": sha256_file(manifest_path),
        "sequence_checkpoint_sha256": sha256_file(checkpoint),
        "replacement_mean_sha256": sha256_file(replacement),
        "roles": roles,
    }


def evaluate_locked_test_target(context: Gate8Context) -> dict[str, Any]:
    if context.smoke:
        raise ValueError("Gate 8 smoke runs cannot evaluate test")
    report = validate_gate8_run(context.workspace, context.run_root)
    if report["status"] == "complete":
        return report
    manifest_path = context.run_root / "run_manifest.json"
    manifest = read_json(manifest_path)
    test_records = _load_role_records(context, "test", permit_test=True)
    if report["status"] == "validation_complete":
        manifest["status"] = "test_running"
        manifest["role_records"]["test"] = [
            record.info.record_key for record in test_records
        ]
        atomic_write_json(manifest_path, manifest)
    elif manifest["role_records"]["test"] != [
        record.info.record_key for record in test_records
    ]:
        raise ValueError("Gate 8 test record set changed while resuming")
    extractor, extractor_hash = load_cnn15_from_e0(context.source_context)
    if extractor_hash != manifest["source_e0_extractor_sha256"]:
        raise ValueError("Gate 8 source E0 extractor changed before test")
    mean = _load_replacement(context, extractor_hash)
    test_features = _feature_sequences_for_role(
        context, "test", test_records, extractor, extractor_hash
    )
    masked_test = mask_feature_sequences(test_features, context.condition, mean)
    model = _make_tcn(context)
    checkpoint = context.run_root / "checkpoints" / "sequence" / "tcn" / "best.pt"
    load_model_checkpoint(
        checkpoint,
        model,
        expected_metadata=_checkpoint_metadata(context, model),
        device=context.device,
    )
    _save_role(context, model.eval(), masked_test, "test", checkpoint)
    manifest["status"] = "complete"
    manifest["metrics_roles"] = ["validation", "test"]
    atomic_write_json(manifest_path, manifest)
    final_report = validate_gate8_run(context.workspace, context.run_root)
    atomic_write_json(context.run_root / "validation_report.json", final_report)
    return final_report


def preflight(workspace: Path, *, seed: int = 42) -> dict[str, Any]:
    workspace = workspace.resolve()
    commit = clean_git_commit(workspace)
    protocol, protocol_hash = load_protocol(workspace)
    if seed != protocol["seed"]:
        raise ValueError("Gate 8 preflight seed differs from protocol")
    _assert_reference_ancestor(workspace, protocol["source_reference_commit"])
    source_reports: dict[str, Any] = {}
    processed_records: set[str] = set()
    for fold in range(10):
        for experiment in ("E0", "E1"):
            root = (
                workspace
                / "runs"
                / "v2"
                / "full"
                / experiment
                / f"fold_{fold:02d}"
                / f"seed_{seed}"
            )
            report = validate_run(workspace, root)
            if "validation" not in report["roles"] or "test" not in report["roles"]:
                raise ValueError(f"{experiment}/fold_{fold:02d} is not fully validated")
            source_reports[f"{experiment}/fold_{fold:02d}"] = {
                "manifest_sha256": report["manifest_sha256"],
                "sequence_checkpoint_sha256": report[
                    "sequence_checkpoint_sha256"
                ],
            }
        context = build_gate8_context(
            workspace, "CP", fold, seed, "cpu", num_workers=0
        )
        partitions = resolve_fold_partitions(
            workspace / "data" / "processed",
            workspace / SPLIT_PATH,
            fold,
            "paper_raw_v1",
        )
        for role in ("train", "validation", "test"):
            for path in partitions.for_role(role).paths:
                if not path.is_file():
                    raise FileNotFoundError(path)
                processed_records.add(path.stem)
        if context.protocol_sha256 != protocol_hash:
            raise AssertionError("Gate 8 protocol hash changed during preflight")
    if len(processed_records) != 153:
        raise ValueError(
            f"Gate 8 expected 153 paper_raw_v1 records, found {len(processed_records)}"
        )
    test_artifacts = list(
        (workspace / "runs" / "v2" / "gate8" / "full").glob(
            "*/fold_*/seed_42/predictions/test.npz"
        )
    )
    if test_artifacts:
        raise FileExistsError("Gate 8 test artifacts already exist; do not prepare anew")
    return {
        "schema_version": 1,
        "status": "prepared",
        "gate": "GATE_8_CONTEXT_GROUP_ABLATION",
        "source_git_commit": commit,
        "gate8_config_sha256": protocol_hash,
        "split_sha256": sha256_file(workspace / SPLIT_PATH),
        "conditions": list(CONDITIONS),
        "folds": list(range(10)),
        "target_count": 30,
        "processed_records": len(processed_records),
        "validated_source_runs": len(source_reports),
        "source_artifacts": source_reports,
    }
