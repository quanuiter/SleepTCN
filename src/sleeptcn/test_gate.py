"""One-time, auditable evaluation of the locked test folds.

This module deliberately does not call ``run_experiment``: training and validation
artifacts are immutable by the time the test gate is opened.
"""

from __future__ import annotations

import json
import hashlib
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .artifacts import combined_sha256, sha256_file
from .engine import load_model_checkpoint
from .experiment import (
    RunContext,
    _checkpoint_metadata,
    _save_role_predictions,
    _selected_records,
    _stage_is_complete,
    _write_json_atomic,
    build_context,
    feature_sequences,
    load_cnn15_from_e0,
)
from .features import expected_15cnn_keys
from .models import BiLSTMSleepNet, EEGResNet1D, SleepCNN, SleepTCN
from .run_validation import validate_run
from .training_data import resolve_fold_partitions


ACTIVE_EXPERIMENTS = ("E0", "E1", "E2", "E3", "E4", "E6")
CONFIRMATION_PHRASE = "OPEN-LOCKED-TEST-ONCE"
CAMPAIGN_SCHEMA_VERSION = 1


@dataclass(frozen=True, order=True)
class TestTarget:
    experiment_id: str
    outer_fold: int

    @property
    def key(self) -> str:
        return f"{self.experiment_id}/fold_{self.outer_fold:02d}"


def campaign_targets() -> tuple[TestTarget, ...]:
    return tuple(
        TestTarget(experiment_id, fold)
        for fold in range(10)
        for experiment_id in ACTIVE_EXPERIMENTS
    )


def campaign_path(workspace: Path, seed: int) -> Path:
    return workspace / "runs" / "v2" / f"test_campaign_seed{seed}.json"


def _git_state(workspace: Path) -> tuple[str, tuple[str, ...]]:
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
    if commit_result.returncode or status_result.returncode:
        raise RuntimeError("workspace must be a readable Git repository")
    return commit_result.stdout.strip(), tuple(
        line.rstrip() for line in status_result.stdout.splitlines() if line.strip()
    )


def _status_path(line: str) -> str:
    path = line[3:].strip().replace("\\", "/")
    if " -> " in path:
        path = path.split(" -> ", 1)[1]
    return path.strip('"')


def _allowed_resume_paths(seed: int) -> set[str]:
    paths: set[str] = set()
    for target in campaign_targets():
        root = (
            f"runs/v2/full/{target.experiment_id}/fold_"
            f"{target.outer_fold:02d}/seed_{seed}"
        )
        paths.add(f"{root}/run_manifest.json")
        paths.add(f"{root}/validation_report.json")
    return paths


def assert_git_state(workspace: Path, seed: int, *, resume: bool) -> str:
    commit, status = _git_state(workspace)
    if not resume and status:
        raise RuntimeError(
            "test campaign requires a clean Git worktree; changed paths: "
            + ", ".join(_status_path(line) for line in status)
        )
    if resume:
        allowed = _allowed_resume_paths(seed)
        unexpected = sorted(
            path for path in map(_status_path, status) if path not in allowed
        )
        if unexpected:
            raise RuntimeError(
                "resume found changes outside prior test artifacts: "
                + ", ".join(unexpected)
            )
    return commit


def _run_root(workspace: Path, target: TestTarget, seed: int) -> Path:
    return (
        workspace
        / "runs"
        / "v2"
        / "full"
        / target.experiment_id
        / f"fold_{target.outer_fold:02d}"
        / f"seed_{seed}"
    )


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _text_sha256_lf(path: Path) -> str:
    """Hash JSON provenance canonically when Git checked it out as CRLF."""
    payload = path.read_bytes().replace(b"\r\n", b"\n")
    return hashlib.sha256(payload).hexdigest()


def _baseline_entry(workspace: Path, target: TestTarget, seed: int) -> dict[str, Any]:
    run_root = _run_root(workspace, target, seed)
    manifest_path = run_root / "run_manifest.json"
    report_path = run_root / "validation_report.json"
    manifest = _read_json(manifest_path)
    if manifest.get("experiment_id") != target.experiment_id:
        raise ValueError(f"{target.key}: experiment mismatch")
    if manifest.get("outer_fold") != target.outer_fold or manifest.get("seed") != seed:
        raise ValueError(f"{target.key}: fold/seed mismatch")
    if manifest.get("smoke") or manifest.get("status") != "complete":
        raise ValueError(f"{target.key}: full training run is not complete")
    if manifest.get("allow_test_evaluation") is not False:
        raise ValueError(f"{target.key}: test gate is already open")
    if manifest.get("metrics_roles") != ["validation"]:
        raise ValueError(f"{target.key}: expected validation-only metrics")
    if manifest.get("role_records", {}).get("test") != "locked_until_best_checkpoint":
        raise ValueError(f"{target.key}: test role is not locked")
    test_paths = (
        run_root / "predictions" / "test.npz",
        run_root / "metrics" / "test.json",
    )
    if any(path.exists() for path in test_paths):
        raise ValueError(f"{target.key}: test artifacts already exist")
    report = validate_run(workspace, run_root)
    if not report.get("passed") or set(report.get("roles", {})) != {"validation"}:
        raise ValueError(f"{target.key}: independent validation did not pass")
    stored_report = _read_json(report_path)
    portable_report = json.loads(json.dumps(report))
    portable_report["manifest_sha256"] = _text_sha256_lf(manifest_path)
    portable_report["roles"]["validation"]["metrics_sha256"] = _text_sha256_lf(
        run_root / "metrics" / "validation.json"
    )
    if stored_report != portable_report:
        raise ValueError(f"{target.key}: stored validation report is stale")
    return {
        "state": "pending",
        "run_manifest_sha256": _text_sha256_lf(manifest_path),
        "validation_report_sha256": _text_sha256_lf(report_path),
        "validation_prediction_sha256": sha256_file(
            run_root / "predictions" / "validation.npz"
        ),
        "validation_metrics_sha256": _text_sha256_lf(
            run_root / "metrics" / "validation.json"
        ),
        "sequence_checkpoint_sha256": manifest["sequence_checkpoint_sha256"],
        "extractor_sha256": manifest["extractor_sha256"],
    }


def preflight(workspace: Path, seed: int = 42) -> dict[str, Any]:
    """Validate all locked runs without resolving or reading the test partition."""
    workspace = workspace.resolve()
    source_commit = assert_git_state(workspace, seed, resume=False)
    entries: dict[str, Any] = {}
    for target in campaign_targets():
        entries[target.key] = _baseline_entry(workspace, target, seed)
    return {
        "schema_version": CAMPAIGN_SCHEMA_VERSION,
        "status": "prepared",
        "source_git_commit": source_commit,
        "seed": seed,
        "target_count": len(entries),
        "target_order": [target.key for target in campaign_targets()],
        "targets": entries,
    }


def _assert_baseline_unchanged(
    workspace: Path, target: TestTarget, seed: int, entry: dict[str, Any]
) -> None:
    run_root = _run_root(workspace, target, seed)
    prediction_path = run_root / "predictions" / "validation.npz"
    metric_path = run_root / "metrics" / "validation.json"
    if sha256_file(prediction_path) != entry["validation_prediction_sha256"]:
        raise ValueError(
            f"{target.key}: immutable validation artifact changed: {prediction_path.name}"
        )
    if _text_sha256_lf(metric_path) != entry["validation_metrics_sha256"]:
        raise ValueError(
            f"{target.key}: immutable validation artifact changed: {metric_path.name}"
        )


def _load_extractor(context: RunContext) -> tuple[str, Any, str]:
    if context.experiment_id == "E1":
        models, digest = load_cnn15_from_e0(context)
        return "cnn15", models, digest
    if context.experiment_id == "E0":
        models: dict[str, SleepCNN] = {}
        hashes: dict[str, str] = {}
        for index, key in enumerate(expected_15cnn_keys()):
            component_seed = context.seed + index
            model = SleepCNN()
            stage = f"cnn15/{key}"
            checkpoint_dir = context.run_root / "checkpoints" / "cnn15" / key
            if not _stage_is_complete(
                context, checkpoint_dir, stage, component_seed
            ):
                raise ValueError(f"incomplete extractor stage: {stage}")
            path = checkpoint_dir / "best.pt"
            load_model_checkpoint(
                path,
                model,
                expected_metadata={
                    **_checkpoint_metadata(context, model, stage, component_seed),
                    "selection_metric": "validation_loss",
                },
                device=context.device,
            )
            models[key] = model.eval()
            hashes[key] = sha256_file(path)
        return "cnn15", models, combined_sha256(hashes)
    model = EEGResNet1D()
    stage = "resnet1d"
    checkpoint_dir = context.run_root / "checkpoints" / stage
    if not _stage_is_complete(context, checkpoint_dir, stage, context.seed):
        raise ValueError(f"incomplete extractor stage: {stage}")
    path = checkpoint_dir / "best.pt"
    load_model_checkpoint(
        path,
        model,
        expected_metadata={
            **_checkpoint_metadata(context, model, stage, context.seed),
            "selection_metric": "validation_macro_f1",
        },
        device=context.device,
    )
    return "resnet1d", model.eval(), sha256_file(path)


def _load_sequence_model(context: RunContext) -> tuple[str, Any, Path]:
    experiment = context.config["experiments"][context.experiment_id]
    if experiment["sequence_model"] == "bilstm":
        kind = "bilstm"
        cfg = context.config["components"]["bilstm"]
        model = BiLSTMSleepNet(input_dim=75, hidden_dim=cfg["hidden_dim"])
    else:
        kind = "tcn"
        cfg = context.config["components"]["common_tcn"]
        feature_dim = 75 if context.experiment_id == "E1" else 128
        model = SleepTCN(
            input_dim=feature_dim,
            hidden_dim=cfg["hidden_dim"],
            kernel_size=cfg["kernel_size"],
            n_blocks=cfg["residual_blocks"],
            dropout=cfg["dropout"],
        )
    stage = f"sequence/{kind}"
    checkpoint_dir = context.run_root / "checkpoints" / "sequence" / kind
    if not _stage_is_complete(context, checkpoint_dir, stage, context.seed):
        raise ValueError(f"incomplete sequence stage: {stage}")
    path = checkpoint_dir / "best.pt"
    load_model_checkpoint(
        path,
        model,
        expected_metadata={
            **_checkpoint_metadata(context, model, stage, context.seed),
            "selection_metric": "validation_macro_f1",
        },
        device=context.device,
    )
    return kind, model.eval(), path


def _unlocked_manifest(
    manifest: dict[str, Any], record_keys: Iterable[str], campaign: dict[str, Any]
) -> dict[str, Any]:
    updated = dict(manifest)
    updated["role_records"] = dict(manifest["role_records"])
    updated["allow_test_evaluation"] = True
    updated["metrics_roles"] = ["validation", "test"]
    updated["role_records"]["test"] = list(record_keys)
    updated["test_evaluation"] = {
        "campaign_schema_version": CAMPAIGN_SCHEMA_VERSION,
        "source_git_commit": campaign["source_git_commit"],
        "selection": "prelocked_best_checkpoint",
    }
    return updated


def _evaluate_target(
    workspace: Path,
    target: TestTarget,
    seed: int,
    device: str,
    num_workers: int,
    campaign: dict[str, Any],
) -> dict[str, Any]:
    entry = campaign["targets"][target.key]
    _assert_baseline_unchanged(workspace, target, seed, entry)
    context = build_context(
        workspace,
        target.experiment_id,
        target.outer_fold,
        seed,
        device,
        smoke=False,
        allow_test_evaluation=True,
        num_workers=num_workers,
        resume=True,
    )
    manifest_path = context.run_root / "run_manifest.json"
    manifest = _read_json(manifest_path)
    extractor_kind, extractor, extractor_hash = _load_extractor(context)
    if extractor_hash != entry["extractor_sha256"]:
        raise ValueError(f"{target.key}: extractor hash changed")
    sequence_kind, sequence_model, sequence_path = _load_sequence_model(context)
    if sha256_file(sequence_path) != entry["sequence_checkpoint_sha256"]:
        raise ValueError(f"{target.key}: sequence checkpoint hash changed")
    partitions = resolve_fold_partitions(
        workspace / "data" / "processed",
        workspace / "data" / "splits" / "sleepedf_sc_10fold_seed42_v2.json",
        target.outer_fold,
        context.data_variant,
    )
    test_records = _selected_records(partitions.test, context.data_variant, None)
    test_sequences = feature_sequences(
        context,
        test_records,
        "test",
        extractor_kind=extractor_kind,
        extractor=extractor,
        extractor_sha256=extractor_hash,
    )
    _save_role_predictions(
        context,
        sequence_model,
        test_sequences,
        sequence_kind,
        "test",
        sequence_path,
    )
    _write_json_atomic(
        manifest_path,
        _unlocked_manifest(
            manifest,
            (record.info.record_key for record in test_records),
            campaign,
        ),
    )
    report = validate_run(workspace, context.run_root)
    _write_json_atomic(context.run_root / "validation_report.json", report)
    return {
        "state": "complete",
        "test_prediction_sha256": report["roles"]["test"]["prediction_sha256"],
        "test_metrics_sha256": report["roles"]["test"]["metrics_sha256"],
        "test_records": report["roles"]["test"]["records"],
        "test_valid_epochs": report["roles"]["test"]["valid_epochs"],
        "final_manifest_sha256": report["manifest_sha256"],
    }


def _resume_or_recover_target(
    workspace: Path, target: TestTarget, seed: int, entry: dict[str, Any]
) -> bool:
    if entry.get("state") != "complete":
        return False
    run_root = _run_root(workspace, target, seed)
    report = validate_run(workspace, run_root)
    observed = report["roles"]["test"]
    if observed["prediction_sha256"] != entry["test_prediction_sha256"]:
        raise ValueError(f"{target.key}: completed test prediction changed")
    if observed["metrics_sha256"] != entry["test_metrics_sha256"]:
        raise ValueError(f"{target.key}: completed test metrics changed")
    _assert_baseline_unchanged(workspace, target, seed, entry)
    return True


def execute_campaign(
    workspace: Path,
    *,
    seed: int,
    device: str,
    num_workers: int,
    confirmation: str,
    resume: bool,
) -> dict[str, Any]:
    if confirmation != CONFIRMATION_PHRASE:
        raise ValueError(f"confirmation must equal {CONFIRMATION_PHRASE!r}")
    workspace = workspace.resolve()
    path = campaign_path(workspace, seed)
    if resume:
        if not path.is_file():
            raise FileNotFoundError(f"campaign journal does not exist: {path}")
        source_commit = assert_git_state(workspace, seed, resume=True)
        campaign = _read_json(path)
        if campaign.get("schema_version") != CAMPAIGN_SCHEMA_VERSION:
            raise ValueError("unsupported campaign schema")
        if campaign.get("source_git_commit") != source_commit:
            raise ValueError("Git HEAD differs from the prepared test campaign")
    else:
        if path.exists():
            raise FileExistsError(
                f"campaign journal already exists; use --resume: {path}"
            )
        campaign = preflight(workspace, seed)
        campaign["status"] = "running"
        _write_json_atomic(path, campaign)
    for index, target in enumerate(campaign_targets(), start=1):
        entry = campaign["targets"][target.key]
        if _resume_or_recover_target(workspace, target, seed, entry):
            print(f"[{index:02d}/60] {target.key}: da kiem dinh, bo qua", flush=True)
            continue
        print(f"[{index:02d}/60] {target.key}: dang danh gia test", flush=True)
        entry["state"] = "running"
        _write_json_atomic(path, campaign)
        result = _evaluate_target(
            workspace, target, seed, device, num_workers, campaign
        )
        entry.update(result)
        _write_json_atomic(path, campaign)
        print(
            f"[{index:02d}/60] {target.key}: dat ({result['test_valid_epochs']} epoch)",
            flush=True,
        )
    campaign["status"] = "complete"
    _write_json_atomic(path, campaign)
    return campaign
