"""Kiem dinh doc lap artifact cua mot lan chay E0-E3."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .io.hashing import combined_sha256, sha256_file
from .evaluation import PredictionTable, load_prediction_table
from .dataset import load_record
from .features import expected_15cnn_keys
from .workflows.provenance import runner_code_sha256


def _prediction_table(path: Path) -> tuple[PredictionTable, dict[str, Any]]:
    return load_prediction_table(path)


def _expected_sequence_checkpoint(run_root: Path, experiment_id: str) -> Path:
    kind = "bilstm" if experiment_id == "E0" else "tcn"
    return run_root / "checkpoints" / "sequence" / kind / "best.pt"


def _extractor_sha256(run_root: Path, experiment_id: str) -> str:
    if experiment_id not in {"E0", "E1"}:
        return sha256_file(run_root / "checkpoints" / "resnet1d" / "best.pt")
    source_root = run_root
    if experiment_id == "E1":
        mode_root = run_root.parents[2]
        source_root = mode_root / "E0" / run_root.parent.name / run_root.name
    hashes = {
        key: sha256_file(
            source_root / "checkpoints" / "cnn15" / key / "best.pt"
        )
        for key in expected_15cnn_keys()
    }
    return combined_sha256(hashes)


def _compare_metrics(observed: dict[str, Any], expected: dict[str, Any]) -> None:
    if observed.keys() != expected.keys():
        raise ValueError("metric keys mismatch")
    for key in observed:
        left, right = observed[key], expected[key]
        if isinstance(left, dict):
            if not isinstance(right, dict):
                raise ValueError(f"metric type mismatch: {key}")
            _compare_metrics(left, right)
        elif isinstance(left, list):
            if left != right:
                raise ValueError(f"metric list mismatch: {key}")
        elif isinstance(left, float):
            if not np.isclose(left, right, rtol=0.0, atol=1e-12):
                raise ValueError(f"metric float mismatch: {key}")
        elif left != right:
            raise ValueError(f"metric value mismatch: {key}")


def _validate_role(
    workspace: Path,
    run_root: Path,
    manifest: dict[str, Any],
    role: str,
) -> dict[str, Any]:
    prediction_path = run_root / "predictions" / f"{role}.npz"
    metric_path = run_root / "metrics" / f"{role}.json"
    if not prediction_path.is_file() or not metric_path.is_file():
        raise FileNotFoundError(f"missing {role} prediction/metric artifact")
    table, metadata = _prediction_table(prediction_path)
    checkpoint_path = _expected_sequence_checkpoint(
        run_root, manifest["experiment_id"]
    )
    expected_metadata = {
        "experiment_id": manifest["experiment_id"],
        "outer_fold": manifest["outer_fold"],
        "seed": manifest["seed"],
        "split_sha256": manifest["split_sha256"],
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "data_variant": manifest["data_variant"],
        "role": role,
        "smoke": manifest["smoke"],
    }
    mismatches = {
        key: (metadata.get(key), value)
        for key, value in expected_metadata.items()
        if metadata.get(key) != value
    }
    if mismatches:
        raise ValueError(f"{role} prediction metadata mismatch: {mismatches}")
    expected_records = manifest["role_records"][role]
    if list(dict.fromkeys(table.record_key.tolist())) != expected_records:
        raise ValueError(f"{role} record order differs from run manifest")
    for record_key in expected_records:
        record = load_record(
            workspace
            / "data"
            / "processed"
            / manifest["data_variant"]
            / f"{record_key}.npz",
            manifest["data_variant"],
        )
        selected = table.record_key == record_key
        valid = record.valid_mask
        if not np.array_equal(
            table.original_epoch_index[selected], record.original_epoch_index[valid]
        ):
            raise ValueError(f"{record_key}: original epoch indices mismatch")
        if not np.array_equal(table.true_label[selected], record.y[valid]):
            raise ValueError(f"{record_key}: true labels mismatch")
        if not np.all(table.subject_id[selected] == record.info.subject_id):
            raise ValueError(f"{record_key}: subject id mismatch")
    metric_payload = json.loads(metric_path.read_text(encoding="utf-8"))
    recomputed = table.metrics()
    _compare_metrics(metric_payload["metrics"], recomputed)
    return {
        "records": len(expected_records),
        "valid_epochs": len(table.true_label),
        "prediction_sha256": sha256_file(prediction_path),
        "metrics_sha256": sha256_file(metric_path),
    }


def validate_run(workspace: Path, run_root: Path) -> dict[str, Any]:
    workspace = workspace.resolve()
    run_root = run_root.resolve()
    manifest_path = run_root / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "complete":
        raise ValueError("run manifest is not complete")
    config_relative = manifest.get("config_path", "configs/experiments_v1.json")
    split_relative = manifest.get(
        "split_path", "data/splits/sleepedf_sc_10fold_seed42_v1.json"
    )
    current_hashes = {
        "config_sha256": sha256_file(workspace / config_relative),
        "split_sha256": sha256_file(workspace / split_relative),
        "runner_code_sha256": runner_code_sha256(workspace),
    }
    for key, value in current_hashes.items():
        if manifest.get(key) != value:
            raise ValueError(f"run manifest {key} differs from current workspace")
    sequence_checkpoint = _expected_sequence_checkpoint(
        run_root, manifest["experiment_id"]
    )
    if sha256_file(sequence_checkpoint) != manifest["sequence_checkpoint_sha256"]:
        raise ValueError("sequence checkpoint hash mismatch")
    if _extractor_sha256(run_root, manifest["experiment_id"]) != manifest[
        "extractor_sha256"
    ]:
        raise ValueError("extractor checkpoint hash mismatch")
    roles = {}
    for role in manifest["metrics_roles"]:
        roles[role] = _validate_role(workspace, run_root, manifest, role)
    if not manifest["allow_test_evaluation"]:
        if manifest["role_records"]["test"] != "locked_until_best_checkpoint":
            raise ValueError("test records were exposed without permission")
        if (run_root / "predictions" / "test.npz").exists():
            raise ValueError("test predictions exist although test was locked")
        if "test" in roles:
            raise ValueError("test metrics exist although test was locked")
    return {
        "schema_version": 1,
        "passed": True,
        "experiment_id": manifest["experiment_id"],
        "outer_fold": manifest["outer_fold"],
        "seed": manifest["seed"],
        "smoke": manifest["smoke"],
        "manifest_sha256": sha256_file(manifest_path),
        "sequence_checkpoint_sha256": sha256_file(sequence_checkpoint),
        "roles": roles,
    }
