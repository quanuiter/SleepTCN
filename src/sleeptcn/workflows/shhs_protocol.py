"""Torch-free contracts for the locked SHHS zero-shot protocol.

The inference runner still owns model loading and prediction.  This module
owns only the filesystem manifests that define which external inputs and
checkpoints may be used.  Hashes are computed from the original JSON bytes on
purpose: the protocol records provenance for the exact file, not for a
re-serialized equivalent document.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


EXPERIMENT_VARIANTS = {
    "E0": "paper_raw_v1",
    "E3": "filtered_v2",
    "E6": "filtered_zscore_v2",
}
FOLDS = tuple(range(10))
TEST_CONFIRMATION = "OPEN-SHHS-ZERO-SHOT-TEST-ONCE"


def _load_json_bytes(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    document = json.loads(raw.decode("utf-8"))
    if not isinstance(document, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return document, hashlib.sha256(raw).hexdigest()


def load_protocol(
    path: Path,
    *,
    experiment_variants: dict[str, str] | None = None,
    expected_status: str = "locked_before_validation_inference",
) -> tuple[dict[str, Any], str]:
    """Load and validate the immutable SHHS zero-shot protocol."""

    selected_variants = EXPERIMENT_VARIANTS if experiment_variants is None else experiment_variants
    protocol, digest = _load_json_bytes(path)
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
    return protocol, digest


def load_inventory(
    path: Path,
    protocol_sha256: str,
    *,
    expected_best_checkpoints: int = 200,
) -> tuple[dict[str, Any], str]:
    """Load the checkpoint inventory bound to a locked protocol."""

    inventory, digest = _load_json_bytes(path)
    if inventory.get("status") != "passed":
        raise ValueError("Checkpoint inventory has not passed")
    if inventory.get("protocol_sha256") != protocol_sha256:
        raise ValueError("Checkpoint inventory points to another protocol")
    if inventory.get("summary", {}).get("best_checkpoints") != expected_best_checkpoints:
        raise ValueError(
            "Checkpoint inventory does not contain "
            f"{expected_best_checkpoints} best checkpoint references"
        )
    return inventory, digest


def load_preprocess_manifest(
    path: Path, protocol: dict[str, Any]
) -> tuple[dict[str, Any], str]:
    """Load the primary SHHS preprocessing manifest and verify its raw hash."""

    manifest, digest = _load_json_bytes(path)
    expected = protocol["preprocessing_provenance"]["manifest_sha256"]
    if digest != expected:
        raise ValueError("Preprocess manifest SHA-256 differs from zero-shot protocol")
    if manifest.get("status") != "complete" or manifest.get("scope") != "primary":
        raise ValueError("Primary preprocessing manifest is not complete")
    return manifest, digest


def input_entries(
    preprocess_manifest: dict[str, Any], role: str, variant: str
) -> list[dict[str, Any]]:
    """Return the locked, deterministically ordered records for one role/variant."""

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


__all__ = [
    "EXPERIMENT_VARIANTS",
    "FOLDS",
    "TEST_CONFIRMATION",
    "input_entries",
    "load_inventory",
    "load_preprocess_manifest",
    "load_protocol",
]
