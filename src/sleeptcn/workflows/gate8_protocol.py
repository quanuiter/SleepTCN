"""Filesystem-only contract for the locked Gate-8 protocol."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..io.hashing import sha256_file
from ..io.serialization import read_json
from .context_ablation import CONDITIONS, GROUP_SLICES


GATE8_CONFIG = Path("configs/gate8_context_ablation.json")
SPLIT_PATH = Path("data/splits/sleepedf_sc_10fold_seed42_v2.json")
UNLOCK_CONFIRMATION = "OPEN-GATE8-LOCKED-TEST-ONCE"


def load_protocol(workspace: Path) -> tuple[dict[str, Any], str]:
    """Load and validate the preregistered Gate-8 protocol and its hash."""

    path = workspace.resolve() / GATE8_CONFIG
    protocol = read_json(path)
    expected = {
        "schema_version": 1,
        "status": "preregistered_before_gate8_training",
        "gate": "GATE_8_CONTEXT_GROUP_ABLATION",
        "source_experiment": "E1",
        "source_extractor_experiment": "E0",
        "seed": 42,
        "outer_folds": 10,
    }
    mismatches = {
        key: (protocol.get(key), value)
        for key, value in expected.items()
        if protocol.get(key) != value
    }
    if mismatches:
        raise ValueError(f"Gate 8 protocol mismatch: {mismatches}")
    if tuple(protocol.get("conditions", {})) != CONDITIONS:
        raise ValueError("Gate 8 condition order must be CP, CN, C")
    if protocol["feature_contract"].get("group_slices") != {
        key: list(value) for key, value in GROUP_SLICES.items()
    }:
        raise ValueError("Gate 8 feature slices differ from the frozen contract")
    if protocol["test_gate"].get("required_completed_validation_runs") != 30:
        raise ValueError("Gate 8 must lock exactly 30 validation runs")
    if protocol["test_gate"].get("unlock_confirmation") != UNLOCK_CONFIRMATION:
        raise ValueError("Gate 8 test confirmation phrase changed")
    return protocol, sha256_file(path)


__all__ = [
    "CONDITIONS",
    "GATE8_CONFIG",
    "GROUP_SLICES",
    "SPLIT_PATH",
    "UNLOCK_CONFIRMATION",
    "load_protocol",
]
