from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from sleeptcn.workflows.shhs_protocol import (
    EXPERIMENT_VARIANTS,
    FOLDS,
    input_entries,
    load_inventory,
    load_preprocess_manifest,
    load_protocol,
)


def test_locked_protocol_preserves_raw_file_digest() -> None:
    path = Path(__file__).resolve().parents[1] / "configs" / "shhs_zero_shot_v1.json"
    protocol, observed = load_protocol(path)
    expected = hashlib.sha256(path.read_bytes()).hexdigest()

    assert observed == expected
    assert tuple(protocol["experiments"]) == tuple(EXPERIMENT_VARIANTS)
    assert protocol["checkpoint_policy"]["outer_folds"] == list(FOLDS)


def test_inventory_and_preprocess_manifest_bind_to_protocol_hash(tmp_path: Path) -> None:
    inventory_path = tmp_path / "inventory.json"
    inventory_raw = json.dumps(
        {
            "status": "passed",
            "protocol_sha256": "protocol-digest",
            "summary": {"best_checkpoints": 0},
        },
        indent=2,
    ).encode("utf-8")
    inventory_path.write_bytes(inventory_raw)
    inventory, inventory_digest = load_inventory(
        inventory_path, "protocol-digest", expected_best_checkpoints=0
    )
    assert inventory["status"] == "passed"
    assert inventory_digest == hashlib.sha256(inventory_raw).hexdigest()

    manifest_path = tmp_path / "manifest.json"
    manifest_raw = json.dumps(
        {"status": "complete", "scope": "primary", "records": []}, indent=2
    ).encode("utf-8")
    manifest_path.write_bytes(manifest_raw)
    manifest, manifest_digest = load_preprocess_manifest(
        manifest_path,
        {"preprocessing_provenance": {"manifest_sha256": hashlib.sha256(manifest_raw).hexdigest()}},
    )
    assert manifest["scope"] == "primary"
    assert manifest_digest == hashlib.sha256(manifest_raw).hexdigest()

    with pytest.raises(ValueError, match="differs from zero-shot protocol"):
        load_preprocess_manifest(
            manifest_path, {"preprocessing_provenance": {"manifest_sha256": "wrong"}}
        )


def test_input_entries_are_sorted_and_require_locked_cardinality() -> None:
    records = [
        {"record_key": f"subject_{index:02d}", "role": "validation", "variant": "paper_raw_v1"}
        for index in reversed(range(15))
    ]
    observed = input_entries({"records": records}, "validation", "paper_raw_v1")
    assert [entry["record_key"] for entry in observed] == [f"subject_{i:02d}" for i in range(15)]

    with pytest.raises(ValueError, match="Expected 180 test"):
        input_entries({"records": records}, "test", "paper_raw_v1")

