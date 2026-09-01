from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from sleeptcn.demo import (
    DemoRecord,
    available_fold_records,
    inspect_edf_demo,
    load_edf_demo_records,
    load_locked_prediction,
    n3_to_n2_mask,
    slice_demo_record,
    stage_transition_mask,
    validate_asset_manifest,
)
from sleeptcn.features import expected_15cnn_keys


def _record(epochs: int = 6) -> DemoRecord:
    return DemoRecord(
        record_key="SC4001E",
        x=np.arange(epochs * 3000, dtype=np.float32).reshape(epochs, 3000),
        labels=np.array([0, 1, 2, 3, 4, -1][:epochs], dtype=np.int8),
        original_epoch_index=np.arange(100, 100 + epochs, dtype=np.int32),
        source="test",
        note="test record",
        data_variant="filtered_v2",
    )


def _asset_manifest(root: Path) -> None:
    def entry(name: str) -> dict[str, str]:
        path = root / "checkpoints" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = f"asset:{name}".encode()
        path.write_bytes(payload)
        return {
            "path": f"checkpoints/{name}",
            "sha256": hashlib.sha256(payload).hexdigest(),
        }

    experiments = {
        "E0": {
            "data_variant": "paper_raw_v1",
            "extractor_kind": "cnn15",
            "sequence_kind": "bilstm",
            "source_run_path": "runs/E0",
            "prediction": entry("E0_prediction.npz"),
            "sequence": entry("E0_sequence.pt"),
            "extractors": {
                key: entry(f"E0_{key}.pt") for key in expected_15cnn_keys()
            },
        },
        "E3": {
            "data_variant": "filtered_v2",
            "extractor_kind": "resnet1d",
            "sequence_kind": "tcn",
            "source_run_path": "runs/E3",
            "prediction": entry("E3_prediction.npz"),
            "sequence": entry("E3_sequence.pt"),
            "extractor": entry("E3_extractor.pt"),
        },
        "E6": {
            "data_variant": "filtered_zscore_v2",
            "extractor_kind": "resnet1d",
            "sequence_kind": "tcn",
            "source_run_path": "runs/E6",
            "prediction": entry("E6_prediction.npz"),
            "sequence": entry("E6_sequence.pt"),
            "extractor": entry("E6_extractor.pt"),
        },
    }
    manifest = {
        "schema_version": 2,
        "source_ref": "run-in-docker",
        "outer_fold": 0,
        "seed": 123,
        "experiments": experiments,
        "test_records": ["SC4001E", "SC4002E"],
    }
    (root / "demo_assets.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_slice_demo_record_preserves_alignment() -> None:
    sliced = slice_demo_record(_record(), start=2, count=3)
    assert sliced.x.shape == (3, 3000)
    assert sliced.labels.tolist() == [2, 3, 4]
    assert sliced.original_epoch_index.tolist() == [102, 103, 104]


def test_slice_demo_record_rejects_out_of_range() -> None:
    with pytest.raises(ValueError, match="invalid demo record slice"):
        slice_demo_record(_record(), start=5, count=2)


def test_validate_asset_manifest_checks_hashes(tmp_path: Path) -> None:
    _asset_manifest(tmp_path)
    manifest = validate_asset_manifest(tmp_path)
    assert manifest["outer_fold"] == 0
    (tmp_path / "checkpoints/E3_extractor.pt").write_bytes(b"changed")
    with pytest.raises(ValueError, match="E3 extractor SHA-256 mismatch"):
        validate_asset_manifest(tmp_path)


def test_available_fold_records_uses_only_manifest_test_records(tmp_path: Path) -> None:
    asset_root = tmp_path / "assets"
    processed_root = tmp_path / "processed"
    asset_root.mkdir()
    processed_root.mkdir()
    _asset_manifest(asset_root)
    for name in ("SC4001E.npz", "SC4999E.npz"):
        (processed_root / name).write_bytes(b"placeholder")
    paths = available_fold_records(asset_root, processed_root)
    assert [path.name for path in paths] == ["SC4001E.npz"]


def test_transition_mask_marks_two_epochs_around_real_change() -> None:
    labels = np.array([3, 3, 3, 2, 2, 2, 4], dtype=np.int8)
    observed = stage_transition_mask(labels, radius=1)
    assert observed.tolist() == [False, False, True, True, True, True, True]


def test_n3_to_n2_mask_only_selects_requested_confusion() -> None:
    labels = np.array([3, 3, 2, 3, 1], dtype=np.int8)
    predicted = np.array([2, 3, 3, 2, 2], dtype=np.int8)
    assert n3_to_n2_mask(labels, predicted).tolist() == [True, False, False, True, False]


def test_locked_prediction_preserves_ignored_epoch_gap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    record = DemoRecord(
        record_key="SC4001E",
        x=np.zeros((4, 3000), dtype=np.float32),
        labels=np.array([3, -1, 2, 1], dtype=np.int8),
        original_epoch_index=np.array([10, 11, 12, 13], dtype=np.int32),
        source="test",
        note="ignored epoch",
        data_variant="filtered_v2",
    )
    logits = np.array(
        [[0, 0, 3, 1, 0], [0, 0, 4, 1, 0], [0, 5, 0, 0, 0]],
        dtype=np.float32,
    )
    np.savez(
        tmp_path / "prediction.npz",
        record_key=np.array(["SC4001E"] * 3),
        true_label=np.array([3, 2, 1], dtype=np.int8),
        original_epoch_index=np.array([10, 12, 13], dtype=np.int32),
        predicted_label=np.array([2, 2, 1], dtype=np.int8),
        logits=logits,
    )
    manifest = {
        "experiments": {
            "E3": {"prediction": {"path": "prediction.npz", "sha256": "unused"}}
        }
    }
    monkeypatch.setattr(
        "sleeptcn.demo.validate_asset_manifest", lambda _asset_root: manifest
    )
    prediction = load_locked_prediction(tmp_path, "E3", record)
    assert prediction.predicted.tolist() == [2, -1, 2, 1]
    assert prediction.probabilities[1].tolist() == pytest.approx([.2] * 5)


def test_edf_inspection_reports_ready_contract(tmp_path: Path) -> None:
    import pyedflib

    path = tmp_path / "sample.edf"
    writer = pyedflib.EdfWriter(str(path), 1, file_type=pyedflib.FILETYPE_EDFPLUS)
    try:
        writer.setSignalHeaders(
            [
                {
                    "label": "EEG Fpz-Cz",
                    "dimension": "uV",
                    "sample_frequency": 100,
                    "physical_min": -200.0,
                    "physical_max": 200.0,
                    "digital_min": -32768,
                    "digital_max": 32767,
                    "transducer": "",
                    "prefilter": "",
                }
            ]
        )
        seconds = np.arange(3000, dtype=np.float64) / 100.0
        writer.writeSamples([25.0 * np.sin(2 * np.pi * 8.0 * seconds)])
    finally:
        writer.close()
    inspection = inspect_edf_demo(path)
    assert inspection["ready"] is True
    assert inspection["complete_epochs"] == 1
    assert inspection["sampling_rate_hz"] == 100.0
    records = load_edf_demo_records(path)
    assert tuple(records) == ("E0", "E3", "E6")
    assert all(record.x.shape == (1, 3000) for record in records.values())
