import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from sleeptcn.io.hashing import sha256_file
from sleeptcn.io.processed_validation import validate_processed_dataset
from sleeptcn.io.serialization import atomic_savez
from sleeptcn.io.split_validation import validate_subject_splits


def write_record(path: Path, variant: str, subject: str) -> None:
    labels = np.arange(5, dtype=np.int8)
    atomic_savez(
        path,
        {
            "x": np.zeros((5, 3000), dtype=np.float32),
            "y": labels,
            "valid_mask": np.ones(5, dtype=np.bool_),
            "original_epoch_index": np.arange(5, dtype=np.int32),
            "record_key": np.asarray(path.stem),
            "subject_id": np.asarray(subject),
            "source_psg_sha256": np.asarray("a" * 64),
            "source_hypnogram_sha256": np.asarray("b" * 64),
            "source_hashes_verified": np.asarray(True),
            "channel": np.asarray("EEG Fpz-Cz"),
            "sampling_rate_hz": np.asarray(100.0),
            "epoch_seconds": np.asarray(30),
            "samples_per_epoch": np.asarray(3000),
            "preprocess_version": np.asarray(variant),
            "filter": np.asarray("none"),
            "normalization": np.asarray("none"),
            "trim_anchor_policy": np.asarray("true_sleep_n1_to_rem"),
            "signal_epochs_before_trim": np.asarray(5),
        },
    )


def subject_summary(record_key: str) -> dict[str, object]:
    return {
        "subject_id": record_key[:5],
        "record_keys": [record_key],
        "epochs": 5,
        "valid_epochs": 5,
        "ignored_epochs": 0,
        "label_counts": {"-1": 0, "0": 1, "1": 1, "2": 1, "3": 1, "4": 1},
    }


class ValidationWrapperTests(unittest.TestCase):
    def test_processed_validation_uses_manifest_hash_and_pairing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            processed = root / "processed" / "paper_raw_v1"
            processed.mkdir(parents=True)
            record = processed / "SC4001E.npz"
            write_record(record, "paper_raw_v1", "SC400")
            preprocess = root / "preprocess.json"
            preprocess.write_text(
                json.dumps(
                    {
                        "records": [
                            {
                                "variant": "paper_raw_v1",
                                "record_key": "SC4001E",
                                "output_sha256": sha256_file(record),
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            report = validate_processed_dataset(
                root / "processed",
                [preprocess],
                ["paper_raw_v1"],
                expected_records=1,
                expected_subjects=1,
            )
            self.assertEqual(report["summary"]["files_with_errors"], 0)
            self.assertEqual(report["summary"]["global_errors"], [])

    def test_canonical_snapshot_supersedes_legacy_container_hash(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            processed = root / "processed" / "paper_raw_v1"
            processed.mkdir(parents=True)
            record = processed / "SC4001E.npz"
            write_record(record, "paper_raw_v1", "SC400")
            source = root / "preprocess.json"
            source.write_text(
                json.dumps(
                    {
                        "records": [
                            {
                                "variant": "paper_raw_v1",
                                "record_key": "SC4001E",
                                "output_sha256": "0" * 64,
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            snapshot = root / "processed_artifact_manifest.json"
            snapshot.write_text(
                json.dumps(
                    {
                        "manifest_kind": "processed_artifact_snapshot",
                        "records": [
                            {
                                "variant": "paper_raw_v1",
                                "record_key": "SC4001E",
                                "output_sha256": sha256_file(record),
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            report = validate_processed_dataset(
                root / "processed",
                [source, snapshot],
                ["paper_raw_v1"],
                expected_records=1,
                expected_subjects=1,
            )
            self.assertEqual(report["summary"]["files_with_errors"], 0)

    def test_split_validation_checks_subject_aggregates(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            processed = root / "processed" / "paper_raw_v1"
            processed.mkdir(parents=True)
            write_record(processed / "SC4001E.npz", "paper_raw_v1", "SC400")
            write_record(processed / "SC4011E.npz", "paper_raw_v1", "SC401")
            subjects = [subject_summary("SC4001E"), subject_summary("SC4011E")]
            split = {
                "subjects": subjects,
                "compatible_variants": ["paper_raw_v1"],
                "folds": [
                    {
                        "fold_index": 0,
                        "subject_ids": ["SC400"],
                        "record_keys": ["SC4001E"],
                        "label_counts": {"0": 1, "1": 1, "2": 1, "3": 1, "4": 1},
                    },
                    {
                        "fold_index": 1,
                        "subject_ids": ["SC401"],
                        "record_keys": ["SC4011E"],
                        "label_counts": {"0": 1, "1": 1, "2": 1, "3": 1, "4": 1},
                    },
                ],
                "outer_runs": [
                    {
                        "outer_fold": 0,
                        "train": {"subject_ids": [], "record_keys": []},
                        "validation": {"subject_ids": ["SC401"], "record_keys": ["SC4011E"]},
                        "test": {"subject_ids": ["SC400"], "record_keys": ["SC4001E"]},
                    },
                    {
                        "outer_fold": 1,
                        "train": {"subject_ids": [], "record_keys": []},
                        "validation": {"subject_ids": ["SC400"], "record_keys": ["SC4001E"]},
                        "test": {"subject_ids": ["SC401"], "record_keys": ["SC4011E"]},
                    },
                ],
            }
            split_path = root / "split.json"
            split_path.write_text(json.dumps(split), encoding="utf-8")
            report = validate_subject_splits(split_path, root / "processed")
            self.assertTrue(report["summary"]["passed"])
            self.assertEqual(
                report["summary"]["variants"]["paper_raw_v1"]["aggregate_mismatches"],
                0,
            )


if __name__ == "__main__":
    unittest.main()
