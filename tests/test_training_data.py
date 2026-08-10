import tempfile
import unittest
from pathlib import Path

import numpy as np

from sleeptcn.dataset import RecordInfo, SleepRecord
from sleeptcn.training_data import (
    FeatureSequence,
    FeatureSequenceDataset,
    RecordEpochDataset,
    class_counts_from_records,
    resolve_fold_partitions,
)


def fake_record(key: str, values: list[float], labels: list[int]) -> SleepRecord:
    x = np.repeat(np.asarray(values, dtype=np.float32)[:, None], 3000, axis=1)
    y = np.asarray(labels, dtype=np.int8)
    info = RecordInfo(
        path=Path(f"{key}.npz"),
        record_key=key,
        subject_id=key[:5],
        preprocess_version="paper_raw_v1",
        epochs=len(y),
        valid_epochs=int(np.sum(y >= 0)),
        ignored_epochs=int(np.sum(y == -1)),
        label_counts={label: int(np.sum(y == label)) for label in range(-1, 5)},
    )
    return SleepRecord(
        info=info,
        x=x,
        y=y,
        valid_mask=y >= 0,
        original_epoch_index=np.arange(100, 100 + len(y), dtype=np.int32),
    )


class RecordEpochDatasetTests(unittest.TestCase):
    def setUp(self) -> None:
        self.first = fake_record("SC4001E", [10, 11, 12, 13], [0, -1, 2, 4])
        self.second = fake_record("SC4011E", [20, 21, 22], [1, 3, 0])

    def test_previous_uses_real_ignored_neighbor_and_never_previous_record(self) -> None:
        dataset = RecordEpochDataset([self.first, self.second], "previous")
        self.assertEqual(len(dataset), 6)
        reference = dataset.sample_ref(1)
        self.assertEqual(reference.target_epoch_position, 2)
        self.assertEqual(reference.source_epoch_position, 1)
        signal, label = dataset[1]
        self.assertEqual(float(signal[0, 0]), 11.0)
        self.assertEqual(int(label), 2)
        first_second_record = dataset.sample_ref(3)
        self.assertEqual(first_second_record.record_key, "SC4011E")
        self.assertEqual(first_second_record.source_epoch_position, 0)
        self.assertEqual(float(dataset[3][0][0, 0]), 20.0)

    def test_next_never_enters_next_record(self) -> None:
        dataset = RecordEpochDataset([self.first, self.second], "next")
        last_first_record = dataset.sample_ref(2)
        self.assertEqual(last_first_record.target_epoch_position, 3)
        self.assertEqual(last_first_record.source_epoch_position, 3)
        self.assertEqual(float(dataset[2][0][0, 0]), 13.0)

    def test_counts_only_valid_targets(self) -> None:
        counts = class_counts_from_records([self.first, self.second])
        np.testing.assert_array_equal(counts, [2, 1, 1, 1, 1])


class FeatureSequenceTests(unittest.TestCase):
    def test_preserves_timeline_and_metadata(self) -> None:
        sequence = FeatureSequence(
            record_key="SC4001E",
            subject_id="SC400",
            preprocess_version="paper_raw_v1",
            extractor_id="cnn15_fold0_seed42",
            features=np.zeros((3, 75), dtype=np.float32),
            labels=np.array([0, -1, 2], dtype=np.int8),
            original_epoch_index=np.array([100, 101, 102], dtype=np.int32),
        )
        dataset = FeatureSequenceDataset([sequence])
        features, labels = dataset[0]
        self.assertEqual(tuple(features.shape), (3, 75))
        self.assertEqual(labels.tolist(), [0, -1, 2])

    def test_rejects_nonconsecutive_timeline(self) -> None:
        with self.assertRaisesRegex(ValueError, "consecutive"):
            FeatureSequence(
                record_key="SC4001E",
                subject_id="SC400",
                preprocess_version="paper_raw_v1",
                extractor_id="x",
                features=np.zeros((2, 75), dtype=np.float32),
                labels=np.array([0, 1], dtype=np.int8),
                original_epoch_index=np.array([100, 102], dtype=np.int32),
            )


class RealFoldPartitionTests(unittest.TestCase):
    def test_fold_zero_is_exact_and_disjoint(self) -> None:
        root = Path(__file__).resolve().parents[1]
        partitions = resolve_fold_partitions(
            root / "data" / "processed",
            root / "data" / "splits" / "sleepedf_sc_10fold_seed42_v1.json",
            0,
            "paper_raw_v1",
        )
        self.assertEqual(len(partitions.train.subject_ids), 62)
        self.assertEqual(len(partitions.validation.subject_ids), 8)
        self.assertEqual(len(partitions.test.subject_ids), 8)
        self.assertEqual(len(partitions.train.record_keys), 121)
        self.assertEqual(len(partitions.validation.record_keys), 16)
        self.assertEqual(len(partitions.test.record_keys), 16)
        self.assertFalse(
            set(partitions.train.subject_ids) & set(partitions.test.subject_ids)
        )


if __name__ == "__main__":
    unittest.main()
