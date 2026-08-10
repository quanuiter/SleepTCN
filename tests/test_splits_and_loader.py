import json
import unittest
from pathlib import Path

import numpy as np

from sleeptcn.dataset import inspect_record, load_record, paths_for_role
from sleeptcn.splits import deterministic_folds, validate_split_structure


WORKSPACE = Path(__file__).resolve().parents[1]


class DeterministicFoldTests(unittest.TestCase):
    def test_deterministic_and_complete(self) -> None:
        subjects = [f"S{i:02d}" for i in range(12)]
        first = deterministic_folds(subjects, n_folds=5, seed=42)
        second = deterministic_folds(reversed(subjects), n_folds=5, seed=42)
        self.assertEqual(first, second)
        flat = [subject for fold in first for subject in fold]
        self.assertEqual(set(flat), set(subjects))
        self.assertEqual(len(flat), len(set(flat)))


class RealLoaderTests(unittest.TestCase):
    def test_inspect_both_variants(self) -> None:
        for variant in ("paper_raw_v1", "filtered_v2"):
            path = WORKSPACE / "data" / "processed" / variant / "SC4002E.npz"
            info = inspect_record(path, variant)
            self.assertEqual(info.subject_id, "SC400")
            self.assertEqual(info.record_key, "SC4002E")
            self.assertGreater(info.ignored_epochs, 0)

    def test_load_record_preserves_ignored_epoch(self) -> None:
        path = WORKSPACE / "data" / "processed" / "paper_raw_v1" / "SC4002E.npz"
        record = load_record(path, "paper_raw_v1")
        ignored = np.flatnonzero(record.y == -1)
        self.assertGreater(len(ignored), 0)
        self.assertTrue(np.all(~record.valid_mask[ignored]))
        self.assertEqual(record.x[ignored].shape[1], 3000)

    def test_rejects_variant_mismatch(self) -> None:
        path = WORKSPACE / "data" / "processed" / "paper_raw_v1" / "SC4001E.npz"
        with self.assertRaises(ValueError):
            inspect_record(path, "filtered_v2")


class RealSplitTests(unittest.TestCase):
    def setUp(self) -> None:
        self.path = (
            WORKSPACE / "data" / "splits" / "sleepedf_sc_10fold_seed42_v1.json"
        )
        if not self.path.exists():
            self.skipTest("split manifest has not been generated yet")
        self.manifest = json.loads(self.path.read_text(encoding="utf-8"))

    def test_manifest_has_no_leakage(self) -> None:
        self.assertEqual(validate_split_structure(self.manifest), [])

    def test_two_nights_stay_in_same_fold(self) -> None:
        assignment = {
            subject: fold["fold_index"]
            for fold in self.manifest["folds"]
            for subject in fold["subject_ids"]
        }
        for subject in self.manifest["subjects"]:
            folds = {assignment[subject["subject_id"]] for _ in subject["record_keys"]}
            self.assertEqual(len(folds), 1)

    def test_paths_for_role_match_manifest(self) -> None:
        paths = paths_for_role(
            WORKSPACE / "data" / "processed",
            self.manifest,
            outer_fold=0,
            role="test",
            expected_variant="paper_raw_v1",
        )
        expected = self.manifest["outer_runs"][0]["test"]["record_keys"]
        self.assertEqual([path.stem for path in paths], expected)


if __name__ == "__main__":
    unittest.main()
