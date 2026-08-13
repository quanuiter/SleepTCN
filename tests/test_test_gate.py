import unittest
import tempfile
from pathlib import Path

from sleeptcn.test_gate import (
    ACTIVE_EXPERIMENTS,
    TestTarget,
    _allowed_resume_paths,
    _status_path,
    _text_sha256_lf,
    _unlocked_manifest,
    campaign_targets,
)


class TestGateContractTests(unittest.TestCase):
    def test_json_hash_is_stable_across_lf_and_crlf(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            lf = root / "lf.json"
            crlf = root / "crlf.json"
            lf.write_bytes(b'{\n  "value": 1\n}\n')
            crlf.write_bytes(b'{\r\n  "value": 1\r\n}\r\n')
            self.assertEqual(_text_sha256_lf(lf), _text_sha256_lf(crlf))

    def test_campaign_is_exactly_six_experiments_by_ten_folds(self) -> None:
        targets = campaign_targets()
        self.assertEqual(len(targets), 60)
        self.assertNotIn("E5", ACTIVE_EXPERIMENTS)
        self.assertEqual(len(set(targets)), 60)
        self.assertEqual(targets[0], TestTarget("E0", 0))
        self.assertEqual(targets[-1], TestTarget("E6", 9))

    def test_resume_paths_only_cover_manifest_and_validation_report(self) -> None:
        paths = _allowed_resume_paths(42)
        self.assertEqual(len(paths), 120)
        self.assertIn(
            "runs/v2/full/E3/fold_07/seed_42/run_manifest.json", paths
        )
        self.assertNotIn("src/sleeptcn/experiment.py", paths)

    def test_git_porcelain_path_is_parsed(self) -> None:
        self.assertEqual(
            _status_path(" M runs/v2/full/E0/fold_00/seed_42/run_manifest.json"),
            "runs/v2/full/E0/fold_00/seed_42/run_manifest.json",
        )

    def test_unlock_preserves_training_provenance(self) -> None:
        original = {
            "git_commit": "abc",
            "sequence_checkpoint_sha256": "s" * 64,
            "allow_test_evaluation": False,
            "metrics_roles": ["validation"],
            "role_records": {
                "train": ["train"],
                "validation": ["validation"],
                "test": "locked_until_best_checkpoint",
            },
        }
        campaign = {"source_git_commit": "def"}
        result = _unlocked_manifest(original, ["test-a", "test-b"], campaign)
        self.assertEqual(result["git_commit"], "abc")
        self.assertEqual(result["sequence_checkpoint_sha256"], "s" * 64)
        self.assertTrue(result["allow_test_evaluation"])
        self.assertEqual(result["metrics_roles"], ["validation", "test"])
        self.assertEqual(result["role_records"]["test"], ["test-a", "test-b"])
        self.assertEqual(
            original["role_records"]["test"], "locked_until_best_checkpoint"
        )


if __name__ == "__main__":
    unittest.main()
