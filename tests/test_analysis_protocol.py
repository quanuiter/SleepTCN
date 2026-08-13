import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "analyze_paired_results", ROOT / "scripts" / "analyze_paired_results.py"
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class AnalysisProtocolTests(unittest.TestCase):
    def config(self):
        return {
            "statistical_analysis": {
                "primary_metric": "macro_f1",
                "multiple_testing_correction": "holm",
                "primary_comparisons": ["E1-E0", "E2-E1", "E3-E2", "E3-E6"],
            }
        }

    def test_primary_and_secondary_families_are_separate(self) -> None:
        primary, secondary = MODULE.locked_comparisons(self.config())
        self.assertEqual(
            primary,
            (("E1", "E0"), ("E2", "E1"), ("E3", "E2"), ("E3", "E6")),
        )
        self.assertEqual(secondary, (("E4", "E2"),))
        self.assertTrue(set(primary).isdisjoint(secondary))

    def test_changed_primary_family_is_rejected(self) -> None:
        config = self.config()
        config["statistical_analysis"]["primary_comparisons"].append("E4-E2")
        with self.assertRaisesRegex(ValueError, "frozen protocol"):
            MODULE.locked_comparisons(config)

    def test_changed_correction_is_rejected(self) -> None:
        config = self.config()
        config["statistical_analysis"]["multiple_testing_correction"] = "none"
        with self.assertRaisesRegex(ValueError, "Holm"):
            MODULE.locked_comparisons(config)


if __name__ == "__main__":
    unittest.main()
