import unittest

from sleeptcn.seed_sensitivity import compare_seed_reports


EXPERIMENTS = ("E0", "E1", "E2", "E3", "E4", "E6")


def report(seed: int, scale: float) -> dict:
    metrics = {
        experiment: {
            "macro_f1": 0.75 + index * 0.005 * scale,
            "accuracy": 0.8 + index * 0.002 * scale,
            "cohen_kappa": 0.7 + index * 0.002 * scale,
        }
        for index, experiment in enumerate(EXPERIMENTS)
    }

    def row(comparison: str, family_size: int) -> dict:
        proposed, reference = comparison.split("-")
        effect = metrics[proposed]["macro_f1"] - metrics[reference]["macro_f1"]
        return {
            "comparison": comparison,
            "proposed": proposed,
            "reference": reference,
            "descriptive": {
                "proposed": metrics[proposed],
                "reference": metrics[reference],
            },
            "cluster_bootstrap_macro_f1": {
                "observed_difference": effect,
                "ci95_low": effect - 0.001,
                "ci95_high": effect + 0.001,
            },
            "subject_wilcoxon_macro_f1": {
                "p_value": 0.01,
                "holm_adjusted_p_value": 0.04 if family_size else None,
                "holm_family_size": family_size,
                "wins": 50,
                "ties": 0,
                "losses": 28,
            },
        }

    return {
        "schema_version": 2,
        "status": "complete",
        "seed": seed,
        "provenance": {"split_sha256": "a" * 64, "config_sha256": "b" * 64},
        "input_coverage": {
            experiment: {"subjects": 78, "records": 153, "valid_epochs": 195469}
            for experiment in EXPERIMENTS
        },
        "primary_results": [
            row(comparison, 4)
            for comparison in ("E1-E0", "E2-E1", "E3-E2", "E3-E6")
        ],
        "secondary_results": [row("E4-E2", 0)],
    }


class SeedSensitivityTests(unittest.TestCase):
    def test_two_seed_summary_preserves_separate_inference(self) -> None:
        result = compare_seed_reports({42: report(42, 1.0), 123: report(123, 0.8)})
        self.assertEqual(result["seeds"], [42, 123])
        self.assertEqual(result["seed_count"], 2)
        comparison = result["comparisons"]["E3-E6"]
        self.assertTrue(comparison["same_nonzero_direction_in_all_seeds"])
        self.assertEqual(comparison["holm_significant_seeds"], [42, 123])
        self.assertIn("No p-values are pooled", result["statistical_boundary"])

    def test_split_mismatch_is_rejected(self) -> None:
        first = report(42, 1.0)
        second = report(123, 1.0)
        second["provenance"]["split_sha256"] = "c" * 64
        with self.assertRaisesRegex(ValueError, "same split"):
            compare_seed_reports({42: first, 123: second})


if __name__ == "__main__":
    unittest.main()
