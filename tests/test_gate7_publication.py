import importlib.util
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "gate7_publication", ROOT / "scripts" / "build_gate7_publication_package.py"
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules["gate7_publication"] = MODULE
SPEC.loader.exec_module(MODULE)


class Gate7PublicationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.gate5 = MODULE.read_json(
            ROOT / "runs/v2/analysis/gate5_paired_results_seed42.json"
        )
        cls.latency = MODULE.read_json(
            ROOT / "runs/v2/analysis/gate6_latency_fold00_seed42.json"
        )
        cls.parameters = MODULE.read_json(
            ROOT / "runs/v2/analysis/gate6_parameters_fold00_seed42.json"
        )
        cls.feature = MODULE.read_json(
            ROOT / "runs/v2/analysis/gate6_feature_space/feature_space_report.json"
        )
        cls.validation = MODULE.read_json(
            ROOT / "runs/v2/analysis/gate6_validation_report.json"
        )

    def test_locked_inputs_validate(self) -> None:
        MODULE.validate_inputs(
            self.gate5,
            self.latency,
            self.parameters,
            self.feature,
            self.validation,
        )

    def test_performance_and_comparison_values_are_exact(self) -> None:
        performance = {
            row["experiment"]: row for row in MODULE.performance_rows(self.gate5)
        }
        comparisons = {
            row["comparison"]: row for row in MODULE.comparison_rows(self.gate5)
        }
        self.assertEqual(len(performance), 6)
        self.assertAlmostEqual(performance["E3"]["macro_f1"], 0.790443093, places=9)
        self.assertAlmostEqual(
            comparisons["E3-E6"]["holm_p"], 0.001185186103850067
        )
        self.assertGreater(comparisons["E3-E6"]["ci95_low"], 0)
        self.assertGreater(comparisons["E1-E0"]["holm_p"], 0.05)
        self.assertGreater(comparisons["E2-E1"]["holm_p"], 0.05)

    def test_tradeoff_is_not_framed_as_parameter_efficiency(self) -> None:
        complexity = {
            row["experiment"]: row
            for row in MODULE.complexity_rows(self.latency, self.parameters)
        }
        self.assertAlmostEqual(complexity["E2"]["speedup_vs_E0"], 3.7570664754496965)
        self.assertAlmostEqual(
            complexity["E2"]["parameter_ratio_vs_E0"], 4.366238989663355
        )
        self.assertGreater(complexity["E2"]["peak_allocated_ratio_vs_E0"], 1)

    def test_claim_matrix_contains_required_boundaries(self) -> None:
        performance = MODULE.performance_rows(self.gate5)
        comparisons = MODULE.comparison_rows(self.gate5)
        complexity = MODULE.complexity_rows(self.latency, self.parameters)
        silhouettes = MODULE.silhouette_rows(self.feature)
        claims = MODULE.evidence_rows(
            performance, comparisons, complexity, silhouettes
        )
        self.assertEqual(len(claims), 8)
        statuses = {row["claim_id"]: row["status"] for row in claims}
        self.assertEqual(statuses["C02"], "supported")
        self.assertEqual(statuses["C03"], "not_supported")
        self.assertEqual(statuses["C07"], "not_evaluated")
        prohibited = " ".join(row["prohibited_wording"] for row in claims)
        self.assertIn("8,2×", prohibited)
        self.assertIn("zero-shot", prohibited)

    def test_silhouette_is_lower_for_E2_in_all_folds(self) -> None:
        rows = MODULE.silhouette_rows(self.feature)
        self.assertEqual(len(rows), 10)
        self.assertTrue(all(row["E2_minus_E1"] < 0 for row in rows))

    def test_manuscript_uses_measured_speedup_and_scope_boundary(self) -> None:
        import tempfile

        performance = MODULE.performance_rows(self.gate5)
        comparisons = MODULE.comparison_rows(self.gate5)
        complexity = MODULE.complexity_rows(self.latency, self.parameters)
        silhouettes = MODULE.silhouette_rows(self.feature)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "draft.md"
            MODULE.write_manuscript_draft(
                path, performance, comparisons, complexity, silhouettes
            )
            text = path.read_text(encoding="utf-8")
        self.assertIn("3.76 lần", text)
        self.assertIn("4.37 lần số tham số", text)
        self.assertIn("chỉ áp dụng in-domain", text)
        self.assertNotIn("nhanh hơn 8,2", text)


if __name__ == "__main__":
    unittest.main()
