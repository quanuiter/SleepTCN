import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "gate8_publication", ROOT / "scripts" / "build_gate8_publication_package.py"
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules["gate8_publication"] = MODULE
SPEC.loader.exec_module(MODULE)


class Gate8PublicationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.analysis = MODULE.read_json(ROOT / "runs/v2/gate8/analysis_seed42.json")
        cls.validation = MODULE.read_json(
            ROOT / "runs/v2/gate8/validation_campaign_seed42.json"
        )
        cls.test = MODULE.read_json(ROOT / "runs/v2/gate8/test_campaign_seed42.json")

    def test_locked_gate8_inputs_validate(self) -> None:
        MODULE.validate_gate8_inputs(self.analysis, self.validation, self.test)

    def test_ablation_values_are_exact(self) -> None:
        conditions = {
            row["condition"]: row
            for row in MODULE.ablation_condition_rows(self.analysis)
        }
        comparisons = {
            row["comparison"]: row
            for row in MODULE.ablation_comparison_rows(self.analysis)
        }
        self.assertEqual(set(conditions), {"FULL_CPN", "C", "CP", "CN"})
        self.assertAlmostEqual(
            conditions["FULL_CPN"]["overall_macro_f1"], 0.7802296650249438
        )
        self.assertAlmostEqual(
            comparisons["FULL_CPN-C"]["delta_transition_macro_f1"],
            0.0009525338121707527,
        )
        self.assertLess(comparisons["FULL_CPN-C"]["ci95_low"], 0)
        self.assertGreater(comparisons["FULL_CPN-C"]["ci95_high"], 0)
        self.assertTrue(all(row["holm_p"] == 1.0 for row in comparisons.values()))

    def test_gate8_claims_forbid_percentage_and_equivalence(self) -> None:
        gate7 = MODULE.load_gate7_builder(ROOT)
        gate5 = MODULE.read_json(
            ROOT / "runs/v2/analysis/gate5_paired_results_seed42.json"
        )
        latency = MODULE.read_json(
            ROOT / "runs/v2/analysis/gate6_latency_fold00_seed42.json"
        )
        parameters = MODULE.read_json(
            ROOT / "runs/v2/analysis/gate6_parameters_fold00_seed42.json"
        )
        feature = MODULE.read_json(
            ROOT / "runs/v2/analysis/gate6_feature_space/feature_space_report.json"
        )
        base = gate7.evidence_rows(
            gate7.performance_rows(gate5),
            gate7.comparison_rows(gate5),
            gate7.complexity_rows(latency, parameters),
            gate7.silhouette_rows(feature),
        )
        claims = MODULE.gate8_evidence_rows(
            base, MODULE.ablation_comparison_rows(self.analysis)
        )
        statuses = {row["claim_id"]: row["status"] for row in claims}
        self.assertEqual(len(claims), 12)
        self.assertEqual(statuses["C09"], "not_supported")
        self.assertEqual(statuses["C10"], "withdrawn_unsupported")
        self.assertEqual(statuses["C11"], "not_established")
        prohibited = " ".join(row["prohibited_wording"] for row in claims)
        self.assertIn("12% thông tin", prohibited)
        self.assertIn("tương đương", prohibited)

    def test_manuscript_integration_adds_required_boundaries(self) -> None:
        gate7 = MODULE.load_gate7_builder(ROOT)
        gate5 = MODULE.read_json(
            ROOT / "runs/v2/analysis/gate5_paired_results_seed42.json"
        )
        latency = MODULE.read_json(
            ROOT / "runs/v2/analysis/gate6_latency_fold00_seed42.json"
        )
        parameters = MODULE.read_json(
            ROOT / "runs/v2/analysis/gate6_parameters_fold00_seed42.json"
        )
        feature = MODULE.read_json(
            ROOT / "runs/v2/analysis/gate6_feature_space/feature_space_report.json"
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "draft.md"
            gate7.write_manuscript_draft(
                path,
                gate7.performance_rows(gate5),
                gate7.comparison_rows(gate5),
                gate7.complexity_rows(latency, parameters),
                gate7.silhouette_rows(feature),
            )
            MODULE.integrate_gate8_manuscript(
                path,
                MODULE.ablation_condition_rows(self.analysis),
                MODULE.ablation_comparison_rows(self.analysis),
            )
            text = path.read_text(encoding="utf-8")
        self.assertIn("### 2.6. Ablation nhóm đặc trưng C/P/N", text)
        self.assertIn("không thiết lập tương đương", text)
        self.assertIn("không phải phép đo phần trăm thông tin", text)
        self.assertNotIn("P/N chỉ chứa 12% thông tin", text)


if __name__ == "__main__":
    unittest.main()
