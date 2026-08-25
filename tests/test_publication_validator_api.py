import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from sleeptcn.evaluation.publication import (  # noqa: E402
    GATE7_EXPECTED_CLAIM_STATUS,
    GATE8_EXPECTED_CLAIM_STATUS,
    validate_gate7_claims,
    validate_gate8_claims,
)


def _claim_rows(statuses: dict[str, str]) -> list[dict[str, str]]:
    return [
        {
            "claim_id": claim_id,
            "claim": f"Claim {claim_id}",
            "status": status,
            "evidence": "locked evidence",
            "source": "locked source",
            "allowed_wording": "allowed wording",
            "prohibited_wording": "prohibited wording",
        }
        for claim_id, status in statuses.items()
    ]


class PublicationValidatorApiTests(unittest.TestCase):
    def test_gate7_claim_contract_is_available_from_package(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            package = Path(temporary)
            (package / "claim_evidence_matrix.json").write_text(
                json.dumps(_claim_rows(GATE7_EXPECTED_CLAIM_STATUS)), encoding="utf-8"
            )
            report = validate_gate7_claims(package)
        self.assertEqual(report["claims"], 8)
        self.assertEqual(report["statuses"], GATE7_EXPECTED_CLAIM_STATUS)

    def test_gate8_claim_contract_rejects_status_drift(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            package = Path(temporary)
            rows = _claim_rows(GATE8_EXPECTED_CLAIM_STATUS)
            rows[-1]["status"] = "supported_with_tradeoff"
            (package / "claim_evidence_matrix.json").write_text(
                json.dumps(rows), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "Gate-8 claim status mismatch"):
                validate_gate8_claims(package)

    def test_cli_wrappers_keep_legacy_validate_alias(self) -> None:
        for filename, package_name in (
            ("validate_gate7_artifacts.py", "validate_gate7_package"),
            ("validate_gate8_artifacts.py", "validate_gate8_package"),
        ):
            spec = importlib.util.spec_from_file_location(filename, ROOT / "scripts" / filename)
            assert spec is not None and spec.loader is not None
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            self.assertEqual(module.validate.__name__, package_name)
            self.assertTrue(callable(module.validate_manifest))


if __name__ == "__main__":
    unittest.main()
