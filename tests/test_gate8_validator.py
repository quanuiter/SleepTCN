import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "gate8_validator", ROOT / "scripts" / "validate_gate8_artifacts.py"
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules["gate8_validator"] = MODULE
SPEC.loader.exec_module(MODULE)


class Gate8ValidatorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.package = ROOT / "runs/v2/publication/gate8"

    def test_current_package_passes(self) -> None:
        report = MODULE.validate(self.package)
        self.assertEqual(report["status"], "passed")
        self.assertEqual(report["tables"]["gate8_conditions"], 4)
        self.assertEqual(report["claims"]["claims"], 12)
        self.assertEqual(len(report["figures"]["png_bytes"]), 4)

    def test_hash_tampering_is_detected(self) -> None:
        import shutil

        with tempfile.TemporaryDirectory() as temporary:
            copied = Path(temporary) / "gate8"
            shutil.copytree(self.package, copied)
            path = copied / "TABLES.md"
            path.write_text(path.read_text(encoding="utf-8") + "tampered\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                MODULE.validate_manifest(copied)

    def test_equivalence_claim_tampering_is_detected_after_rehash(self) -> None:
        import hashlib
        import json
        import shutil

        with tempfile.TemporaryDirectory() as temporary:
            copied = Path(temporary) / "gate8"
            shutil.copytree(self.package, copied)
            evidence_path = copied / "claim_evidence_matrix.json"
            evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
            evidence[10]["status"] = "supported"
            evidence_path.write_text(
                json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            manifest_path = copied / "publication_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["output_sha256"]["evidence_json"] = hashlib.sha256(
                evidence_path.read_bytes()
            ).hexdigest()
            manifest_path.write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "claim status mismatch"):
                MODULE.validate_claims(copied)


if __name__ == "__main__":
    unittest.main()
