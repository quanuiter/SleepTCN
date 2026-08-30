import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from sleeptcn.io.artifact_audit import audit_artifact_manifest
from sleeptcn.io.canonical import canonicalize_processed
from sleeptcn.io.hashing import sha256_file
from sleeptcn.io.manifest_builder import build_artifact_manifest
from sleeptcn.io.serialization import NPZ_SERIALIZATION_FORMAT, atomic_savez


class IoWrapperTests(unittest.TestCase):
    def test_npz_serialization_contract_is_explicit(self) -> None:
        self.assertEqual(NPZ_SERIALIZATION_FORMAT, "npz_zip_stored_v1")

    def test_canonicalize_check_and_rewrite(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            processed = root / "processed" / "paper_raw_v1"
            processed.mkdir(parents=True)
            path = processed / "SC4001E.npz"
            np.savez_compressed(path, value=np.arange(3, dtype=np.int16))
            checked = canonicalize_processed(
                root / "processed", ["paper_raw_v1"], rewrite=False
            )
            self.assertEqual(checked["drifted"], [path])
            rewritten = canonicalize_processed(
                root / "processed", ["paper_raw_v1"], rewrite=True
            )
            self.assertEqual(rewritten["remaining"], [])
            self.assertEqual(
                canonicalize_processed(
                    root / "processed", ["paper_raw_v1"], rewrite=False
                )["drifted"],
                [],
            )

    def test_manifest_builder_and_audit_share_portable_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            processed = root / "data" / "processed" / "paper_raw_v1"
            processed.mkdir(parents=True)
            artifact = processed / "SC4001E.npz"
            atomic_savez(artifact, {"value": np.arange(3, dtype=np.int16)})
            preprocess_manifest = root / "data" / "preprocess.json"
            preprocess_manifest.parent.mkdir(parents=True, exist_ok=True)
            preprocess_manifest.write_text(
                json.dumps(
                    {
                        "dataset": "fixture",
                        "config": {"version": 1},
                        "records": [
                            {
                                "variant": "paper_raw_v1",
                                "record_key": "SC4001E",
                                "subject_id": "SC400",
                                "output_sha256": sha256_file(artifact),
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            report = build_artifact_manifest(
                root / "data" / "processed",
                [preprocess_manifest],
                workspace=root,
                expected_records=1,
            )
            self.assertEqual(report["summary"]["errors"], [])
            manifest_path = root / "manifest.json"
            manifest_path.write_text(
                json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            audit = audit_artifact_manifest(manifest_path, root)
            self.assertTrue(audit["summary"]["passed"])
            self.assertEqual(audit["summary"]["files_checked"], 1)


if __name__ == "__main__":
    unittest.main()
