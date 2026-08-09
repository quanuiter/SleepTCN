import unittest
from pathlib import Path

from sleeptcn.experiment import build_context


class ContextTests(unittest.TestCase):
    def test_smoke_cannot_unlock_test_role(self) -> None:
        root = Path(__file__).resolve().parents[1]
        with self.assertRaisesRegex(ValueError, "must not evaluate"):
            build_context(
                root,
                "E2",
                0,
                42,
                "cpu",
                smoke=True,
                allow_test_evaluation=True,
                num_workers=0,
            )

    def test_full_context_hashes_config_and_split(self) -> None:
        root = Path(__file__).resolve().parents[1]
        context = build_context(
            root,
            "E3",
            9,
            123,
            "cpu",
            smoke=False,
            allow_test_evaluation=False,
            num_workers=0,
        )
        self.assertEqual(context.data_variant, "filtered_v2")
        self.assertEqual(len(context.config_sha256), 64)
        self.assertEqual(len(context.split_sha256), 64)
        self.assertIn("full", context.run_root.parts)


if __name__ == "__main__":
    unittest.main()
