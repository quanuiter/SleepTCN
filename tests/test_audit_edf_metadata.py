import unittest
from pathlib import Path

from scripts.audit_edf_metadata import is_integer_multiple, record_key


class RecordKeyTests(unittest.TestCase):
    def test_valid_record_keys(self) -> None:
        cases = {
            "SC4001E0-PSG.edf": "SC4001E",
            "SC4001EC-Hypnogram.edf": "SC4001E",
            "SC4362F0-PSG.edf": "SC4362F",
        }
        for name, expected in cases.items():
            with self.subTest(name=name):
                self.assertEqual(record_key(Path(name)), expected)

    def test_rejects_non_sc_file(self) -> None:
        with self.assertRaises(ValueError):
            record_key(Path("README.txt"))


class MultipleTests(unittest.TestCase):
    def test_accepts_valid_values(self) -> None:
        for value in (30.0, 60.0, 90.0, 3000.0):
            with self.subTest(value=value):
                self.assertTrue(is_integer_multiple(value, 30.0))

    def test_rejects_invalid_values(self) -> None:
        for value in (0.0, -30.0, 29.0, 30.5, 61.0):
            with self.subTest(value=value):
                self.assertFalse(is_integer_multiple(value, 30.0))


if __name__ == "__main__":
    unittest.main()

