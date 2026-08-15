from __future__ import annotations

import csv
import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "select_shhs_subjects.py"
SPEC = importlib.util.spec_from_file_location("select_shhs_subjects", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def write_metadata(path: Path, rows: int = 300) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["nsrrid", "overall_shhs1", "gender", "age_s1"],
        )
        writer.writeheader()
        for index in range(rows):
            writer.writerow(
                {
                    "nsrrid": str(200001 + index),
                    "overall_shhs1": str(3 + index % 5),
                    "gender": str(index % 2),
                    "age_s1": str(40 + index % 50),
                }
            )


def test_selection_is_deterministic_and_has_locked_counts(tmp_path: Path) -> None:
    metadata = tmp_path / "shhs1.csv"
    write_metadata(metadata)
    eligible, audit = MODULE.read_eligible_subjects(
        metadata, "nsrrid", "overall_shhs1"
    )
    first = MODULE.assign_subjects(eligible, 42)
    second = MODULE.assign_subjects(eligible, 42)

    assert first == second
    assert audit["eligible_rows"] == 300
    assert len(first) == 220
    assert sum(item["role"] == "adaptation" for item in first) == 5
    assert sum(item["role"] == "validation" for item in first) == 15
    assert sum(item["role"] == "test" for item in first) == 180
    assert sum(item["role"] == "reserve" for item in first) == 20
    assert sum(bool(item["pilot"]) for item in first) == 10
    assert not any(item["role"] in {"test", "reserve"} and item["pilot"] for item in first)


def test_missing_quality_is_excluded_before_sampling(tmp_path: Path) -> None:
    metadata = tmp_path / "shhs1.csv"
    write_metadata(metadata, rows=221)
    rows = list(csv.DictReader(metadata.open("r", encoding="utf-8")))
    rows[0]["overall_shhs1"] = ""
    with metadata.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    eligible, audit = MODULE.read_eligible_subjects(
        metadata, "nsrrid", "overall_shhs1"
    )
    assert len(eligible) == 220
    assert audit["excluded_missing_quality"] == 1


def test_download_script_bypasses_cmd_batch_parser(tmp_path: Path) -> None:
    metadata = tmp_path / "shhs1.csv"
    write_metadata(metadata)
    eligible, _ = MODULE.read_eligible_subjects(
        metadata, "nsrrid", "overall_shhs1"
    )
    assignments = MODULE.assign_subjects(eligible, 42)
    script = MODULE.powershell_download_script(
        Path(r"E:\research\Dataset\SHHS_v1"), assignments, pilot_only=True
    )
    assert "nsrr.bat" not in script
    assert '$Ruby $Nsrr download' in script
    assert "|" in script
