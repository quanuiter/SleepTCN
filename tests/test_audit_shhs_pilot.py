from __future__ import annotations

import importlib.util
import xml.etree.ElementTree as ET
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "audit_shhs_pilot.py"
SPEC = importlib.util.spec_from_file_location("audit_shhs_pilot", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_profusion_parser_reads_epoch_and_stage_domain(tmp_path: Path) -> None:
    root = ET.Element("CMPStudyConfig")
    ET.SubElement(root, "EpochLength").text = "30"
    sleep_stages = ET.SubElement(root, "SleepStages")
    for stage in (0, 1, 2, 3, 4, 5, 6, 9):
        ET.SubElement(sleep_stages, "SleepStage").text = str(stage)
    path = tmp_path / "sample.xml"
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)

    result = MODULE.read_profusion_xml(path)

    assert result["epoch_seconds"] == 30
    assert result["stage_count"] == 8
    assert result["duration_seconds"] == 240
    assert set(result["stages"]) == MODULE.ALLOWED_RAW_STAGES


def test_expected_pilot_rejects_test_subject() -> None:
    subjects = [
        {
            "subject_id": str(200000 + index),
            "pilot": True,
            "role": "adaptation" if index < 5 else "validation",
        }
        for index in range(10)
    ]
    subjects[-1]["role"] = "test"
    manifest = {
        "dataset": "SHHS Visit 1",
        "selection_seed": 42,
        "subjects": subjects,
    }

    try:
        MODULE.expected_pilot(manifest)
    except ValueError as exc:
        assert "adaptation/validation only" in str(exc)
    else:
        raise AssertionError("test subject was incorrectly accepted in pilot")


def test_selected_scope_requires_locked_role_counts() -> None:
    roles = (
        [("adaptation", 5), ("validation", 15), ("test", 180), ("reserve", 20)]
    )
    subjects = []
    subject_id = 200000
    for role, count in roles:
        for _ in range(count):
            subjects.append(
                {"subject_id": str(subject_id), "pilot": False, "role": role}
            )
            subject_id += 1
    manifest = {
        "dataset": "SHHS Visit 1",
        "selection_seed": 42,
        "subjects": subjects,
    }

    selected = MODULE.expected_subjects(manifest, "selected")

    assert len(selected) == 220
