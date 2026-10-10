"""End-to-end check of the SHHS1 holdout audit and preprocessing scripts on synthetic files."""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pyedflib
import pytest

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs" / "shhs_holdout_confirmatory_v1.json"
CONFIG = ROOT / "configs" / "shhs_preprocessing_v1.json"


def write_edf(path: Path, epochs: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    fs = 125
    t = np.arange(epochs * 30 * fs) / fs
    signal = 40 * np.sin(2 * np.pi * 1.5 * t) + 15 * rng.standard_normal(t.size)
    writer = pyedflib.EdfWriter(str(path), 1, file_type=pyedflib.FILETYPE_EDF)
    try:
        writer.setSignalHeaders([
            {
                "label": "EEG",
                "dimension": "uV",
                "sample_frequency": fs,
                "physical_min": -250.0,
                "physical_max": 250.0,
                "digital_min": -32768,
                "digital_max": 32767,
                "transducer": "",
                "prefilter": "",
            }
        ])
        writer.writeSamples([signal])
    finally:
        writer.close()


def write_xml(path: Path, stages: list[int]) -> None:
    body = "".join(f"<SleepStage>{s}</SleepStage>" for s in stages)
    path.write_text(
        "<?xml version='1.0' encoding='UTF-8'?><CMPStudyConfig><EpochLength>30</EpochLength>"
        f"<SleepStages>{body}</SleepStages></CMPStudyConfig>",
        encoding="utf-8",
    )


def make_dataset(tmp_path: Path, cohort: list[str], spares: list[str], broken: str | None) -> dict[str, Path]:
    edf_dir = tmp_path / "edfs"
    xml_dir = tmp_path / "xml"
    edf_dir.mkdir()
    xml_dir.mkdir()
    stages = [0] * 70 + [1] * 5 + [2] * 20 + [3] * 10 + [4] * 5 + [5] * 10 + [2] * 10 + [0] * 70
    for index, sid in enumerate(cohort + spares):
        write_edf(edf_dir / f"shhs1-{sid}.edf", len(stages), index)
        if sid != broken:
            write_xml(xml_dir / f"shhs1-{sid}-profusion.xml", stages)

    rows = [
        {
            "subject_id": sid,
            "selection_rank": 221 + i,
            "batch": 1,
            "overall_shhs1": 5,
            "gender": 1,
            "age_s1": 60,
            "edf_filename": f"shhs1-{sid}.edf",
            "annotation_filename": f"shhs1-{sid}-profusion.xml",
            "status": "included",
        }
        for i, sid in enumerate(cohort)
    ]
    csv_path = tmp_path / "cohort.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    protocol["cohort"]["n_target"] = len(cohort)
    protocol["cohort"]["final_list_sha256"] = hashlib.sha256(csv_path.read_bytes()).hexdigest()
    protocol["cohort"]["spares_in_rank_order"] = spares
    protocol_path = tmp_path / "protocol.json"
    protocol_path.write_text(json.dumps(protocol), encoding="utf-8")
    return {"edf": edf_dir, "xml": xml_dir, "csv": csv_path, "protocol": protocol_path}


def run(script: str, *args: str | Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(ROOT / "scripts" / script), *map(str, args)],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )


@pytest.mark.parametrize("broken", [None, "900002"])
def test_holdout_audit_and_preprocessing(tmp_path: Path, broken: str | None) -> None:
    cohort = ["900001", "900002", "900003"]
    spares = ["900009"]
    paths = make_dataset(tmp_path, cohort, spares, broken)
    audit_path = tmp_path / "audit.json"
    result = run(
        "audit_shhs_holdout.py",
        "--protocol", paths["protocol"], "--cohort-csv", paths["csv"],
        "--edf-dir", paths["edf"], "--xml-dir", paths["xml"],
        "--output", audit_path, "--workers", "2",
    )
    assert result.returncode == 0, result.stdout + result.stderr
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    expected = ["900001", "900009", "900003"] if broken else cohort
    assert audit["analysis_cohort"] == expected
    assert audit["replacements"] == ([{"replaced": "900002", "by": "900009"}] if broken else [])

    manifest_path = tmp_path / "preprocess.json"
    out_root = tmp_path / "processed"
    result = run(
        "preprocess_shhs_holdout.py",
        "--config", CONFIG, "--protocol", paths["protocol"], "--technical-audit", audit_path,
        "--edf-dir", paths["edf"], "--xml-dir", paths["xml"],
        "--output-root", out_root, "--output-manifest", manifest_path, "--workers", "2",
    )
    assert result.returncode == 0, result.stdout + result.stderr
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "complete"
    assert manifest["analysis_cohort"] == expected
    assert manifest["summary"]["records_per_variant"] == {
        "paper_raw_v1": 3, "filtered_v2": 3, "filtered_zscore_v2": 3, "bandpass_v2": 3,
    }
    # 60 wake epochs kept on each side of the first/last sleep epoch.
    sid = expected[0]
    with np.load(out_root / "filtered_v2" / f"shhs1-{sid}.npz") as f3, \
         np.load(out_root / "bandpass_v2" / f"shhs1-{sid}.npz") as f4:
        assert f3["x"].shape == (60 + 60 + 60, 3000)
        assert str(f3["role"].item()) == "holdout_test"
        # E3 input equals E4 input divided by 100 because clipping is inactive.
        np.testing.assert_allclose(f3["x"] * 100.0, f4["x"], rtol=1e-5, atol=1e-3)
        np.testing.assert_array_equal(f3["y"], f4["y"])
    assert manifest["summary"]["s6_zscore_input_std_in_window"]["n"] == 3
