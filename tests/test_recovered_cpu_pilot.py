import importlib.util
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location(
    "recovered_pilot", Path(__file__).resolve().parents[1] / "scripts/run_recovered_e3_cpu_pilot.py")
pilot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pilot)


def test_evaluation_mask_uses_absolute_full_record_indices():
    np.testing.assert_array_equal(pilot.benchmark_positions(np.arange(20, 30), 40), np.arange(20, 30))
    for invalid in [np.array([-1, 0]), np.arange(39, 41), np.array([2, 4]), np.array([2., 3.])]:
        with pytest.raises(ValueError, match="benchmark"):
            pilot.benchmark_positions(invalid, 40)


def test_adaptation_and_test_roles_must_be_disjoint():
    subjects = [{"subject_id": str(i), "role": "adaptation" if i < 5 else "test",
                 "edf_filename": f"synthetic-{i}.edf"} for i in range(185)]
    selection = {"dataset": "SHHS Visit 1", "selection_seed": 42, "subjects": subjects}
    audit = {"status": "passed", "manifest_sha256": "x",
             "subjects": {s["subject_id"]: {**s, "passed": True} for s in subjects}}
    assert len(pilot.checked_roles(selection, audit, "x")["test"]) == 180
    subjects[5]["subject_id"] = subjects[0]["subject_id"]
    with pytest.raises(ValueError, match="disjoint"):
        pilot.checked_roles(selection, audit, "x")


def test_paired_scoring_distinguishes_n3_recall_and_false_positives():
    cm = np.stack([np.eye(5, dtype=int) * 10 for _ in range(3)])
    changed = cm.copy()
    changed[:, 2, 2] -= 2
    changed[:, 2, 3] += 2
    result = pilot.summarize_confusions({"raw": cm, "calibrated": cm, "calibrated_em": changed})
    metrics = result["comparisons"]["calibrated_em_minus_calibrated"]["metrics"]
    assert metrics["n3_recall"]["difference"] == 0
    assert metrics["n3_precision"]["difference"] < 0
    assert metrics["n2_to_n3_rate"]["difference"] > 0
    assert result["comparisons"]["calibrated_minus_raw"]["subject_mean_macro_f1"]["ci95"] == [0., 0.]
    changed[0, 3, 3] += 1
    with pytest.raises(ValueError, match="support"):
        pilot.summarize_confusions({"raw": cm, "calibrated": cm, "calibrated_em": changed})
