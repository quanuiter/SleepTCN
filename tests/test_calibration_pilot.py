import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location("calibration_pilot", Path(__file__).resolve().parents[1]
                                           / "scripts/run_calibration_pilot.py")
pilot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pilot)


def make_inputs(tmp_path):
    p = np.full((5, 5), 0.05)
    np.fill_diagonal(p, 0.8)
    ids = np.array([f"synthetic-{i}" for i in range(5)])
    np.savez(tmp_path / "source.npz", probabilities=p, labels=np.arange(5), epoch_id=ids)
    np.savez(tmp_path / "target.npz", probabilities=p, epoch_id=ids)
    manifest = dict(schema_version=1, class_order=["W", "N1", "N2", "N3", "REM"],
                    source_role="validation", source_split_sha256="a" * 64,
                    checkpoint_sha256="b" * 64, outer_fold=0,
                    adaptation_window_policy="label_independent", source_validation="source.npz",
                    target_adaptation="target.npz", target_inference="target.npz")
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    return path, p, ids


def test_end_to_end_three_arms_and_no_overwrite(tmp_path):
    manifest, p, ids = make_inputs(tmp_path)
    out = tmp_path / "pilot.npz"
    pilot.run(manifest, out)
    with np.load(out, allow_pickle=False) as z:
        for arm in ["raw", "calibrated", "calibrated_em"]:
            assert z[arm].shape == p.shape
            np.testing.assert_allclose(z[arm].sum(1), 1)
        np.testing.assert_array_equal(z["epoch_id"], ids)
        assert not json.loads(str(z["metadata_json"]))["target_labels_used"]
    with pytest.raises(FileExistsError):
        pilot.run(manifest, out)


def test_target_labels_and_label_based_window_rejected(tmp_path):
    manifest, p, ids = make_inputs(tmp_path)
    np.savez(tmp_path / "target.npz", probabilities=p, labels=np.arange(5), epoch_id=ids)
    with pytest.raises(ValueError, match="target labels are forbidden"):
        pilot.run(manifest, tmp_path / "forbidden.npz")
    data = json.loads(manifest.read_text())
    data["adaptation_window_policy"] = "first_last_true_sleep"
    manifest.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="window"):
        pilot.run(manifest, tmp_path / "forbidden.npz")
    assert not (tmp_path / "forbidden.npz").exists()
