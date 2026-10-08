import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location("weighted_recovery_test", Path(__file__).resolve().parents[1] / "scripts/resume_weighted_cpu_pilot_evaluation.py")
recovery = importlib.util.module_from_spec(spec)
spec.loader.exec_module(recovery)


def test_recovery_rejects_misaligned_or_wrong_checkpoint_predictions(tmp_path):
    path = tmp_path / "predictions.npz"
    entry = {"subject_id": "synthetic", "epochs": 3}
    selections = {"unweighted": {"hash": "a"}, "weighted": {"hash": "b"}}
    values = dict(unweighted=np.full((3, 5), .2), weighted=np.full((3, 5), .2),
                  original_epoch_index=np.arange(3), metadata_json=np.array(json.dumps({
                      "subject_id": "synthetic", "selected_checkpoints": selections})))
    np.savez(path, **values)
    recovery.validate_prediction(path, entry, selections)
    with pytest.raises(ValueError, match="identity/selection"):
        recovery.validate_prediction(path, entry, {"unweighted": {"hash": "changed"}})
    values["original_epoch_index"] = np.arange(1, 4)
    np.savez(path, **values)
    with pytest.raises(ValueError, match="indices"):
        recovery.validate_prediction(path, entry, selections)
    values["original_epoch_index"] = np.arange(3)
    values["weighted"][0, 0] = np.nan
    np.savez(path, **values)
    with pytest.raises(ValueError, match="probabilities"):
        recovery.validate_prediction(path, entry, selections)
