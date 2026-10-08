import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location(
    "label_free_prepare", Path(__file__).resolve().parents[1]
    / "scripts/prepare_label_free_shhs_adaptation.py")
prepare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare)


def selection_and_audit():
    subjects = [{"subject_id": f"synthetic-{i}", "role": "adaptation",
                 "edf_filename": f"synthetic-{i}.edf"} for i in range(5)]
    raw = json.dumps({"dataset": "SHHS Visit 1", "selection_seed": 42,
                      "subjects": subjects}).encode()
    audit = {"status": "passed", "manifest_sha256": hashlib.sha256(raw).hexdigest(),
             "subjects": {s["subject_id"]: {**s, "passed": True} for s in subjects}}
    return raw, audit


def test_locked_selection_and_hash_guard():
    raw, audit = selection_and_audit()
    assert len(prepare.locked_adaptation_subjects(raw, audit)) == 5
    audit["manifest_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="linked"):
        prepare.locked_adaptation_subjects(raw, audit)


def test_locked_selection_rejects_other_role_or_duplicate():
    raw, audit = selection_and_audit()
    audit["subjects"]["synthetic-0"]["role"] = "test"
    with pytest.raises(ValueError, match="identity"):
        prepare.locked_adaptation_subjects(raw, audit)


def test_output_validator_rejects_labels_and_trimmed_indices(tmp_path):
    path = tmp_path / "synthetic.npz"
    metadata = json.dumps({"role": "adaptation", "uses_sleep_annotations": False,
                           "epoch_selection": "full_record_signal_length_no_labels"})
    fields = dict(x=np.zeros((2, 3000), dtype=np.float32),
                  original_epoch_index=np.arange(2), metadata_json=np.asarray(metadata))
    np.savez(path, **fields)
    assert prepare.validate_output(path, 2)["bytes"] > 0
    np.savez(path, y=np.zeros(2, dtype=int), **fields)
    with pytest.raises(ValueError, match="schema"):
        prepare.validate_output(path, 2)
    fields["original_epoch_index"] = np.array([5, 6])
    np.savez(path, **fields)
    with pytest.raises(ValueError, match="alignment"):
        prepare.validate_output(path, 2)
