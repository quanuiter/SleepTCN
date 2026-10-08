import importlib.util
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location("input_audit", Path(__file__).resolve().parents[1]
                                           / "scripts/audit_teacher_revision_inputs.py")
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def make_pair(tmp_path):
    e3, e4 = tmp_path / "e3", tmp_path / "e4"
    e3.mkdir()
    e4.mkdir()
    fields = dict(y=np.array([2, 3]), valid_mask=np.ones(2, dtype=bool),
                  original_epoch_index=np.array([8, 9]), source_psg_sha256=np.array("a" * 64))
    signal = np.arange(12, dtype=np.float32).reshape(2, 6)
    np.savez(e3 / "synthetic.npz", x=signal / 100, **fields)
    np.savez(e4 / "synthetic.npz", x=signal, **fields)
    return e3, e4, signal, fields


def test_signal_pair_audit_and_declared_scope(tmp_path):
    e3, e4, _, _ = make_pair(tmp_path)
    result = audit.signal_pairs(e3, e4, 1)
    assert result["records"] == 1 and result["samples_in_saved_windows"] == 12
    assert "not all continuous" in result["scope"]


@pytest.mark.parametrize("fault", ["alignment", "signal", "source", "clipping"])
def test_signal_pair_rejects_mismatched_inputs(tmp_path, fault):
    e3, e4, signal, fields = make_pair(tmp_path)
    if fault == "clipping":
        np.savez(e3 / "synthetic.npz", x=signal / 100, clip_fraction=np.array(0.1), **fields)
    else:
        if fault == "alignment":
            fields["original_epoch_index"] = np.array([9, 10])
        elif fault == "source":
            fields["source_psg_sha256"] = np.array("b" * 64)
        elif fault == "signal":
            signal = signal + 1
        np.savez(e4 / "synthetic.npz", x=signal, **fields)
    with pytest.raises(ValueError):
        audit.signal_pairs(e3, e4, 1)
