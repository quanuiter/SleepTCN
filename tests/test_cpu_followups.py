import numpy as np
import pytest

from sleeptcn.cpu_followups import (
    confusion_diagnostics, label_free_signal_variants, paired_diagnostic_bootstrap,
)
from sleeptcn.shhs_preprocessing import SHHSPreprocessConfig


def test_confusion_orientation_and_missing_stage():
    cm = np.zeros((5, 5), dtype=int)
    cm[3, 3], cm[3, 2], cm[2, 3], cm[2, 2] = 2, 8, 1, 9
    metrics = confusion_diagnostics(cm)
    assert metrics["n3_recall"] == .2
    assert metrics["n3_precision"] == pytest.approx(2/3)
    assert metrics["n3_to_n2_rate"] == .8
    assert metrics["n2_to_n3_rate"] == .1
    empty = confusion_diagnostics(np.zeros((5, 5)))
    assert np.isnan(empty["n3_recall"]) and empty["n3_f1"] == 0


def test_paired_cluster_bootstrap_identity_and_support_guard():
    cm = np.stack([np.eye(5, dtype=int), 2 * np.eye(5, dtype=int)])
    result = paired_diagnostic_bootstrap(cm, cm, resamples=50)
    for metric in result["metrics"].values():
        assert metric["difference"] == 0 and metric["ci95"] == [0., 0.]
    wrong = cm.copy()
    wrong[0, 3, 3] += 1
    with pytest.raises(ValueError, match="support"):
        paired_diagnostic_bootstrap(cm, wrong)


def test_full_signal_preparation_uses_all_epochs_and_scale_relation():
    cfg = SHHSPreprocessConfig()
    time = np.arange(3 * cfg.source_samples_per_epoch) / cfg.source_sampling_rate_hz
    signal = 40 * np.sin(2 * np.pi * 2 * time)
    result = label_free_signal_variants(signal, cfg)
    e3, clip, _ = result["filtered_v2"]
    e4, _, _ = result["bandpass_v2"]
    assert e3.shape == (3, 3000) and e3.dtype == np.float32 and clip == 0
    np.testing.assert_allclose(e3.astype(float) * 100, e4, rtol=3e-7, atol=2e-5)
    with pytest.raises(ValueError, match="divisible"):
        label_free_signal_variants(signal[:-1], cfg)
