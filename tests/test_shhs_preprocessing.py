from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from sleeptcn.shhs_preprocessing import (
    RAW_STAGE_MAP,
    SHHSPreprocessConfig,
    raw_stages_to_labels,
    resample_continuous_eeg,
    select_manifest_subjects,
)


def test_raw_stage_mapping_merges_n3_and_preserves_ignored_positions() -> None:
    raw = np.asarray([0, 1, 2, 3, 4, 5, 6, 9], dtype=np.int8)
    labels = raw_stages_to_labels(raw)
    np.testing.assert_array_equal(labels, [0, 1, 2, 3, 3, 4, -1, -1])
    assert labels.dtype == np.int8
    assert set(RAW_STAGE_MAP) == {0, 1, 2, 3, 4, 5, 6, 9}


def test_resampling_125_to_100_is_exact_and_suppresses_above_target_nyquist() -> None:
    config = SHHSPreprocessConfig()
    seconds = 30
    time = np.arange(seconds * 125, dtype=np.float64) / 125.0
    low = np.sin(2 * np.pi * 10 * time)
    high = np.sin(2 * np.pi * 55 * time)
    low_out = resample_continuous_eeg(low, config)
    high_out = resample_continuous_eeg(high, config)
    assert low_out.shape == (seconds * 100,)
    assert high_out.shape == low_out.shape
    assert float(np.std(low_out[200:-200])) > 0.65
    attenuation_ratio = float(
        np.std(high_out[200:-200]) / np.std(high[250:-250])
    )
    assert attenuation_ratio < 0.10


def test_primary_scope_excludes_reserve_and_locks_roles() -> None:
    subjects = []
    subject_id = 200000
    for role, count in (("adaptation", 5), ("validation", 15), ("test", 180), ("reserve", 20)):
        for role_index in range(1, count + 1):
            subjects.append(
                {
                    "subject_id": str(subject_id),
                    "role": role,
                    "role_index": role_index,
                    "pilot": role == "adaptation" or (role == "validation" and role_index <= 5),
                }
            )
            subject_id += 1
    manifest = {"dataset": "SHHS Visit 1", "selection_seed": 42, "subjects": subjects}
    primary = select_manifest_subjects(manifest, "primary")
    pilot = select_manifest_subjects(manifest, "pilot")
    assert len(primary) == 200 and not any(item["role"] == "reserve" for item in primary)
    assert len(pilot) == 10 and not any(item["role"] == "test" for item in pilot)
