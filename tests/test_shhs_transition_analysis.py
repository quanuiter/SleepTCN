from __future__ import annotations

import numpy as np

from sleeptcn.shhs_transition_analysis import (
    EnsemblePredictions,
    _binary_auc,
    annotate_transitions,
)


def _predictions(labels: list[int]) -> EnsemblePredictions:
    true = np.asarray(labels, dtype=np.int8)
    probabilities = np.full((len(true), 5), 0.025, dtype=np.float32)
    probabilities[np.arange(len(true)), true] = 0.9
    return EnsemblePredictions(
        subject_id=np.full(len(true), "s1"),
        record_key=np.full(len(true), "r1"),
        original_epoch_index=np.arange(len(true), dtype=np.int32),
        true_label=true,
        predicted_label=true.copy(),
        probabilities=probabilities,
    )


def test_transition_distance_is_symmetric_and_requires_persistence() -> None:
    value = _predictions([2, 2, 2, 3, 3, 3, 3])
    result = annotate_transitions(value, persistence_epochs=3)
    np.testing.assert_array_equal(result.distance_any_change, [2, 1, 0, 0, 1, 2, 3])
    np.testing.assert_array_equal(
        result.distance_persistent_n2_n3, [2, 1, 0, 0, 1, 2, 3]
    )
    np.testing.assert_array_equal(result.n2_n3_direction, [1, 1, 1, 1, 1, 1, 1])
    np.testing.assert_array_equal(result.legacy_radius_1, [False, False, True, True, True, False, False])
    assert result.persistent_n2_n3_count == 1


def test_short_stage_flicker_is_not_a_persistent_transition() -> None:
    value = _predictions([2, 2, 2, 3, 2, 2, 2])
    result = annotate_transitions(value, persistence_epochs=3)
    assert result.raw_change_count == 2
    assert result.persistent_n2_n3_count == 0
    assert np.all(result.distance_persistent_n2_n3 == np.iinfo(np.int32).max)


def test_low_confidence_auc_has_expected_direction() -> None:
    target = np.asarray([False, False, True, True])
    low_confidence = np.asarray([0.1, 0.2, 0.8, 0.9])
    assert _binary_auc(target, low_confidence) == 1.0
