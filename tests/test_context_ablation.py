from dataclasses import dataclass

import numpy as np
import pytest

from sleeptcn.workflows.context_ablation import (
    context_groups,
    mask_feature_sequences,
    train_replacement_mean,
)


@dataclass(frozen=True)
class FakeSequence:
    extractor_id: str
    features: np.ndarray
    labels: np.ndarray


def _sequence(features: np.ndarray, labels: list[int]) -> FakeSequence:
    return FakeSequence(
        extractor_id="cnn15",
        features=features.astype(np.float32),
        labels=np.asarray(labels, dtype=np.int8),
    )


def test_replacement_mean_uses_only_valid_epochs() -> None:
    features = np.vstack(
        [
            np.zeros(75, dtype=np.float32),
            np.full(75, 2.0, dtype=np.float32),
            np.full(75, 1000.0, dtype=np.float32),
        ]
    )
    mean, count = train_replacement_mean([_sequence(features, [0, 1, -1])])
    assert count == 2
    np.testing.assert_array_equal(mean, np.ones(75, dtype=np.float32))


def test_mask_replaces_only_groups_not_retained() -> None:
    features = np.arange(150, dtype=np.float32).reshape(2, 75)
    replacement = np.arange(75, dtype=np.float32) + 500
    masked = mask_feature_sequences(
        [_sequence(features, [0, 1])], "CP", replacement
    )[0]
    assert masked.extractor_id == "cnn15_gate8_CP"
    np.testing.assert_array_equal(masked.features[:, :50], features[:, :50])
    np.testing.assert_array_equal(
        masked.features[:, 50:], np.broadcast_to(replacement[50:], (2, 25))
    )
    np.testing.assert_array_equal(features, np.arange(150, dtype=np.float32).reshape(2, 75))


def test_unknown_condition_is_rejected() -> None:
    assert context_groups("C") == ("C",)
    with pytest.raises(ValueError, match="unknown Gate 8 condition"):
        context_groups("FULL_CPN")
