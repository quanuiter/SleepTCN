from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from sleeptcn.workflows.stages import (
    checkpoint_metadata,
    mark_stage_complete,
    stage_is_complete,
)


@dataclass(frozen=True)
class FakeContext:
    experiment_id: str = "E3"
    outer_fold: int = 2
    config_sha256: str = "a" * 64
    split_sha256: str = "b" * 64
    data_variant: str = "filtered_v2"
    smoke: bool = False


class FakeModel:
    pass


def test_checkpoint_metadata_is_stable() -> None:
    metadata = checkpoint_metadata(FakeContext(), FakeModel(), "resnet1d", 123)
    assert metadata == {
        "experiment_id": "E3",
        "stage": "resnet1d",
        "outer_fold": 2,
        "seed": 123,
        "config_sha256": "a" * 64,
        "split_sha256": "b" * 64,
        "data_variant": "filtered_v2",
        "model_class": "FakeModel",
    }


def test_stage_marker_round_trip_and_hash_tamper_detection(tmp_path: Path) -> None:
    checkpoint_dir = tmp_path / "checkpoints" / "resnet1d"
    checkpoint_dir.mkdir(parents=True)
    best = checkpoint_dir / "best.pt"
    best.write_bytes(b"checkpoint-v1")
    context = FakeContext()

    assert not stage_is_complete(context, checkpoint_dir, "resnet1d", 123)
    mark_stage_complete(context, checkpoint_dir, "resnet1d", 123)
    assert stage_is_complete(context, checkpoint_dir, "resnet1d", 123)

    best.write_bytes(b"checkpoint-tampered")
    with pytest.raises(ValueError, match="checkpoint mismatch"):
        stage_is_complete(context, checkpoint_dir, "resnet1d", 123)
