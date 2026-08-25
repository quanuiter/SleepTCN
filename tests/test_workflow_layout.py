from __future__ import annotations

from pathlib import Path

import pytest

from sleeptcn.workflows.layout import build_experiment_layout


def test_experiment_layout_is_portable_and_canonical(tmp_path: Path) -> None:
    layout = build_experiment_layout(
        tmp_path,
        "E3",
        2,
        123,
        smoke=False,
    )
    root = tmp_path.resolve()
    assert layout.workspace == root
    assert layout.run_root == root / "runs/v2/full/E3/fold_02/seed_123"
    assert layout.cache_root == root / "data/cache/features/v2/full/fold_02/seed_123"


def test_experiment_layout_rejects_invalid_identifiers(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="experiment_id"):
        build_experiment_layout(tmp_path, "E9", 0, 1, smoke=True)
    with pytest.raises(ValueError, match="invalid fold or seed"):
        build_experiment_layout(tmp_path, "E0", 10, 1, smoke=True)
    with pytest.raises(ValueError, match="invalid fold or seed"):
        build_experiment_layout(tmp_path, "E0", 0, -1, smoke=True)
