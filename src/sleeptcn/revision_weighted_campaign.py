"""Resource limits and artifact checks for the matched ten-fold loss campaign."""
import json
from pathlib import Path
import time

import numpy as np


class CampaignStop(RuntimeError):
    pass


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.json")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(data, stream, indent=2, allow_nan=False)
        stream.write("\n")
    temporary.replace(path)


class CampaignBudget:
    def __init__(self, seconds, progress_path, clock=time.perf_counter):
        if not 0 < seconds <= 18000:
            raise ValueError("require a positive budget of at most five hours")
        self.clock, self.started, self.limit = clock, clock(), seconds
        self.progress_path = Path(progress_path)
        self.feature_seconds = 0.
        self.last_projection = None
        self.context = {}

    def remaining(self):
        return max(0., self.limit - (self.clock() - self.started))

    def publish(self, **values):
        self.context.update(values)
        atomic_json(self.progress_path, {**self.context,
            "elapsed_attempt_seconds": self.clock() - self.started,
            "remaining_attempt_seconds": self.remaining()})

    def __call__(self, progress=None):
        if self.remaining() <= 0:
            raise CampaignStop("five-hour attempt budget reached; completed artifacts/checkpoints retained")
        if progress and progress["phase"] == "epoch_saved":
            row = progress["history"][-1]
            self.publish(phase="training", arm=progress["arm"], epoch=row["epoch"],
                best_epoch=row["best_epoch"], stale_epochs=row["stale_epochs"],
                validation_macro_f1=row["validation_macro_f1"], epoch_seconds=row["seconds"])


def audit_probabilities(parts, epochs):
    parts = np.asarray(parts)
    if parts.shape != (10, epochs, 5) or not np.isfinite(parts).all() or (parts < 0).any() or (parts > 1).any():
        raise ValueError("require valid probabilities for all ten folds")
    np.testing.assert_allclose(parts.sum(-1), 1., atol=1e-6, rtol=0)


def audit_prediction(path, entry, frozen_hash, spec_hash):
    with np.load(path, allow_pickle=False) as z:
        if set(z.files) != {"fold_unweighted", "fold_weighted", "unweighted", "weighted",
                            "original_epoch_index", "metadata_json"}:
            raise ValueError("unexpected target prediction schema")
        expected = {"subject_id": entry["subject_id"], "source_edf_sha256": entry["source_edf_sha256"],
                    "frozen_checkpoints_sha256": frozen_hash, "specification_sha256": spec_hash}
        if json.loads(str(z["metadata_json"])) != expected:
            raise ValueError("prediction provenance differs")
        np.testing.assert_array_equal(z["original_epoch_index"], np.arange(entry["epochs"]))
        result = {}
        for arm in ["unweighted", "weighted"]:
            parts = z["fold_" + arm]
            audit_probabilities(parts, entry["epochs"])
            expected_p = np.mean(parts.astype(np.float64), axis=0).astype(np.float32)
            np.testing.assert_array_equal(z[arm], expected_p)
            result[arm] = z[arm].copy()
        return result


def require_ten_folds(records):
    if [r["fold"] for r in records] != list(range(10)):
        raise ValueError("must freeze exactly folds zero through nine in order")
    if any(set(r["arms"]) != {"unweighted", "weighted"} for r in records):
        raise ValueError("both source-selected arms are required in every fold")
