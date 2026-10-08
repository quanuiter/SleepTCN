import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from sleeptcn.revision_weighted_campaign import CampaignBudget, CampaignStop, audit_prediction, require_ten_folds


def test_shared_deadline_and_progress(tmp_path):
    tick = [0.]
    budget = CampaignBudget(100, tmp_path/"progress.json", clock=lambda: tick[0])
    budget.publish(fold=2, phase="source_cache")
    tick[0] = 90
    budget({"phase": "epoch_saved", "arm": "weighted", "history": [{"epoch": 3,
        "best_epoch": 2, "stale_epochs": 1, "validation_macro_f1": .7, "seconds": 9.5}]})
    state = json.loads((tmp_path/"progress.json").read_bytes())
    assert state["fold"] == 2 and state["remaining_attempt_seconds"] == 10
    tick[0] = 100
    with pytest.raises(CampaignStop):
        budget()
    with pytest.raises(ValueError):
        CampaignBudget(18001, tmp_path/"bad.json")


def test_freeze_requires_twenty_in_order():
    records = [{"fold": f, "arms": {"weighted": {}, "unweighted": {}}} for f in range(10)]
    require_ten_folds(records)
    with pytest.raises(ValueError):
        require_ten_folds(records[:-1])
    records[9]["arms"].pop("weighted")
    with pytest.raises(ValueError):
        require_ten_folds(records)


def test_target_ensemble_audit_rejects_corruption(tmp_path):
    p = np.random.default_rng(9).dirichlet(np.ones(5), size=(10, 4)).astype(np.float32)
    entry = {"subject_id": "synthetic", "source_edf_sha256": "edf", "epochs": 4}
    meta = {"subject_id": "synthetic", "source_edf_sha256": "edf",
        "frozen_checkpoints_sha256": "frozen", "specification_sha256": "spec"}
    arrays = {"fold_unweighted": p, "fold_weighted": p,
        "unweighted": p.astype(np.float64).mean(0).astype(np.float32),
        "weighted": p.astype(np.float64).mean(0).astype(np.float32),
        "original_epoch_index": np.arange(4), "metadata_json": np.array(json.dumps(meta))}
    path = tmp_path/"pred.npz"
    np.savez(path, **arrays)
    audit_prediction(path, entry, "frozen", "spec")
    arrays["weighted"][0] = .2
    np.savez(path, **arrays)
    with pytest.raises(AssertionError):
        audit_prediction(path, entry, "frozen", "spec")


def test_reused_result_requires_matching_verification(tmp_path):
    loader = importlib.util.spec_from_file_location("campaign_runner", Path(__file__).resolve().parents[1]/"scripts/run_revision_weighted_campaign.py")
    module = importlib.util.module_from_spec(loader)
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
    loader.loader.exec_module(module)
    (tmp_path/"aggregate_results.json").write_text('{"status":"done"}', encoding="utf-8")
    (tmp_path/"verification.json").write_text('{"status":"passed","aggregate_results_sha256":"wrong"}', encoding="utf-8")
    with pytest.raises(ValueError):
        module.checked_result(tmp_path, "verification.json")
