"""Data routing and failure handling tests; no model training is performed."""
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


BASE = ROOT / "scripts"
load("adast_reference_adapter", BASE / "run_colab_adast_training.py")
load("development_metrics", ROOT / "src/sleeptcn/metrics.py")
load("run_adast_development_cuda", BASE / "run_adast_development_cuda.py")
runner = load("fullsource_completion_test", ROOT / "scripts/run_adast_fullsource_completion_cuda.py")
builder = load("fullsource_completion_builder_test", ROOT / "scripts/build_adast_fullsource_completion_bundle.py")


def config():
    return json.loads((ROOT / "configs/adast_fullsource_completion_v1_20261006.json").read_bytes())


def pair():
    rows = []
    for i, arm in enumerate(runner.ARMS):
        history = [{"global_epoch": epoch + 1, "updates": 3, "source_presentations": 257,
                    "source_unique_seen": 257, "source_order_sha256": str(epoch),
                    "target_order_sha256": str(epoch + 10000) if i else None,
                    "target_training_presentations": 257 if i else 0, "target_unique_seen": 5 if i else 0,
                    "optimizer_lr_before_epoch": .001 if epoch < 10 else .0001,
                    "loss_coefficients": {"source_ce": 1 if epoch < 15 else .1, "similarity": .001,
                                          "adversarial": i, "target_pseudo_ce": .01 if i and epoch >= 15 else 0},
                    "validation": {"source": {"macro_f1": .7 if epoch < 2 else .6}}}
                   for epoch in range(30)]
        rows.append({"arm": arm, "initial_state_sha256": "matched", "history": history,
                     "epochs_completed": 30, "total_updates": 90, "best_epoch": 1})
    return rows


def test_protocol_is_original_reference_and_exact_remaining_folds():
    runner.validate_protocol(config())
    for field, value in [("folds_to_train", [0, 1]), ("arms", runner.ARMS[::-1]),
                         ("source_loss_weights_by_round", [1, 1]), ("cpu_training_fallback", True),
                         ("max_seconds", 36000), ("automatic_continuation", True)]:
        altered = config()
        altered[field] = value
        with pytest.raises(ValueError, match="protocol differs"):
            runner.validate_protocol(altered)


def test_role_view_equals_sanitized_subset_in_original_order():
    full = np.arange(21, dtype=np.float32).reshape(7, 3)
    view = runner.RoleArray(full, np.array([5, 0, 4, 2]))
    subset = full[[5, 0, 4, 2]]
    assert view.shape == subset.shape and view.dtype == subset.dtype
    np.testing.assert_array_equal(view[:], subset)
    np.testing.assert_array_equal(view[1:3], subset[1:3])
    np.testing.assert_array_equal(view[np.array([3, 0, 1])], subset[[3, 0, 1]])
    np.testing.assert_array_equal(view[-1], subset[-1])


def test_outer_test_is_not_routed_to_train_or_validation():
    x = np.arange(21, dtype=np.float32).reshape(7, 3)
    y = np.arange(7, dtype=np.int64) % 5
    roles = {"train": np.array([4, 0, 6]), "validation": np.array([2, 1]), "test": np.array([3, 5])}
    target = x[:2].copy()
    train, labels, validation, val_labels, adaptation = runner.fold_arrays(x, y, target, roles)
    np.testing.assert_array_equal(train[:], x[roles["train"]])
    np.testing.assert_array_equal(labels, y[roles["train"]])
    np.testing.assert_array_equal(validation[:], x[roles["validation"]])
    np.testing.assert_array_equal(val_labels, y[roles["validation"]])
    assert adaptation is target
    assert not set(train.indices) & set(roles["test"])
    assert not set(validation.indices) & set(roles["test"])


@pytest.mark.parametrize("kind", ["overlap", "missing", "duplicate", "negative", "float"])
def test_invalid_roles_are_rejected(kind):
    roles = {"train": np.array([0, 2]), "validation": np.array([1]), "test": np.array([3])}
    if kind == "overlap": roles["validation"] = np.array([2])
    if kind == "missing": roles["test"] = np.array([], dtype=np.int64)
    if kind == "duplicate": roles["train"] = np.array([0, 0])
    if kind == "negative": roles["test"] = np.array([-1])
    if kind == "float": roles["train"] = np.array([0., 2.])
    with pytest.raises(ValueError):
        runner.validate_roles(roles, 4)


def test_complete_pair_and_first_tie_best_selection():
    runner.check_pair(pair(), 257)


@pytest.mark.parametrize("kind", ["init", "order", "partial", "loss", "lr", "target_leak", "selection"])
def test_incompatible_pair_rejected(kind):
    rows = pair()
    if kind == "init": rows[1]["initial_state_sha256"] = "other"
    if kind == "order": rows[1]["history"][0]["source_order_sha256"] = "other"
    if kind == "partial": rows[1]["history"][0]["source_presentations"] = 256
    if kind == "loss": rows[1]["history"][15]["loss_coefficients"]["source_ce"] = 1
    if kind == "lr": rows[1]["history"][10]["optimizer_lr_before_epoch"] = .001
    if kind == "target_leak": rows[0]["history"][0]["target_training_presentations"] = 257
    if kind == "selection": rows[1]["best_epoch"] = 2
    with pytest.raises(ValueError):
        runner.check_pair(rows, 257)


def test_no_production_cpu_training_or_output_creation(tmp_path, monkeypatch):
    monkeypatch.setattr(runner.torch.cuda, "is_available", lambda: False)
    args = type("Args", (), {"output": tmp_path / "results", "max_seconds": 18000})()
    with pytest.raises(RuntimeError, match="CUDA required"):
        runner.main(args)
    assert not args.output.exists()


def test_matching_torch_required_before_training(tmp_path, monkeypatch):
    monkeypatch.setattr(runner.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(runner.torch, "__version__", "other")
    args = type("Args", (), {"output": tmp_path / "results", "max_seconds": 18000})()
    with pytest.raises(RuntimeError, match="matching PyTorch"):
        runner.main(args)
    assert not args.output.exists()


def test_shared_guard_stops_at_limit_and_keeps_fold_progress(tmp_path, monkeypatch):
    clock = [0.]
    monkeypatch.setattr(runner.base.reference.time, "monotonic", lambda: clock[0])
    guard = runner.CampaignGuard(tmp_path, 10, config()["remaining_fold_sizes"])
    guard.publish(fold=2, arm=runner.ARMS[0], epochs_completed=1, epoch_seconds=4.)
    clock[0] = 10.
    with pytest.raises(runner.base.reference.BudgetStop):
        guard()
    state = json.loads((tmp_path / "progress.json").read_bytes())
    assert state["fold"] == 2 and state["epochs_completed"] == 1
    assert guard.seconds == 10 and state["measured_epoch_count"] == 1
    assert state["eta_seconds_from_observed_epochs"] is None


def test_export_preserves_partial_checkpoint_and_manifest(tmp_path):
    output = tmp_path / "results"
    folder = output / "fold_02" / runner.ARMS[0]
    folder.mkdir(parents=True)
    (folder / "latest.pt").write_bytes(b"fixture-checkpoint")
    (output / "progress.json").write_text(json.dumps({"status": "stopped_resource_budget"}), encoding="utf8")
    archive = runner.export(output)
    with runner.zipfile.ZipFile(archive) as zipped:
        manifest = json.loads(zipped.read("export_manifest.json"))
        for name, digest in manifest.items():
            assert runner.base.hashlib.sha256(zipped.read(name)).hexdigest() == digest
        assert zipped.read(f"fold_02/{runner.ARMS[0]}/latest.pt") == b"fixture-checkpoint"
    assert (folder / "latest.pt").exists()


def test_launcher_binds_archive_and_manifest_without_overlapping_tokens(tmp_path):
    binding = {"archive_sha256": "a" * 64, "manifest_sha256": "b" * 64, "archive_bytes": 123}
    parts = [{"name": "input.part00", "path": "private/local/path", "bytes": 123, "sha256": "c" * 64}]
    destination = tmp_path / "colab_launch.py"
    builder.write_launcher(binding, parts, destination)
    text = destination.read_text(encoding="utf8")
    compile(text, "bound_launcher.py", "exec")
    assert "PLACEHOLDER" not in text
    assert repr("a" * 64) in text and repr("b" * 64) in text
    assert "private/local/path" not in text
    assert "torch.cuda.is_available()" in text and "'18000'" in text
    with pytest.raises(FileExistsError):
        builder.write_launcher(binding, parts, destination)
