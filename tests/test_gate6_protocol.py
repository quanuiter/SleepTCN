import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from sleeptcn.models import BiLSTMSleepNet, EEGResNet1D, SleepCNN, SleepTCN


ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


BENCHMARK = load_script(
    "gate6_benchmark", ROOT / "scripts" / "benchmark_model_complexity.py"
)
FEATURES = load_script(
    "gate6_features", ROOT / "scripts" / "analyze_feature_space.py"
)


def fake_record(key: str, offset: int = 0):
    labels = np.tile(np.arange(5, dtype=np.int8), 6)
    x = np.stack(
        [np.full(3000, offset + index, dtype=np.float32) for index in range(len(labels))]
    )
    return SimpleNamespace(
        info=SimpleNamespace(record_key=key, subject_id=key[:5]),
        x=x,
        y=labels,
        valid_mask=np.ones(len(labels), dtype=bool),
        original_epoch_index=np.arange(100, 100 + len(labels), dtype=np.int32),
    )


class Gate6BenchmarkTests(unittest.TestCase):
    def extractors(self):
        return {
            f"{prefix}_{stage}": SleepCNN()
            for prefix in ("C", "P", "N")
            for stage in ("W", "N1", "N2", "N3", "REM")
        }

    def test_pipeline_parameter_counts_match_frozen_architectures(self) -> None:
        e0 = BENCHMARK.CNN15Pipeline(
            self.extractors(), BiLSTMSleepNet(), "bilstm"
        )
        e1 = BENCHMARK.CNN15Pipeline(
            self.extractors(), SleepTCN(input_dim=75), "tcn"
        )
        e2 = BENCHMARK.ResNetTCNPipeline(
            EEGResNet1D(), SleepTCN(input_dim=128)
        )
        self.assertEqual(BENCHMARK.parameter_report(e0)["parameters"], 248_630)
        self.assertEqual(BENCHMARK.parameter_report(e1)["parameters"], 640_950)
        self.assertEqual(BENCHMARK.parameter_report(e2)["parameters"], 1_085_578)
        self.assertEqual(BENCHMARK.parameter_report(e0)["component_models"], 16)
        self.assertEqual(BENCHMARK.parameter_report(e2)["component_models"], 2)

    def test_temporal_shifts_do_not_cross_record_boundaries(self) -> None:
        import torch

        x = torch.tensor([[[[1.0]], [[2.0]], [[3.0]]]])
        previous = BENCHMARK.CNN15Pipeline._shift(x, "previous")
        following = BENCHMARK.CNN15Pipeline._shift(x, "next")
        self.assertEqual(previous.flatten().tolist(), [1.0, 1.0, 2.0])
        self.assertEqual(following.flatten().tolist(), [2.0, 3.0, 3.0])

    def test_latency_protocol_rejects_cherry_picked_dimensions(self) -> None:
        import unittest.mock

        with unittest.mock.patch.object(BENCHMARK, "_git_commit", return_value="a" * 40):
            with self.assertRaisesRegex(ValueError, "official latency protocol"):
                BENCHMARK.build_report(
                    Path("."),
                    mode="latency",
                    device_name="cpu",
                    fold=0,
                    seed=42,
                    batch_records=1,
                    sequence_length=50,
                    warmup=20,
                    repeats=100,
                    rounds=3,
                )


class Gate6FeatureTests(unittest.TestCase):
    def test_balanced_sample_is_deterministic_unique_and_subject_distributed(self) -> None:
        records = (fake_record("SC4001E", 0), fake_record("SC4011E", 100))
        first = FEATURES.select_balanced_epochs(records, per_class=8, seed=42)
        second = FEATURES.select_balanced_epochs(records, per_class=8, seed=42)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 40)
        self.assertEqual({sum(item.label == label for item in first) for label in range(5)}, {8})
        keys = {(x.subject_id, x.record_key, x.original_epoch_index) for x in first}
        self.assertEqual(len(keys), len(first))
        for label in range(5):
            self.assertEqual(
                {item.subject_id for item in first if item.label == label},
                {"SC400", "SC401"},
            )

    def test_sampled_previous_and_next_signals_use_true_record_neighbors(self) -> None:
        record = fake_record("SC4001E")
        samples = (
            FEATURES.SampleEpoch("SC400", "SC4001E", 0, 100, 0),
            FEATURES.SampleEpoch("SC400", "SC4001E", 5, 105, 0),
            FEATURES.SampleEpoch("SC400", "SC4001E", 29, 129, 4),
        )
        previous = FEATURES._signals_for_samples((record,), samples, "previous")
        following = FEATURES._signals_for_samples((record,), samples, "next")
        self.assertEqual(previous[:, 0].tolist(), [0.0, 4.0, 28.0])
        self.assertEqual(following[:, 0].tolist(), [1.0, 6.0, 29.0])

    def test_manifest_loader_requires_all_ten_balanced_folds(self) -> None:
        import json
        import tempfile

        sample = {
            "subject_id": "SC400",
            "record_key": "SC4001E",
            "position": 0,
            "original_epoch_index": 100,
            "label": 0,
        }
        manifest = {
            "schema_version": 3,
            "status": "prepared",
            "folds": list(range(10)),
            "role": "test",
            "seed": 42,
            "data_variant": "paper_raw_v1",
            "sample_per_class_per_fold": 1,
            "total_sample_count": 50,
            "fold_samples": {},
        }
        for fold in range(10):
            samples = [{**sample, "label": label} for label in range(5)]
            manifest["fold_samples"][f"fold_{fold:02d}"] = {
                "sample_count": 5,
                "samples": samples,
            }
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "samples.json"
            path.write_text(json.dumps(manifest), encoding="utf-8")
            loaded, folds = FEATURES._load_sample_manifest(path)
            self.assertEqual(loaded["total_sample_count"], 50)
            self.assertEqual(len(folds), 10)
            self.assertTrue(all(len(samples) == 5 for samples in folds.values()))


if __name__ == "__main__":
    unittest.main()
