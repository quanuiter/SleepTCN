import unittest
import hashlib
import tempfile
import zipfile
from pathlib import Path

import numpy as np

from sleeptcn.preprocessing import (
    PreprocessConfig,
    annotations_to_labels,
    atomic_savez,
    filtered_v2,
    preprocess_signal_variant,
    trim_sleep_window,
)


class AnnotationTests(unittest.TestCase):
    def test_mapping_merge_and_ignored_labels(self) -> None:
        labels = annotations_to_labels(
            onsets=[0, 30, 60, 90, 120, 150, 180, 210],
            durations=[30] * 8,
            annotations=[
                "Sleep stage W",
                "Sleep stage 1",
                "Sleep stage 2",
                "Sleep stage 3",
                "Sleep stage 4",
                "Sleep stage R",
                "Movement time",
                "Sleep stage ?",
            ],
            epoch_seconds=30,
        )
        np.testing.assert_array_equal(labels, [0, 1, 2, 3, 3, 4, -1, -1])
        self.assertEqual(labels.dtype, np.int8)

    def test_rejects_timeline_gap(self) -> None:
        with self.assertRaises(ValueError):
            annotations_to_labels(
                [0, 60], [30, 30], ["Sleep stage W", "Sleep stage 1"], 30
            )


class TrimTests(unittest.TestCase):
    def test_ignored_label_does_not_define_sleep_boundary(self) -> None:
        x = np.arange(8 * 3, dtype=np.float32).reshape(8, 3)
        y = np.array([0, -1, 0, 1, 2, 0, -1, 0], dtype=np.int8)
        x_t, y_t, idx, start, stop = trim_sleep_window(x, y, edge_epochs=1)
        self.assertEqual((start, stop), (2, 6))
        np.testing.assert_array_equal(idx, [2, 3, 4, 5])
        np.testing.assert_array_equal(y_t, [0, 1, 2, 0])
        np.testing.assert_array_equal(x_t, x[2:6])


class FilterTests(unittest.TestCase):
    def test_e3_e4_are_only_a_scale_change_when_clipping_is_inactive(self) -> None:
        config = PreprocessConfig()
        time = np.arange(6000, dtype=np.float64) / 100.0
        signal = 70 * np.sin(2 * np.pi * 1.2 * time) + 20 * np.sin(2 * np.pi * 11 * time)
        e3, fraction3, _ = preprocess_signal_variant(signal, "filtered_v2", config)
        e4, fraction4, _ = preprocess_signal_variant(signal, "bandpass_v2", config)
        self.assertEqual(fraction3, 0.0)
        self.assertEqual(fraction4, 0.0)
        np.testing.assert_allclose(e3 * config.scale_factor, e4, rtol=2e-7, atol=1e-5)

    def test_filtered_v2_is_finite_and_preserves_length(self) -> None:
        config = PreprocessConfig()
        time = np.arange(30 * 100, dtype=np.float64) / 100.0
        signal = 50 * np.sin(2 * np.pi * 10 * time) + 20 * np.sin(2 * np.pi * 40 * time)
        result, clip_fraction = filtered_v2(signal, config)
        self.assertEqual(result.shape, signal.shape)
        self.assertEqual(result.dtype, np.float32)
        self.assertTrue(np.isfinite(result).all())
        self.assertEqual(clip_fraction, 0.0)

    def test_ablation_variants_are_finite_and_distinct(self) -> None:
        config = PreprocessConfig(clip_uv=100.0, scale_factor=10.0)
        time = np.arange(30 * 100, dtype=np.float64) / 100.0
        signal = 150 * np.sin(2 * np.pi * 10 * time)
        outputs = {}
        for variant in (
            "bandpass_v2",
            "bandpass_clip_v2",
            "filtered_v2",
            "filtered_zscore_v2",
        ):
            output, clip_fraction, metadata = preprocess_signal_variant(
                signal, variant, config
            )
            self.assertTrue(np.isfinite(output).all())
            self.assertGreater(clip_fraction, 0.0)
            self.assertIn("normalization", metadata)
            outputs[variant] = output
        self.assertGreater(float(np.max(np.abs(outputs["bandpass_v2"]))), 100.0)
        self.assertLessEqual(float(np.max(np.abs(outputs["bandpass_clip_v2"]))), 100.0)
        self.assertLessEqual(float(np.max(np.abs(outputs["filtered_v2"]))), 10.0)
        self.assertAlmostEqual(
            float(outputs["filtered_zscore_v2"].mean()), 0.0, places=5
        )
        self.assertAlmostEqual(
            float(outputs["filtered_zscore_v2"].std()), 1.0, places=5
        )

    def test_window_zscore_uses_only_the_evaluation_window(self) -> None:
        config = PreprocessConfig()
        rng = np.random.default_rng(0)
        signal = rng.normal(0.0, 20.0, 10 * 3000)
        signal[:3000] *= 5.0  # loud first epoch outside the window
        output, _, metadata = preprocess_signal_variant(
            signal, "filtered_zscore_window_v3", config, window_epochs=(2, 8)
        )
        window = output[2 * 3000:8 * 3000].astype(np.float64)
        self.assertAlmostEqual(float(window.mean()), 0.0, places=4)
        self.assertAlmostEqual(float(window.std()), 1.0, places=4)
        self.assertEqual(
            metadata["normalization_scope"], "evaluation_window_after_filter_clip"
        )
        with self.assertRaises(ValueError):
            preprocess_signal_variant(signal, "filtered_zscore_window_v3", config)

    def test_rolling_zscore_matches_brute_force_and_floor(self) -> None:
        from sleeptcn.preprocessing import rolling_zscore

        rng = np.random.default_rng(1)
        values = rng.normal(5.0, 3.0, 101)
        output, _, _ = rolling_zscore(values, 10, 1e-9)
        for i in (0, 4, 50, 97, 100):
            part = values[max(0, i - 5):min(101, i + 5)]
            expected = (values[i] - part.mean()) / part.std()
            self.assertAlmostEqual(float(output[i]), float(expected), places=8)
        flat, _, std = rolling_zscore(np.zeros(50), 10, 1.0)
        self.assertTrue(np.all(flat == 0.0))
        self.assertEqual(std, 1.0)
        with self.assertRaises(ValueError):
            rolling_zscore(values, 9, 1.0)

    def test_rolling90_variant_is_finite_and_locally_standardised(self) -> None:
        config = PreprocessConfig()
        rng = np.random.default_rng(2)
        # 4 h record whose amplitude doubles halfway: a full-record z-score keeps
        # the step, the 90-minute rolling z-score removes most of it.
        signal = rng.normal(0.0, 10.0, 4 * 3600 * 100)
        signal[2 * 3600 * 100:] *= 2.0
        output, _, metadata = preprocess_signal_variant(
            signal, "filtered_zscore_rolling90_v3", config
        )
        self.assertTrue(np.isfinite(output).all())
        first = float(output[: 3600 * 100].std())
        last = float(output[-3600 * 100:].std())
        self.assertAlmostEqual(first, last, delta=0.1)
        self.assertEqual(
            metadata["normalization_scope"], "centred_rolling_window_after_filter_clip"
        )


class SerializationTests(unittest.TestCase):
    def test_npz_serialization_is_byte_stable(self) -> None:
        arrays = {
            "x": np.arange(12, dtype=np.float32).reshape(3, 4),
            "label": np.array("Sleep stage 2"),
            "mask": np.array([True, False, True]),
        }
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "first.npz"
            second = Path(directory) / "second.npz"
            atomic_savez(first, arrays)
            atomic_savez(second, arrays)
            self.assertEqual(
                hashlib.sha256(first.read_bytes()).digest(),
                hashlib.sha256(second.read_bytes()).digest(),
            )
            with zipfile.ZipFile(first) as archive:
                self.assertTrue(all(info.create_system == 3 for info in archive.infolist()))
                self.assertTrue(
                    all(info.date_time == (1980, 1, 1, 0, 0, 0) for info in archive.infolist())
                )


if __name__ == "__main__":
    unittest.main()
