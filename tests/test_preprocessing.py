import unittest

import numpy as np

from sleeptcn.preprocessing import (
    PreprocessConfig,
    annotations_to_labels,
    filtered_v2,
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
    def test_filtered_v2_is_finite_and_preserves_length(self) -> None:
        config = PreprocessConfig()
        time = np.arange(30 * 100, dtype=np.float64) / 100.0
        signal = 50 * np.sin(2 * np.pi * 10 * time) + 20 * np.sin(2 * np.pi * 40 * time)
        result, clip_fraction = filtered_v2(signal, config)
        self.assertEqual(result.shape, signal.shape)
        self.assertEqual(result.dtype, np.float32)
        self.assertTrue(np.isfinite(result).all())
        self.assertEqual(clip_fraction, 0.0)


if __name__ == "__main__":
    unittest.main()
