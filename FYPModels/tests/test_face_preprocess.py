from __future__ import annotations

import unittest

import numpy as np

from seizure_multimodal.facial.face_preprocess import (
    _patient_split,
    _smooth_box,
    _window_starts,
)


class FacePreprocessHelpersTest(unittest.TestCase):
    def test_complete_overlapping_windows_only(self) -> None:
        self.assertEqual(list(_window_starts(90, 40, 20)), [0, 20, 40])
        self.assertEqual(list(_window_starts(39, 40, 20)), [])

    def test_split_is_stable_and_patient_level(self) -> None:
        self.assertEqual(_patient_split("S47"), _patient_split("S47"))
        self.assertIn(_patient_split("S47"), {"train", "validation", "test"})

    def test_invalid_window_parameters_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "positive"):
            list(_window_starts(100, 0, 20))

    def test_face_box_smoothing_reduces_detector_jitter(self) -> None:
        previous = np.asarray([10, 20, 110, 120], dtype=np.float32)
        current = np.asarray([20, 30, 120, 130], dtype=np.float32)
        np.testing.assert_allclose(
            _smooth_box(previous, current, .7), [13, 23, 113, 123]
        )


if __name__ == "__main__":
    unittest.main()
