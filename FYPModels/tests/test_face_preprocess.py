from __future__ import annotations

import unittest

import numpy as np

from FYPModels.preprocess._facial.face_preprocess import (
    _pose_head_box,
    _pose_head_center,
    _select_patient_pose,
    _patient_split,
    _smooth_box,
    _window_starts,
)


class _Point:
    def __init__(self, x: float, y: float, visibility: float = 1.0):
        self.x = x
        self.y = y
        self.z = 0.0
        self.visibility = visibility
        self.presence = 1.0


class _PoseResult:
    def __init__(self, poses):
        self.pose_landmarks = poses


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

    def test_patient_roi_selects_pose_with_head_inside_region(self) -> None:
        left = [_Point(.2, .2)] * 33
        right = [_Point(.8, .2)] * 33
        result = _PoseResult([left, right])
        self.assertEqual(_select_patient_pose(result, (0, 0, .5, 1)), 0)
        self.assertEqual(_select_patient_pose(result, (.5, 0, 1, 1)), 1)

    def test_pose_tracking_rejects_identity_switch(self) -> None:
        patient = [_Point(.2, .2)] * 33
        caretaker = [_Point(.8, .2)] * 33
        result = _PoseResult([patient, caretaker])
        self.assertEqual(
            _select_patient_pose(result, (0, 0, 1, 1), (.21, .2), .12),
            0,
        )
        self.assertIsNone(
            _select_patient_pose(_PoseResult([caretaker]), (0, 0, 1, 1), (.2, .2), .12)
        )
        self.assertEqual(_pose_head_center(result, 0), (.2, .2))

    def test_pose_head_box_is_clipped_to_frame(self) -> None:
        pose = [_Point(.05 + index * .002, .05 + index * .002) for index in range(33)]
        result = _PoseResult([pose])
        box = _pose_head_box(result, 0, 1920, 1080)
        self.assertIsNotNone(box)
        assert box is not None
        self.assertGreaterEqual(float(box.min()), 0.0)
        self.assertLessEqual(float(box[2]), 1920.0)
        self.assertLessEqual(float(box[3]), 1080.0)


if __name__ == "__main__":
    unittest.main()
