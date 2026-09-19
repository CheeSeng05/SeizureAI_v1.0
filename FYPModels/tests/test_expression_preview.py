import unittest
from unittest.mock import patch

import cv2
import numpy as np

from FYPModels.preprocess._facial.preview_microexpressions import (
    FACE_3D_POINTS, PNP_INDICES, LEFT_EYE, RIGHT_EYE, camera_matrix,
    estimate_pose, expression_ratios, pose_filter, measure_face,
)
from FYPModels.preprocess._facial.face_localization import (
    LocalizedFace, select_face, face_crop_box, landmark_agreement_error,
)


class ExpressionPreviewTests(unittest.TestCase):
    def test_pose_recovers_known_angles_in_degrees(self):
        for pitch, yaw in ((0, 0), (15, 30), (-20, -40)):
            with self.subTest(pitch=pitch, yaw=yaw):
                x, y = np.deg2rad([pitch, yaw])
                rx = np.array([[1, 0, 0], [0, np.cos(x), -np.sin(x)], [0, np.sin(x), np.cos(x)]])
                ry = np.array([[np.cos(y), 0, np.sin(y)], [0, 1, 0], [-np.sin(y), 0, np.cos(y)]])
                rotation, _ = cv2.Rodrigues(ry @ rx)
                projected, _ = cv2.projectPoints(FACE_3D_POINTS, rotation, np.array([0., 0., 1600.]), camera_matrix(1920, 1080), np.zeros(4))
                points = np.zeros((478, 2))
                points[PNP_INDICES] = projected.reshape(-1, 2)
                pose = estimate_pose(points, 1920, 1080)
                self.assertIsNotNone(pose)
                self.assertAlmostEqual(pose["pitch_deg"], pitch, places=4)
                self.assertAlmostEqual(pose["yaw_deg"], yaw, places=4)
                self.assertLess(pose["reprojection_rmse_px"], .001)

    def test_failed_pose_is_rejected(self):
        with patch("cv2.solvePnP", return_value=(False, None, None)):
            self.assertIsNone(estimate_pose(np.ones((478, 2)), 100, 100))

    def test_pose_limits_include_boundary_and_both_signs(self):
        self.assertEqual(pose_filter(dict(yaw_deg=45, pitch_deg=-30), 45, 30), "")
        self.assertEqual(pose_filter(dict(yaw_deg=-46, pitch_deg=31), 45, 30), "yaw_limit|pitch_limit")

    def test_eyes_are_classified_separately(self):
        points = np.zeros((478, 2))
        eye = np.array([[0, 0], [2, -1], [8, -1], [10, 0], [8, 1], [2, 1]], float)
        points[LEFT_EYE] = eye * [1, .25]
        points[RIGHT_EYE] = eye * [1, 1.5]
        points[[61, 291, 0, 17]] = [[0, 0], [10, 0], [5, -2], [5, 2]]
        result = expression_ratios(points, .2, .3)
        self.assertEqual(result["eye_state"], "left_closed")
        self.assertEqual(result["mouth_active"], 1)
        points[RIGHT_EYE] = points[LEFT_EYE]
        self.assertEqual(expression_ratios(points, .2, .3)["eye_state"], "both_closed_candidate")

    def test_degenerate_geometry_is_not_a_blink(self):
        self.assertIsNone(expression_ratios(np.zeros((478, 2)), .2, .3))

    def mesh_and_detection(self):
        anchors = np.array([[480, 295], [510, 295], [475, 330], [480, 350], [510, 350]], float)
        face = LocalizedFace(np.array([460, 240, 580, 380]), anchors, .95)
        points = np.tile([500., 310.], (478, 1))
        points[[33, 133]] = anchors[0]
        points[[362, 263]] = anchors[1]
        points[1], points[61], points[291] = anchors[2:]
        return points, face

    def test_mesh_on_cap_cannot_pass_with_a_plausible_pose(self):
        points, face = self.mesh_and_detection()
        points[:, 1] -= 100
        with patch('FYPModels.preprocess._facial.preview_microexpressions.estimate_pose') as pose, \
             patch('FYPModels.preprocess._facial.preview_microexpressions.expression_ratios') as expressions:
            row = measure_face(points, face, 1920, 1080)
        self.assertEqual(row['accepted'], 0)
        self.assertEqual(row['reason'], 'landmark_mismatch')
        self.assertGreater(row['landmark_error_fraction'], .2)
        self.assertNotIn('left_ear', row)
        self.assertNotIn('yaw_deg', row)
        pose.assert_not_called()
        expressions.assert_not_called()

    def test_agreement_uses_all_five_anchors_and_detected_face_scale(self):
        points, face = self.mesh_and_detection()
        self.assertEqual(landmark_agreement_error(points, face), 0)
        points[1, 0] += 42
        self.assertAlmostEqual(landmark_agreement_error(points, face), .3)
        self.assertIsNone(landmark_agreement_error(points[:100], face))
        points[0, 0] = np.nan
        self.assertIsNone(landmark_agreement_error(points, face))

    def test_yunet_coordinates_are_translated_back_from_search_roi(self):
        row = np.array([[10, 20, 40, 50, 15, 30, 35, 30, 20, 40, 18, 55, 33, 55, .9]])
        face = select_face(row, (100, 50), (200, 400, 3), (.25, .25, .75, 1))
        np.testing.assert_array_equal(face.box, [110, 70, 150, 120])
        np.testing.assert_array_equal(face.keypoints[2], [120, 90])
        self.assertIsNone(select_face(row, (100, 50), (200, 400, 3), (.5, .5, 1, 1)))

    def test_missing_and_distant_faces_are_not_replaced_with_broad_roi(self):
        self.assertIsNone(select_face(None, (0, 0), (200, 400, 3), (0, 0, 1, 1)))
        row = np.array([[250, 20, 40, 50, 260, 30, 280, 30, 270, 40, 260, 55, 280, 55, .99]])
        self.assertIsNone(select_face(row, (0, 0), (200, 400, 3), (0, 0, 1, 1), np.array([.2, .3])))

    def test_face_crop_is_square_clipped_and_does_not_modify_box(self):
        face = LocalizedFace(np.array([0., 0., 40., 60.]), np.zeros((5, 2)), .9)
        box = face_crop_box(face, (100, 200, 3))
        np.testing.assert_array_equal(box, [0, 0, 78, 78])
        np.testing.assert_array_equal(face.box, [0, 0, 40, 60])


if __name__ == "__main__":
    unittest.main()
