import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import cv2
import numpy as np

from FYPModels.preprocess._facial.preview_microexpressions import (
    FACE_3D_POINTS, PNP_INDICES, LEFT_EYE, RIGHT_EYE, camera_matrix,
    estimate_pose, expression_ratios, pose_filter, measure_face,
)
from FYPModels.preprocess._facial.face_localization import (
    LocalizedFace, select_face, face_crop_box, landmark_agreement_error,
)
from FYPModels.preprocess._facial import preview_microexpressions as preview
from FYPModels.preprocess._facial.face_feature_extraction import ASSET_NAMES
from FYPModels.preprocess.io import VideoMetadata


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

    def test_video_facemap_uses_existing_unannotated_crop_and_records_missing_frames(self):
        for enabled in (True, False):
            with self.subTest(facemap=enabled), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                source = root / "source.mp4"
                source.touch()
                models = root / "models"
                models.mkdir()
                for name in ASSET_NAMES:
                    (models / name).touch()
                frame = np.full((100, 200, 3), (10, 80, 160), np.uint8)
                capture = SimpleNamespace(read=Mock(side_effect=[
                    (True, frame.copy()), (True, frame.copy()), (False, None),
                ]))
                metadata = VideoMetadata(str(source), 5, 2, 200, 100)
                face = LocalizedFace(np.array([40, 20, 80, 40]), np.zeros((5, 2)), .9)
                localizer = SimpleNamespace(detect=Mock(side_effect=[face, None]))
                seen = {}

                def extract(crop, *, crop_box):
                    seen["facemap_crop"] = crop.copy()
                    seen["box"] = crop_box
                    return SimpleNamespace(
                        landmarks_frame=np.tile([60., 30.], (68, 1)),
                        as_dict=lambda: {"landmarks_frame": [[60., 30.]] * 68},
                    )

                def mesh_process(rgb):
                    seen["mesh_crop"] = rgb.copy()
                    return SimpleNamespace(multi_face_landmarks=[])

                extractor = SimpleNamespace(extract=Mock(side_effect=extract))
                mesh = SimpleNamespace(process=Mock(side_effect=mesh_process))
                mediapipe = SimpleNamespace(solutions=SimpleNamespace(face_mesh=SimpleNamespace(
                    FaceMesh=Mock(return_value=contextlib.nullcontext(mesh)),
                )))
                writer = Mock()
                writer.isOpened.return_value = True
                args = ["preview_microexpressions.py", str(source), "--output", str(root / "output"),
                        "--face-detector-model", str(source)]
                if enabled:
                    args += ["--facemap", "--facemap-model-dir", str(models)]
                with patch("sys.argv", args), patch.dict("sys.modules", {"mediapipe": mediapipe}), \
                     patch.object(preview, "open_video", return_value=contextlib.nullcontext((capture, metadata))), \
                     patch.object(preview, "YuNetLocalizer", return_value=localizer), \
                     patch("FYPModels.preprocess._facial.face_feature_extraction.FaceMapFeatureExtractor",
                           return_value=extractor) as factory, \
                     patch.object(cv2, "VideoWriter", return_value=writer), \
                     contextlib.redirect_stdout(io.StringIO()):
                    preview.main()
                self.assertEqual(writer.write.call_count, 2)
                writer.release.assert_called_once_with()
                mesh.process.assert_called_once()
                features_path = root / "output/facemap.jsonl"
                summary = json.loads((root / "output/summary.json").read_text())
                if enabled:
                    factory.assert_called_once_with(models, device="cpu")
                    extractor.extract.assert_called_once()
                    np.testing.assert_array_equal(seen["facemap_crop"][:, :, ::-1], seen["mesh_crop"])
                    x1, y1, x2, y2 = face_crop_box(face, frame.shape)
                    np.testing.assert_array_equal(seen["facemap_crop"], frame[y1:y2, x1:x2])
                    self.assertEqual(seen["box"], [x1, y1, x2, y2])
                    rows = [json.loads(line) for line in features_path.read_text().splitlines()]
                    self.assertEqual([row["time_sec"] for row in rows], [0, .2])
                    self.assertEqual(len(rows[0]["features"]["landmarks_frame"]), 68)
                    self.assertIsNone(rows[1]["features"])
                    self.assertIsNone(rows[1]["crop_box_xyxy"])
                    self.assertEqual(summary["counts"]["facemap_frames"], 1)
                    self.assertEqual(len(summary["facemap"]["assets"]), 4)
                else:
                    factory.assert_not_called()
                    self.assertFalse(features_path.exists())
                    self.assertNotIn("facemap", summary)


if __name__ == "__main__":
    unittest.main()
