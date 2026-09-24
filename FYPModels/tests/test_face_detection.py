from __future__ import annotations

import contextlib
import csv
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import cv2
import numpy as np

from FYPModels.preprocess._facial import face_preprocess as preprocessing
from FYPModels.preprocess.io import VideoMetadata
from FYPModels.preprocess._facial.face_localization import select_face, face_crop_box


def detection(x=.2, y=.2, width=.2, height=.2, score=.9):
    x, y, width, height = x * 200, y * 100, width * 200, height * 100
    anchors = np.array([[.25, .3], [.75, .3], [.5, .5], [.3, .8], [.7, .8]])
    return np.r_[x, y, width, height, (anchors * [width, height] + [x, y]).ravel(), score]


class FaceDetectionTest(unittest.TestCase):
    def select(self, detections, **kwargs):
        return select_face(
            None if detections is None else np.asarray(detections), (0, 0), (100, 200, 3),
            kwargs.pop("patient_roi", (0, 0, 1, 1)), **kwargs,
        )

    def test_roi_excludes_higher_confidence_caregiver(self):
        patient, caregiver = detection(score=.6), detection(x=.7, score=.99)
        selected = self.select([caregiver, patient], patient_roi=(.1, .1, .45, .7))
        self.assertEqual(selected.score, .6)
        np.testing.assert_allclose(face_crop_box(selected, (100, 200, 3), .25), [30, 0, 90, 60])

    def test_tracking_prefers_nearby_face_and_rejects_identity_switch(self):
        patient, caregiver = detection(score=.6), detection(x=.7, score=.99)
        selected = self.select([caregiver, patient], previous_center=np.array([.3, .3]))
        self.assertEqual(selected.score, .6)
        self.assertIsNone(self.select([caregiver], previous_center=np.array([.3, .3])))

    def test_border_boxes_are_square_and_clipped(self):
        selected = self.select([detection(x=-.05, y=-.1, width=.2, height=.3)])
        np.testing.assert_allclose(face_crop_box(selected, (100, 200, 3), .25), [0, 0, 45, 45])

    def test_absent_and_invalid_detections_are_ignored(self):
        for detections in (None, [], [detection(width=0)], [detection(x=2)],
                           [detection(y=float("nan"))], [detection(score=float("nan"))]):
            with self.subTest(detections=detections):
                self.assertIsNone(self.select(detections))

    def test_yunet_localizer_honors_custom_jump_limit(self):
        detector = SimpleNamespace(setInputSize=Mock(), detect=Mock(return_value=(None, np.array([detection()]))))
        with tempfile.TemporaryDirectory() as directory, patch.object(cv2.FaceDetectorYN, "create", return_value=detector):
            model = Path(directory) / "yunet.onnx"
            model.touch()
            localizer = preprocessing.YuNetLocalizer(model)
            frame = np.zeros((100, 200, 3), np.uint8)
            previous = np.array([.05, .3])
            self.assertIsNone(localizer.detect(frame, (0, 0, 1, 1), previous))
            face = localizer.detect(frame, (0, 0, 1, 1), previous, max_jump=.3)
            self.assertIsNotNone(face)
            np.testing.assert_allclose(face.center, [60, 30])

    def test_invalid_yunet_settings_fail_before_loading_models(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "model.task"
            model.touch()
            for name in ("face_confidence", "max_landmark_error"):
                for value in (-.1, 1.1, float("nan"), float("inf")):
                    with self.subTest(name=name, value=value), self.assertRaisesRegex(ValueError, name):
                        preprocessing.extract_video(
                            root / "video.mp4", root / "out", "sample", pose_model_path=model,
                            face_model_path=model, face_detector_model=model, **{name: value},
                        )
            with self.assertRaisesRegex(FileNotFoundError, "YuNet model not found"):
                preprocessing.extract_video(root / "video.mp4", root / "out", "sample",
                                            pose_model_path=model, face_model_path=model,
                                            face_detector_model=root / "missing.onnx")

    def run_pipeline(self, root, *, preview=False, save_preview=False,
                     diagnostics_only=False, fail_detector=False, landmarks=None, pose_found=False):
        # YuNet expects BGR while landmark inference expects RGB.
        frame = np.zeros((100, 200, 3), np.uint8)
        frame[:, :] = (10, 80, 160)
        capture = SimpleNamespace(read=Mock(side_effect=[(True, frame.copy())] * 3 + [(False, None)]),
                                  release=Mock())
        point = SimpleNamespace(x=.3, y=.3, z=0, visibility=1, presence=1)
        pose_api = SimpleNamespace(detect_for_video=Mock(return_value=SimpleNamespace(
            pose_landmarks=[[point] * 33] if pose_found else [])), close=Mock())
        face_api = SimpleNamespace(detect=Mock(return_value=SimpleNamespace(
            face_landmarks=[landmarks] if landmarks is not None else [])), close=Mock())
        patient, caregiver = detection(score=.6), detection(x=.7, score=.99)
        # Detector outputs are relative to the search crop at (20, 10).
        def roi_row(row):
            row = row.copy()
            row[:2] -= [20, 10]
            row[4:14] = (row[4:14].reshape(5, 2) - [20, 10]).ravel()
            return row
        detector = SimpleNamespace(setInputSize=Mock(), detect=Mock(side_effect=[
            (None, np.array([roi_row(caregiver), roi_row(patient)])),
            (None, np.array([roi_row(caregiver)])), (None, None),
        ]))
        factory = Mock(side_effect=RuntimeError("detector creation failed")) if fail_detector else Mock(return_value=detector)
        face_factory = Mock(return_value=face_api)
        vision = SimpleNamespace(
            RunningMode=SimpleNamespace(VIDEO="video", IMAGE="image"),
            PoseLandmarkerOptions=SimpleNamespace, FaceLandmarkerOptions=SimpleNamespace,
            PoseLandmarker=SimpleNamespace(create_from_options=lambda options: pose_api),
            FaceLandmarker=SimpleNamespace(create_from_options=face_factory),
        )
        mediapipe = SimpleNamespace(
            tasks=SimpleNamespace(vision=vision, BaseOptions=SimpleNamespace),
            Image=lambda **kwargs: kwargs, ImageFormat=SimpleNamespace(SRGB=1),
        )
        video, model = root / "input.mp4", root / "model.task"
        video.touch()
        model.touch()

        @contextlib.contextmanager
        def open_video(*args, **kwargs):
            try:
                yield capture, VideoMetadata(str(video), 5, 3, 200, 100)
            finally:
                capture.release()

        with patch.dict("sys.modules", {"mediapipe": mediapipe}), \
             patch.object(cv2.FaceDetectorYN, "create", factory), \
             patch.object(preprocessing, "open_video", side_effect=open_video), \
             patch.object(preprocessing, "_show_preprocessing_preview", return_value=False) as show, \
             patch.object(cv2, "destroyAllWindows") as destroy, \
             contextlib.redirect_stdout(io.StringIO()):
            if fail_detector:
                with self.assertRaisesRegex(RuntimeError, "detector creation failed"):
                    preprocessing.extract_video(video, root / "out", "sample", pose_model_path=model,
                                                face_model_path=model, face_detector_model=model)
                pose_api.close.assert_called_once_with()
                face_api.close.assert_called_once_with()
                capture.release.assert_called_once_with()
                return
            summary = preprocessing.extract_video(
                video, root / "out", "sample", pose_model_path=model, face_model_path=model, face_detector_model=model,
                fps=5, window_seconds=.6, face_size=16, patient_roi=(.1, .1, .45, .7),
                max_box_reuse_frames=1, face_confidence=.6, preview=preview,
                save_preview=save_preview, diagnostics_only=diagnostics_only,
            )
        factory.assert_called_once_with(str(model.resolve()), "", (320, 320), .6, .3, 5000)
        self.assertEqual(face_factory.call_args.args[0].running_mode, "image")
        pose_api.close.assert_called_once_with()
        face_api.close.assert_called_once_with()
        capture.release.assert_called_once_with()
        if preview:
            show.assert_called_once()
            destroy.assert_called_once_with()
        else:
            show.assert_not_called()
            destroy.assert_not_called()
        np.testing.assert_array_equal(detector.detect.call_args_list[0].args[0], frame[10:70, 20:90])
        np.testing.assert_array_equal(face_api.detect.call_args.args[0]["data"],
                                      frame[0:60, 30:90, ::-1])
        face_api.detect.assert_called_once()
        return summary

    def test_detector_supplies_crops_without_pose_or_landmarks_and_reuse_expires(self):
        with tempfile.TemporaryDirectory() as directory:
            summary = self.run_pipeline(Path(directory))
            report = json.loads(Path(summary.diagnostics_path).read_text())
            self.assertEqual(report["metrics"]["face_detector_fraction"], 1 / 3)
            self.assertEqual(report["metrics"]["face_landmark_fraction"], 0)
            self.assertEqual(report["metrics"]["pose_head_fallback_fraction"], 0)
            self.assertEqual(report["metrics"]["face_region_fraction"], 2 / 3)
            faces = np.load(next((Path(directory) / "out/face").glob("*.npy")))
            self.assertEqual(faces.shape, (3, 4, 16, 16))
            self.assertGreater(faces[0, :3].sum(), 0)
            self.assertGreater(faces[1, :3].sum(), 0)
            self.assertEqual(faces[2].sum(), 0)
            geometry = np.load(next((Path(directory) / "out/face_geometry").glob("*.npy")))
            self.assertFalse(geometry.any())
            with Path(summary.manifest_path).open() as stream:
                row = next(csv.DictReader(stream))
            self.assertAlmostEqual(float(row["face_detection_fraction"]), 1 / 3, places=6)
            self.assertEqual(float(row["face_landmark_fraction"]), 0)

    def test_pose_cannot_supply_face_crops_when_yunet_misses(self):
        with tempfile.TemporaryDirectory() as directory:
            summary = self.run_pipeline(Path(directory), pose_found=True)
            report = json.loads(Path(summary.diagnostics_path).read_text())
            self.assertEqual(report["metrics"]["pose_detection_fraction"], 1)
            self.assertEqual(report["metrics"]["pose_head_fallback_fraction"], 0)
            self.assertEqual(report["metrics"]["face_region_fraction"], 2 / 3)

    def test_only_anchor_agreement_allows_geometry_and_never_changes_yunet_crop(self):
        anchors = detection()[4:14].reshape(5, 2)
        points = np.tile([60., 30.], (478, 1))
        points[[33, 133]] = anchors[0]
        points[[362, 263]] = anchors[1]
        points[1], points[61], points[291] = anchors[2:]
        for offset in (0, 40):
            landmarks = [SimpleNamespace(x=(x - 30) / 60, y=(y + offset) / 60)
                         for x, y in points]
            with self.subTest(offset=offset), tempfile.TemporaryDirectory() as directory:
                summary = self.run_pipeline(Path(directory), landmarks=landmarks)
                report = json.loads(Path(summary.diagnostics_path).read_text())
                self.assertEqual(report["metrics"]["face_landmark_fraction"], 1 / 3 if offset == 0 else 0)
                geometry = np.load(next((Path(directory) / "out/face_geometry").glob("*.npy")))
                self.assertEqual(bool(geometry.any()), offset == 0)
                with Path(report["frames_path"]).open() as stream:
                    row = next(csv.DictReader(stream))
                self.assertAlmostEqual(float(row["landmark_error_fraction"]), offset / 40)
                self.assertEqual([float(row[f"raw_{axis}"]) for axis in ("x1", "y1", "x2", "y2")],
                                 [30, 0, 90, 60])
                record = json.loads(Path(summary.run_record_path).read_text())
                self.assertEqual(record["settings"]["face_detector"], "opencv_yunet")
                self.assertIn("sha256", record["inputs"]["face_detector_model"])

    def test_preview_quit_stops_decoding_and_cleans_up(self):
        with tempfile.TemporaryDirectory() as directory:
            summary = self.run_pipeline(Path(directory), preview=True)
            self.assertEqual(summary.sampled_frames, 1)
            self.assertEqual(summary.windows_written, 0)

    def test_detector_initialization_failure_closes_existing_resources(self):
        with tempfile.TemporaryDirectory() as directory:
            self.run_pipeline(Path(directory), fail_detector=True)

    def test_saved_preview_is_playable_without_gui_in_both_output_modes(self):
        for diagnostics_only in (False, True):
            with self.subTest(diagnostics_only=diagnostics_only), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                summary = self.run_pipeline(root, save_preview=True, diagnostics_only=diagnostics_only)
                path = root / "out/preview.mp4"
                capture = cv2.VideoCapture(str(path))
                try:
                    self.assertTrue(capture.isOpened())
                    self.assertEqual(capture.get(cv2.CAP_PROP_FPS), 5)
                    frames = []
                    while True:
                        ok, frame = capture.read()
                        if not ok:
                            break
                        frames.append(frame)
                    self.assertEqual(len(frames), summary.sampled_frames)
                    self.assertEqual(len(frames), 3)
                    self.assertEqual(frames[0].shape, (448, 648, 3))
                    # Away from captions: model crop is BGR and turns black
                    # when reuse expires, while the source panel remains visible.
                    np.testing.assert_allclose(frames[0][250, 400], [10, 80, 160], atol=8)
                    self.assertLess(frames[2][250, 400].max(), 8)
                finally:
                    capture.release()
                record = json.loads(Path(summary.run_record_path).read_text())
                self.assertEqual(record["settings"]["preview_path"], str(path.resolve()))

    def test_saved_preview_is_finalized_when_user_quits(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            summary = self.run_pipeline(root, preview=True, save_preview=True)
            capture = cv2.VideoCapture(str(root / "out/preview.mp4"))
            try:
                self.assertTrue(capture.read()[0])
                self.assertFalse(capture.read()[0])
            finally:
                capture.release()
            self.assertEqual(summary.sampled_frames, 1)

    def test_unavailable_preview_encoder_fails_and_releases_writer(self):
        writer = Mock()
        writer.isOpened.return_value = False
        with tempfile.TemporaryDirectory() as directory, patch.object(cv2, "VideoWriter", return_value=writer):
            with self.assertRaisesRegex(OSError, "Cannot create preview video"):
                self.run_pipeline(Path(directory), save_preview=True)
        writer.release.assert_called_once_with()
        writer.write.assert_not_called()

    def test_preview_draws_selected_detection_on_a_copy(self):
        frame = np.zeros((100, 200, 3), np.uint8)
        fake_cv2 = Mock(wraps=cv2)
        fake_cv2.FONT_HERSHEY_SIMPLEX = cv2.FONT_HERSHEY_SIMPLEX
        fake_cv2.LINE_AA = cv2.LINE_AA
        fake_cv2.INTER_NEAREST = cv2.INTER_NEAREST
        fake_cv2.imshow = Mock()
        fake_cv2.waitKey = Mock(return_value=ord("q"))
        fake_cv2.destroyWindow = Mock()
        face = self.select([detection()])
        panels = preprocessing._render_preprocessing_preview(
            frame, np.zeros((4, 16, 16)), np.array([30, 0, 90, 60]), time_sec=0,
            face_landmarks_found=False, box_reused=False, cv2=fake_cv2,
            face_detection=face,
        )
        keep_running = preprocessing._show_preprocessing_preview(*panels, delay_ms=5, cv2=fake_cv2)
        self.assertFalse(keep_running)
        self.assertEqual(fake_cv2.circle.call_count, 5)
        self.assertFalse(np.shares_memory(panels[0], frame))
        self.assertFalse(frame.any())
        self.assertTrue(any("FACE BOX (NO LANDMARKS)" in call.args[1]
                            for call in fake_cv2.putText.call_args_list))


if __name__ == "__main__":
    unittest.main()
