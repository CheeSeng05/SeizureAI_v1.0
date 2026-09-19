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
from FYPModels.seizure_multimodal.video.io import VideoMetadata


def detection(x=.2, y=.2, width=.2, height=.2, score=.9):
    return SimpleNamespace(
        score=[score], location_data=SimpleNamespace(relative_bounding_box=SimpleNamespace(
            xmin=x, ymin=y, width=width, height=height,
        )),
    )


class FaceDetectionTest(unittest.TestCase):
    def select(self, detections, **kwargs):
        return preprocessing._select_face_detection(
            SimpleNamespace(detections=detections), 200, 100,
            kwargs.pop("patient_roi", (0, 0, 1, 1)), **kwargs,
        )

    def test_roi_excludes_higher_confidence_caregiver(self):
        patient, caregiver = detection(score=.6), detection(x=.7, score=.99)
        selected = self.select([caregiver, patient], patient_roi=(.1, .1, .45, .7))
        self.assertIs(selected.detection, patient)
        np.testing.assert_allclose(selected.box, [30, 0, 90, 60], atol=1e-5)

    def test_tracking_prefers_nearby_face_and_rejects_identity_switch(self):
        patient, caregiver = detection(score=.6), detection(x=.7, score=.99)
        selected = self.select([caregiver, patient], previous_head_center=(.3, .3))
        self.assertIs(selected.detection, patient)
        self.assertIsNone(self.select([caregiver], previous_head_center=(.3, .3)))

    def test_border_boxes_are_square_and_clipped(self):
        selected = self.select([detection(x=-.05, y=-.1, width=.2, height=.3)])
        np.testing.assert_allclose(selected.box, [0, 0, 60, 60])

    def test_absent_and_invalid_detections_are_ignored(self):
        for detections in (None, [], [detection(width=0)], [detection(x=2)],
                           [detection(y=float("nan"))], [detection(score=float("nan"))]):
            with self.subTest(detections=detections):
                self.assertIsNone(self.select(detections))

    def run_pipeline(self, root, *, preview=False, fail_detector=False):
        # Distinct channels make an accidental BGR input to MediaPipe observable.
        frame = np.zeros((100, 200, 3), np.uint8)
        frame[:, :] = (10, 80, 160)
        capture = SimpleNamespace(read=Mock(side_effect=[(True, frame.copy())] * 3 + [(False, None)]),
                                  release=Mock())
        pose_api = SimpleNamespace(detect_for_video=Mock(return_value=SimpleNamespace(pose_landmarks=[])),
                                   close=Mock())
        face_api = SimpleNamespace(detect_for_video=Mock(return_value=SimpleNamespace(face_landmarks=[])),
                                   close=Mock())
        patient, caregiver = detection(score=.6), detection(x=.7, score=.99)
        detector = SimpleNamespace(process=Mock(side_effect=[
            SimpleNamespace(detections=[caregiver, patient]),
            SimpleNamespace(detections=[caregiver]), SimpleNamespace(detections=[]),
        ]), close=Mock())
        factory = Mock(side_effect=RuntimeError("detector creation failed")) if fail_detector else Mock(return_value=detector)
        vision = SimpleNamespace(
            RunningMode=SimpleNamespace(VIDEO="video"),
            PoseLandmarkerOptions=SimpleNamespace, FaceLandmarkerOptions=SimpleNamespace,
            PoseLandmarker=SimpleNamespace(create_from_options=lambda options: pose_api),
            FaceLandmarker=SimpleNamespace(create_from_options=lambda options: face_api),
        )
        mediapipe = SimpleNamespace(
            tasks=SimpleNamespace(vision=vision, BaseOptions=SimpleNamespace),
            solutions=SimpleNamespace(face_detection=SimpleNamespace(FaceDetection=factory),
                                      drawing_utils=SimpleNamespace(draw_detection=Mock())),
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
             patch.object(preprocessing, "open_video", side_effect=open_video), \
             patch.object(preprocessing, "_show_preprocessing_preview", return_value=False) as show, \
             patch.object(cv2, "destroyAllWindows") as destroy, \
             contextlib.redirect_stdout(io.StringIO()):
            if fail_detector:
                with self.assertRaisesRegex(RuntimeError, "detector creation failed"):
                    preprocessing.extract_video(video, root / "out", "sample", pose_model_path=model,
                                                face_model_path=model)
                pose_api.close.assert_called_once_with()
                face_api.close.assert_called_once_with()
                capture.release.assert_called_once_with()
                return
            summary = preprocessing.extract_video(
                video, root / "out", "sample", pose_model_path=model, face_model_path=model,
                fps=5, window_seconds=.6, face_size=16, patient_roi=(.1, .1, .45, .7),
                max_box_reuse_frames=1, min_face_detection_confidence=.6, preview=preview,
            )
        factory.assert_called_once_with(model_selection=1, min_detection_confidence=.6)
        detector.close.assert_called_once_with()
        pose_api.close.assert_called_once_with()
        face_api.close.assert_called_once_with()
        capture.release.assert_called_once_with()
        if preview:
            show.assert_called_once()
            self.assertIs(show.call_args.kwargs["face_detection"], patient)
            destroy.assert_called_once_with()
        np.testing.assert_array_equal(detector.process.call_args_list[0].args[0], frame[:, :, ::-1])
        np.testing.assert_array_equal(face_api.detect_for_video.call_args.args[0]["data"],
                                      frame[0:60, 30:90, ::-1])
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

    def test_preview_quit_stops_decoding_and_cleans_up(self):
        with tempfile.TemporaryDirectory() as directory:
            summary = self.run_pipeline(Path(directory), preview=True)
            self.assertEqual(summary.sampled_frames, 1)
            self.assertEqual(summary.windows_written, 0)

    def test_detector_initialization_failure_closes_existing_resources(self):
        with tempfile.TemporaryDirectory() as directory:
            self.run_pipeline(Path(directory), fail_detector=True)

    def test_preview_draws_selected_detection_on_a_copy(self):
        frame = np.zeros((100, 200, 3), np.uint8)
        fake_cv2 = Mock(wraps=cv2)
        fake_cv2.FONT_HERSHEY_SIMPLEX = cv2.FONT_HERSHEY_SIMPLEX
        fake_cv2.LINE_AA = cv2.LINE_AA
        fake_cv2.INTER_NEAREST = cv2.INTER_NEAREST
        fake_cv2.imshow = Mock()
        fake_cv2.waitKey = Mock(return_value=ord("q"))
        fake_cv2.destroyWindow = Mock()
        face = detection()
        draw = Mock()
        keep_running = preprocessing._show_preprocessing_preview(
            frame, np.zeros((4, 16, 16)), np.array([30, 0, 90, 60]), time_sec=0,
            face_landmarks_found=False, box_reused=False, delay_ms=5, cv2=fake_cv2,
            face_detection=face, draw_detection=draw,
        )
        self.assertFalse(keep_running)
        self.assertIs(draw.call_args.args[1], face)
        self.assertFalse(np.shares_memory(draw.call_args.args[0], frame))
        self.assertFalse(frame.any())
        self.assertIn("FACE BOX (NO LANDMARKS)", fake_cv2.putText.call_args_list[0].args[1])


if __name__ == "__main__":
    unittest.main()
