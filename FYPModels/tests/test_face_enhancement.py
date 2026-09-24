from __future__ import annotations

import argparse
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
from FYPModels.preprocess import frame_enhancement
from FYPModels.preprocess.io import VideoMetadata


class FaceEnhancementTest(unittest.TestCase):
    def setUp(self):
        self.image = np.full((24, 32, 3), (30, 60, 90), np.uint8)
        self.image[:, 16:] = (140, 170, 200)

    def test_disabled_enhancement_preserves_pixels_without_aliasing(self):
        result = preprocessing.enhance_face_crop(self.image)
        np.testing.assert_array_equal(result, self.image)
        self.assertFalse(np.shares_memory(result, self.image))

    def test_sharpening_increases_edge_contrast_and_preserves_input(self):
        original = self.image.copy()
        result = preprocessing.enhance_face_crop(self.image, sharpen_amount=.5)
        self.assertEqual(result.shape, self.image.shape)
        self.assertEqual(result.dtype, np.uint8)
        self.assertTrue(np.all(result[:, 15] < self.image[:, 15]))
        self.assertTrue(np.all(result[:, 16] > self.image[:, 16]))
        np.testing.assert_array_equal(result[:, 0], self.image[:, 0])
        np.testing.assert_array_equal(self.image, original)

    def test_uniform_color_and_clipped_edges_do_not_wrap(self):
        constant = np.full((12, 16, 3), (10, 80, 210), np.uint8)
        np.testing.assert_array_equal(
            preprocessing.enhance_face_crop(constant, sharpen_amount=2), constant,
        )
        edge = np.zeros_like(constant)
        edge[:, 8:] = 255
        np.testing.assert_array_equal(
            preprocessing.enhance_face_crop(edge, sharpen_amount=3), edge,
        )

    def test_enhancement_reuses_existing_upscaler_on_native_crop(self):
        crop = self.image[4:16, 6:18]
        with patch.object(frame_enhancement, "_upscaling", wraps=frame_enhancement._upscaling) as upscale:
            result = preprocessing.enhance_face_crop(
                crop, upscale_model=" BICUBIC ", upscale_scale=2, sharpen_amount=.5,
            )
        self.assertEqual(result.shape, (24, 24, 3))
        np.testing.assert_array_equal(upscale.call_args.args[0], crop)
        self.assertEqual(upscale.call_args.args[1], "bicubic")
        self.assertEqual(upscale.call_args.kwargs["scale"], 2)

    def test_invalid_images_and_settings_fail_clearly(self):
        for image in (None, np.zeros((0, 3, 3), np.uint8), self.image[:, :, 0],
                      self.image.astype(np.float32)):
            with self.subTest(image=getattr(image, "shape", None)), self.assertRaises(ValueError):
                preprocessing.enhance_face_crop(image)
        for settings in (
            {"sharpen_amount": -1}, {"sharpen_amount": float("nan")},
            {"sharpen_sigma": 0}, {"sharpen_sigma": float("inf")},
            {"upscale_model": "unknown"}, {"upscale_scale": True},
            {"upscale_scale": 1}, {"upscale_scale": 2.5},
            {"upscale_model": "lapsrn", "upscale_scale": 3},
        ):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                preprocessing.enhance_face_crop(self.image, **settings)

    def test_tensor_enhances_rgb_but_keeps_source_grayscale_and_flow(self):
        box = np.array([0, 0, 32, 24])
        _, previous_gray = preprocessing._crop_face_input(
            np.roll(self.image, 2, axis=1), box, None, face_size=32, cv2=cv2,
        )
        baseline, original_gray = preprocessing._crop_face_input(
            self.image, box, previous_gray, face_size=32, cv2=cv2,
        )
        enhanced, enhanced_gray = preprocessing._crop_face_input(
            self.image, box, previous_gray, face_size=32, cv2=cv2,
            enhancement={"upscale_model": "bicubic", "upscale_scale": 2, "sharpen_amount": .5},
        )
        self.assertEqual(enhanced.shape, (4, 32, 32))
        self.assertEqual(enhanced.dtype, np.float32)
        self.assertTrue(np.isfinite(enhanced).all())
        self.assertGreaterEqual(float(enhanced.min()), 0)
        self.assertLessEqual(float(enhanced.max()), 1)
        self.assertFalse(np.array_equal(enhanced[:3], baseline[:3]))
        np.testing.assert_array_equal(enhanced_gray, original_gray)
        np.testing.assert_array_equal(enhanced[3], baseline[3])
        # A constant region remains RGB (red = original BGR channel 2).
        np.testing.assert_allclose(enhanced[:3, 0, 0], np.array([90, 60, 30]) / 255, atol=1e-7)

    def test_missing_face_skips_enhancement_and_returns_empty_tensor(self):
        with patch.object(preprocessing, "enhance_face_crop") as enhance:
            tensor, gray = preprocessing._crop_face_input(
                self.image, None, None, face_size=16, cv2=cv2,
                enhancement={"sharpen_amount": .5},
            )
        enhance.assert_not_called()
        self.assertFalse(tensor.any())
        self.assertIsNone(gray)

    def test_cli_exposes_opt_in_settings(self):
        parser = argparse.ArgumentParser()
        preprocessing.add_enhancement_arguments(parser)
        defaults = preprocessing.enhancement_kwargs(parser.parse_args([]))
        self.assertIsNone(defaults["upscale_model"])
        self.assertEqual(defaults["sharpen_amount"], 0)
        settings = preprocessing.enhancement_kwargs(parser.parse_args([
            "--upscale-model", "fsrcnn", "--upscale-scale", "2", "--sharpen-amount", ".5",
        ]))
        self.assertEqual(settings["upscale_model"], "fsrcnn")
        self.assertEqual(settings["upscale_scale"], 2)
        self.assertEqual(settings["sharpen_amount"], .5)

    def test_pipeline_enhances_detector_crop_and_keeps_original_coordinates(self):
        anchors = np.array([[.3, .3], [.7, .3], [.5, .5], [.35, .7], [.65, .7]])
        points = np.full((478, 2), .5)
        points[[33, 133]] = anchors[0]
        points[[362, 263]] = anchors[1]
        points[1], points[61], points[291] = anchors[2:]
        landmarks = [SimpleNamespace(x=x, y=y, z=0) for x, y in points]
        face_result = SimpleNamespace(face_landmarks=[landmarks], face_blendshapes=[],
                                      facial_transformation_matrixes=[])
        pose_api = SimpleNamespace(detect_for_video=Mock(return_value=object()), close=Mock())
        face_api = SimpleNamespace(detect=Mock(return_value=face_result), close=Mock())
        vision = SimpleNamespace(
            RunningMode=SimpleNamespace(VIDEO="video", IMAGE="image"),
            PoseLandmarkerOptions=SimpleNamespace, FaceLandmarkerOptions=SimpleNamespace,
            PoseLandmarker=SimpleNamespace(create_from_options=lambda options: pose_api),
            FaceLandmarker=SimpleNamespace(create_from_options=lambda options: face_api),
        )
        mediapipe = SimpleNamespace(tasks=SimpleNamespace(vision=vision, BaseOptions=SimpleNamespace),
                                    Image=lambda **kwargs: kwargs, ImageFormat=SimpleNamespace(SRGB=1))
        capture = SimpleNamespace(read=Mock(side_effect=[(True, self.image), (False, None)]))
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, model = root / "input.avi", root / "model.task"
            source.touch()
            model.touch()
            metadata = VideoMetadata(str(source), 5, 1, 32, 24)
            with patch.dict("sys.modules", {"mediapipe": mediapipe}), \
                 patch.object(preprocessing, "open_video", return_value=contextlib.nullcontext((capture, metadata))), \
                 patch.object(preprocessing, "_select_patient_pose", return_value=0), \
                 patch.object(preprocessing, "_normalized_pose", return_value=(np.zeros((33, 4)), True)), \
                 patch.object(preprocessing, "YuNetLocalizer", return_value=SimpleNamespace(detect=Mock(
                     return_value=preprocessing.LocalizedFace(np.array([8, 4, 24, 20]), anchors * 16 + [8, 4], .9)))), \
                 contextlib.redirect_stdout(io.StringIO()):
                summary = preprocessing.extract_video(
                    source, root / "output", "sample", fps=5, window_seconds=.2,
                    pose_model_path=model, face_model_path=model, face_detector_model=model, face_margin=0, face_size=16,
                    upscale_model="bicubic", upscale_scale=2, sharpen_amount=.5,
                )
            detected_rgb = face_api.detect.call_args.args[0]["data"]
            self.assertEqual(detected_rgb.shape, (32, 32, 3))
            expected_bgr = preprocessing.enhance_face_crop(
                self.image[4:20, 8:24], upscale_model="bicubic", upscale_scale=2, sharpen_amount=.5,
            )
            np.testing.assert_array_equal(detected_rgb, cv2.cvtColor(expected_bgr, cv2.COLOR_BGR2RGB))
            # Pose detection sees the original full frame.
            self.assertEqual(pose_api.detect_for_video.call_args.args[0]["data"].shape, self.image.shape)
            report = json.loads(Path(summary.diagnostics_path).read_text())
            with Path(report["frames_path"]).open() as stream:
                row = next(csv.DictReader(stream))
            self.assertEqual([float(row[f"crop_{name}"]) for name in ("x1", "y1", "x2", "y2")],
                             [8, 4, 24, 20])
            self.assertEqual(row["face_landmarks_detected"], "1")
            self.assertAlmostEqual(float(row["landmark_error_fraction"]), 0)
            self.assertEqual(summary.windows_written, 1)
            face = np.load(next((root / "output/face").glob("*.npy")))
            self.assertEqual(face.shape, (1, 4, 16, 16))
            record = json.loads(Path(summary.run_record_path).read_text())
            self.assertEqual(record["settings"]["upscale_model"], "bicubic")
            self.assertEqual(record["settings"]["sharpen_amount"], .5)
            pose_api.close.assert_called_once_with()
            face_api.close.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
