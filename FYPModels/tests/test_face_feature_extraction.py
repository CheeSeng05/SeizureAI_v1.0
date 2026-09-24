from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import torch

from FYPModels.preprocess._facial.face_feature_extraction import (
    FaceMapFeatureExtractor, map_landmarks, prepare_face_crop, project_landmarks,
)


class FaceMapFeatureExtractionTests(unittest.TestCase):
    def test_input_is_existing_crop_in_rgb_and_normalized_once(self):
        crop = np.full((24, 40, 3), (10, 80, 160), np.uint8)
        original = crop.copy()
        tensor = prepare_face_crop(crop)
        self.assertEqual(tuple(tensor.shape), (1, 3, 128, 128))
        self.assertEqual(tensor.dtype, torch.float32)
        np.testing.assert_allclose(tensor[0, :, 64, 64], np.array([160, 80, 10]) / 255, atol=1e-7)
        np.testing.assert_array_equal(tensor, prepare_face_crop(crop[:, :, ::-1], color_space="rgb"))
        np.testing.assert_array_equal(crop, original)

    def test_invalid_crops_fail_before_inference(self):
        for crop in (None, np.empty((0, 10, 3), np.uint8), np.zeros((10, 10), np.uint8),
                     np.zeros((10, 10, 4), np.uint8), np.zeros((10, 10, 3), np.float32)):
            with self.subTest(shape=getattr(crop, "shape", None)), self.assertRaises(ValueError):
                prepare_face_crop(crop)
        with self.assertRaisesRegex(ValueError, "color_space"):
            prepare_face_crop(np.zeros((2, 2, 3), np.uint8), color_space="gray")

    def test_exclusive_bounds_match_inclusive_upstream_mapping(self):
        points = np.zeros((68, 2), np.float32)
        points[0], points[1] = (-64, -64), (64, 64)
        original = points.copy()
        actual = map_landmarks(points, (100, 50, 300, 150))
        # Qualcomm bbox=(100,50,299,149), width=x1-x0+1=200, height=100.
        expected = (points + 64) * [200 / 128, 100 / 128] + [100, 50]
        np.testing.assert_allclose(actual, expected)
        np.testing.assert_array_equal(actual[:3], [[100, 50], [300, 150], [200, 100]])
        np.testing.assert_array_equal(points, original)

    def test_invalid_bounds_and_parameters_are_rejected(self):
        for box in ((0, 0, 0, 10), (0, 5, 10, 0), (0, 0, 1), (0, 0, np.nan, 1)):
            with self.subTest(box=box), self.assertRaises(ValueError):
                map_landmarks(np.zeros((68, 2)), box)
        for parameters in (np.zeros(264), np.full(265, np.nan)):
            with self.assertRaisesRegex(ValueError, "265 finite"):
                project_landmarks(parameters, np.zeros(204), np.zeros((204, 219)), np.zeros((204, 39)))

    def test_projection_denormalization_axes_and_fixed_depth(self):
        mean = np.tile([10., 20., 30.], (68, 1)).astype(np.float32)
        shape = np.zeros((204, 219), np.float32)
        expression = np.zeros((204, 39), np.float32)
        shape[0::3, 0] = 1
        expression[1::3, 0] = 2
        parameters = np.zeros(265, np.float32)
        parameters[0] = 1  # identity coefficient becomes 3
        parameters[219] = 1  # expression coefficient becomes 1
        parameters[261:264] = [.5, -.25, 1]  # translation=(30,-15), focal=600
        projected = project_landmarks(parameters, mean, shape, expression)
        np.testing.assert_allclose(projected, np.tile([(13 + 30) * 1.2, (-22 - 15) * 1.2], (68, 1)), atol=1e-5)
        parameters[264] = 1e6  # undocumented extra output must not affect projection
        np.testing.assert_array_equal(projected, project_landmarks(parameters, mean, shape, expression))

    def test_projection_roll_and_yaw_follow_facemap_convention(self):
        mean = np.tile([10., 20., 30.], (68, 1)).astype(np.float32)
        shape, expression = np.zeros((204, 219)), np.zeros((204, 39))
        for index, expected in ((258, [10, -30]), (259, [30, -20]), (260, [20, 10])):
            parameters = np.zeros(265, np.float32)
            parameters[index] = 1  # 90 degrees before reference sign conversion
            with self.subTest(axis=index):
                actual = project_landmarks(parameters, mean, shape, expression)
                np.testing.assert_allclose(actual, np.tile(np.array(expected) * .9, (68, 1)), atol=1e-5)

    def fake_extractor(self):
        # Isolate crop and coordinate handling from expensive checkpoint loading.
        extractor = FaceMapFeatureExtractor.__new__(FaceMapFeatureExtractor)
        extractor.device = torch.device("cpu")
        extractor.model = Mock(return_value=torch.zeros((1, 265)))
        extractor.mean_face = np.zeros(204, np.float32)
        extractor.shape_basis = np.zeros((204, 219), np.float32)
        extractor.expression_basis = np.zeros((204, 39), np.float32)
        return extractor

    def test_enlarged_crop_maps_to_native_frame_box_without_recropping(self):
        extractor = self.fake_extractor()
        crop = np.full((400, 400, 3), 127, np.uint8)
        features = extractor.extract(crop, crop_box=(10, 20, 110, 120))
        np.testing.assert_array_equal(features.landmarks_crop, np.tile([200, 200], (68, 1)))
        np.testing.assert_array_equal(features.landmarks_frame, np.tile([60, 70], (68, 1)))
        self.assertEqual(features.identity_coefficients.shape, (219,))
        self.assertEqual(features.expression_coefficients.shape, (39,))
        self.assertEqual(features.parameters.shape, (265,))
        np.testing.assert_array_equal(features.expression_coefficients, np.full(39, .5))
        self.assertEqual(len(json.loads(json.dumps(features.as_dict(), allow_nan=False))["landmarks_frame"]), 68)
        extractor.model.assert_called_once()
        self.assertFalse(extractor.model.call_args.args[0].requires_grad)

    def test_frame_coordinates_are_absent_without_source_box(self):
        features = self.fake_extractor().extract(np.zeros((40, 60, 3), np.uint8))
        self.assertIsNone(features.landmarks_frame)
        np.testing.assert_array_equal(features.landmarks_crop, np.tile([30, 20], (68, 1)))

    def test_invalid_model_output_is_not_reported_as_landmarks(self):
        for output in (torch.zeros((1, 264)), torch.full((1, 265), float("nan"))):
            extractor = self.fake_extractor()
            extractor.model.return_value = output
            with self.assertRaises(ValueError):
                extractor.extract(np.zeros((20, 20, 3), np.uint8))

    def test_missing_assets_fail_without_implicit_downloads(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(FileNotFoundError, "Missing FaceMap assets"):
                FaceMapFeatureExtractor(directory)
            self.assertEqual(list(Path(directory).iterdir()), [])


if __name__ == "__main__":
    unittest.main()
