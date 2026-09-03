from __future__ import annotations

import unittest

import numpy as np

from seizure_multimodal.evaluation.biomarker_validation import (
    validate_binary_biomarker,
    validate_continuous_biomarker,
)
from seizure_multimodal.evaluation.facial_biomarkers import extract_facial_biomarkers


class FacialBiomarkerTests(unittest.TestCase):
    def test_extracts_motion_and_geometry_proxies(self) -> None:
        face = np.zeros((3, 4, 4, 4), dtype=np.float32)
        face[1:, 3] = 0.5
        pose = np.zeros((3, 33, 4), dtype=np.float32)
        pose[:, :, 3] = 1.0
        pose[1, :11, 0] = 0.1
        pose[2, :11, 0] = 0.2
        geometry = np.ones((3, 20), dtype=np.float32)
        geometry[1:] *= 1.1

        result = extract_facial_biomarkers(
            face_window=face,
            pose_window=pose,
            face_geometry_window=geometry,
            sampled_fps=5.0,
        )

        self.assertGreater(result["optical_flow_mean"], 0)
        self.assertGreater(result["head_speed_mean_per_sec"], 0)
        self.assertIsNotNone(result["mouth_opening_proxy"])

    def test_validates_against_independent_reference_scores(self) -> None:
        continuous = validate_continuous_biomarker([1, 2, 3], [1, 2, 4])
        self.assertEqual(continuous["n"], 3)
        self.assertGreater(continuous["pearson_r"], 0.9)
        binary = validate_binary_biomarker([0, 1, 1, 0], [0, 1, 0, 0])
        self.assertEqual(binary["n"], 4)
        self.assertIsNotNone(binary["cohen_kappa"])


if __name__ == "__main__":
    unittest.main()
