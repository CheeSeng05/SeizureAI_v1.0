from __future__ import annotations

import importlib.util
import unittest

import numpy as np

from seizure_multimodal.fusion.feature_fusion import FeatureFusionSpec, build_feature_fusion_model
from seizure_multimodal.fusion.late_fusion import LateFusion


class FusionTests(unittest.TestCase):
    def test_equal_weight_late_fusion_dimensions(self) -> None:
        result = LateFusion().predict(video_probability=[0.2, 0.8], eeg_probability=[0.4, 0.6])
        np.testing.assert_allclose(result, [0.3, 0.7])

    def test_missing_modality_fails_explicitly(self) -> None:
        with self.assertRaisesRegex(ValueError, "missing data is not zero-filled"):
            LateFusion().predict(video_probability=[0.2])

    def test_late_fusion_shape_mismatch(self) -> None:
        with self.assertRaisesRegex(ValueError, "shapes differ"):
            LateFusion().predict(video_probability=[0.2, 0.3], eeg_probability=[0.2])

    def test_feature_fusion_supports_unequal_embedding_dimensions(self) -> None:
        spec = FeatureFusionSpec(384, 32, projection_dim=128)
        video, eeg = spec.validate_inputs(np.zeros((3, 384)), np.zeros((3, 32)))
        self.assertEqual(video.shape, (3, 384))
        self.assertEqual(eeg.shape, (3, 32))
        self.assertEqual(spec.concatenated_dim, 256)

    def test_embedding_dimension_error(self) -> None:
        spec = FeatureFusionSpec(384, 32)
        with self.assertRaisesRegex(ValueError, "video embedding dim"):
            spec.validate_inputs(np.zeros((2, 128)), np.zeros((2, 32)))

    @unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch unavailable")
    def test_feature_fusion_forward_dimension(self) -> None:
        import torch

        model = build_feature_fusion_model(FeatureFusionSpec(384, 32))
        model.eval()
        with torch.no_grad():
            output = model(torch.zeros(4, 384), torch.zeros(4, 32))
        self.assertEqual(tuple(output.shape), (4,))


if __name__ == "__main__":
    unittest.main()
