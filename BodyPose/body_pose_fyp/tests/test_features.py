from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from body_pose.extract_features import interpolate_and_smooth


class MissingPoseFeatureTest(unittest.TestCase):
    def test_all_missing_landmark_is_neutral_and_finite(self) -> None:
        result = interpolate_and_smooth(pd.Series([np.nan] * 10), limit=3, window=5)
        self.assertTrue(np.isfinite(result.to_numpy()).all())
        self.assertEqual(result.tolist(), [0.0] * 10)


if __name__ == "__main__":
    unittest.main()
