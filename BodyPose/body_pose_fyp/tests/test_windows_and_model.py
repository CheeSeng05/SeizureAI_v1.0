from __future__ import annotations

import unittest

import pandas as pd
import torch

from body_pose.build_dataset import (
    assert_patient_disjoint,
    deterministic_window_id,
    split_patients,
)
from body_pose.models.lstm import PoseLSTMClassifier


class WindowManifestTest(unittest.TestCase):
    def test_window_id_uses_milliseconds(self) -> None:
        self.assertEqual(
            deterministic_window_id("Patient001", 0.0, 3.0),
            "Patient001__0000000000__0000003000",
        )

    def test_patient_split_has_no_leakage(self) -> None:
        metadata = pd.DataFrame({
            "patient_id": [f"P{index:02d}" for index in range(9) for _ in range(3)]
        })
        config = {
            "split_seed": 42,
            "train_fraction": 0.67,
            "validation_fraction": 0.11,
            "test_fraction": 0.22,
        }
        metadata["split"] = split_patients(metadata, config)
        assert_patient_disjoint(metadata)
        self.assertEqual(set(metadata["split"]), {"train", "validation", "test"})
        patients_per_split = metadata.groupby("split")["patient_id"].nunique().to_dict()
        self.assertEqual(patients_per_split, {"test": 2, "train": 6, "validation": 1})


class PoseModelTest(unittest.TestCase):
    def test_encode_and_forward_shapes(self) -> None:
        model = PoseLSTMClassifier(
            input_size=5,
            num_classes=2,
            hidden_size=4,
            num_layers=1,
            bidirectional=True,
            dropout=0.0,
        )
        x = torch.randn(3, 10, 5)
        self.assertEqual(tuple(model.encode(x).shape), (3, 8))
        self.assertEqual(tuple(model(x).shape), (3, 2))


if __name__ == "__main__":
    unittest.main()
