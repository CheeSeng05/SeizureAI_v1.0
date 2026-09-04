from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from seizure_multimodal.facial_mhan.model import (
    MHANBiGRUSemiology,
    MultiHeadAttentionPooling,
)
from seizure_multimodal.facial_mhan.train import (
    MicroExpressionDataset,
    multitask_loss,
)


class MHANBiGRUTests(unittest.TestCase):
    def test_multitask_shapes_and_normalized_attention(self) -> None:
        model = MHANBiGRUSemiology(
            emotion_classes=("neutral", "fear", "anger"),
            semiology_classes=("fear", "grimacing"),
            frame_feature_dim=32,
            gru_hidden_size=8,
            attention_heads=2,
        ).eval()
        with torch.inference_mode():
            output = model(torch.rand(2, 4, 4, 32, 32), torch.rand(2, 4, 20))
        self.assertEqual(tuple(output["emotion_logits"].shape), (2, 3))
        self.assertEqual(tuple(output["semiology_logits"].shape), (2, 2))
        self.assertEqual(tuple(output["seizure_logits"].shape), (2, 2))
        self.assertEqual(tuple(output["attention_weights"].shape), (2, 2, 4))
        torch.testing.assert_close(
            output["attention_weights"].sum(dim=-1), torch.ones(2, 2)
        )

    def test_attention_masks_invalid_frames(self) -> None:
        attention = MultiHeadAttentionPooling(4, 2).eval()
        _, weights = attention(
            torch.rand(1, 3, 4), torch.tensor([[True, True, False]])
        )
        torch.testing.assert_close(weights[:, :, 2], torch.zeros(1, 2))

    def test_masked_multitask_loss_allows_missing_semiology(self) -> None:
        model = MHANBiGRUSemiology(
            emotion_classes=("neutral", "fear"), semiology_classes=("fear",),
            frame_feature_dim=16, gru_hidden_size=4, attention_heads=2,
        )
        output = model(torch.rand(2, 2, 4, 32, 32), torch.rand(2, 2, 20))
        loss, parts = multitask_loss(output, {
            "emotion": torch.tensor([0, 1]),
            "semiology": torch.full((2, 1), -1.0),
            "seizure": torch.tensor([0, 1]),
        })
        self.assertTrue(torch.isfinite(loss))
        self.assertNotIn("semiology_loss", parts)

    def test_dataset_rejects_low_fps_microexpression_data(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            np.save(root / "face.npy", np.zeros((2, 4, 8, 8), np.float32))
            np.save(root / "geometry.npy", np.zeros((2, 20), np.float32))
            manifest = root / "manifest.csv"
            with manifest.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=(
                    "window_id", "split", "sampled_fps", "face_landmark_fraction",
                    "face_detection_fraction", "face_path", "face_geometry_path", "label",
                ))
                writer.writeheader(); writer.writerow({
                    "window_id": "W1", "split": "train", "sampled_fps": "5",
                    "face_landmark_fraction": "1", "face_detection_fraction": "1",
                    "face_path": "face.npy", "face_geometry_path": "geometry.npy", "label": "1",
                })
            annotations = root / "annotations.csv"
            annotations.write_text(
                "window_id,emotion,semiologies,annotator_id\nW1,fear,fear,C01\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "requires at least 25 fps"):
                MicroExpressionDataset(manifest, annotations, "train")


if __name__ == "__main__":
    unittest.main()
