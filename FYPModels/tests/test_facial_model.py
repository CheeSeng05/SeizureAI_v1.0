from __future__ import annotations

import unittest

import torch

from seizure_multimodal.facial_resnet.facial_model import (
    ResNet50GRUAttention,
)


class FacialModelTest(unittest.TestCase):
    def test_attention_model_returns_class_logits(self) -> None:
        model = ResNet50GRUAttention(
            num_classes=2,
            gru_hidden_size=16,
            attention_heads=4,
            pretrained=False,
        )
        model.eval()

        video = torch.rand(1, 2, 4, 64, 64)
        with torch.inference_mode():
            logits = model(video)

        self.assertEqual(tuple(logits.shape), (1, 2))
        self.assertIsInstance(
            model.temporal_attention,
            torch.nn.MultiheadAttention,
        )
        self.assertEqual(model.temporal_attention.num_heads, 4)

    def test_attention_heads_must_divide_gru_output(self) -> None:
        with self.assertRaisesRegex(ValueError, "divisible"):
            ResNet50GRUAttention(
                gru_hidden_size=10,
                attention_heads=3,
                pretrained=False,
            )

    def test_input_requires_rgb_plus_flow(self) -> None:
        model = ResNet50GRUAttention(pretrained=False)
        with self.assertRaisesRegex(ValueError, "RGB\\+flow"):
            model(torch.rand(1, 2, 3, 64, 64))


if __name__ == "__main__":
    unittest.main()
