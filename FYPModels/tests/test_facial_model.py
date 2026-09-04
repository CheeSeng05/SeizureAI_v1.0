from __future__ import annotations

import unittest

import torch

from seizure_multimodal.facial_resnet.facial_model import (
    ResNet50BiGRUSemiology,
    ResNet50GRUAttention,
)


class FacialModelTest(unittest.TestCase):
    def test_resnet_emotion_model_is_independent_of_mhan(self) -> None:
        model = ResNet50BiGRUSemiology(gru_hidden_size=8, pretrained=False).eval()
        with torch.inference_mode():
            output = model(torch.rand(1, 2, 4, 64, 64), torch.rand(1, 2, 20))
        self.assertEqual(tuple(output["emotion_logits"].shape), (1, 7))
        self.assertEqual(tuple(output["semiology_logits"].shape), (1, 6))
        self.assertFalse(hasattr(model, "mhan"))

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
