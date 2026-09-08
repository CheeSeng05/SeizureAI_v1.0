from __future__ import annotations

import unittest
from unittest.mock import patch

import torch

from seizure_multimodal.facial_resnet.facial_model import (
    InceptionResnetV1BiGRUSemiology,
    InceptionResnetV1GRUAttention,
)


class FacialModelTest(unittest.TestCase):
    def test_inception_emotion_model_is_independent_of_mhan(self) -> None:
        model = InceptionResnetV1BiGRUSemiology(gru_hidden_size=8, pretrained=False).eval()
        with torch.inference_mode():
            output = model(torch.rand(1, 2, 4, 64, 64), torch.rand(1, 2, 20))
        self.assertEqual(tuple(output["emotion_logits"].shape), (1, 7))
        self.assertEqual(tuple(output["semiology_logits"].shape), (1, 6))
        self.assertFalse(hasattr(model, "mhan"))

    def test_attention_model_returns_class_logits(self) -> None:
        model = InceptionResnetV1GRUAttention(
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

    def test_vggface2_filters_and_checkpoint_structure(self) -> None:
        from facenet_pytorch import InceptionResnetV1

        backbone = InceptionResnetV1(pretrained=None)
        original = backbone.conv2d_1a.conv.weight.detach().clone()
        backbone.logits = torch.nn.Linear(512, 8631)
        with patch(
            "seizure_multimodal.facial_resnet.facial_model.InceptionResnetV1",
            return_value=backbone,
        ) as constructor:
            model = InceptionResnetV1GRUAttention(pretrained=True)
        constructor.assert_called_once_with(pretrained="vggface2", classify=False)
        torch.testing.assert_close(model.backbone.conv2d_1a.conv.weight[:, :3], original)
        torch.testing.assert_close(
            model.backbone.conv2d_1a.conv.weight[:, 3:4], original.mean(1, keepdim=True)
        )
        restored = InceptionResnetV1GRUAttention(pretrained=False)
        restored.load_state_dict(model.state_dict(), strict=True)
        self.assertEqual(model.gru.input_size, 512)

    def test_facenet_standardization_preserves_flow(self) -> None:
        model = InceptionResnetV1GRUAttention(pretrained=False)
        frames = torch.zeros(1, 4, 224, 224)
        frames[:, 1] = 0.5
        frames[:, 2] = 1.0
        frames[:, 3] = 0.25
        normalized = model.normalize_frames(frames)
        self.assertEqual(tuple(normalized.shape), (1, 4, 160, 160))
        torch.testing.assert_close(
            normalized[0, :, 0, 0],
            torch.tensor([-127.5 / 128, 0, 127.5 / 128, 0.25]),
        )

    def test_attention_heads_must_divide_gru_output(self) -> None:
        with self.assertRaisesRegex(ValueError, "divisible"):
            InceptionResnetV1GRUAttention(
                gru_hidden_size=10,
                attention_heads=3,
                pretrained=False,
            )

    def test_input_requires_rgb_plus_flow(self) -> None:
        model = InceptionResnetV1GRUAttention(pretrained=False)
        with self.assertRaisesRegex(ValueError, "RGB\\+flow"):
            model(torch.rand(1, 2, 3, 64, 64))


if __name__ == "__main__":
    unittest.main()
