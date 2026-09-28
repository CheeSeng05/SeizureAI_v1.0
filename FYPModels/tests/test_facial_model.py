from __future__ import annotations

import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch

import torch

from seizure_multimodal.facial.facial_model import (
    InceptionResnetV1Semiology,
    InceptionResnetV1Attention,
)
from seizure_multimodal.facial.inference import load_model
from seizure_multimodal.facial.train import build_optimizer, configure_trainable_layers, save_checkpoint


class FacialModelTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls) -> None:
        torch.set_num_threads(cls.previous_threads)

    def test_inception_emotion_model_is_independent_of_mhan(self) -> None:
        model = InceptionResnetV1Semiology(pretrained=False).eval()
        with torch.inference_mode():
            output = model(torch.rand(1, 2, 4, 64, 64), torch.rand(1, 2, 20))
        self.assertEqual(tuple(output["emotion_logits"].shape), (1, 7))
        self.assertEqual(tuple(output["semiology_logits"].shape), (1, 6))
        self.assertEqual(tuple(output["seizure_logits"].shape), (1, 2))
        self.assertEqual(tuple(output["embedding"].shape), (1, 640))
        self.assertFalse(any(isinstance(module, torch.nn.GRU) for module in model.modules()))
        self.assertFalse(hasattr(model, "mhan"))

    def test_attention_model_returns_class_logits(self) -> None:
        model = InceptionResnetV1Attention(
            num_classes=2,
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
        self.assertFalse(any(isinstance(module, torch.nn.GRU) for module in model.modules()))

    def test_vggface2_filters_and_checkpoint_structure(self) -> None:
        from facenet_pytorch import InceptionResnetV1

        backbone = InceptionResnetV1(pretrained=None)
        original = backbone.conv2d_1a.conv.weight.detach().clone()
        backbone.logits = torch.nn.Linear(512, 8631)
        with patch(
            "seizure_multimodal.facial.facial_model.InceptionResnetV1",
            return_value=backbone,
        ) as constructor:
            model = InceptionResnetV1Attention(pretrained=True)
        constructor.assert_called_once_with(pretrained="vggface2", classify=False)
        torch.testing.assert_close(model.backbone.conv2d_1a.conv.weight[:, :3], original)
        torch.testing.assert_close(
            model.backbone.conv2d_1a.conv.weight[:, 3:4], original.mean(1, keepdim=True)
        )
        restored = InceptionResnetV1Attention(pretrained=False)
        restored.load_state_dict(model.state_dict(), strict=True)
        self.assertEqual(model.temporal_attention.embed_dim, 512)

    def test_facenet_standardization_preserves_flow(self) -> None:
        model = InceptionResnetV1Attention(pretrained=False)
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

    def test_invalid_attention_settings_fail_before_loading_backbone(self) -> None:
        with patch("seizure_multimodal.facial.facial_model.InceptionResnetV1") as constructor:
            for model_type in (InceptionResnetV1Attention, InceptionResnetV1Semiology):
                for heads in (0, -1, 3):
                    with self.subTest(model=model_type.__name__, heads=heads):
                        with self.assertRaisesRegex(ValueError, "positive and divide"):
                            model_type(attention_heads=heads)
                with self.assertRaisesRegex(ValueError, "max_frames"):
                    model_type(max_frames=0)
            constructor.assert_not_called()

    def test_input_requires_rgb_plus_flow(self) -> None:
        model = InceptionResnetV1Attention(pretrained=False)
        with self.assertRaisesRegex(ValueError, "RGB\\+flow"):
            model(torch.rand(1, 2, 3, 64, 64))

    def test_empty_and_oversized_windows_fail_before_backbone(self) -> None:
        for model_type in (InceptionResnetV1Attention, InceptionResnetV1Semiology):
            model = model_type(pretrained=False, max_frames=2)
            with patch.object(model.backbone, "forward", side_effect=AssertionError("Unexpected encoding")):
                for frames in (0, 3):
                    with self.assertRaisesRegex(ValueError, "1–2 frames"):
                        video = torch.zeros(1, frames, 4, 8, 8)
                        if isinstance(model, InceptionResnetV1Semiology):
                            model(video, torch.zeros(1, frames, 20))
                        else:
                            model(video)

    def test_training_updates_positions_but_keeps_frozen_weights(self) -> None:
        torch.manual_seed(4)
        model = InceptionResnetV1Attention(num_classes=2, pretrained=False, max_frames=4).eval()
        configure_trainable_layers(model)
        optimizer = build_optimizer(model)
        expected = {id(parameter) for parameter in model.parameters() if parameter.requires_grad}
        actual = [id(parameter) for group in optimizer.param_groups for parameter in group["params"]]
        self.assertEqual(set(actual), expected)
        self.assertEqual(len(actual), len(set(actual)))
        positions = model.position_embedding.detach().clone()
        frozen = next(parameter for parameter in model.backbone.parameters() if not parameter.requires_grad)
        original_frozen = frozen.detach().clone()
        loss = torch.nn.functional.cross_entropy(model(torch.rand(1, 2, 4, 32, 32)), torch.tensor([1]))
        loss.backward()
        self.assertTrue(torch.isfinite(model.position_embedding.grad).all())
        self.assertGreater(model.position_embedding.grad.abs().sum().item(), 0)
        optimizer.step()
        self.assertFalse(torch.equal(positions, model.position_embedding))
        torch.testing.assert_close(original_frozen, frozen)

    def test_checkpoint_roundtrip_preserves_capacity_and_predictions(self) -> None:
        model = InceptionResnetV1Attention(
            num_classes=2, pretrained=False, max_frames=5, attention_heads=4, dropout=.1,
        ).eval()
        configure_trainable_layers(model)
        optimizer = build_optimizer(model)
        video = torch.rand(1, 2, 4, 32, 32)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "attention.pth"
            save_checkpoint(path, model, optimizer, 1, {"loss": .2}, .6)
            restored, classes = load_model(path, torch.device("cpu"))
            self.assertEqual(classes, ("normal", "seizure"))
            self.assertEqual(restored.max_frames, 5)
            self.assertEqual(restored.dropout, .1)
            with torch.inference_mode():
                torch.testing.assert_close(restored(video), model(video))
            metadata = torch.load(path, weights_only=True)
            self.assertNotIn("bidirectional", metadata)
            metadata["model_name"] = "InceptionResnetV1GRUAttention"
            torch.save(metadata, path)
            with self.assertRaisesRegex(ValueError, "BiGRU checkpoints are incompatible"):
                load_model(path, torch.device("cpu"))


if __name__ == "__main__":
    unittest.main()
