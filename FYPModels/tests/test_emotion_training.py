from __future__ import annotations

import contextlib
import csv
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from torch.utils.data import DataLoader

from seizure_multimodal.facial.emotion_training import MicroExpressionDataset, multitask_loss, run_epoch
from seizure_multimodal.facial.facial_model import EMOTIONS, InceptionResnetV1Semiology
from seizure_multimodal.facial import train_emotions


SIGNS = ("fear", "crying", "laughter", "grimacing", "oral_automatism", "motor_asymmetry")


def fixture(root: Path, fps: int = 25) -> tuple[Path, Path]:
    np.save(root / "face.npy", np.random.default_rng(4).random((2, 4, 16, 16)).astype(np.float32))
    np.save(root / "geometry.npy", np.ones((2, 20), np.float32))
    rows = [{"window_id": name, "patient_id": split, "recording_id": split, "split": split,
             "start_sec": index, "end_sec": index + 1, "sampled_fps": fps,
             "face_detection_fraction": 1, "face_path": "face.npy", "face_geometry_path": "geometry.npy",
             "label": label}
            for index, (name, split, label) in enumerate([
                ("W1", "train", "1"), ("W2", "train", ""), ("W3", "validation", "0"), ("W4", "test", ""),
            ])]
    rows[-1]["face_path"] = "test_must_not_be_loaded.npy"
    manifest = root / "manifest.csv"
    with manifest.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    annotations = root / "annotations.csv"
    annotations.write_text("window_id,emotion,semiologies,annotator_id\n"
                           "W1,fear,,C1\nW2,,none,C1\nW3,neutral,none,C1\n", encoding="utf-8")
    return manifest, annotations


class EmotionTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls) -> None:
        torch.set_num_threads(cls.previous_threads)

    def test_dataset_distinguishes_missing_and_negative_labels(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest, annotations = fixture(Path(directory))
            data = MicroExpressionDataset(manifest, annotations, "train", emotion_classes=EMOTIONS,
                                          semiology_classes=SIGNS)
            self.assertEqual(len(data), 2)
            self.assertEqual(data[0]["emotion"].item(), EMOTIONS.index("fear"))
            self.assertTrue((data[0]["semiology"] == -1).all())
            self.assertTrue((data[1]["semiology"] == 0).all())
            self.assertEqual(data[1]["emotion"].item(), -1)
            self.assertEqual(data[1]["seizure"].item(), -1)

    def test_dataset_rejects_low_fps_and_ambiguous_signs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, annotations = fixture(root, fps=5)
            with self.assertRaisesRegex(ValueError, "requires at least 25 fps"):
                MicroExpressionDataset(manifest, annotations, "train", emotion_classes=EMOTIONS,
                                       semiology_classes=SIGNS)
            manifest, annotations = fixture(root)
            annotations.write_text("window_id,emotion,semiologies\nW1,fear,none|fear\n")
            with self.assertRaisesRegex(ValueError, "ambiguous"):
                MicroExpressionDataset(manifest, annotations, "train", emotion_classes=EMOTIONS,
                                       semiology_classes=SIGNS)

    def test_missing_task_labels_contribute_no_gradients(self) -> None:
        outputs = {"emotion_logits": torch.randn(2, 7, requires_grad=True),
                   "semiology_logits": torch.randn(2, 6, requires_grad=True),
                   "seizure_logits": torch.randn(2, 2, requires_grad=True)}
        targets = {"emotion": torch.tensor([1, -1]), "semiology": torch.full((2, 6), -1.),
                   "seizure": torch.tensor([0, 1])}
        loss, parts = multitask_loss(outputs, targets)
        loss.backward()
        self.assertNotIn("semiology_loss", parts)
        self.assertIsNone(outputs["semiology_logits"].grad)
        torch.testing.assert_close(outputs["emotion_logits"].grad[1], torch.zeros(7))
        self.assertTrue(torch.isfinite(loss))
        for value in targets.values():
            value.fill_(-1)
        with self.assertRaisesRegex(ValueError, "no observed"):
            multitask_loss(outputs, targets)

    def test_validation_loss_does_not_depend_on_partial_label_batching(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest, annotations = fixture(Path(directory))
            data = MicroExpressionDataset(manifest, annotations, "train", emotion_classes=EMOTIONS,
                                          semiology_classes=SIGNS)
            model = InceptionResnetV1Semiology(pretrained=False, max_frames=4)
            single = run_epoch(model, DataLoader(data, batch_size=1), torch.device("cpu"), None)
            together = run_epoch(model, DataLoader(data, batch_size=2), torch.device("cpu"), None)
            self.assertAlmostEqual(single["total_loss"], together["total_loss"], places=5)

    def test_emotion_training_runs_and_saves_attention_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, annotations = fixture(root)
            checkpoint = root / "emotion.pth"
            argv = ["train_emotions", "--manifest", str(manifest), "--annotations", str(annotations),
                    "--checkpoint", str(checkpoint), "--epochs", "1", "--batch-size", "2",
                    "--max-frames", "4", "--device", "cpu"]
            model = InceptionResnetV1Semiology(pretrained=False, max_frames=4)
            original_positions = model.position_embedding.detach().clone()
            with patch("sys.argv", argv), contextlib.redirect_stdout(io.StringIO()), patch(
                "seizure_multimodal.facial.facial_model.InceptionResnetV1Semiology", return_value=model,
            ):
                train_emotions.main()
            saved = torch.load(checkpoint, weights_only=True)
            self.assertEqual(saved["model_name"], "InceptionResnetV1Semiology")
            self.assertEqual(saved["max_frames"], 4)
            self.assertIn("validation_metrics", saved)
            self.assertFalse(torch.equal(original_positions, saved["model_state_dict"]["position_embedding"]))
            self.assertFalse(any("gru" in key for key in saved["model_state_dict"]))


if __name__ == "__main__":
    unittest.main()
