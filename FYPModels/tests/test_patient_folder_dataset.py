from __future__ import annotations

import contextlib
import csv
import io
import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import create_autospec, patch

import numpy as np

from FYPModels.preprocess import prepare_dataset
from FYPModels.seizure_multimodal.facial_resnet.train import FaceWindowDataset


class PatientFolderDatasetTest(unittest.TestCase):
    def make_dataset(self, root: Path, count: int = 10) -> None:
        for index in range(count):
            directory = root / f"Patient {index}" / "session" / "video"
            directory.mkdir(parents=True)
            for filename in ("clip.AVI", "clip.mp4", "report.doc", "._hidden.avi"):
                (directory / filename).touch()
            hidden = root / f"Patient {index}" / ".hidden"
            hidden.mkdir()
            (hidden / "ignored.avi").touch()

    def test_nested_videos_have_blank_labels_and_disjoint_patient_splits(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_dataset(root)
            rows = prepare_dataset.patient_folder_plan(root, seed=7)
            self.assertEqual(len(rows), 20)
            self.assertEqual({row["label"] for row in rows}, {""})
            self.assertEqual({row["class_name"] for row in rows}, {"unlabelled"})
            self.assertEqual(len({row["recording_id"] for row in rows}), 20)
            self.assertEqual(Counter(row["split"] for row in rows),
                             {"train": 14, "validation": 4, "test": 2})
            summary = prepare_dataset.summarize_plan(rows)
            self.assertEqual(summary["patient_overlap"], [])
            self.assertEqual(summary["train_patients"], 7)
            self.assertEqual(rows, prepare_dataset.patient_folder_plan(root, seed=7))
            moved = root.parent / (root.name + "_moved")
            root.rename(moved)
            try:
                moved_rows = prepare_dataset.patient_folder_plan(moved, seed=7)
                for before, after in zip(rows, moved_rows):
                    self.assertEqual({k: v for k, v in before.items() if k != "source_video"},
                                     {k: v for k, v in after.items() if k != "source_video"})
            finally:
                moved.rename(root)

    def test_three_patients_keep_all_splits_nonempty(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_dataset(root, count=3)
            rows = prepare_dataset.patient_folder_plan(root, train_fraction=.9, validation_fraction=.05)
            self.assertEqual(Counter(row["split"] for row in rows),
                             {"train": 2, "validation": 2, "test": 2})

    def test_missing_patient_group_and_too_few_patients_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_dataset(root, count=2)
            with self.assertRaisesRegex(ValueError, "three patient"):
                prepare_dataset.patient_folder_plan(root)
            (root / "unassigned.avi").touch()
            with self.assertRaisesRegex(ValueError, "patient subfolder"):
                prepare_dataset.patient_folder_plan(root)

    def test_plan_only_writes_inventory_without_decoding_videos(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_dataset(root / "input")
            output = root / "output"
            argv = ["prepare_dataset", "--dataset-root", str(root / "input"),
                    "--layout", "patient-folders", "--output", str(output), "--plan-only"]
            with patch("sys.argv", argv), contextlib.redirect_stdout(io.StringIO()), \
                 patch("cv2.VideoCapture", side_effect=AssertionError("Unexpected video decoding")):
                prepare_dataset.main()
            with (output / "video_split.csv").open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 20)
            self.assertTrue(all(row["label"] == "" for row in rows))
            self.assertFalse((output / "video_manifest.csv").exists())
            self.assertEqual(json.loads((output / "split_summary.json").read_text())["patient_overlap"], [])

    def test_cli_passes_blank_label_and_duration_to_current_extractor(self) -> None:
        from FYPModels.preprocess._facial import face_preprocess
        import cv2

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_dataset(root / "input")
            argv = ["prepare_dataset", "--dataset-root", str(root / "input"),
                    "--layout", "patient-folders", "--output", str(root / "output"),
                    "--limit", "1", "--max-duration-seconds", "4"]
            capture = SimpleNamespace(get=lambda key: {cv2.CAP_PROP_FPS: 25,
                                                       cv2.CAP_PROP_FRAME_COUNT: 250}[key],
                                      release=lambda: None)
            extract = create_autospec(face_preprocess.extract_video,
                                      return_value=SimpleNamespace(windows_written=1))
            with patch("sys.argv", argv), contextlib.redirect_stdout(io.StringIO()), \
                 patch("cv2.VideoCapture", return_value=capture) as open_capture, \
                 patch.object(face_preprocess, "extract_video", extract):
                prepare_dataset.main()
            open_capture.assert_called_once()
            extract.assert_called_once()
            self.assertEqual(extract.call_args.kwargs["label"], "")
            self.assertEqual(extract.call_args.kwargs["max_duration_seconds"], 4)
            self.assertIn(extract.call_args.kwargs["split"], {"train", "validation", "test"})

    def test_existing_manifest_prevents_plan_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_dataset(root / "input")
            output = root / "output"
            output.mkdir()
            (output / "video_manifest.csv").write_text("existing manifest")
            (output / "video_split.csv").write_text("existing plan")
            argv = ["prepare_dataset", "--dataset-root", str(root / "input"),
                    "--layout", "patient-folders", "--output", str(output), "--plan-only"]
            with patch("sys.argv", argv), self.assertRaises(FileExistsError):
                prepare_dataset.main()
            self.assertEqual((output / "video_split.csv").read_text(), "existing plan")


class UnlabelledTrainingTest(unittest.TestCase):
    def write_manifest(self, root: Path, rows: list[dict[str, str]]) -> Path:
        path = root / "video_manifest.csv"
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=("split", "label", "face_detection_fraction", "face_path"))
            writer.writeheader()
            writer.writerows(rows)
        return path

    def test_unknown_labels_fail_before_loading_arrays(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for label in ("", "unknown", "-1", "2"):
                with self.subTest(label=label):
                    manifest = self.write_manifest(root, [{"split": "train", "label": label,
                        "face_detection_fraction": "1", "face_path": "missing.npy"}])
                    with self.assertRaisesRegex(ValueError, "requires annotated labels"):
                        FaceWindowDataset(manifest, "train")

    def test_annotated_split_still_loads_without_inspecting_test_labels(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            np.save(root / "face.npy", np.zeros((2, 4, 4, 4), np.float32))
            manifest = self.write_manifest(root, [
                {"split": "train", "label": "1", "face_detection_fraction": "1", "face_path": "face.npy"},
                {"split": "test", "label": "", "face_detection_fraction": "1", "face_path": "missing.npy"},
            ])
            dataset = FaceWindowDataset(manifest, "train")
            self.assertEqual(len(dataset), 1)
            video, label = dataset[0]
            self.assertEqual(tuple(video.shape), (2, 4, 4, 4))
            self.assertEqual(label.item(), 1)


if __name__ == "__main__":
    unittest.main()
