from __future__ import annotations

import tempfile
import unittest
from collections import Counter
from pathlib import Path

from seizure_multimodal.facial_resnet.prepare_dataset import (
    stratified_video_plan,
    summarize_plan,
)
from seizure_multimodal.facial_resnet.train import resolve_device


class FacialDatasetPlanTest(unittest.TestCase):
    def test_class_stratified_three_way_video_split_is_reproducible(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for class_name in ("Normal", "Seizure"):
                directory = root / class_name
                directory.mkdir()
                for index in range(10):
                    (directory / f"S{index % 3}_{index}.mp4").touch()

            first = stratified_video_plan(root, seed=7, split_unit="video")
            second = stratified_video_plan(root, seed=7, split_unit="video")
            self.assertEqual(first, second)
            counts = Counter((row["split"], row["label"]) for row in first)
            self.assertEqual(counts[("train", "0")], 7)
            self.assertEqual(counts[("validation", "0")], 2)
            self.assertEqual(counts[("test", "0")], 1)
            self.assertEqual(counts[("train", "1")], 7)
            self.assertEqual(counts[("validation", "1")], 2)
            self.assertEqual(counts[("test", "1")], 1)

    def test_patient_split_has_no_patient_overlap(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for class_name in ("Normal", "Seizure"):
                directory = root / class_name
                directory.mkdir()
                for index in range(10):
                    (directory / f"S{index}_{index}.mp4").touch()

            rows = stratified_video_plan(root, seed=7, split_unit="patient")
            counts = Counter((row["split"], row["label"]) for row in rows)
            self.assertEqual(counts[("train", "0")], 7)
            self.assertEqual(counts[("validation", "0")], 2)
            self.assertEqual(counts[("test", "0")], 1)
            self.assertEqual(counts[("train", "1")], 7)
            self.assertEqual(counts[("validation", "1")], 2)
            self.assertEqual(counts[("test", "1")], 1)
            self.assertFalse(summarize_plan(rows)["patient_leakage_warning"])

    def test_summary_reports_patient_leakage(self) -> None:
        rows = [
            {"split": "train", "class_name": "normal", "patient_id": "S1"},
            {"split": "test", "class_name": "seizure", "patient_id": "S1"},
        ]
        summary = summarize_plan(rows)
        self.assertTrue(summary["patient_leakage_warning"])
        self.assertEqual(summary["patient_overlap"], ["S1"])

    def test_explicit_cpu_device(self) -> None:
        self.assertEqual(str(resolve_device("cpu")), "cpu")


if __name__ == "__main__":
    unittest.main()
