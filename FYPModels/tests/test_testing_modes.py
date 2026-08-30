from __future__ import annotations

import json
import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np

from seizure_multimodal.checkpoints.provenance import sha256_file
from seizure_multimodal.config import load_config
from seizure_multimodal.data_inspection import (
    inspect_eeg_csv,
    inspect_label_workbook,
    inspect_mp4,
    resolved_label_records,
)
from seizure_multimodal.discovery import ModalitySource, discover_sources, validate_verified_pair
from seizure_multimodal.eeg.csv_inference import (
    adapt_csv_for_notebook_model,
    load_eeg_csv,
    normalized_label_mapping,
)
from seizure_multimodal.testing import (
    SIMULATED_WARNING,
    WorkflowResult,
    _video_metrics,
    result_metadata,
    run_simulated_workflow,
    run_test_mode,
)
from seizure_multimodal.video.raw_pipeline import load_video_annotation, reference_label_for_interval


ROOT = Path(__file__).resolve().parents[1]


def _configured_path(config: dict, key: str) -> Path:
    return ROOT / str(config["data"][key])


class ActualDataInspectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = load_config(ROOT / "configs" / "inference.yaml")
        cls.video_path = _configured_path(cls.config, "video")
        cls.label_path = _configured_path(cls.config, "video_labels")
        cls.eeg_path = _configured_path(cls.config, "eeg_csv")

    def setUp(self) -> None:
        missing = [
            path
            for path in (self.video_path, self.label_path, self.eeg_path)
            if not path.is_file()
        ]
        if missing:
            self.skipTest(
                "Local sample data is not present: "
                + ", ".join(str(path.relative_to(ROOT)) for path in missing)
            )

    def test_actual_mp4_metadata(self) -> None:
        metadata = inspect_mp4(self.video_path)
        self.assertEqual(metadata.frame_count, 25081)
        self.assertAlmostEqual(metadata.fps, 29.97002997)
        self.assertAlmostEqual(metadata.duration_sec, 836.86936667)
        self.assertEqual((metadata.width, metadata.height, metadata.codec), (1920, 1080, "avc1"))

    def test_label_workbook_catalog_and_verified_row(self) -> None:
        workbook = inspect_label_workbook(self.label_path)
        self.assertEqual(workbook["sheet_names"], ["Sheet1"])
        self.assertEqual(workbook["sheets"][0]["data_rows"], 33)
        records = resolved_label_records(self.label_path)
        selected = [
            row
            for row in records
            if row["resolved_PatID"] == "Pat01"
            and row["#Seizure"] == "Sz2"
            and row["resolved_Seizure Type"] == "PG"
        ]
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0]["EEG onset seconds"], 624.0)
        self.assertEqual(selected[0]["Clinical Onset seconds"], 640.0)
        self.assertEqual(
            sha256_file(self.video_path),
            "819634aac6b8a6fe5779b2da786b6b5697524eee761acccd270c4968af135c67",
        )

    def test_video_annotation_policy_does_not_invent_offset(self) -> None:
        annotation = load_video_annotation(
            self.label_path,
            836.87,
            patient_id="Pat01",
            seizure_id="Sz2",
            seizure_type="PG",
        )
        self.assertEqual(annotation.alignment_status, "onsets_aligned_but_offset_missing")
        self.assertEqual(reference_label_for_interval(annotation, 0, 1), 0)
        self.assertIsNone(reference_label_for_interval(annotation, 630, 631))
        self.assertEqual(reference_label_for_interval(annotation, 641, 642), 1)
        self.assertIsNone(annotation.seizure_end_sec)

    def test_actual_eeg_csv_schema_and_label_mapping(self) -> None:
        inspection = inspect_eeg_csv(self.eeg_path)
        self.assertEqual((inspection["rows"], inspection["columns"]), (11500, 180))
        self.assertEqual(inspection["feature_count"], 178)
        self.assertEqual(inspection["label_values"], {str(i): 2300 for i in range(1, 6)})
        self.assertEqual(inspection["timestamp_columns"], [])
        self.assertEqual(inspection["patient_id_columns"], [])
        mapping, positive = normalized_label_mapping(self.config)
        self.assertEqual(positive, 1)
        self.assertEqual(mapping[1], "ictal")
        data = load_eeg_csv(self.eeg_path, self.config)
        self.assertEqual(data.features.shape, (11500, 178))
        self.assertEqual(int(data.binary_labels.sum()), 2300)

    @unittest.skipUnless(importlib.util.find_spec("scipy"), "SciPy unavailable")
    def test_notebook_csv_adaptation_shape_and_global_normalization(self) -> None:
        values = np.vstack((np.arange(178), np.arange(178) + 10)).astype(np.float32)
        adapted = adapt_csv_for_notebook_model(values)
        self.assertEqual(adapted.shape, (2, 1024, 18))
        self.assertAlmostEqual(float(adapted.mean()), 0.0, places=5)
        self.assertAlmostEqual(float(adapted.std()), 1.0, places=5)


class ModeSafeguardTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load_config(ROOT / "configs" / "inference.yaml")

    def test_auto_detection_keeps_sources_independent(self) -> None:
        required = [
            ROOT / str(self.config["data"]["video"]),
            ROOT / str(self.config["data"]["video_labels"]),
            ROOT / str(self.config["data"]["eeg_csv"]),
        ]
        missing = [path for path in required if not path.is_file()]
        if missing:
            self.skipTest(
                "Local sample data is not present: "
                + ", ".join(str(path.relative_to(ROOT)) for path in missing)
            )
        sources = discover_sources(ROOT, self.config)
        self.assertEqual(len(sources), 2)
        video = next(source for source in sources if source.has_video)
        eeg = next(source for source in sources if source.has_eeg)
        self.assertFalse(video.has_eeg)
        self.assertFalse(eeg.has_video)
        self.assertFalse(video.is_verified_pair)
        self.assertFalse(eeg.is_verified_pair)
        self.assertEqual(video.patient_id, "Pat01")
        valid, reasons = validate_verified_pair(video, eeg)
        self.assertFalse(valid)
        self.assertIn("is_verified_pair is false", reasons)

    def test_missing_modalities_do_not_fail_unimodal_mode(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            result = run_test_mode(root=directory, config=self.config, mode="unimodal")
            self.assertEqual(result, {})
            self.assertTrue((Path(directory) / "results" / "data_inspection.json").exists())

    def test_null_video_labels_disables_label_autodiscovery(self) -> None:
        sample = ROOT / str(self.config["data"]["video"])
        if not sample.is_file():
            self.skipTest(f"Local sample video is not present: {sample.relative_to(ROOT)}")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            label = root / "Label.xlsx"
            label.write_bytes(b"should not be discovered")
            config = {
                "data": {
                    "video": str(sample),
                    "video_labels": None,
                }
            }
            sources = discover_sources(root, config)
            self.assertEqual(sources[0].label_path, None)

    def test_video_metrics_are_prediction_only_without_label_workbook(self) -> None:
        rows = [
            {
                "recording_id": "recording-001",
                "start_sec": 0.0,
                "end_sec": 1.0,
                "video_probability": 0.8,
            }
        ]
        metrics = _video_metrics(
            rows,
            {
                "annotation": {
                    "alignment_status": "annotation_file_missing",
                    "patient_id": None,
                    "eeg_onset_sec": None,
                    "clinical_onset_sec": None,
                    "seizure_end_sec": None,
                }
            },
        )
        self.assertTrue(metrics["prediction_only"])
        self.assertEqual(metrics["sensitivity"], "N/A")
        self.assertEqual(metrics["predicted_event_count"], 1)

    def test_video_metrics_keep_model_events_when_labels_are_present(self) -> None:
        rows = [
            {"recording_id": "recording-001", "start_sec": 0.0, "end_sec": 1.0,
             "video_probability": 0.2, "reference_label": 0},
            {"recording_id": "recording-001", "start_sec": 1.0, "end_sec": 2.0,
             "video_probability": 0.8, "reference_label": 1},
        ]
        metrics = _video_metrics(rows, {"annotation": {
            "alignment_status": "aligned", "patient_id": "Pat01",
            "eeg_onset_sec": 0.5, "clinical_onset_sec": 1.0,
            "seizure_end_sec": None,
        }})

        self.assertEqual(metrics["predicted_event_count"], 1)
        self.assertEqual(len(metrics["predicted_events"]), 1)
        self.assertEqual(metrics["predicted_events"][0]["start_sec"], 1.0)

    def test_simulated_pairing_requires_explicit_acknowledgement(self) -> None:
        metadata = result_metadata(
            test_mode="video_only",
            video_model_used=True,
            eeg_model_used=False,
            fusion_used=False,
            pairing_type="none",
            video_checkpoint="video.pth",
            eeg_checkpoint="eeg.h5",
        )
        video = WorkflowResult(
            "video", "completed", [], np.array([0.2]), np.array([0]), np.zeros((1, 384)), {}, metadata
        )
        eeg = WorkflowResult(
            "eeg", "completed", [], np.array([0.4]), np.array([0]), np.zeros((1, 32)), {}, metadata
        )
        with tempfile.TemporaryDirectory() as directory, self.assertRaises(PermissionError):
            run_simulated_workflow(
                Path(directory), self.config, video, eeg, allow_simulated_pairing=False
            )

    def test_simulated_pairing_is_deterministic_and_marked(self) -> None:
        video_metadata = result_metadata(
            test_mode="video_only",
            video_model_used=True,
            eeg_model_used=False,
            fusion_used=False,
            pairing_type="none",
            video_checkpoint="video.pth",
            eeg_checkpoint="eeg.h5",
        )
        eeg_metadata = {**video_metadata, "test_mode": "eeg_only"}
        video = WorkflowResult(
            "video",
            "completed",
            [],
            np.array([0.2, 0.8, 0.9]),
            np.array([0, 1, 1]),
            np.zeros((3, 384)),
            {},
            video_metadata,
        )
        eeg = WorkflowResult(
            "eeg",
            "completed",
            [],
            np.array([0.4, 0.6]),
            np.array([0, 1]),
            np.zeros((2, 32)),
            {},
            eeg_metadata,
        )
        with tempfile.TemporaryDirectory() as directory:
            result = run_simulated_workflow(
                Path(directory), self.config, video, eeg, allow_simulated_pairing=True
            )
            manifest = (Path(directory) / "results" / "simulated_multimodal" / "pairing_manifest.csv")
            warning = (Path(directory) / "results" / "simulated_multimodal" / "WARNING.txt")
            payload = json.loads(
                (Path(directory) / "results" / "simulated_multimodal" / "metrics.json").read_text()
            )
            manifest_text = manifest.read_text()
            warning_text = warning.read_text()
        np.testing.assert_allclose(result.probabilities, [0.3, 0.7])
        self.assertEqual(result.rows[0]["sim_window_id"], "sim_window_0001")
        self.assertIn("first N", manifest_text)
        self.assertIn("SIMULATED MULTIMODAL PAIRING", warning_text)
        self.assertEqual(payload["metadata"]["pairing_type"], "simulated")
        self.assertFalse(payload["artificial_metrics_enabled"])
        self.assertIn("SIMULATED MULTIMODAL PAIRING", SIMULATED_WARNING)

    def test_result_metadata_contains_all_required_fields(self) -> None:
        metadata = result_metadata(
            test_mode="video_only",
            video_model_used=True,
            eeg_model_used=False,
            fusion_used=False,
            pairing_type="none",
            video_checkpoint="v",
            eeg_checkpoint="e",
        )
        self.assertEqual(
            set(metadata),
            {
                "test_mode",
                "video_model_used",
                "EEG_model_used",
                "fusion_used",
                "pairing_type",
                "training_performed",
                "video_checkpoint",
                "EEG_checkpoint",
            },
        )
        self.assertFalse(metadata["training_performed"])


if __name__ == "__main__":
    unittest.main()
