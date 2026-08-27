from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from body_pose.annotation_adapters.generic_tabular import GenericTabularAdapter
from body_pose.annotation_adapters.seizeit2 import SeizeIT2Adapter


class GenericTabularAdapterTest(unittest.TestCase):
    def test_frame_times_and_labels_are_configured(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "hukm.csv"
            pd.DataFrame({
                "Patient": ["P001", "P001"],
                "VideoName": ["videoA", "videoA"],
                "BeginFrame": [0, 250],
                "FinishFrame": [250, 300],
                "ClinicalState": ["Normal", "Seizure"],
            }).to_csv(path, index=False)
            adapter = GenericTabularAdapter(path, {
                "dataset": "hukm",
                "column_map": {
                    "media_id": "VideoName",
                    "patient_id": "Patient",
                    "start_sec": "BeginFrame",
                    "end_sec": "FinishFrame",
                    "label": "ClinicalState",
                },
                "label_map": {"Normal": "normal", "Seizure": "ictal"},
                "time_unit": "frames",
                "fps": 25,
            })
            result = adapter.canonical(active_labels=["normal", "ictal"])
            self.assertEqual(result["start_sec"].tolist(), [0.0, 10.0])
            self.assertEqual(result["end_sec"].tolist(), [10.0, 12.0])
            self.assertEqual(result["label"].tolist(), ["normal", "ictal"])
            self.assertEqual(result["recording_id"].tolist(), ["videoA", "videoA"])


class SeizeIT2AdapterTest(unittest.TestCase):
    def test_real_schema_mapping_and_impedance_exclusion(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            eeg = root / "sub-001" / "ses-01" / "eeg"
            eeg.mkdir(parents=True)
            path = eeg / "sub-001_ses-01_task-szMonitoring_run-03_events.tsv"
            pd.DataFrame({
                "onset": [10.0, 20.0, 30.0],
                "duration": [2.0, 5.0, 7.0],
                "eventType": ["impd", "bckg", "sz_foc_ia_nm"],
                "lateralization": ["n/a", "n/a", "left"],
                "localization": ["n/a", "n/a", "temp"],
                "vigilance": ["n/a", "awake", "asleep"],
                "confidence": ["n/a", "n/a", 0.9],
                "channels": ["n/a", "n/a", "n/a"],
                "dateTime": ["2000-01-01 00:00:00"] * 3,
                "recordingDuration": [100.0] * 3,
            }).to_csv(path, sep="\t", index=False)
            result = SeizeIT2Adapter(root).canonical(active_labels=["normal", "ictal"])
            self.assertEqual(len(result), 2)
            self.assertEqual(result["label"].tolist(), ["normal", "ictal"])
            self.assertEqual(result["start_sec"].tolist(), [20.0, 30.0])
            self.assertEqual(result["end_sec"].tolist(), [25.0, 37.0])
            self.assertEqual(result["patient_id"].tolist(), ["sub-001", "sub-001"])
            self.assertEqual(result.iloc[1]["source_event_type"], "sz_foc_ia_nm")
            self.assertEqual(result.iloc[1]["localization"], "temp")
            self.assertTrue(str(result.iloc[1]["source_tsv_path"]).endswith("_events.tsv"))


if __name__ == "__main__":
    unittest.main()
