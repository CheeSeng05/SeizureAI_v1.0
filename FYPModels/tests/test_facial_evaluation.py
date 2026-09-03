from __future__ import annotations

import unittest

from seizure_multimodal.facial_resnet.evaluation import (
    evaluate_clip_records,
    evaluate_records,
    metric_summary,
    patient_bootstrap_clip_ci,
    patient_bootstrap_timeline_ci,
)


CONFIG = {
    "event_generation": {
        "probability_threshold": 0.5,
        "minimum_event_duration_sec": 0.0,
        "gap_tolerance_sec": 0.0,
        "refractory_period_sec": 0.0,
    },
    "onset_reference": "clinical_onset",
    "offset_annotations_available": True,
}


class FacialEvaluationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.records = [
            {"patient_id": "P1", "recording_id": "R1", "start_sec": 0, "end_sec": 5, "seizure_probability": 0.1},
            {"patient_id": "P1", "recording_id": "R1", "start_sec": 5, "end_sec": 10, "seizure_probability": 0.9},
            {"patient_id": "P2", "recording_id": "R2", "start_sec": 0, "end_sec": 5, "seizure_probability": 0.1},
            {"patient_id": "P2", "recording_id": "R2", "start_sec": 5, "end_sec": 10, "seizure_probability": 0.8},
        ]
        self.annotations = [
            {"patient_id": "P1", "recording_id": "R1", "monitor_start_sec": 0, "monitor_end_sec": 10, "seizure_onset_sec": 5, "seizure_offset_sec": 10},
            {"patient_id": "P2", "recording_id": "R2", "monitor_start_sec": 0, "monitor_end_sec": 10, "seizure_onset_sec": 5, "seizure_offset_sec": 10},
        ]

    def test_timeline_metrics_use_probabilities_and_timestamps(self) -> None:
        result = evaluate_records(self.records, self.annotations, CONFIG)
        summary = metric_summary(result)
        self.assertEqual(summary["sensitivity"], 1.0)
        self.assertEqual(summary["specificity"], 1.0)
        self.assertEqual(summary["event_sensitivity"], 1.0)
        self.assertEqual(summary["onset_mae_sec"], 0.0)
        self.assertEqual(summary["offset_mae_sec"], 0.0)

    def test_patient_bootstrap_returns_95_percent_intervals(self) -> None:
        result = patient_bootstrap_timeline_ci(
            self.records, self.annotations, CONFIG, n_resamples=20, seed=4
        )
        self.assertEqual(result["sensitivity"]["confidence"], 0.95)
        self.assertEqual(result["sensitivity"]["valid_resamples"], 20)

    def test_missing_monitoring_annotation_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires a monitoring annotation"):
            evaluate_records(self.records, self.annotations[:1], CONFIG)

    def test_clip_mode_does_not_invent_temporal_metrics(self) -> None:
        records = [
            {**row, "reference_label": int(index % 2)}
            for index, row in enumerate(self.records)
        ]
        result = evaluate_clip_records(records, threshold=0.5)
        self.assertFalse(result["temporal_metrics_available"])
        self.assertIsNone(result["event_level_sensitivity"])
        self.assertIsNone(result["false_positives_per_hour"]["false_positives_per_hour"])
        self.assertIsNone(result["localization"]["onset"])

    def test_clip_bootstrap_resamples_patients(self) -> None:
        records = [
            {**row, "reference_label": int(index % 2)}
            for index, row in enumerate(self.records)
        ]
        result = patient_bootstrap_clip_ci(
            records, threshold=0.5, n_resamples=20, seed=3
        )
        self.assertEqual(result["f1"]["confidence"], 0.95)


if __name__ == "__main__":
    unittest.main()
