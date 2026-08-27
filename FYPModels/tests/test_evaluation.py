from __future__ import annotations

import unittest

from seizure_multimodal.evaluation.agreement import cohen_kappa
from seizure_multimodal.evaluation.bootstrap import patient_bootstrap_confidence_intervals
from seizure_multimodal.evaluation.classification import binary_classification_metrics
from seizure_multimodal.evaluation.events import (
    TemporalEvent,
    TemporalWindow,
    event_level_sensitivity,
    false_positives_per_hour,
    probabilities_to_events,
)
from seizure_multimodal.evaluation.localization import localization_metrics
from seizure_multimodal.evaluation.thresholds import select_threshold


class ClassificationTests(unittest.TestCase):
    def test_sensitivity_specificity_auroc_f1(self) -> None:
        metrics = binary_classification_metrics([0, 0, 1, 1], [0.1, 0.8, 0.7, 0.9])
        self.assertEqual((metrics["tn"], metrics["fp"], metrics["fn"], metrics["tp"]), (1, 1, 0, 2))
        self.assertAlmostEqual(metrics["sensitivity"], 1.0)
        self.assertAlmostEqual(metrics["specificity"], 0.5)
        self.assertAlmostEqual(metrics["f1"], 0.8)
        self.assertAlmostEqual(metrics["auroc"], 0.75)

    def test_auroc_is_not_based_on_hard_predictions(self) -> None:
        metrics = binary_classification_metrics([0, 1, 0, 1], [0.49, 0.51, 0.48, 0.52])
        self.assertAlmostEqual(metrics["auroc"], 1.0)

    def test_cohen_kappa_model_vs_reference(self) -> None:
        self.assertAlmostEqual(cohen_kappa([0, 0, 1, 1], [0, 1, 1, 1]), 0.5)

    def test_threshold_selection_rejects_test_set(self) -> None:
        with self.assertRaisesRegex(ValueError, "validation split"):
            select_threshold([0, 1], [0.1, 0.9], split="test")

    def test_threshold_selection_on_validation(self) -> None:
        result = select_threshold([0, 1, 1], [0.1, 0.6, 0.7], split="validation", candidates=[0.5, 0.8])
        self.assertEqual(result["threshold"], 0.5)


class EventTests(unittest.TestCase):
    def test_thresholding_and_adjacent_window_merging(self) -> None:
        windows = [TemporalWindow("P1", "R1", i * 5, (i + 1) * 5) for i in range(5)]
        events = probabilities_to_events(windows, [0.1, 0.8, 0.9, 0.2, 0.7])
        self.assertEqual([(e.start_sec, e.end_sec) for e in events], [(5, 15), (20, 25)])

    def test_gap_tolerance_and_minimum_duration(self) -> None:
        windows = [TemporalWindow("P1", "R1", i * 5, (i + 1) * 5) for i in range(4)]
        events = probabilities_to_events(
            windows,
            [0.8, 0.1, 0.9, 0.1],
            gap_tolerance_sec=5,
            minimum_event_duration_sec=10,
        )
        self.assertEqual([(e.start_sec, e.end_sec) for e in events], [(0, 15)])

    def test_event_sensitivity_false_alarms_per_hour_and_localization(self) -> None:
        predictions = [
            TemporalEvent("P1", "R1", 12, 19),
            TemporalEvent("P1", "R1", 30, 40),
        ]
        references = [TemporalEvent("P1", "R1", 10, 20)]
        monitoring = [TemporalEvent("P1", "R1", 0, 3600)]
        self.assertEqual(event_level_sensitivity(predictions, references), 1.0)
        fp = false_positives_per_hour(predictions, references, monitoring)
        self.assertEqual(fp["false_alarm_events"], 1)
        self.assertAlmostEqual(fp["nonseizure_monitoring_hours"], 3590 / 3600)
        self.assertAlmostEqual(fp["false_positives_per_hour"], 3600 / 3590)
        localization = localization_metrics(
            predictions,
            references,
            onset_reference="eeg_onset",
            offset_annotations_available=True,
        )
        self.assertEqual(localization["onset"]["mean_signed_error_sec"], 2.0)
        self.assertEqual(localization["offset"]["mean_signed_error_sec"], -1.0)

    def test_missing_offset_annotation_is_na(self) -> None:
        result = localization_metrics(
            [TemporalEvent("P1", "R1", 12, 19)],
            [TemporalEvent("P1", "R1", 10, 20)],
            onset_reference="clinical_onset",
            offset_annotations_available=False,
        )
        self.assertIsNone(result["offset"])

    def test_patient_bootstrap_returns_ci(self) -> None:
        records = [("P1", 1.0), ("P1", 0.0), ("P2", 1.0), ("P3", 1.0)]
        result = patient_bootstrap_confidence_intervals(
            records,
            patient_id=lambda row: row[0],
            metric=lambda rows: {"mean": sum(row[1] for row in rows) / len(rows)},
            n_resamples=50,
        )
        self.assertIn("lower", result["mean"])
        self.assertEqual(result["mean"]["method"], "patient-level percentile bootstrap")


if __name__ == "__main__":
    unittest.main()
