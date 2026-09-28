from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from seizure_multimodal.facial.anomaly_evaluation import evaluate_anomalies, summarize_anomalies
from seizure_multimodal.facial.movement_features import (
    FEATURE_NAMES, UnusableWindow, extract_movement_features, validate_manifest,
)
from seizure_multimodal.facial.unsupervised import (
    load_detector, predict_manifest, save_detector, train_detector,
)


EVENTS = {"minimum_event_duration_sec": 0., "gap_tolerance_sec": 0., "refractory_period_sec": 0.}


def arrays(scale: float = 1.) -> tuple[np.ndarray, np.ndarray]:
    geometry = np.zeros((5, 10, 2), np.float32)
    geometry[:, :, 0] = 1
    geometry[:, [2, 3, 5], 0] = np.linspace(.1, .2 * scale, 5)[:, None]
    face = np.zeros((5, 4, 4, 4), np.float32)
    face[:, 3] = np.linspace(.01, .03 * scale, 5)[:, None, None]
    return face, geometry.reshape(5, 20)


def write_rows(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def make_manifest(root: Path) -> Path:
    rows = []
    for split, count in (("train", 8), ("validation", 3), ("test", 3)):
        for index in range(count):
            identifier = f"{split}_{index}"
            face, geometry = arrays(1 + index * .05)
            np.save(root / f"{identifier}_face.npy", face)
            np.save(root / f"{identifier}_geometry.npy", geometry)
            rows.append({
                "window_id": identifier, "patient_id": split, "recording_id": split,
                "start_sec": index * 5, "end_sec": (index + 1) * 5,
                "split": split, "label": "", "sampled_fps": 1,
                "face_path": f"{identifier}_face.npy", "face_geometry_path": f"{identifier}_geometry.npy",
                "face_region_fraction": 1,
            })
    path = root / "video_manifest.csv"
    write_rows(path, rows)
    return path


def prediction(identifier: str, start: float, end: float, score: float | None, *, status: str = "scored") -> dict:
    return {"window_id": identifier, "patient_id": "P", "recording_id": "R", "split": "test",
            "start_sec": start, "end_sec": end, "anomaly_score": score,
            "status": status, "exclusion_reason": "tracking" if status == "excluded" else ""}


class MovementFeatureTests(unittest.TestCase):
    def test_missing_geometry_is_not_a_closed_eye_or_velocity_spike(self) -> None:
        face, geometry = arrays()
        geometry[2] = 0
        features = extract_movement_features(face, geometry, 2, minimum_valid_geometry=.8)
        self.assertEqual(features.shape, (len(FEATURE_NAMES),))
        self.assertAlmostEqual(features[FEATURE_NAMES.index("right_eye_aperture_mean")], .15)
        self.assertAlmostEqual(features[FEATURE_NAMES.index("right_eye_aperture_speed_mean")], .05)
        geometry[:] = 0
        with self.assertRaisesRegex(UnusableWindow, "insufficient_valid_geometry"):
            extract_movement_features(face, geometry, 2)

    def test_disconnected_observations_do_not_create_motion(self) -> None:
        face, geometry = arrays()
        geometry[1::2] = 0
        with self.assertRaisesRegex(UnusableWindow, "no_adjacent_valid_geometry"):
            extract_movement_features(face, geometry, 1, minimum_valid_geometry=.5)

    def test_invalid_flow_transition_is_ignored_and_bad_geometry_shape_fails(self) -> None:
        face, geometry = arrays()
        face[0, 3] = np.nan
        self.assertTrue(np.isfinite(extract_movement_features(face, geometry, 1)).all())
        face[3, 3] = np.nan
        with self.assertRaisesRegex(UnusableWindow, "nonfinite_flow"):
            extract_movement_features(face, geometry, 1)
        with self.assertRaisesRegex(ValueError, "aligned"):
            extract_movement_features(face, geometry[:3], 1)


class DetectorTests(unittest.TestCase):
    def test_train_ignores_labels_and_test_arrays_and_persists_threshold(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = make_manifest(root)
            original = train_detector(manifest, n_estimators=10)
            with manifest.open() as stream:
                rows = list(csv.DictReader(stream))
            for row in rows:
                row["label"] = "unknown-not-a-target"
                if row["split"] == "test":
                    row["face_path"] = "does_not_exist.npy"
            write_rows(manifest, rows)
            changed = train_detector(manifest, n_estimators=10)
            self.assertEqual(original["anomaly_threshold"], changed["anomaly_threshold"])
            self.assertEqual(original["reports"], changed["reports"])
            make_manifest(root)
            save_detector(changed, root / "model.joblib")
            restored = load_detector(root / "model.joblib")
            before, summary = predict_manifest(manifest, changed)
            after, _ = predict_manifest(manifest, restored)
            self.assertEqual(before, after)
            self.assertEqual(summary["scored_windows"], 3)
            self.assertIsNone(summary["reference_evaluation"])
            self.assertNotIn("seizure_probability", before[0])

    def test_split_leakage_fails_before_opening_arrays(self) -> None:
        records = [prediction("a", 0, 1, .2), {**prediction("b", 1, 2, .2), "split": "train"}]
        with self.assertRaisesRegex(ValueError, "multiple splits"):
            validate_manifest(records)
        with tempfile.TemporaryDirectory() as directory:
            manifest = make_manifest(Path(directory))
            checkpoint = train_detector(manifest, n_estimators=5)
            checkpoint["training_patient_ids"].append("test")
            with self.assertRaisesRegex(ValueError, "overlap"):
                predict_manifest(manifest, checkpoint)

    def test_quality_failures_remain_in_predictions_without_negative_labels(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = make_manifest(root)
            checkpoint = train_detector(manifest, n_estimators=5)
            for index in range(3):
                np.save(root / f"test_{index}_geometry.npy", np.zeros((5, 20)))
            records, summary = predict_manifest(manifest, checkpoint)
            self.assertEqual(summary["scored_windows"], 0)
            self.assertIsNone(summary["candidate_events_per_scored_hour"])
            self.assertTrue(all(row["anomaly_score"] is None and row["candidate_window"] is None for row in records))

    def test_missing_validation_and_nonfinite_settings_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest = make_manifest(Path(directory))
            with self.assertRaisesRegex(ValueError, "validation_quantile"):
                train_detector(manifest, validation_quantile=np.nan)
            with manifest.open() as stream:
                rows = [row for row in csv.DictReader(stream) if row["split"] != "validation"]
            write_rows(manifest, rows)
            with self.assertRaisesRegex(ValueError, "validation patients"):
                train_detector(manifest)

    def test_cli_train_predict_evaluate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = make_manifest(root)
            checkpoint, output = root / "model.joblib", root / "predictions"
            prefix = [sys.executable, "-m", "seizure_multimodal.facial.unsupervised"]
            commands = [
                ["train", "--manifest", str(manifest), "--checkpoint", str(checkpoint), "--trees", "5"],
                ["predict", "--manifest", str(manifest), "--checkpoint", str(checkpoint), "--output", str(output)],
            ]
            write_rows(root / "labels.csv", [{"window_id": "test_0", "label": "0"},
                                            {"window_id": "test_1", "label": "1"}])
            commands.append(["evaluate", "--predictions", str(output / "anomaly_scores.csv"),
                             "--checkpoint", str(checkpoint), "--labels", str(root / "labels.csv"),
                             "--output", str(root / "evaluation.json")])
            for command in commands:
                completed = subprocess.run(prefix + command, capture_output=True, text=True,
                                           cwd=Path(__file__).resolve().parents[1], timeout=60)
                self.assertEqual(completed.returncode, 0, completed.stderr)
            result = json.loads((root / "evaluation.json").read_text())
            self.assertEqual(result["reference_evaluation"]["labelled_scored_windows"], 2)
            self.assertEqual(result["reference_evaluation"]["unlabelled_scored_windows"], 1)
            self.assertIsNone(result["reference_evaluation"]["false_positives_per_hour"])
            other_checkpoint = train_detector(manifest, n_estimators=5)
            save_detector(other_checkpoint, checkpoint)
            completed = subprocess.run(prefix + commands[-1], capture_output=True, text=True,
                                       cwd=Path(__file__).resolve().parents[1], timeout=60)
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("not generated by this detector", completed.stderr)


class AnomalyEvaluationTests(unittest.TestCase):
    def test_coverage_is_a_union_and_events_cannot_cross_unknown_gaps(self) -> None:
        rows = [prediction("a", 0, 5, .8), prediction("b", 2, 7, .9),
                prediction("c", 7, 10, None, status="excluded"), prediction("d", 10, 15, .8)]
        summary = summarize_anomalies(rows, threshold=.5, event_settings={**EVENTS, "gap_tolerance_sec": 100.})
        self.assertAlmostEqual(summary["scored_window_hours"], 12 / 3600)
        self.assertAlmostEqual(summary["time_coverage_fraction"], 12 / 15)
        self.assertEqual([(row["start_sec"], row["end_sec"]) for row in summary["candidate_events"]],
                         [(0, 7), (10, 15)])
        self.assertIsNone(summary["reference_evaluation"])

    def test_full_annotation_coverage_enables_event_metrics(self) -> None:
        rows = [prediction("a", 0, 5, .2), prediction("b", 5, 10, .8)]
        annotations = [{"patient_id": "P", "recording_id": "R", "monitor_start_sec": 0,
                        "monitor_end_sec": 10, "seizure_onset_sec": 5, "seizure_offset_sec": 10}]
        result = evaluate_anomalies(rows, threshold=.5, event_settings=EVENTS, annotations=annotations)
        evaluation = result["reference_evaluation"]
        self.assertTrue(evaluation["temporal_metrics_available"])
        self.assertEqual(evaluation["classification"]["auroc"], 1)
        self.assertEqual(evaluation["event_level_sensitivity"], 1)
        rows[0].update(status="excluded", anomaly_score=None, exclusion_reason="tracking")
        result = evaluate_anomalies(rows, threshold=.5, event_settings=EVENTS, annotations=annotations)
        self.assertFalse(result["reference_evaluation"]["temporal_metrics_available"])
        self.assertIsNone(result["reference_evaluation"]["false_positives_per_hour"])

    def test_partial_labels_and_exclusions_never_become_negative_targets(self) -> None:
        rows = [prediction("a", 0, 5, .2), prediction("b", 5, 10, .8),
                prediction("c", 10, 15, None, status="excluded")]
        labels = [{"window_id": "a", "label": ""}, {"window_id": "b", "label": "1"},
                  {"window_id": "c", "label": "0"}]
        result = evaluate_anomalies(rows, threshold=.5, event_settings=EVENTS, labels=labels)
        metrics = result["reference_evaluation"]["classification"]
        self.assertEqual(metrics["evaluated_samples"], 1)
        self.assertEqual(metrics["tn"], 0)
        self.assertIsNone(metrics["specificity"])
        self.assertIsNone(metrics["auroc"])
        with self.assertRaisesRegex(ValueError, "Duplicate or unknown"):
            evaluate_anomalies(rows, threshold=.5, labels=[{"window_id": "other", "label": "0"}])

    def test_unpaired_timestamps_and_duplicate_events_are_rejected(self) -> None:
        rows = [prediction("a", 0, 5, .8)]
        annotation = {"patient_id": "P", "recording_id": "R", "monitor_start_sec": 0,
                      "monitor_end_sec": 5, "seizure_onset_sec": "", "seizure_offset_sec": 4}
        with self.assertRaisesRegex(ValueError, "both"):
            evaluate_anomalies(rows, threshold=.5, annotations=[annotation])
        annotation["seizure_onset_sec"] = 1
        with self.assertRaisesRegex(ValueError, "Duplicate reference"):
            evaluate_anomalies(rows, threshold=.5, annotations=[annotation, annotation])

    def test_event_duration_and_refractory_are_applied(self) -> None:
        rows = [prediction("a", 0, 1, .8), prediction("b", 1, 3, .1),
                prediction("c", 3, 6, .8), prediction("d", 6, 7, .1),
                prediction("e", 7, 10, .8)]
        result = summarize_anomalies(rows, threshold=.5, event_settings={
            "minimum_event_duration_sec": 2., "gap_tolerance_sec": 0., "refractory_period_sec": 2.,
        })
        self.assertEqual([(row["start_sec"], row["end_sec"]) for row in result["candidate_events"]], [(3, 6)])


if __name__ == "__main__":
    unittest.main()
