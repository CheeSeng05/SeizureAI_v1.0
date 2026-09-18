from __future__ import annotations

import argparse
import contextlib
import csv
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from FYPModels.preprocess._facial import face_preprocess as preprocessing
from FYPModels.preprocess._facial.compare_preprocessing import build_experiments, comparison_row
from FYPModels.preprocess.preprocess_diagnostics import frame_diagnostics, summarize_frames
from FYPModels.preprocess.preprocess_options import (
    BASELINE_TUNING, add_tuning_arguments, can_reuse_box, tuning_kwargs, validate_tuning,
)


class PreprocessingComparisonTest(unittest.TestCase):
    def test_experiments_change_exactly_one_setting_and_deduplicate(self):
        experiments = build_experiments()
        self.assertEqual(len(experiments), 18)
        baseline = experiments[0]["tuning"]
        for experiment in experiments[1:]:
            differences = [key for key in baseline if baseline[key] != experiment["tuning"][key]]
            self.assertEqual(differences, [experiment["changed_parameter"]])
        self.assertEqual(len(build_experiments((.5, .6, .6), (.7,), ())), 7)
        self.assertEqual(BASELINE_TUNING["max_box_reuse_frames"], 15)

    def test_cli_seconds_and_frames_cannot_be_combined(self):
        parser = argparse.ArgumentParser()
        add_tuning_arguments(parser)
        args = parser.parse_args(["--max-box-reuse-seconds", ".6", "--min-face-tracking-confidence", ".7"])
        settings = tuning_kwargs(args)
        validate_tuning(settings)
        self.assertEqual(settings["max_box_reuse_seconds"], .6)
        self.assertEqual(settings["min_face_tracking_confidence"], .7)
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parser.parse_args(["--max-box-reuse-seconds", ".6", "--max-box-reuse-frames", "2"])

    def test_invalid_settings_are_rejected(self):
        for key, value in (("min_face_detection_confidence", 1.1),
                           ("min_pose_tracking_confidence", float("nan")),
                           ("max_box_reuse_seconds", -1),
                           ("max_box_reuse_seconds", float("inf")),
                           ("box_smoothing", 1), ("max_box_reuse_frames", 1.5)):
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                validate_tuning({**BASELINE_TUNING, key: value})

    def test_reuse_seconds_uses_elapsed_time_even_after_frame_gaps(self):
        self.assertFalse(can_reuse_box(missed_frames=1, elapsed_seconds=1.2, max_frames=15, max_seconds=.6))
        self.assertTrue(can_reuse_box(missed_frames=8, elapsed_seconds=.4, max_frames=2, max_seconds=.6))
        self.assertTrue(can_reuse_box(missed_frames=3, elapsed_seconds=.6000000000000001, max_frames=15, max_seconds=.6))
        self.assertFalse(can_reuse_box(missed_frames=1, elapsed_seconds=.01, max_frames=15, max_seconds=0))
        self.assertTrue(can_reuse_box(missed_frames=15, elapsed_seconds=3, max_frames=15, max_seconds=None))
        self.assertFalse(can_reuse_box(missed_frames=16, elapsed_seconds=3.2, max_frames=15, max_seconds=None))

    def test_metrics_separate_fallbacks_from_landmarks_and_handle_missing_boxes(self):
        first = frame_diagnostics(time_sec=0, raw_box=np.array([0, 0, 20, 20]),
                                  crop_box=np.array([0, 0, 20, 20]), previous_box=None,
                                  frame_shape=(60, 80, 3), flags=(True, True, True, False, False))
        second = frame_diagnostics(time_sec=.2, raw_box=np.array([20, 0, 40, 20]),
                                   crop_box=np.array([10, 0, 30, 20]), previous_box=np.array([0, 0, 20, 20]),
                                   frame_shape=(60, 80, 3), flags=(True, False, True, True, False))
        missing = frame_diagnostics(time_sec=.4, raw_box=None, crop_box=None, previous_box=None,
                                    frame_shape=(60, 80, 3), flags=(False,) * 5)
        metrics = summarize_frames([first, second, missing], 2)
        self.assertEqual(metrics["face_landmark_fraction"], 1 / 3)
        self.assertEqual(metrics["face_region_fraction"], 2 / 3)
        self.assertAlmostEqual(metrics["mean_box_center_step_fraction"], .1)
        self.assertAlmostEqual(metrics["mean_box_center_offset_fraction"], .05)
        self.assertIsNone(summarize_frames([missing], 1)["mean_box_center_step_fraction"])
        row = comparison_row(build_experiments()[0], Path("clip.mp4"), metrics, metrics)
        self.assertEqual(row["delta_face_landmark_fraction"], 0)

    def _run_extraction(self, root: Path, diagnostics_only: bool, max_duration_seconds=None,
                        *, progress_config=None, fail_face_creation=False,
                        fail_inference=False, resources=None, video_enhancement=None):
        frame = np.zeros((20, 20, 3), dtype=np.uint8)
        frames = iter([frame.copy() for _ in range(7)])
        capture = SimpleNamespace(isOpened=lambda: True, get=lambda key: {1: 5.0, 3: 7, 4: 20, 5: 20}[key],
                                  read=lambda: (True, image) if (image := next(frames, None)) is not None else (False, None),
                                  release=Mock())
        cv2 = SimpleNamespace(VideoCapture=lambda path: capture, CAP_PROP_FPS=1,
                              CAP_PROP_FRAME_COUNT=3, CAP_PROP_FRAME_WIDTH=4, CAP_PROP_FRAME_HEIGHT=5,
                              COLOR_BGR2RGB=2, cvtColor=lambda image, code: image)
        point = SimpleNamespace(x=.5, y=.5, z=0, visibility=1, presence=1)
        results = iter([SimpleNamespace(pose_landmarks=[[point] * 33] if found else [])
                        for found in [True, True, False, False, False, False, True]])
        pose_api = SimpleNamespace(
            detect_for_video=Mock(side_effect=RuntimeError("inference failed") if fail_inference else results),
            close=Mock(),
        )
        face_api = SimpleNamespace(
            detect_for_video=Mock(return_value=SimpleNamespace(face_landmarks=[])), close=Mock(),
        )
        if resources is not None:
            resources.update(capture=capture, pose=pose_api, face=face_api)
        options = {}

        def create(kind, value):
            options[kind] = vars(value)
            if kind == "face" and fail_face_creation:
                raise RuntimeError("face initialization failed")
            return pose_api if kind == "pose" else face_api

        vision = SimpleNamespace(
            RunningMode=SimpleNamespace(VIDEO="video"),
            PoseLandmarkerOptions=SimpleNamespace, FaceLandmarkerOptions=SimpleNamespace,
            PoseLandmarker=SimpleNamespace(create_from_options=lambda value: create("pose", value)),
            FaceLandmarker=SimpleNamespace(create_from_options=lambda value: create("face", value)),
        )
        mediapipe = SimpleNamespace(tasks=SimpleNamespace(vision=vision, BaseOptions=SimpleNamespace),
                                    Image=lambda **kwargs: kwargs, ImageFormat=SimpleNamespace(SRGB=1))
        video = root / "input.mp4"
        model = root / "model.task"
        video.touch(); model.touch()
        with patch.dict("sys.modules", {"cv2": cv2, "mediapipe": mediapipe}), \
             patch.object(preprocessing, "_pose_head_box", side_effect=lambda result, index, *args: None if index is None else np.array([2, 2, 18, 18])), \
             patch.object(preprocessing, "_face_box", return_value=np.array([0, 0, 10, 10], dtype=np.float32)), \
             patch.object(preprocessing, "_face_geometry", return_value=(np.ones(20), True)), \
             patch.object(preprocessing, "_crop_face_input", side_effect=lambda image, box, gray, **kw: (np.full((4, 16, 16), int(box is not None)), None)), \
             contextlib.redirect_stdout(io.StringIO()):
            result = preprocessing.extract_video(
                video, root / "output", "sample", pose_model_path=model, face_model_path=model,
                window_seconds=1, face_size=16, max_box_reuse_seconds=.4,
                min_pose_detection_confidence=.55, min_pose_presence_confidence=.65,
                min_pose_tracking_confidence=.6, min_face_detection_confidence=.6,
                min_face_presence_confidence=.65, min_face_tracking_confidence=.7,
                diagnostics_only=diagnostics_only,
                max_duration_seconds=max_duration_seconds,
                progress_config=progress_config,
                video_enhancement=video_enhancement,
            )
        return result, options

    def test_video_enhancement_precedes_both_pose_and_face_detection(self):
        from FYPModels.preprocess import SPEINetConfig
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = root / "speinet.pt"
            checkpoint.touch()
            config = SPEINetConfig(checkpoint=checkpoint)
            enhancer = SimpleNamespace(
                config=config, record={"model": "speinet", "checkpoint": str(checkpoint)},
                is_sharp=Mock(return_value=False),
                enhance=Mock(side_effect=lambda frames: np.full_like(frames[1], 123)),
            )
            resources = {}
            with patch.object(preprocessing, "SPEINetEnhancer", return_value=enhancer):
                result, _ = self._run_extraction(root, True, resources=resources, video_enhancement=config)
            self.assertEqual(result.sampled_frames, 7)
            for call in resources["pose"].detect_for_video.call_args_list:
                self.assertTrue(np.all(call.args[0]["data"] == 123))
            for call in resources["face"].detect_for_video.call_args_list:
                self.assertTrue(np.all(call.args[0]["data"] == 123))
            record = json.loads(Path(result.run_record_path).read_text())
            self.assertEqual(record["settings"]["video_enhancement"]["model"], "speinet")
            self.assertIn("speinet_checkpoint", record["inputs"])
            self.assertEqual(record["inputs"]["video"]["path"], str((root / "input.mp4").resolve()))

    def test_extraction_forwards_thresholds_and_expires_reused_boxes(self):
        with tempfile.TemporaryDirectory() as temporary:
            result, options = self._run_extraction(Path(temporary), diagnostics_only=True)
            self.assertEqual(options["pose"]["min_tracking_confidence"], .6)
            self.assertEqual(options["face"]["min_tracking_confidence"], .7)
            self.assertEqual(options["pose"]["min_pose_detection_confidence"], .55)
            self.assertEqual(options["face"]["min_face_presence_confidence"], .65)
            report = json.loads(Path(result.diagnostics_path).read_text())
            self.assertEqual(report["metrics"]["face_landmark_fraction"], 3 / 7)
            self.assertEqual(report["metrics"]["face_region_fraction"], 5 / 7)
            self.assertEqual(report["metrics"]["face_box_reused_fraction"], 2 / 7)
            with Path(report["frames_path"]).open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual([row["box_reused"] for row in rows], ["0", "0", "1", "1", "0", "0", "0"])
            self.assertIsNone(result.manifest_path)
            self.assertFalse(list(Path(temporary).rglob("*.npy")))

    def test_normal_extraction_still_writes_training_windows_and_manifest(self):
        with tempfile.TemporaryDirectory() as temporary:
            result, _ = self._run_extraction(Path(temporary), diagnostics_only=False)
            self.assertEqual(result.windows_written, 2)
            with Path(result.manifest_path).open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 2)
            face = np.load(Path(result.manifest_path).parent / rows[0]["face_path"])
            self.assertEqual(face.shape, (5, 4, 16, 16))
            self.assertFalse(face[-1].any())

    def test_duration_limit_uses_video_time(self):
        with tempfile.TemporaryDirectory() as temporary:
            result, _ = self._run_extraction(Path(temporary), diagnostics_only=True, max_duration_seconds=1)
            self.assertEqual(result.sampled_frames, 5)
            report = json.loads(Path(result.diagnostics_path).read_text())
            with Path(report["frames_path"]).open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(float(rows[-1]["time_sec"]), .8)

    def test_run_record_and_progress_are_written_in_both_output_modes(self):
        for diagnostics_only in (False, True):
            with self.subTest(diagnostics_only=diagnostics_only), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                progress = root / "progress.json"
                result, _ = self._run_extraction(
                    root, diagnostics_only=diagnostics_only,
                    progress_config={"testing": {"progress_file": str(progress)}},
                )
                record = json.loads(Path(result.run_record_path).read_text())
                self.assertEqual(record["record_type"], "preprocessing")
                self.assertEqual(record["inputs"]["video"]["sha256"],
                                 "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855")
                self.assertEqual(record["source_metadata"]["fps"], 5.0)
                self.assertEqual(record["source_metadata"]["frame_count"], 7)
                self.assertIn("No moov", record["source_metadata"]["inspection_error"])
                self.assertEqual(record["settings"]["split_seed"], 42)
                self.assertEqual(record["settings"]["diagnostics_only"], diagnostics_only)
                self.assertEqual(record["outputs"]["manifest_path"], result.manifest_path)
                self.assertEqual(record["software_versions"]["numpy"], np.__version__)
                snapshot = json.loads(progress.read_text())
                self.assertEqual(snapshot["stage"], "facial_preprocessing")
                self.assertEqual(snapshot["completed_units"], 7)
                self.assertEqual(snapshot["percent"], 100)

    def test_failed_face_initialization_closes_pose_and_capture(self):
        with tempfile.TemporaryDirectory() as temporary:
            resources = {}
            with self.assertRaisesRegex(RuntimeError, "face initialization failed"):
                self._run_extraction(Path(temporary), True, fail_face_creation=True, resources=resources)
            resources["pose"].close.assert_called_once_with()
            resources["capture"].release.assert_called_once_with()
            resources["face"].close.assert_not_called()
            self.assertFalse(list(Path(temporary).rglob("runs/*.json")))

    def test_failed_inference_closes_all_resources_without_reporting_completion(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            progress = root / "progress.json"
            resources = {}
            with self.assertRaisesRegex(RuntimeError, "inference failed"):
                self._run_extraction(
                    root, True, fail_inference=True, resources=resources,
                    progress_config={"testing": {"progress_file": str(progress)}},
                )
            resources["pose"].close.assert_called_once_with()
            resources["face"].close.assert_called_once_with()
            resources["capture"].release.assert_called_once_with()
            self.assertEqual(json.loads(progress.read_text())["percent"], 0)
            self.assertFalse(list(root.rglob("runs/*.json")))


if __name__ == "__main__":
    unittest.main()
