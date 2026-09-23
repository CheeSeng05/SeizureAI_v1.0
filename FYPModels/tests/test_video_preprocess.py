from __future__ import annotations

import contextlib
import csv
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import cv2
import numpy as np

from FYPModels.preprocess import video_preprocess as preprocessing
from FYPModels.preprocess.io import VideoMetadata


def pose(x):
    points = [SimpleNamespace(x=x, y=.2, z=0., visibility=1., presence=1.) for _ in range(33)]
    points[11].x, points[11].y = x - .1, .4
    points[12].x, points[12].y = x + .1, .4
    points[23].x, points[23].y = x - .05, .65
    points[24].x, points[24].y = x + .05, .65
    return points


class VideoPreprocessTest(unittest.TestCase):
    def run_pipeline(self, root, *, count=5, **kwargs):
        video, model = root / "input.mp4", root / "pose.task"
        video.touch()
        model.touch()
        frame = np.full((100, 200, 3), (10, 80, 160), dtype=np.uint8)
        capture = SimpleNamespace(
            read=Mock(side_effect=[(True, frame.copy()) for _ in range(count)] + [(False, None)]),
            release=Mock(),
        )
        results = [[pose(.8), pose(.2)], [pose(.8)], [pose(.21)], [], [pose(.2)]]
        api = SimpleNamespace(
            detect_for_video=Mock(side_effect=[SimpleNamespace(pose_landmarks=p) for p in results[:count]]),
            close=Mock(),
        )
        factory = Mock(return_value=api)
        # Only the pose API exists: this run cannot fall back to any facial API.
        mp = SimpleNamespace(
            tasks=SimpleNamespace(
                BaseOptions=SimpleNamespace,
                vision=SimpleNamespace(
                    RunningMode=SimpleNamespace(VIDEO="video"),
                    PoseLandmarkerOptions=SimpleNamespace,
                    PoseLandmarker=SimpleNamespace(create_from_options=factory),
                ),
            ),
            Image=lambda **values: values,
            ImageFormat=SimpleNamespace(SRGB=1),
        )

        @contextlib.contextmanager
        def open_video(*args, **options):
            try:
                yield capture, VideoMetadata(str(video), 5, count, 200, 100)
            finally:
                capture.release()

        with patch.dict("sys.modules", {"mediapipe": mp}), \
             patch.object(preprocessing, "open_video", side_effect=open_video), \
             contextlib.redirect_stdout(io.StringIO()):
            summary = preprocessing.extract_video(
                video, root / "out", "patient_recording", pose_model_path=model,
                fps=5, window_seconds=.6, stride_seconds=.4,
                patient_roi=(0, 0, .5, 1), **kwargs,
            )
        api.close.assert_called_once_with()
        capture.release.assert_called_once_with()
        if count:
            np.testing.assert_array_equal(api.detect_for_video.call_args_list[0].args[0]["data"],
                                          frame[:, :, ::-1])
        return summary

    def test_writes_pose_windows_and_metadata_without_face_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            summary = self.run_pipeline(root)
            self.assertEqual(summary.sampled_frames, 5)
            self.assertEqual(summary.windows_written, 2)
            with Path(summary.manifest_path).open() as stream:
                reader = csv.DictReader(stream)
                rows = list(reader)
                self.assertFalse(any("face" in name for name in reader.fieldnames))
            self.assertEqual([float(row["start_sec"]) for row in rows], [0., .4])
            for row in rows:
                data = np.load(root / "out" / row["pose_path"], allow_pickle=False)
                self.assertEqual(data.shape, (3, 33, 4))
                self.assertEqual(data.dtype, np.float32)
                self.assertFalse(data[1].any())
                np.testing.assert_allclose((data[0, 23, :3] + data[0, 24, :3]) / 2, 0., atol=1e-6)
                self.assertAlmostEqual(float(np.linalg.norm(data[0, 11, :3] - data[0, 12, :3])), 1.)
                self.assertAlmostEqual(float(row["pose_detection_fraction"]), 2 / 3, places=6)
            self.assertFalse((root / "out/face").exists())
            self.assertFalse((root / "out/face_geometry").exists())
            report = json.loads(Path(summary.diagnostics_path).read_text())
            self.assertEqual(report["metrics"]["pose_detection_fraction"], 3 / 5)
            record = json.loads(Path(summary.run_record_path).read_text())
            self.assertEqual(set(record["inputs"]), {"video", "pose_model"})
            self.assertEqual(record["pipeline"], "pose")
            self.assertFalse(any("face" in key for key in record["settings"]))

    def test_diagnostics_only_and_empty_video(self):
        for count in (0, 5):
            with self.subTest(count=count), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                summary = self.run_pipeline(root, count=count, diagnostics_only=True)
                self.assertIsNone(summary.manifest_path)
                self.assertEqual(summary.windows_written, 0)
                self.assertFalse((root / "out/pose").exists())
                report = json.loads(Path(summary.diagnostics_path).read_text())
                self.assertEqual(report["metrics"]["sampled_frames"], count)
                if count == 0:
                    self.assertIsNone(report["metrics"]["pose_detection_fraction"])
                with Path(report["frames_path"]).open() as stream:
                    reader = csv.DictReader(stream)
                    self.assertEqual(reader.fieldnames, ["time_sec", "pose_detected"])
                    self.assertEqual(len(list(reader)), count)

    def test_preview_quit_cleans_up_and_discards_partial_window(self):
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(cv2, "imshow") as show, \
             patch.object(cv2, "waitKey", return_value=ord("q")), \
             patch.object(cv2, "destroyWindow"), \
             patch.object(cv2, "destroyAllWindows") as destroy:
            summary = self.run_pipeline(Path(directory), preview=True)
            self.assertEqual(summary.sampled_frames, 1)
            self.assertEqual(summary.windows_written, 0)
            show.assert_called_once()
            self.assertEqual(show.call_args.args[0], "Pose preprocessing - source")
            destroy.assert_called_once_with()

    def test_pose_confidences_are_validated(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "pose.task"
            model.touch()
            for name in preprocessing.POSE_CONFIDENCE_OPTIONS:
                for value in (-.1, 1.1, float("nan")):
                    with self.subTest(name=name, value=value), self.assertRaisesRegex(ValueError, name):
                        preprocessing.extract_video(root / "missing.mp4", root / "out", "sample",
                                                    pose_model_path=model, **{name: value})

    def test_manifest_append_preserves_header_and_rejects_other_schema(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.csv"
            options = {"fieldnames": preprocessing.MANIFEST_FIELDS}
            preprocessing.write_rows(path, [{"window_id": "first"}], append=False, **options)
            preprocessing.write_rows(path, [{"window_id": "second"}], append=True, **options)
            with path.open() as stream:
                self.assertEqual([row["window_id"] for row in csv.DictReader(stream)], ["first", "second"])
            original = path.read_bytes()
            with self.assertRaisesRegex(ValueError, "Manifest columns"):
                preprocessing.write_rows(path, [], append=True)
            self.assertEqual(path.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
