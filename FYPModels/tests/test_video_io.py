from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from seizure_multimodal.data_inspection import MP4VideoMetadata
from FYPModels.preprocess.io import iter_sampled_frames, open_video


class VideoIOTest(unittest.TestCase):
    def _decoder(self, *, fps=10.0, opened=True, frame_count=10):
        frames = iter(range(10))

        def read():
            frame = next(frames, None)
            return frame is not None, frame

        capture = SimpleNamespace(
            isOpened=lambda: opened, release=Mock(), read=Mock(side_effect=read),
            get=lambda key: {1: fps, 2: frame_count, 3: 640, 4: 480}[key],
        )
        cv2 = SimpleNamespace(
            VideoCapture=Mock(return_value=capture), CAP_PROP_FPS=1,
            CAP_PROP_FRAME_COUNT=2, CAP_PROP_FRAME_WIDTH=3, CAP_PROP_FRAME_HEIGHT=4,
        )
        return capture, cv2

    def test_sampling_non_integer_rate_keeps_source_timestamps(self):
        capture, _ = self._decoder()
        self.assertEqual(list(iter_sampled_frames(capture, source_fps=10, fps=3)),
                         [(0.0, 0), (0.4, 4), (0.7, 7)])

    def test_duration_limit_does_not_decode_past_boundary(self):
        capture, _ = self._decoder()
        self.assertEqual(list(iter_sampled_frames(
            capture, source_fps=10, fps=5, max_duration_seconds=.4,
        )), [(0.0, 0), (.2, 2)])
        self.assertEqual(capture.read.call_count, 4)

    def test_sampling_does_not_duplicate_frames_when_requested_rate_is_higher(self):
        capture, _ = self._decoder()
        self.assertEqual([frame for _, frame in iter_sampled_frames(capture, source_fps=10, fps=20)],
                         list(range(10)))

    def test_open_video_reuses_mp4_inspection_without_changing_decoder_clock(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "input.mp4"
            path.touch()
            track = MP4VideoMetadata(str(path), 1.25, 8.0, 10, 640, 480, "avc1", True)
            capture, cv2 = self._decoder()
            with patch("seizure_multimodal.video.io.inspect_mp4", return_value=track) as inspect:
                with open_video(path, cv2=cv2) as (decoder, metadata):
                    self.assertIs(decoder, capture)
                    self.assertEqual(metadata.fps, 10)
                    self.assertEqual(metadata.duration_sec, 1)
                    self.assertEqual(metadata.mp4, track)
                    self.assertTrue(metadata.as_dict()["mp4"]["variable_frame_rate"])
                    self.assertEqual(metadata.as_dict()["timestamp_policy"], "frame_index / source_fps")
                    capture.release.assert_not_called()
                inspect.assert_called_once_with(path.resolve())
            capture.release.assert_called_once_with()

    def test_mp4_inspection_failure_is_recorded_and_decoding_remains_available(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "input.mp4"
            path.touch()
            capture, cv2 = self._decoder()
            with open_video(path, cv2=cv2) as (_, metadata):
                self.assertIsNone(metadata.mp4)
                self.assertIn("No moov", metadata.inspection_error)
                self.assertEqual(metadata.frame_count, 10)
            capture.release.assert_called_once_with()

    def test_non_mp4_and_unknown_frame_count_do_not_require_mp4_parser(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "input.avi"
            path.touch()
            _, cv2 = self._decoder(frame_count=float("nan"))
            with patch("seizure_multimodal.video.io.inspect_mp4") as inspect:
                with open_video(path, cv2=cv2) as (_, metadata):
                    self.assertIsNone(metadata.frame_count)
                    self.assertIsNone(metadata.duration_sec)
                inspect.assert_not_called()

    def test_open_and_validation_failures_release_decoder(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "input.avi"
            path.touch()
            for fps, opened, error in ((0, True, ValueError), (float("nan"), True, ValueError),
                                       (10, False, OSError)):
                with self.subTest(fps=fps, opened=opened):
                    capture, cv2 = self._decoder(fps=fps, opened=opened)
                    with self.assertRaises(error), open_video(path, cv2=cv2):
                        self.fail("Invalid input should not enter the context")
                    capture.release.assert_called_once_with()

    def test_consumer_failure_releases_decoder(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "input.avi"
            path.touch()
            capture, cv2 = self._decoder()
            with self.assertRaisesRegex(RuntimeError, "consumer"), open_video(path, cv2=cv2):
                raise RuntimeError("consumer failed")
            capture.release.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
