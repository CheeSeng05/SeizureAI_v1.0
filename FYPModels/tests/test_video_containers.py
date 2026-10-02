import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from seizure_multimodal.data_inspection import inspect_mp4, inspect_video
from seizure_multimodal.discovery import discover_sources, inspection_bundle


class VideoContainerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def make_video(self, name, codec):
        path = self.root / name
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*codec), 25, (64, 48))
        self.assertTrue(writer.isOpened())
        try:
            for index in range(50):
                writer.write(np.full((48, 64, 3), index * 4, dtype=np.uint8))
        finally:
            writer.release()
        return path

    def test_real_avi_inspection_and_discovery(self):
        path = self.make_video('sample.AVI', 'MJPG')
        config = {'data': {'video': str(path), 'video_labels': None}}
        metadata = inspect_video(path)
        self.assertEqual((metadata.width, metadata.height, metadata.frame_count), (64, 48, 50))
        self.assertAlmostEqual(metadata.fps, 25)
        self.assertAlmostEqual(metadata.duration_sec, 2)
        self.assertIsNone(metadata.variable_frame_rate)
        self.assertEqual(discover_sources(self.root, config)[0].end_sec, 2)
        self.assertEqual(inspection_bundle(self.root, config)['video']['frame_count'], 50)

    def test_mp4_keeps_sample_table_metadata(self):
        path = self.make_video('sample.mp4', 'mp4v')
        self.assertEqual(inspect_video(path), inspect_mp4(path))

    def test_corrupt_avi_reports_video_error(self):
        path = self.root / 'broken.avi'
        path.write_bytes(b'not a video')
        with self.assertRaisesRegex(ValueError, 'Video cannot be opened'):
            inspect_video(path)

    def test_invalid_fps_is_rejected_and_capture_released(self):
        path = self.root / 'invalid.avi'
        path.touch()
        with patch.object(cv2, 'VideoCapture') as capture:
            capture.return_value.isOpened.return_value = True
            capture.return_value.get.return_value = float('nan')
            with self.assertRaisesRegex(ValueError, 'invalid FPS or frame count'):
                inspect_video(path)
            capture.return_value.release.assert_called_once()


if __name__ == '__main__':
    unittest.main()
