from __future__ import annotations

import argparse
import importlib.util
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

from FYPModels.preprocess import (
    SPEINetConfig, SPEINetEnhancer, add_video_enhancement_arguments,
    iter_enhanced_frames, video_enhancement_kwargs,
)


class VideoEnhancementTest(unittest.TestCase):
    def _stream(self, count=12, sharp_ids=(), radius=7):
        images = [np.full((20, 24, 3), index + 1, np.uint8) for index in range(count)]
        iterator = iter(images)
        capture = SimpleNamespace(read=Mock(side_effect=lambda: (True, frame)
                                  if (frame := next(iterator, None)) is not None else (False, None)))
        enhancer = SimpleNamespace(
            config=SPEINetConfig(reference_radius=radius),
            is_sharp=Mock(side_effect=lambda frame: int(frame[0, 0, 0]) in sharp_ids),
            enhance=Mock(side_effect=lambda frames: frames[1].copy()),
        )
        return images, capture, enhancer

    def test_sampling_uses_native_neighbours_and_source_timestamps(self):
        images, capture, enhancer = self._stream()
        result = list(iter_enhanced_frames(capture, source_fps=10, fps=3, enhancer=enhancer))
        self.assertEqual([round(time, 5) for time, _ in result], [0, .4, .7, 1])
        self.assertEqual([int(frame[0, 0, 0]) for _, frame in result], [1, 5, 8, 11])
        second = enhancer.enhance.call_args_list[1].args[0]
        self.assertEqual([int(frame[0, 0, 0]) for frame in second[:3]], [4, 5, 6])
        self.assertIsNone(second[3])
        self.assertIsNone(second[4])
        self.assertEqual(enhancer.is_sharp.call_count, len(images))

    def test_short_clips_reflect_endpoints_and_never_duplicate_output(self):
        for count, expected in ((1, [[1, 1, 1]]), (2, [[2, 1, 2], [1, 2, 1]])):
            with self.subTest(count=count):
                _, capture, enhancer = self._stream(count)
                result = list(iter_enhanced_frames(capture, source_fps=5, fps=30, enhancer=enhancer))
                self.assertEqual(len(result), count)
                actual = [[int(frame[0, 0, 0]) for frame in call.args[0][:3]]
                          for call in enhancer.enhance.call_args_list]
                self.assertEqual(actual, expected)

    def test_sharp_references_are_nearest_and_single_reference_is_duplicated(self):
        _, capture, enhancer = self._stream(12, sharp_ids=(1, 7), radius=3)
        list(iter_enhanced_frames(capture, source_fps=10, fps=10, enhancer=enhancer))
        first = enhancer.enhance.call_args_list[0].args[0]
        middle = enhancer.enhance.call_args_list[3].args[0]
        last = enhancer.enhance.call_args_list[-1].args[0]
        self.assertEqual([int(frame[0, 0, 0]) for frame in first[3:]], [1, 1])
        self.assertEqual([int(frame[0, 0, 0]) for frame in middle[3:]], [1, 7])
        self.assertEqual(last[3:], [None, None])

    def test_duration_limit_bounds_context_reads(self):
        _, capture, enhancer = self._stream(100)
        result = list(iter_enhanced_frames(capture, source_fps=10, fps=5, enhancer=enhancer,
                                          max_duration_seconds=.5))
        self.assertEqual([time for time, _ in result], [0, .2, .4])
        self.assertEqual(capture.read.call_count, 5)
        self.assertEqual(enhancer.is_sharp.call_count, 5)

    def test_stream_rejects_wrong_output_and_invalid_timing(self):
        for values in ({"source_fps": 0, "fps": 5}, {"source_fps": 5, "fps": float("nan")},
                       {"source_fps": 5, "fps": 5, "max_duration_seconds": -1}):
            _, capture, enhancer = self._stream()
            with self.subTest(values=values), self.assertRaises(ValueError):
                list(iter_enhanced_frames(capture, enhancer=enhancer, **values))
            capture.read.assert_not_called()
        _, capture, enhancer = self._stream(1)
        enhancer.enhance.side_effect = lambda frames: frames[1].astype(np.float32)
        with self.assertRaisesRegex(RuntimeError, "preserve"):
            list(iter_enhanced_frames(capture, source_fps=5, fps=5, enhancer=enhancer))

    def test_configuration_and_missing_checkpoint_fail_early(self):
        for kwargs in ({"tile_size": 0}, {"tile_size": 21}, {"tile_overlap": 200},
                       {"tile_overlap": -1}, {"reference_radius": 0}, {"reference_radius": 8},
                       {"device": "invalid"}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                SPEINetConfig(**kwargs)
        with self.assertRaisesRegex(FileNotFoundError, "SPEINet checkpoint"):
            SPEINetEnhancer(SPEINetConfig(checkpoint="/nonexistent/speinet.pt"))

    def test_cli_enables_stage_and_default_needs_no_checkpoint(self):
        parser = argparse.ArgumentParser()
        add_video_enhancement_arguments(parser)
        self.assertIsNone(video_enhancement_kwargs(parser.parse_args([]))["video_enhancement"])
        config = video_enhancement_kwargs(parser.parse_args([
            "--video-enhancement", "speinet", "--speinet-device", "cpu",
            "--speinet-checkpoint", "weights.pt", "--speinet-tile-size", "40",
            "--speinet-tile-overlap", "20",
        ]))["video_enhancement"]
        self.assertEqual(config.device, "cpu")
        self.assertEqual(config.tile_size, 40)
        self.assertEqual(str(config.checkpoint), "weights.pt")


@unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch is optional")
class TiledSPEINetTest(unittest.TestCase):
    def _enhancer(self, model):
        import torch
        enhancer = SPEINetEnhancer.__new__(SPEINetEnhancer)
        enhancer.config = SPEINetConfig(tile_size=40, tile_overlap=20)
        enhancer.torch = torch
        enhancer.device = torch.device("cpu")
        enhancer.model = model
        return enhancer

    def test_tiling_padding_and_color_roundtrip_keep_all_pixels(self):
        model = Mock(side_effect=lambda tensor: tensor[:, 1].clone())
        enhancer = self._enhancer(model)
        for shape in ((1, 1, 3), (39, 53, 3), (63, 87, 3)):
            frame = np.random.default_rng(4).integers(0, 256, shape, dtype=np.uint8)
            original = frame.copy()
            result = enhancer.enhance([frame, frame, frame, None, None])
            np.testing.assert_array_equal(result, original)
            np.testing.assert_array_equal(frame, original)
        for call in model.call_args_list:
            tensor = call.args[0]
            self.assertEqual(tensor.shape[:3], (1, 5, 3))
            self.assertEqual(tensor.shape[-1] % 20, 0)
            self.assertEqual(tensor.shape[-2] % 20, 0)
            self.assertLessEqual(tensor.shape[-1], 40)
            self.assertFalse(bool(tensor[:, 3:].any()))

    def test_nonfinite_model_output_is_rejected(self):
        enhancer = self._enhancer(lambda tensor: tensor[:, 1] * float("nan"))
        frame = np.full((20, 20, 3), 128, np.uint8)
        with self.assertRaisesRegex(RuntimeError, "non-finite"):
            enhancer.enhance([frame, frame, frame, None, None])


if __name__ == "__main__":
    unittest.main()
