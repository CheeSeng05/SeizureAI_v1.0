"""Upscaling tests and a six-algorithm comparison workflow.

From FYPModels:
    python -m unittest tests.test_upscaling -v
    python -m tests.test_upscaling face.png --output results/upscaling --download-models
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch
from urllib.request import urlopen

import cv2
import numpy as np

if not __package__:
    # Also support running python tests/test_upscaling.py directly.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from seizure_multimodal.checkpoints.provenance import sha256_file
from FYPModels.preprocess.io import open_video
from FYPModels.preprocess import frame_enhancement as preprocessing
from FYPModels.preprocess.frame_enhancement import (
    DEFAULT_SR_MODEL_DIR, SR_MODELS, _sr_model_path, _upscaling,
)


# OpenCV tutorial model implementations; weights are downloaded only on request.
MODEL_URLS = {
    "edsr": "https://raw.githubusercontent.com/Saafke/EDSR_Tensorflow/master/models/",
    "espcn": "https://raw.githubusercontent.com/fannymonori/TF-ESPCN/master/export/",
    "fsrcnn": "https://raw.githubusercontent.com/Saafke/FSRCNN_Tensorflow/master/models/",
    "lapsrn": "https://raw.githubusercontent.com/fannymonori/TF-LapSRN/master/export/",
}


def prepare_models(model_dir: Path, scale: int, *, download: bool) -> dict:
    paths = {name: _sr_model_path(name, scale, model_dir) for name in MODEL_URLS}
    missing = [path for path in paths.values() if not path.is_file()]
    if missing and not download:
        raise FileNotFoundError(
            "Missing model weights: " + ", ".join(str(path) for path in missing)
            + ". Add --download-models to download them."
        )
    records = {}
    for name, path in paths.items():
        url = MODEL_URLS[name] + path.name
        if not path.is_file():
            path.parent.mkdir(parents=True, exist_ok=True)
            print(f"Downloading {path.name} ...", flush=True)
            temporary = None
            try:
                with urlopen(url, timeout=60) as source, tempfile.NamedTemporaryFile(
                    dir=path.parent, suffix=".part", delete=False,
                ) as target:
                    temporary = Path(target.name)
                    shutil.copyfileobj(source, target)
                temporary.replace(path)
            finally:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)
        records[name] = {"path": str(path), "sha256": sha256_file(path), "download_url": url}
    return records


def load_input(path: Path, *, video_time: float | None, roi: list[float] | None) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(
            f"Input file not found: {path}. Supply an existing image, "
            "or an existing video with --video-time SECONDS."
        )
    if video_time is None:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Cannot read image: {path}. For video, supply --video-time SECONDS.")
    else:
        if not math.isfinite(video_time) or video_time < 0:
            raise ValueError("--video-time must be finite and non-negative")
        with open_video(path, cv2=cv2) as (capture, metadata):
            frame_index = round(video_time * metadata.fps)
            if not capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index):
                raise ValueError(f"Cannot seek to frame {frame_index}")
            ok, image = capture.read()
            if not ok:
                raise ValueError(f"Cannot read video at {video_time:g} seconds")
    if roi is not None:
        x1, y1, x2, y2 = roi
        if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
            raise ValueError("--roi must be normalized x1 y1 x2 y2 within [0, 1]")
        height, width = image.shape[:2]
        image = image[round(y1 * height):round(y2 * height), round(x1 * width):round(x2 * width)]
        if image.size == 0:
            raise ValueError("--roi selects an empty crop")
    return np.ascontiguousarray(image)


def _save_image(path: Path, image: np.ndarray) -> None:
    if not cv2.imwrite(str(path), image):
        raise OSError(f"Cannot write image: {path}")


def _comparison_panel(image: np.ndarray, title: str, subtitle: str) -> np.ndarray:
    panel = np.full((300, 320, 3), 245, np.uint8)
    height, width = image.shape[:2]
    panel[55:55 + height, (320 - width) // 2:(320 - width) // 2 + width] = image
    cv2.putText(panel, title, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, .65, (20, 20, 20), 1, cv2.LINE_AA)
    cv2.putText(panel, subtitle, (10, 44), cv2.FONT_HERSHEY_SIMPLEX, .40, (60, 60, 60), 1, cv2.LINE_AA)
    return panel


def write_comparison_views(image: np.ndarray, output: Path, *, scale: int) -> None:
    """Render saved outputs with dimensions and a crop at native output pixels.

    Can also refresh previews from an existing run without repeating inference.
    The input and six full-resolution algorithm files are read only.
    """
    height, width = image.shape[:2]
    crop_width, crop_height = min(240 // scale, width), min(240 // scale, height)
    x, y = (width - crop_width) // 2, (height - crop_height) // 2
    # Pixel replication is an explicitly labeled reference, not a seventh SR model.
    reference = cv2.resize(image[y:y + crop_height, x:x + crop_width],
                           (crop_width * scale, crop_height * scale), interpolation=cv2.INTER_NEAREST)
    details = [_comparison_panel(reference, "Input pixel reference", f"Source pixels repeated {scale}x (nearest)")]
    previews = []
    for name in SR_MODELS:
        path = output / f"{name}_x{scale}.png"
        result = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if result is None or result.shape != (height * scale, width * scale, 3):
            raise ValueError(f"Missing or incorrectly sized upscaled image: {path}")
        ratio = min(300 / result.shape[1], 240 / result.shape[0], 1.0)
        preview = cv2.resize(result, (max(1, round(result.shape[1] * ratio)),
                                      max(1, round(result.shape[0] * ratio))), interpolation=cv2.INTER_AREA)
        previews.append(_comparison_panel(
            preview, f"{name} x{scale}",
            f"{result.shape[1]} x {result.shape[0]} px | preview {preview.shape[1]} x {preview.shape[0]}",
        ))
        detail = result[y * scale:(y + crop_height) * scale, x * scale:(x + crop_width) * scale]
        details.append(_comparison_panel(detail, f"{name} x{scale}",
                                        f"{detail.shape[1]} x {detail.shape[0]} output pixels; no resize"))
    header = np.full((65, 960, 3), 245, np.uint8)
    cv2.putText(header, f"Input {width} x {height} px -> outputs {width * scale} x {height * scale} px ({scale}x per side)",
                (10, 24), cv2.FONT_HERSHEY_SIMPLEX, .65, (20, 20, 20), 1, cv2.LINE_AA)
    cv2.putText(header, "Overview resized to fit. Use detail.png or open individual PNGs at 100% zoom.",
                (10, 49), cv2.FONT_HERSHEY_SIMPLEX, .5, (60, 60, 60), 1, cv2.LINE_AA)
    _save_image(output / "comparison.png", np.vstack((header, np.hstack(previews[:3]), np.hstack(previews[3:]))))
    details.append(np.full_like(details[0], 245))
    header = np.full((65, 1280, 3), 245, np.uint8)
    cv2.putText(header, f"Same center region: input x={x}:{x + crop_width}, y={y}:{y + crop_height}",
                (10, 24), cv2.FONT_HERSHEY_SIMPLEX, .65, (20, 20, 20), 1, cv2.LINE_AA)
    cv2.putText(header, "Algorithm crops preserve output pixels 1:1. View this file at 100% zoom to avoid viewer resizing.",
                (10, 49), cv2.FONT_HERSHEY_SIMPLEX, .5, (60, 60, 60), 1, cv2.LINE_AA)
    _save_image(output / "detail.png", np.vstack((header, np.hstack(details[:4]), np.hstack(details[4:]))))


def compare_upscaling(image: np.ndarray, output: Path, *, model_dir: Path, scale: int, repeats: int) -> list[dict]:
    if repeats < 1:
        raise ValueError("repeats must be positive")
    # A fresh directory prevents mixing results from different inputs or scales.
    output.mkdir(parents=True, exist_ok=False)
    _save_image(output / "input.png", image)
    rows = []
    for name in SR_MODELS:
        print(f"Running {name} x{scale} ...", flush=True)
        started = time.perf_counter()
        result = _upscaling(image, name, scale=scale, model_dir=model_dir)
        first_call_ms = (time.perf_counter() - started) * 1000
        timings = []
        for _ in range(repeats):
            started = time.perf_counter()
            result = _upscaling(image, name, scale=scale, model_dir=model_dir)
            timings.append((time.perf_counter() - started) * 1000)
        expected_shape = (image.shape[0] * scale, image.shape[1] * scale, 3)
        if result.shape != expected_shape or result.dtype != np.uint8:
            raise ValueError(f"Unexpected {name} output: {result.shape}, {result.dtype}; expected {expected_shape}, uint8")
        _save_image(output / f"{name}_x{scale}.png", result)
        print(f"  {image.shape[1]} x {image.shape[0]} -> {result.shape[1]} x {result.shape[0]} pixels "
              f"({scale * scale}x pixel count)", flush=True)
        rows.append({"algorithm": name, "scale": scale, "input_width": image.shape[1],
                     "input_height": image.shape[0], "output_width": result.shape[1],
                     "output_height": result.shape[0], "first_call_ms": first_call_ms,
                     "median_warm_ms": float(np.median(timings)), "repeats": repeats})
    write_comparison_views(image, output, scale=scale)
    with (output / "timings.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Image path, or video path with --video-time.")
    parser.add_argument("--video-time", type=float, help="Select one video frame at this time in seconds.")
    parser.add_argument("--roi", type=float, nargs=4, metavar=("X1", "Y1", "X2", "Y2"),
                        help="Crop a normalized region before upscaling; use the patient's face.")
    parser.add_argument("--output", type=Path, required=True, help="New directory for the comparison.")
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_SR_MODEL_DIR)
    parser.add_argument("--scale", type=int, choices=(2, 4), default=4,
                        help="Common scale supported by all four neural models.")
    parser.add_argument("--repeats", type=int, default=3, help="Timed calls after one warm-up per algorithm.")
    parser.add_argument("--download-models", action="store_true", help="Download missing .pb weights from model authors.")
    args = parser.parse_args()
    try:
        if args.repeats < 1:
            raise ValueError("--repeats must be positive")
        if args.output.exists():
            raise FileExistsError(f"Output already exists; choose a new directory: {args.output}")
        image = load_input(args.input.resolve(), video_time=args.video_time, roi=args.roi)
        models = prepare_models(args.model_dir, args.scale, download=args.download_models)
        rows = compare_upscaling(image, args.output, model_dir=args.model_dir, scale=args.scale, repeats=args.repeats)
        (args.output / "run.json").write_text(json.dumps({
            "source": str(args.input.resolve()), "video_time": args.video_time, "roi": args.roi,
            "input_sha256": sha256_file(args.output / "input.png"), "models": models,
            "opencv_version": cv2.__version__, "scale": args.scale, "results": rows,
            "interpretation": "Timings and visual outputs only; no landmark accuracy or PSNR/SSIM claim. "
                              "The comparison preview may be reduced; detail.png preserves native output pixels. "
                              "Inspect detail.png and individual PNGs at 100% zoom. "
                              "First-call time includes model loading; warm timings exclude loading and file writes.",
        }, indent=2) + "\n", encoding="utf-8")
        print(f"Saved all six results to {args.output.resolve()}")
    except (OSError, ValueError, RuntimeError, cv2.error) as exc:
        parser.error(str(exc))


class ImageUpscaleTest(unittest.TestCase):
    def setUp(self):
        self.image = np.zeros((12, 16, 3), np.uint8)
        self.image[3:9, 4:12] = (20, 100, 220)
        preprocessing._load_sr_model.cache_clear()
        self.addCleanup(preprocessing._load_sr_model.cache_clear)

    def test_interpolation_runs_without_neural_weights_or_factory(self):
        with patch.object(preprocessing, "_load_sr_model") as loader:
            for model in ("bilinear", "bicubic"):
                result = preprocessing._upscaling(self.image, model, scale=4)
                self.assertEqual(result.shape, (48, 64, 3))
                self.assertEqual(result.dtype, np.uint8)
                self.assertTrue(result.any())
            loader.assert_not_called()

    def test_bilinear_and_bicubic_produce_different_interpolations(self):
        linear = preprocessing._upscaling(self.image, "bilinear")
        cubic = preprocessing._upscaling(self.image, "bicubic")
        self.assertFalse(np.array_equal(linear, cubic))

    def test_neural_models_use_integer_scales_and_load_once(self):
        for name, scale, filename in (("edsr", 4, "EDSR_x4.pb"), ("espcn", 3, "ESPCN_x3.pb"),
                                       ("fsrcnn", 3, "FSRCNN_x3.pb"), ("lapsrn", 8, "LapSRN_x8.pb")):
            with self.subTest(model=name), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                path = root / filename
                path.touch()
                expected = np.zeros((12 * scale, 16 * scale, 3), np.uint8)
                network = SimpleNamespace(readModel=Mock(), setModel=Mock(), upsample=Mock(return_value=expected))
                dnn = SimpleNamespace(DnnSuperResImpl_create=Mock(return_value=network))
                with patch.object(cv2, "dnn_superres", dnn, create=True):
                    for _ in range(2):
                        self.assertIs(preprocessing._upscaling(self.image, name, scale=scale, model_dir=root), expected)
                dnn.DnnSuperResImpl_create.assert_called_once_with()
                network.readModel.assert_called_once_with(str(path.resolve()))
                network.setModel.assert_called_once_with(name, scale)
                self.assertIsInstance(network.setModel.call_args.args[1], int)
                self.assertEqual(network.upsample.call_count, 2)

    def test_unknown_algorithm_and_invalid_scale_have_clear_errors(self):
        for name in ("", "invalid"):
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "Unknown upscaling"):
                preprocessing._upscaling(self.image, name)
        for scale in (0, -1, 1, 2.5, "4", True):
            with self.subTest(scale=scale), self.assertRaisesRegex(ValueError, "integer"):
                preprocessing._upscaling(self.image, "bicubic", scale=scale)
        with self.assertRaisesRegex(ValueError, "lapsrn supports scales"):
            preprocessing._upscaling(self.image, "lapsrn", scale=3)

    def test_input_validation(self):
        for image in (None, np.zeros((0, 0, 3), np.uint8), np.zeros((12, 16), np.uint8),
                      np.zeros((12, 16, 4), np.uint8), self.image.astype(np.float32)):
            with self.subTest(shape=getattr(image, "shape", None)), self.assertRaises(ValueError):
                preprocessing._upscaling(image, "bicubic")

    def test_missing_weights_fail_before_loading_network(self):
        with tempfile.TemporaryDirectory() as temporary:
            dnn = SimpleNamespace(DnnSuperResImpl_create=Mock())
            with patch.object(cv2, "dnn_superres", dnn, create=True), \
                 self.assertRaisesRegex(FileNotFoundError, "EDSR_x4.pb"):
                preprocessing._upscaling(self.image, "edsr", model_dir=temporary)
            dnn.DnnSuperResImpl_create.assert_not_called()
            with patch(f"{__name__}.urlopen") as request, \
                 self.assertRaisesRegex(FileNotFoundError, "download-models"):
                prepare_models(Path(temporary), 4, download=False)
            request.assert_not_called()

    def test_cropping_happens_before_upscaling(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "input.png"
            self.assertTrue(cv2.imwrite(str(path), self.image))
            crop = load_input(path, video_time=None, roi=[.25, .25, .75, .75])
            np.testing.assert_array_equal(crop, self.image[3:9, 4:12])
            self.assertEqual(preprocessing._upscaling(crop, "bicubic").shape, (24, 32, 3))

    def test_comparison_writes_all_six_outputs_and_timings(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with patch(f"{__name__}._upscaling",
                       side_effect=lambda image, name, **kwargs: cv2.resize(image, (64, 48))) as upscale:
                rows = compare_upscaling(self.image, root / "results", model_dir=root, scale=4, repeats=2)
            self.assertEqual([row["algorithm"] for row in rows], list(preprocessing.SR_MODELS))
            self.assertEqual(upscale.call_count, 18)
            for name in preprocessing.SR_MODELS:
                self.assertEqual(cv2.imread(str(root / "results" / f"{name}_x4.png")).shape, (48, 64, 3))
            self.assertTrue((root / "results/comparison.png").is_file())
            self.assertEqual(len((root / "results/timings.csv").read_text().splitlines()), 7)

    def test_installed_neural_models_upsample_real_pixels(self):
        paths = [preprocessing._sr_model_path(name, 4, preprocessing.DEFAULT_SR_MODEL_DIR)
                 for name in preprocessing.SR_SCALES]
        if not hasattr(cv2, "dnn_superres") or not all(path.is_file() for path in paths):
            self.skipTest("Download the four x4 models with tests.test_upscaling --download-models first")
        for name in preprocessing.SR_SCALES:
            with self.subTest(model=name):
                result = preprocessing._upscaling(self.image, name)
                self.assertEqual(result.shape, (48, 64, 3))
                self.assertEqual(result.dtype, np.uint8)
                self.assertTrue(result.any())


if __name__ == "__main__":
    main()
