"""Convert raw videos to stabilized RGB + optical-flow face windows.

Face crops are taken from full-resolution frames before resizing. Upscaling
standardizes the tensor shape; it is not treated as recovered image detail.
Optional crop enhancement reuses the upscaling models and applies unsharp
masking before face landmark detection and RGB tensor creation.
SPEINet video enhancement, when enabled, runs on full frames before this pipeline.
MediaPipe Face Detection selects a patient face before landmark extraction;
pose-derived head crops remain available when the face detector misses.
"""
from __future__ import annotations

import cv2
import argparse
import csv
import hashlib
import json
import math
import sys
import time
from contextlib import ExitStack
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

if not __package__:
    # Preserve the original standalone `python face_preprocess.py` entry point.
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    __package__ = "FYPModels.preprocess._facial"

from ...seizure_multimodal.progress import report_progress
from ...seizure_multimodal.reproducibility import build_preprocessing_record, save_run_record
from ...seizure_multimodal.video.io import iter_sampled_frames, open_video
from .. import (
    SPEINetConfig, SPEINetEnhancer, add_video_enhancement_arguments,
    iter_enhanced_frames, video_enhancement_kwargs,
)
from ..preprocess_diagnostics import frame_diagnostics, write_diagnostics
from ..preprocess_options import (
    add_tuning_arguments, can_reuse_box, tuning_kwargs, validate_tuning,
)


FACE_LANDMARK_PAIRS = (
    (33, 133),
    (362, 263),
    (159, 145),
    (386, 374),
    (61, 291),
    (13, 14),
    (78, 308),
    (1, 152),
    (10, 152),
    (234, 454),
)

MANIFEST_FIELDS = (
    "window_id",
    "dataset",
    "patient_id",
    "recording_id",
    "source_video",
    "start_sec",
    "end_sec",
    "label",
    "split",
    "sampled_fps",
    "frame_count",
    "pose_path",
    "face_path",
    "face_geometry_path",
    "face_size",
    "pose_detection_fraction",
    "face_detection_fraction",
    "face_region_fraction",
    "face_landmark_fraction",
    "pose_head_fallback_fraction",
    "face_box_reused_fraction",
)

SR_MODELS = {
    # OpenCV interpolation methods
    "bilinear": cv2.INTER_LINEAR,
    "bicubic": cv2.INTER_CUBIC,

    # Supported deep-learning super-resolution models
    "edsr": "EDSR_x4.pb",
    "espcn": "ESPCN_x4.pb",
    "fsrcnn": "FSRCNN_x4.pb",
    "lapsrn": "LapSRN_x4.pb",
}
SR_SCALES = {
    "edsr": (2, 3, 4), "espcn": (2, 3, 4),
    "fsrcnn": (2, 3, 4), "lapsrn": (2, 4, 8),
}
DEFAULT_SR_MODEL_DIR = Path(__file__).resolve().parents[2] / "checkpoints/pretrained/super_resolution"

@dataclass(frozen=True)
class PreprocessSummary:
    source_video: str
    source_fps: float
    sampled_fps: float
    sampled_frames: int
    windows_written: int
    manifest_path: str | None
    diagnostics_path: str | None = None
    run_record_path: str | None = None


def _patient_split(patient_id: str, seed: int = 42) -> str:
    """Assign all recordings from a patient to one deterministic split."""
    value = int.from_bytes(hashlib.sha256(f"{seed}:{patient_id}".encode()).digest()[:8], "big") / 2**64
    return "train" if value < .7 else "validation" if value < .85 else "test"


def _window_starts(frame_count: int, length: int, stride: int) -> range:
    if length <= 0 or stride <= 0:
        raise ValueError("Window length and stride must be positive")
    return range(0, max(0, frame_count - length + 1), stride)


def _write_rows(path: Path, rows: Iterable[dict[str, Any]], *, append: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    has_header = append and path.is_file() and path.stat().st_size > 0
    with path.open("a" if append else "w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=MANIFEST_FIELDS)
        if not has_header:
            writer.writeheader()
        writer.writerows(rows)

def _sr_model_path(model: str, scale: int, model_dir: str | Path) -> Path:
    if scale not in SR_SCALES[model]:
        raise ValueError(f"{model} supports scales {SR_SCALES[model]}, received {scale}")
    prefix = SR_MODELS[model].split("_x", 1)[0]
    return Path(model_dir).resolve() / f"{prefix}_x{scale}.pb"


@lru_cache(maxsize=4)
def _load_sr_model(model: str, scale: int, model_path: str) -> Any:
    """Reuse loaded networks for sequential frame processing in this process."""
    if not hasattr(cv2, "dnn_superres"):
        raise RuntimeError(
            "Neural upscaling requires an OpenCV contrib build with dnn_superres. "
            "Install opencv-contrib-python or opencv-contrib-python-headless."
        )
    if not Path(model_path).is_file():
        raise FileNotFoundError(
            f"Super-resolution model not found: {model_path}. "
            "Run python -m tests.test_upscaling with --download-models to fetch the weights."
        )
    sr = cv2.dnn_superres.DnnSuperResImpl_create()
    sr.readModel(model_path)
    sr.setModel(model, scale)
    return sr


def _upscaling(
    image: np.ndarray, model: str, *, scale: int = 4,
    model_dir: str | Path = DEFAULT_SR_MODEL_DIR,
) -> np.ndarray:
    """Upscale a uint8 BGR image by an integer factor (default: 4 for all six).

    Interpolation needs no weights. Neural methods use the checkpoint matching
    ``scale`` and cache up to four networks; do not share them across threads.
    """
    model = model.strip().lower()
    if model not in SR_MODELS:
        raise ValueError(f"Unknown upscaling model {model!r}; choose from {tuple(SR_MODELS)}")
    if isinstance(scale, bool) or not isinstance(scale, (int, np.integer)) or scale < 2:
        raise ValueError("scale must be an integer of at least 2")
    scale = int(scale)
    if not isinstance(image, np.ndarray) or image.ndim != 3 or image.shape[2] != 3 or image.size == 0:
        raise ValueError("image must be a non-empty BGR image with shape [H, W, 3]")
    if image.dtype != np.uint8:
        raise ValueError("image must have dtype uint8 (pixel values 0 to 255)")
    if model in {"bilinear", "bicubic"}:
        return cv2.resize(
            image, (image.shape[1] * scale, image.shape[0] * scale),
            interpolation=SR_MODELS[model],
        )
    model_path = _sr_model_path(model, scale, model_dir)
    sr = _load_sr_model(model, scale, str(model_path))
    return sr.upsample(np.ascontiguousarray(image))


def _validate_enhancement(
    *, upscale_model: str | None, upscale_scale: int,
    sharpen_amount: float, sharpen_sigma: float,
) -> None:
    if upscale_model is not None and upscale_model not in SR_MODELS:
        raise ValueError(f"Unknown upscaling model {upscale_model!r}; choose from {tuple(SR_MODELS)}")
    if isinstance(upscale_scale, bool) or not isinstance(upscale_scale, (int, np.integer)) or upscale_scale < 2:
        raise ValueError("upscale_scale must be an integer of at least 2")
    if upscale_model in SR_SCALES and upscale_scale not in SR_SCALES[upscale_model]:
        raise ValueError(f"{upscale_model} supports scales {SR_SCALES[upscale_model]}")
    if not math.isfinite(sharpen_amount) or sharpen_amount < 0:
        raise ValueError("sharpen_amount must be finite and nonnegative")
    if not math.isfinite(sharpen_sigma) or sharpen_sigma <= 0:
        raise ValueError("sharpen_sigma must be finite and positive")


def enhance_face_crop(
    image: np.ndarray, *, upscale_model: str | None = None,
    upscale_scale: int = 4, sr_model_dir: str | Path = DEFAULT_SR_MODEL_DIR,
    sharpen_amount: float = 0.0, sharpen_sigma: float = 1.0,
) -> np.ndarray:
    """Optionally upscale, then sharpen a uint8 BGR crop without modifying it.

    Reuses ``_upscaling`` and its cached neural weights. Unsharp masking adds
    ``amount * (image - GaussianBlur(image))`` to enhance existing edges; it
    cannot recover missing detail. Sigma is measured in pixels after upscaling.
    Defaults preserve the input pixels. Apply to a crop, before RGB conversion.
    """
    if upscale_model is not None:
        upscale_model = upscale_model.strip().lower()
    _validate_enhancement(
        upscale_model=upscale_model, upscale_scale=upscale_scale,
        sharpen_amount=sharpen_amount, sharpen_sigma=sharpen_sigma,
    )
    if not isinstance(image, np.ndarray) or image.ndim != 3 or image.shape[2] != 3 or image.size == 0:
        raise ValueError("image must be a non-empty BGR image with shape [H, W, 3]")
    if image.dtype != np.uint8:
        raise ValueError("image must have dtype uint8 (pixel values 0 to 255)")
    result = image.copy() if upscale_model is None else _upscaling(
        image, upscale_model, scale=upscale_scale, model_dir=sr_model_dir,
    )
    if sharpen_amount > 0:
        pixels = result.astype(np.float32)
        blurred = cv2.GaussianBlur(pixels, (0, 0), sigmaX=sharpen_sigma)
        # Floating-point subtraction and clipping avoid uint8 wraparound.
        sharpened = cv2.addWeighted(pixels, 1 + sharpen_amount, blurred, -sharpen_amount, 0)
        result = np.clip(np.rint(sharpened), 0, 255).astype(np.uint8)
    return np.ascontiguousarray(result)


def add_enhancement_arguments(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("Face crop enhancement (disabled by default)")
    group.add_argument("--upscale-model", choices=tuple(SR_MODELS),
                       help="Upscale head/face crops before landmark detection and RGB extraction.")
    group.add_argument("--upscale-scale", type=int, default=4)
    group.add_argument("--sr-model-dir", type=Path, default=DEFAULT_SR_MODEL_DIR)
    group.add_argument("--sharpen-amount", type=float, default=0.0,
                       help="Unsharp-mask strength; 0 disables sharpening (try 0.5).")
    group.add_argument("--sharpen-sigma", type=float, default=1.0,
                       help="Gaussian blur sigma in crop pixels after upscaling.")


def enhancement_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    return {name: getattr(args, name) for name in (
        "upscale_model", "upscale_scale", "sr_model_dir", "sharpen_amount", "sharpen_sigma",
    )}


def _normalized_pose(result: Any, pose_index: int = 0) -> tuple[np.ndarray, bool]:
    if not result.pose_landmarks:
        return np.zeros((33, 4), dtype=np.float32), False

    if pose_index < 0 or pose_index >= len(result.pose_landmarks):
        return np.zeros((33, 4), dtype=np.float32), False
    landmarks = result.pose_landmarks[pose_index]

    pose = np.asarray(
        [
            [
                float(point.x or 0.0),
                float(point.y or 0.0),
                float(point.z or 0.0),
                float(point.visibility or 0.0),
            ]
            for point in landmarks
        ],
        dtype=np.float32,
    )

    if pose.shape != (33, 4):
        return np.zeros((33, 4), dtype=np.float32), False

    center = (pose[23, :3] + pose[24, :3]) / 2
    scale = float(
        np.linalg.norm(pose[11, :3] - pose[12, :3])
    )

    if scale > 1e-6:
        pose[:, :3] = (pose[:, :3] - center) / scale

    return pose, True


def _select_patient_pose(
    result: Any,
    patient_roi: tuple[float, float, float, float],
    previous_head_center: tuple[float, float] | None = None,
    max_head_jump: float = 0.18,
) -> int | None:
    """Select/track one patient's head without switching to another person."""
    if not result.pose_landmarks:
        return None
    x1, y1, x2, y2 = patient_roi
    candidates: list[tuple[float, float, int]] = []
    for index, landmarks in enumerate(result.pose_landmarks):
        head = landmarks[:11]
        visible = [
            point
            for point in head
            if point.x is not None
            and point.y is not None
            and (point.visibility is None or point.visibility >= 0.2)
            and (point.presence is None or point.presence >= 0.2)
        ]
        if not visible:
            continue
        center_x = float(np.mean([point.x for point in visible]))
        center_y = float(np.mean([point.y for point in visible]))
        if not (x1 <= center_x <= x2 and y1 <= center_y <= y2):
            continue
        visibility = float(
            np.mean([
                point.visibility if point.visibility is not None else 1.0
                for point in visible
            ])
        )
        if previous_head_center is None:
            # First acquisition: prefer the most complete, visible head in ROI.
            candidates.append((0.0, -(visibility * len(visible)), index))
            continue
        distance = float(
            np.hypot(
                center_x - previous_head_center[0],
                center_y - previous_head_center[1],
            )
        )
        if distance <= max_head_jump:
            # Tracking: spatial continuity takes priority over visibility.
            candidates.append((distance, -(visibility * len(visible)), index))
    return min(candidates)[2] if candidates else None


def _pose_head_center(result: Any, pose_index: int | None) -> tuple[float, float] | None:
    if pose_index is None or not result.pose_landmarks:
        return None
    visible = [
        point
        for point in result.pose_landmarks[pose_index][:11]
        if point.x is not None
        and point.y is not None
        and (point.visibility is None or point.visibility >= 0.2)
        and (point.presence is None or point.presence >= 0.2)
    ]
    if not visible:
        return None
    return (
        float(np.mean([point.x for point in visible])),
        float(np.mean([point.y for point in visible])),
    )


def _square_box(
    center_x: float,
    center_y: float,
    side: float,
    frame_width: int,
    frame_height: int,
) -> np.ndarray:
    side = min(max(side, 2.0), float(min(frame_width, frame_height)))
    x1 = float(np.clip(center_x - side / 2, 0, frame_width - side))
    y1 = float(np.clip(center_y - side / 2, 0, frame_height - side))
    return np.asarray([x1, y1, x1 + side, y1 + side], dtype=np.float32)


def _pose_head_box(
    result: Any,
    pose_index: int | None,
    frame_width: int,
    frame_height: int,
    margin: float = 0.75,
) -> np.ndarray | None:
    """Estimate an enlarged square head region from pose face/shoulder points."""
    if pose_index is None or not result.pose_landmarks:
        return None
    landmarks = result.pose_landmarks[pose_index]
    head = [
        point
        for point in landmarks[:11]
        if point.x is not None
        and point.y is not None
        and (point.visibility is None or point.visibility >= 0.2)
        and (point.presence is None or point.presence >= 0.2)
    ]
    if len(head) < 3:
        return None
    points = np.asarray(
        [[point.x * frame_width, point.y * frame_height] for point in head],
        dtype=np.float32,
    )
    center = points.mean(axis=0)
    side = float(np.max(points.max(axis=0) - points.min(axis=0)))
    if len(landmarks) > 12:
        shoulders = landmarks[11:13]
        if all(point.x is not None and point.y is not None for point in shoulders):
            shoulder_points = np.asarray(
                [[point.x * frame_width, point.y * frame_height] for point in shoulders],
                dtype=np.float32,
            )
            side = max(side, float(np.linalg.norm(shoulder_points[0] - shoulder_points[1])) * 0.8)
    side = max(side * (1 + 2 * margin), min(frame_width, frame_height) * 0.08)
    return _square_box(float(center[0]), float(center[1]), side, frame_width, frame_height)


def _crop_box(image: np.ndarray, box: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Crop a clipped integer box and return the crop and actual integer box."""
    height, width = image.shape[:2]
    x1, y1, x2, y2 = np.rint(box).astype(int)
    x1, x2 = int(np.clip(x1, 0, width - 1)), int(np.clip(x2, 1, width))
    y1, y2 = int(np.clip(y1, 0, height - 1)), int(np.clip(y2, 1, height))
    actual = np.asarray([x1, y1, x2, y2], dtype=np.float32)
    return image[y1:y2, x1:x2], actual


@dataclass(frozen=True)
class _DetectedFace:
    box: np.ndarray
    center: tuple[float, float]
    detection: Any


def _select_face_detection(
    result: Any,
    frame_width: int,
    frame_height: int,
    patient_roi: tuple[float, float, float, float],
    previous_head_center: tuple[float, float] | None = None,
    max_head_jump: float = 0.18,
    margin: float = 0.25,
) -> _DetectedFace | None:
    """Choose one face inside the patient ROI and convert its box to pixels.

    The highest-confidence face starts a track. Subsequent frames prefer the
    nearest face and reject jumps to other people. Padding produces a square,
    frame-clipped crop suitable for the existing landmark and RGB/flow stages.
    """
    x1, y1, x2, y2 = patient_roi
    candidates = []
    for index, detection in enumerate(result.detections or []):
        rect = detection.location_data.relative_bounding_box
        if not np.isfinite([rect.xmin, rect.ymin, rect.width, rect.height]).all():
            continue
        if rect.width <= 0 or rect.height <= 0:
            continue
        # Boxes can extend beyond the image; reject those entirely outside it.
        if (rect.xmin >= 1 or rect.ymin >= 1
                or rect.xmin + rect.width <= 0 or rect.ymin + rect.height <= 0):
            continue
        center = (rect.xmin + rect.width / 2, rect.ymin + rect.height / 2)
        if not (x1 <= center[0] <= x2 and y1 <= center[1] <= y2):
            continue
        distance = 0.0 if previous_head_center is None else float(
            np.hypot(center[0] - previous_head_center[0], center[1] - previous_head_center[1])
        )
        if previous_head_center is not None and distance > max_head_jump:
            continue
        score = float(detection.score[0]) if len(detection.score) else 0.0
        if not math.isfinite(score):
            continue
        side = max(rect.width * frame_width, rect.height * frame_height) * (1 + 2 * margin)
        box = _square_box(center[0] * frame_width, center[1] * frame_height,
                          side, frame_width, frame_height)
        candidates.append((distance, -score, index, _DetectedFace(box, center, detection)))
    return min(candidates, key=lambda item: item[:3])[3] if candidates else None


def _translate_box(local_box: np.ndarray, parent_box: np.ndarray) -> np.ndarray:
    translated = local_box.copy()
    translated[[0, 2]] += parent_box[0]
    translated[[1, 3]] += parent_box[1]
    return translated


def _face_geometry(
    result: Any,
) -> tuple[np.ndarray, bool]:
    empty = np.zeros(
        len(FACE_LANDMARK_PAIRS) * 2,
        dtype=np.float32,
    )

    if not result.face_landmarks:
        return empty, False

    landmarks = result.face_landmarks[0]

    points = np.asarray(
        [
            [float(point.x), float(point.y)]
            for point in landmarks
        ],
        dtype=np.float32,
    )

    required_index = max(
        max(pair) for pair in FACE_LANDMARK_PAIRS
    )

    if len(points) <= required_index:
        return empty, False

    scale = float(
        np.linalg.norm(points[33] - points[263])
    )

    if scale <= 1e-6:
        return empty, False

    geometry = np.asarray(
        [
            (points[first] - points[second]) / scale
            for first, second in FACE_LANDMARK_PAIRS
        ],
        dtype=np.float32,
    ).reshape(-1)

    return geometry, True


def _face_box(
    result: Any,
    frame_width: int,
    frame_height: int,
    margin: float,
) -> np.ndarray | None:
    if not result.face_landmarks:
        return None

    landmarks = result.face_landmarks[0]

    points = np.asarray(
        [
            [
                float(point.x) * frame_width,
                float(point.y) * frame_height,
            ]
            for point in landmarks
        ],
        dtype=np.float32,
    )

    if (
        points.ndim != 2
        or points.shape[1] != 2
        or not np.isfinite(points).all()
    ):
        return None

    minimum = points.min(axis=0)
    maximum = points.max(axis=0)
    center = (minimum + maximum) / 2

    side = float(np.max(maximum - minimum))
    side *= 1 + 2 * margin
    side = min(
        max(side, 2.0),
        float(min(frame_width, frame_height)),
    )

    x1 = float(
        np.clip(center[0] - side / 2, 0, frame_width - side)
    )
    y1 = float(
        np.clip(center[1] - side / 2, 0, frame_height - side)
    )

    return np.asarray(
        [x1, y1, x1 + side, y1 + side],
        dtype=np.float32,
    )


def _smooth_box(previous: np.ndarray | None, current: np.ndarray, alpha: float) -> np.ndarray:
    """Suppress detector jitter; ``alpha`` is the weight of the prior box."""
    return current if previous is None else alpha * previous + (1 - alpha) * current


def _crop_face_input(
    frame_bgr: np.ndarray,
    box: np.ndarray | None,
    previous_gray: np.ndarray | None,
    *,
    face_size: int,
    cv2: Any,
    enhancement: Mapping[str, Any] | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Create one normalized ``[4,H,W]`` RGB + flow-magnitude tensor."""
    if box is None:
        return np.zeros((4, face_size, face_size), dtype=np.float32), None
    x1, y1, x2, y2 = np.rint(box).astype(int)
    crop = frame_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        return np.zeros((4, face_size, face_size), dtype=np.float32), None
    interpolation = cv2.INTER_AREA if min(crop.shape[:2]) >= face_size else cv2.INTER_CUBIC
    resized = cv2.resize(crop, (face_size, face_size), interpolation=interpolation)
    # Flow uses the incoming video frame before optional crop sharpening/upscaling.
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    if enhancement is not None:
        enhanced = enhance_face_crop(crop, **enhancement)
        interpolation = cv2.INTER_AREA if min(enhanced.shape[:2]) >= face_size else cv2.INTER_CUBIC
        resized = cv2.resize(enhanced, (face_size, face_size), interpolation=interpolation)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    magnitude = np.zeros((face_size, face_size), dtype=np.float32)
    if previous_gray is not None:
        flow = cv2.calcOpticalFlowFarneback(
            previous_gray, gray, None, .5, 3, 15, 3, 5, 1.2, 0
        )
        magnitude = np.linalg.norm(flow, axis=2).astype(np.float32)
        # Robust scaling prevents one extreme motion vector dominating a frame.
        scale = float(np.percentile(magnitude, 99))
        if scale > 1e-6:
            magnitude = np.clip(magnitude / scale, 0, 1)
        else:
            magnitude.fill(0)
    return np.concatenate((rgb.transpose(2, 0, 1), magnitude[None]), axis=0), gray


def _show_preprocessing_preview(
    frame_bgr: np.ndarray,
    face_input: np.ndarray,
    box: np.ndarray | None,
    *,
    time_sec: float,
    face_landmarks_found: bool,
    box_reused: bool,
    delay_ms: int,
    cv2: Any,
    face_detection: Any | None = None,
    draw_detection: Any | None = None,
) -> bool:
    """Display the selected face and exact RGB tensor crop; return False on q."""
    source = frame_bgr.copy()
    color = (0, 200, 0) if face_landmarks_found or face_detection is not None else (0, 165, 255)
    status = "POSE HEAD FALLBACK" if box is not None else "NO FACE"
    if face_detection is not None:
        status = "FACE BOX (NO LANDMARKS)"
        if draw_detection is not None:
            draw_detection(source, face_detection)
    if face_landmarks_found:
        status = "FACE LANDMARKS"
    if box_reused:
        status = "REUSED PREVIOUS BOX"
    if box is not None:
        x1, y1, x2, y2 = np.rint(box).astype(int)
        cv2.rectangle(source, (x1, y1), (x2, y2), color, 3)

    cv2.putText(
        source,
        f"{time_sec:.2f}s | {status} | q: stop",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        color,
        2,
        cv2.LINE_AA,
    )

    max_width = 960
    if source.shape[1] > max_width:
        scale = max_width / source.shape[1]
        source = cv2.resize(
            source,
            (max_width, round(source.shape[0] * scale)),
            interpolation=cv2.INTER_AREA,
        )

    crop_rgb = np.clip(face_input[:3].transpose(1, 2, 0), 0, 1)
    crop_bgr = (crop_rgb[:, :, ::-1] * 255).astype(np.uint8)
    crop_bgr = cv2.resize(crop_bgr, (448, 448), interpolation=cv2.INTER_NEAREST)
    cv2.putText(
        crop_bgr,
        "MODEL RGB CROP",
        (12, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    cv2.imshow("Facial preprocessing - source", source)
    cv2.imshow("Facial preprocessing - model crop", crop_bgr)
    key = cv2.waitKey(max(1, delay_ms)) & 0xFF
    if key in (ord("q"), 27):
        cv2.destroyWindow("Facial preprocessing - source")
        cv2.destroyWindow("Facial preprocessing - model crop")
        return False
    return True


def extract_video(
    video_path: str | Path,
    output: str | Path,
    sample_id: str,
    fps: float = 5.0,
    window_seconds: float = 8.0,
    *,
    pose_model_path: str | Path,
    face_model_path: str | Path,
    stride_seconds: float | None = None,
    patient_id: str | None = None,
    label: int | str = 0,
    dataset: str = "seizure_video",
    split: str | None = None,
    split_seed: int = 42,
    append_manifest: bool = False,
    face_size: int = 224,
    face_margin: float = 0.25,
    upscale_model: str | None = None,
    upscale_scale: int = 4,
    sr_model_dir: str | Path = DEFAULT_SR_MODEL_DIR,
    sharpen_amount: float = 0.0,
    sharpen_sigma: float = 1.0,
    box_smoothing: float = 0.7,
    max_box_reuse_frames: int = 15,
    max_box_reuse_seconds: float | None = None,
    min_pose_detection_confidence: float = 0.5,
    min_pose_presence_confidence: float = 0.5,
    min_pose_tracking_confidence: float = 0.5,
    min_face_detection_confidence: float = 0.5,
    min_face_presence_confidence: float = 0.5,
    min_face_tracking_confidence: float = 0.5,
    face_detection_model_selection: int = 1,
    patient_roi: tuple[float, float, float, float] = (0.0, 0.0, 1.0, 1.0),
    max_poses: int = 4,
    head_margin: float = 0.75,
    max_head_jump: float = 0.18,
    preview: bool = False,
    preview_delay_ms: int = 1,
    diagnostics_only: bool = False,
    max_duration_seconds: float | None = None,
    progress_config: Mapping[str, Any] | None = None,
    video_enhancement: SPEINetConfig | None = None,
) -> PreprocessSummary:
    """Extract model-ready face crops and landmarks; discard partial windows.

    The caller supplies discovered source/identity information. Optional progress
    uses the shared ``testing.progress_file`` configuration. Each successful run
    records input hashes, source metadata, settings and software versions.
    Enhancement is opt-in; face boxes remain in original-frame coordinates.
    ``video_enhancement`` restores full frames before pose/face detection, RGB
    crops and optical flow. The manifest continues to identify the source video.
    Face Detection runs on RGB frames with the selected short/full-range model.
    Its boxes can supply crops even if pose or face landmarks are unavailable.
    """
    if upscale_model is not None:
        upscale_model = upscale_model.strip().lower()
    enhancement = dict(
        upscale_model=upscale_model, upscale_scale=upscale_scale,
        sharpen_amount=sharpen_amount, sharpen_sigma=sharpen_sigma,
    )
    _validate_enhancement(**enhancement)
    enhancement["upscale_scale"] = int(upscale_scale)
    enhancement["sr_model_dir"] = str(Path(sr_model_dir).resolve())
    crop_enhancement = enhancement if upscale_model is not None or sharpen_amount > 0 else None
    sr_path = None
    if upscale_model in SR_SCALES:
        sr_path = _sr_model_path(upscale_model, upscale_scale, sr_model_dir)
        # Fail before decoding if the requested checkpoint/build is unavailable.
        _load_sr_model(upscale_model, int(upscale_scale), str(sr_path))
    pose_model_path = Path(pose_model_path).resolve()
    face_model_path = Path(face_model_path).resolve()

    if not pose_model_path.is_file():
        raise FileNotFoundError(
            f"Pose task model not found: {pose_model_path}"
        )

    if not face_model_path.is_file():
        raise FileNotFoundError(
            f"Face task model not found: {face_model_path}"
        )
    
    if not math.isfinite(fps) or not math.isfinite(window_seconds) or fps <= 0 or window_seconds <= 0:
        raise ValueError("fps and window_seconds must be positive")
    
    stride_seconds = window_seconds / 2 if stride_seconds is None else stride_seconds
    if not math.isfinite(stride_seconds) or stride_seconds <= 0:
        raise ValueError("stride_seconds must be positive")
    
    if face_size <= 0:
        raise ValueError("face_size must be positive")
    if not math.isfinite(face_margin) or not math.isfinite(head_margin) or face_margin < 0 or head_margin < 0:
        raise ValueError("Invalid face or head margin")
    tuning = dict(
        min_pose_detection_confidence=min_pose_detection_confidence,
        min_pose_presence_confidence=min_pose_presence_confidence,
        min_pose_tracking_confidence=min_pose_tracking_confidence,
        min_face_detection_confidence=min_face_detection_confidence,
        min_face_presence_confidence=min_face_presence_confidence,
        min_face_tracking_confidence=min_face_tracking_confidence,
        box_smoothing=box_smoothing,
        max_box_reuse_frames=max_box_reuse_frames,
        max_box_reuse_seconds=max_box_reuse_seconds,
    )
    validate_tuning(tuning)
    if type(face_detection_model_selection) is not int or face_detection_model_selection not in (0, 1):
        raise ValueError("face_detection_model_selection must be 0 (short range) or 1 (full range)")
    if max_duration_seconds is not None and (
        not math.isfinite(max_duration_seconds) or max_duration_seconds <= 0
    ):
        raise ValueError("max_duration_seconds must be finite and positive")
    if max_poses <= 0:
        raise ValueError("max_poses must be positive")
    if not 0 < max_head_jump <= 1:
        raise ValueError("max_head_jump must be in the range (0, 1]")
    if not (
        len(patient_roi) == 4
        and 0 <= patient_roi[0] < patient_roi[2] <= 1
        and 0 <= patient_roi[1] < patient_roi[3] <= 1
    ):
        raise ValueError("patient_roi must be normalized x1,y1,x2,y2 coordinates")
    
    if not sample_id or "__" in sample_id or Path(sample_id).name != sample_id or sample_id in {".", ".."}:
        raise ValueError("sample_id must be a non-empty filename without '__' or directory components")

    patient = patient_id or sample_id.split("_", 1)[0]
    assigned_split = split or _patient_split(patient, split_seed)
    if assigned_split not in {"train", "validation", "test", "unassigned"}:
        raise ValueError(f"Invalid split: {assigned_split!r}")
    
    try:
        import cv2
        import mediapipe as mp
    except ImportError as exc:
        raise RuntimeError("Video preprocessing requires opencv-python-headless and mediapipe") from exc
    if not hasattr(mp, "solutions") or not hasattr(mp.solutions, "face_detection"):
        raise RuntimeError("Face Detection requires the project's pinned mediapipe==0.10.21 with mp.solutions.")

    source = Path(video_path).resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Video not found: {source}")

    video_enhancer = SPEINetEnhancer(video_enhancement) if video_enhancement is not None else None
    
    destination = Path(output).resolve()
    pose_dir, face_dir = destination / "pose", destination / "face"
    geometry_dir = destination / "face_geometry"
    if not diagnostics_only:
        pose_dir.mkdir(parents=True, exist_ok=True)
        face_dir.mkdir(parents=True, exist_ok=True)
        geometry_dir.mkdir(parents=True, exist_ok=True)

    poses: list[np.ndarray] = []
    faces: list[np.ndarray] = []

    face_geometries: list[np.ndarray] = []

    pose_hits: list[bool] = []
    face_hits: list[bool] = []
    face_detection_hits: list[bool] = []
    face_region_hits: list[bool] = []
    pose_head_fallbacks: list[bool] = []

    face_box_reused: list[bool] = []
    times: list[float] = []
    diagnostic_rows: list[dict[str, Any]] = []

    base_options = mp.tasks.BaseOptions
    FaceLandmarker = mp.tasks.vision.FaceLandmarker
    FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
    PoseLandmarker = mp.tasks.vision.PoseLandmarker
    PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
    RunningMode = mp.tasks.vision.RunningMode

    pose_options = PoseLandmarkerOptions(
        base_options=base_options(
            model_asset_path=str(pose_model_path),
        ),
        running_mode=RunningMode.VIDEO,
        num_poses=max_poses,
        min_pose_detection_confidence=min_pose_detection_confidence,
        min_pose_presence_confidence=min_pose_presence_confidence,
        min_tracking_confidence=min_pose_tracking_confidence,
    )

    face_options = FaceLandmarkerOptions(
        base_options=base_options(
            model_asset_path=str(face_model_path),
        ),
        running_mode=RunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=min_face_detection_confidence,
        min_face_presence_confidence=min_face_presence_confidence,
        min_tracking_confidence=min_face_tracking_confidence,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=True,
    )

    with ExitStack() as resources:
        capture, metadata = resources.enter_context(open_video(source, cv2=cv2))
        source_fps = metadata.fps
        settings = {
            **tuning, **enhancement,
            "source_video": str(source), "pose_model_path": str(pose_model_path),
            "face_model_path": str(face_model_path), "fps": fps, "source_fps": source_fps,
            "window_seconds": window_seconds, "stride_seconds": stride_seconds,
            "face_size": face_size, "face_margin": face_margin, "patient_roi": patient_roi,
            "max_poses": max_poses, "head_margin": head_margin, "max_head_jump": max_head_jump,
            "running_mode": "VIDEO", "diagnostics_only": diagnostics_only,
            "max_duration_seconds": max_duration_seconds,
            "sample_id": sample_id, "dataset": dataset, "patient_id": patient,
            "patient_id_source": "argument" if patient_id else "sample_id_prefix",
            "label": label, "split": assigned_split, "split_seed": split_seed,
            "split_policy": "explicit" if split else "patient_sha256_70_15_15",
            "append_manifest": append_manifest, "preview": preview,
            "preview_delay_ms": preview_delay_ms,
            "face_detector": "mediapipe.solutions.face_detection",
            "face_detection_model_selection": face_detection_model_selection,
            "video_enhancement": video_enhancer.record if video_enhancer is not None else None,
        }
        run_record = build_preprocessing_record(
            pipeline="facial_resnet", sample_id=sample_id,
            inputs={"video": source, "pose_model": pose_model_path, "face_model": face_model_path,
                    **({"super_resolution_model": sr_path} if sr_path is not None else {}),
                    **({"speinet_checkpoint": video_enhancer.record["checkpoint"]}
                       if video_enhancer is not None else {})},
            settings=settings, source_metadata=metadata.as_dict(),
            software_versions={
                "python": sys.version.split()[0], "numpy": np.__version__,
                "opencv": getattr(cv2, "__version__", "unknown"),
                "mediapipe": getattr(mp, "__version__", "unknown"),
            },
        )
        pose_api = PoseLandmarker.create_from_options(pose_options)
        resources.callback(pose_api.close)
        face_api = FaceLandmarker.create_from_options(face_options)
        resources.callback(face_api.close)
        face_detector = mp.solutions.face_detection.FaceDetection(
            model_selection=face_detection_model_selection,
            min_detection_confidence=min_face_detection_confidence,
        )
        resources.callback(face_detector.close)
        if preview:
            resources.callback(cv2.destroyAllWindows)

        duration = metadata.duration_sec
        if max_duration_seconds is not None:
            duration = min(duration, max_duration_seconds) if duration is not None else max_duration_seconds
        estimated_samples = math.ceil(duration * min(fps, source_fps)) if duration is not None else 0
        processing_start = time.perf_counter()
        report_progress(
            progress_config or {}, stage="facial_preprocessing", label="Extracting face windows",
            completed=0, total=estimated_samples, elapsed_sec=0.0,
        )
        last_progress_time = processing_start
        smoothed_box: np.ndarray | None = None
        previous_face_gray: np.ndarray | None = None
        missed_face_frames = 0
        tracked_head_center: tuple[float, float] | None = None
        last_detected_box_sec: float | None = None

        frame_options = dict(source_fps=source_fps, fps=fps, max_duration_seconds=max_duration_seconds)
        # Full-frame video restoration precedes all pose/face preprocessing.
        frame_stream = (
            iter_enhanced_frames(capture, enhancer=video_enhancer, **frame_options)
            if video_enhancer is not None else iter_sampled_frames(capture, **frame_options)
        )
        for time_sec, bgr in frame_stream:

            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            rgb = np.ascontiguousarray(rgb)

            media_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb,
            )

            timestamp_ms = int(round(time_sec * 1000))

            selected_face = _select_face_detection(
                face_detector.process(rgb), bgr.shape[1], bgr.shape[0], patient_roi,
                tracked_head_center, max_head_jump, face_margin,
            )
            if selected_face is not None:
                tracked_head_center = selected_face.center

            pose_result = pose_api.detect_for_video(
                media_image,
                timestamp_ms,
            )

            pose_index = _select_patient_pose(
                pose_result,
                patient_roi,
                tracked_head_center,
                max_head_jump,
            )
            current_head_center = _pose_head_center(pose_result, pose_index)
            if selected_face is None and current_head_center is not None:
                tracked_head_center = current_head_center
            pose, has_pose = _normalized_pose(
                pose_result, pose_index if pose_index is not None else -1
            )
            head_box = _pose_head_box(
                pose_result,
                pose_index,
                bgr.shape[1],
                bgr.shape[0],
                head_margin,
            )

            face_result = None
            detected_box = selected_face.box if selected_face is not None else None
            landmark_crop_box = detected_box if detected_box is not None else head_box
            has_face = False
            geometry = np.zeros(len(FACE_LANDMARK_PAIRS) * 2, np.float32)
            if landmark_crop_box is not None:
                head_rgb, actual_head_box = _crop_box(rgb, landmark_crop_box)
                if head_rgb.size:
                    face_rgb = head_rgb
                    if crop_enhancement is not None:
                        head_bgr, _ = _crop_box(bgr, actual_head_box)
                        face_rgb = cv2.cvtColor(
                            enhance_face_crop(head_bgr, **crop_enhancement), cv2.COLOR_BGR2RGB,
                        )
                    head_image = mp.Image(
                        image_format=mp.ImageFormat.SRGB,
                        data=np.ascontiguousarray(face_rgb),
                    )
                    face_result = face_api.detect_for_video(head_image, timestamp_ms)

                    if face_result.face_landmarks:
                        # Face 0: landmarks with x, y, z attributes
                        landmarks = face_result.face_landmarks[0]

                        original_points = []
                        for point in landmarks:
                            x_original = point.x * head_rgb.shape[1] + actual_head_box[0]
                            y_original = point.y * head_rgb.shape[0] + actual_head_box[1]
                            original_points.append([x_original, y_original])

                        landmark_xy_original = np.asarray(
                            original_points, dtype=np.float32
                        )
                        first_point = landmarks[0]
                        print("Landmark 0:", first_point.x, first_point.y, first_point.z)

                        # Convert all landmarks to a NumPy array: [478, 3]
                        landmark_xyz = np.asarray(
                            [[point.x, point.y, point.z] for point in landmarks],
                            dtype=np.float32,
                        )

                        # Face 0: expression names mapped to scores
                        if face_result.face_blendshapes:
                            blendshapes = {
                                category.category_name: category.score
                                for category in face_result.face_blendshapes[0]
                            }
                            print("Blendshapes:", blendshapes)

                        # Face 0: transformation matrix
                        if face_result.facial_transformation_matrixes:
                            matrix = face_result.facial_transformation_matrixes[0]
                            print("Transformation matrix:", matrix)
                    else:
                        fallback = "detected face box" if selected_face is not None else "pose head box"
                        print(f"[{time_sec:.2f}s] No face landmarks; using {fallback}")
                    geometry, has_face = _face_geometry(face_result)
                    local_box = _face_box(
                        face_result,
                        # Landmarks are normalized: map to the original crop dimensions.
                        head_rgb.shape[1],
                        head_rgb.shape[0],
                        face_margin,
                    )
                    if local_box is not None:
                        detected_box = _translate_box(local_box, actual_head_box)
                if detected_box is None:
                    detected_box = head_box

            used_pose_fallback = detected_box is not None and selected_face is None and not has_face

            reused = False
            previous_box = smoothed_box
            if detected_box is not None:
                smoothed_box = _smooth_box(smoothed_box, detected_box, box_smoothing)
                missed_face_frames = 0
                last_detected_box_sec = time_sec
            else:
                missed_face_frames += 1
                reused = smoothed_box is not None and last_detected_box_sec is not None and can_reuse_box(
                    missed_frames=missed_face_frames,
                    elapsed_seconds=time_sec - last_detected_box_sec,
                    max_frames=max_box_reuse_frames,
                    max_seconds=max_box_reuse_seconds,
                )
                if not reused:
                    smoothed_box = None
            face_input, current_gray = _crop_face_input(
                bgr, smoothed_box, previous_face_gray,
                face_size=face_size, cv2=cv2,
                enhancement=crop_enhancement,
            )
            stop_requested = False
            if preview:
                stop_requested = not _show_preprocessing_preview(
                    bgr,
                    face_input,
                    smoothed_box,
                    time_sec=time_sec,
                    face_landmarks_found=has_face,
                    box_reused=reused,
                    delay_ms=preview_delay_ms,
                    cv2=cv2,
                    face_detection=selected_face.detection if selected_face is not None else None,
                    draw_detection=mp.solutions.drawing_utils.draw_detection,
                )
            previous_face_gray = current_gray
            diagnostic_rows.append(frame_diagnostics(
                time_sec=time_sec, raw_box=detected_box, crop_box=smoothed_box,
                previous_box=previous_box, frame_shape=bgr.shape,
                flags=(has_pose, has_face, smoothed_box is not None, used_pose_fallback, reused),
                face_detector_detected=selected_face is not None,
            ))
            if not diagnostics_only:
                poses.append(pose); faces.append(face_input); face_geometries.append(geometry)
            pose_hits.append(has_pose); face_hits.append(has_face)
            face_detection_hits.append(selected_face is not None or has_face)
            face_region_hits.append(smoothed_box is not None)
            pose_head_fallbacks.append(used_pose_fallback)
            face_box_reused.append(reused); times.append(time_sec)
            now = time.perf_counter()
            if now - last_progress_time >= 1.0:
                report_progress(
                    progress_config or {}, stage="facial_preprocessing", label="Extracting face windows",
                    completed=len(times), total=estimated_samples,
                    elapsed_sec=now - processing_start,
                )
                last_progress_time = now
            if stop_requested:
                break

    processing_seconds = time.perf_counter() - processing_start
    diagnostics_path = write_diagnostics(
        destination / "diagnostics", sample_id, diagnostic_rows,
        settings=settings, processing_seconds=processing_seconds,
    )
    rows: list[dict[str, Any]] = []
    manifest = None
    if not diagnostics_only:
        length = max(1, round(fps * window_seconds))
        stride = max(1, round(fps * stride_seconds))
        pose_array, face_array = np.asarray(poses, np.float32), np.asarray(faces, np.float32)
        geometry_array = np.asarray(face_geometries, np.float32)

        for start in _window_starts(len(poses), length, stride):
            stop, start_sec = start + length, times[start]
            end_sec = start_sec + window_seconds
            window_id = f"{sample_id}__{round(start_sec * 1000):010d}__{round(end_sec * 1000):010d}"
            pose_path, face_path = pose_dir / f"{window_id}.npy", face_dir / f"{window_id}.npy"
            geometry_path = geometry_dir / f"{window_id}.npy"
            np.save(pose_path, pose_array[start:stop], allow_pickle=False)
            np.save(face_path, face_array[start:stop], allow_pickle=False)
            np.save(geometry_path, geometry_array[start:stop], allow_pickle=False)
            rows.append({
                "window_id": window_id, "dataset": dataset, "patient_id": patient,
                "recording_id": sample_id, "source_video": str(source),
                "start_sec": f"{start_sec:.6f}", "end_sec": f"{end_sec:.6f}",
                "label": label, "split": assigned_split, "sampled_fps": fps,
                "frame_count": length, "pose_path": str(pose_path.relative_to(destination)),
                "face_path": str(face_path.relative_to(destination)),
                "face_geometry_path": str(geometry_path.relative_to(destination)),
                "face_size": face_size,
                "pose_detection_fraction": f"{np.mean(pose_hits[start:stop]):.6f}",
                "face_detection_fraction": f"{np.mean(face_detection_hits[start:stop]):.6f}",
                "face_region_fraction": f"{np.mean(face_region_hits[start:stop]):.6f}",
                "face_landmark_fraction": f"{np.mean(face_hits[start:stop]):.6f}",
                "pose_head_fallback_fraction": f"{np.mean(pose_head_fallbacks[start:stop]):.6f}",
                "face_box_reused_fraction": f"{np.mean(face_box_reused[start:stop]):.6f}",
            })
        manifest = destination / "video_manifest.csv"
        _write_rows(manifest, rows, append=append_manifest)
    run_record_path = destination / "runs" / f"{sample_id}.json"
    summary = PreprocessSummary(
        str(source), source_fps, fps, len(times), len(rows),
        str(manifest) if manifest is not None else None,
        str(diagnostics_path), str(run_record_path),
    )
    save_run_record(
        {**run_record, "outputs": asdict(summary), "processing_seconds": processing_seconds},
        run_record_path,
    )
    report_progress(
        progress_config or {}, stage="facial_preprocessing", label="Face preprocessing complete",
        completed=len(times), total=len(times), elapsed_sec=time.perf_counter() - processing_start,
    )
    print(json.dumps(asdict(summary), indent=2))
    return summary



def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video"); parser.add_argument("--output", required=True)
    parser.add_argument("--sample-id"); parser.add_argument("--patient-id")
    parser.add_argument("--label", default="0"); parser.add_argument("--dataset", default="seizure_video")
    parser.add_argument("--split", choices=("train", "validation", "test", "unassigned"))
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--progress-file", help="Write shared JSON progress snapshots to this path.")
    parser.add_argument("--fps", type=float, default=5.0)
    parser.add_argument("--window-seconds", type=float, default=8.0)
    parser.add_argument("--stride-seconds", type=float)

    parser.add_argument("--face-size", type=int, default=224)
    parser.add_argument("--face-margin", type=float, default=.25)
    parser.add_argument("--face-detection-model-selection", type=int, choices=(0, 1), default=1,
                        help="MediaPipe Face Detection model: 0 short range, 1 full range (default).")

    add_tuning_arguments(parser)
    add_enhancement_arguments(parser)
    add_video_enhancement_arguments(parser)
    parser.add_argument("--diagnostics-only", action="store_true",
                        help="Save frame diagnostics and metrics without training window arrays.")
    parser.add_argument("--max-duration-seconds", type=float,
                        help="Process only this many seconds from the start of the video.")
    parser.add_argument("--max-poses", type=int, default=4)
    parser.add_argument("--head-margin", type=float, default=.75)
    parser.add_argument(
        "--max-head-jump",
        type=float,
        default=.18,
        help="Maximum normalized frame-to-frame head movement before rejecting an identity switch.",
    )
    parser.add_argument(
        "--patient-roi",
        default="0,0,1,1",
        help="Normalized patient region x1,y1,x2,y2 used to reject caregivers.",
    )

    parser.add_argument(
        "--pose-model",
        default=(
            "checkpoints/pretrained/mediapipe/"
            "pose_landmarker_full.task"
        ),
    )

    parser.add_argument(
        "--face-model",
        default=(
            "checkpoints/pretrained/mediapipe/"
            "face_landmarker.task"
        ),
    )

    parser.add_argument("--append-manifest", action="store_true",)
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Show the source selection and exact model crop while preprocessing.",
    )
    parser.add_argument(
        "--preview-delay-ms",
        type=int,
        default=30,
        help="Delay between sampled preview frames; q or Escape stops processing and saves complete windows.",
    )

    args = parser.parse_args()
    try:
        patient_roi = tuple(float(value) for value in args.patient_roi.split(","))
    except ValueError as exc:
        parser.error(f"--patient-roi must contain four comma-separated numbers: {exc}")
    if len(patient_roi) != 4:
        parser.error("--patient-roi must contain exactly four comma-separated numbers")
    extract_video(args.video, args.output,
            args.sample_id or Path(args.video).stem,
            args.fps, args.window_seconds,
            pose_model_path=args.pose_model,
            face_model_path=args.face_model,
            stride_seconds=args.stride_seconds,
            patient_id=args.patient_id, label=args.label, dataset=args.dataset, split=args.split,
            split_seed=args.split_seed,
            append_manifest=args.append_manifest,
            face_size=args.face_size, face_margin=args.face_margin,
            face_detection_model_selection=args.face_detection_model_selection,
            **enhancement_kwargs(args),
            **video_enhancement_kwargs(args),
            **tuning_kwargs(args),
            patient_roi=patient_roi,
            max_poses=args.max_poses,
            head_margin=args.head_margin,
            max_head_jump=args.max_head_jump,
            preview=args.preview,
            preview_delay_ms=args.preview_delay_ms,
            diagnostics_only=args.diagnostics_only,
            max_duration_seconds=args.max_duration_seconds,
            progress_config={"testing": {"progress_file": args.progress_file}})
    


if __name__ == "__main__":
    main()
