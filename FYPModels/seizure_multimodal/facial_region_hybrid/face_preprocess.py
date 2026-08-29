"""Convert raw videos to stabilized RGB + optical-flow face windows.

Face crops are taken from the original decoded frame before resizing. Upscaling
standardizes the tensor shape; it is not treated as recovered image detail.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np


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
    "face_box_reused_fraction",
)

@dataclass(frozen=True)
class PreprocessSummary:
    source_video: str
    source_fps: float
    sampled_fps: float
    sampled_frames: int
    windows_written: int
    manifest_path: str


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


def _normalized_pose(result: Any) -> tuple[np.ndarray, bool]:
    if not result.pose_landmarks:
        return np.zeros((33, 4), dtype=np.float32), False

    landmarks = result.pose_landmarks[0]

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
) -> tuple[np.ndarray, np.ndarray | None]:
    """Create one normalized ``[4,H,W]`` RGB + flow-magnitude tensor."""
    if box is None:
        return np.zeros((4, face_size, face_size), dtype=np.float32), None
    x1, y1, x2, y2 = np.rint(box).astype(int)
    crop = frame_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        return np.zeros((4, face_size, face_size), dtype=np.float32), None
    interpolation = cv2.INTER_AREA if min(crop.shape[:2]) >= face_size else cv2.INTER_CUBIC
    crop = cv2.resize(crop, (face_size, face_size), interpolation=interpolation)
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
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
    box_smoothing: float = 0.7,
    max_box_reuse_frames: int = 5,
) -> PreprocessSummary:
    """Extract model-ready face crops and landmarks; discard partial windows."""
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
    
    if fps <= 0 or window_seconds <= 0:
        raise ValueError("fps and window_seconds must be positive")
    
    stride_seconds = window_seconds / 2 if stride_seconds is None else stride_seconds
    if stride_seconds <= 0:
        raise ValueError("stride_seconds must be positive")
    
    if face_size <= 0:
        raise ValueError("face_size must be positive")
    if face_margin < 0 or not 0 <= box_smoothing < 1 or max_box_reuse_frames < 0:
        raise ValueError("Invalid face margin, box smoothing, or box reuse setting")
    
    if not sample_id or "__" in sample_id:
        raise ValueError("sample_id must be non-empty and cannot contain '__'")
    
    try:
        import cv2
        import mediapipe as mp
    except ImportError as exc:
        raise RuntimeError("Video preprocessing requires opencv-python-headless and mediapipe") from exc

    source = Path(video_path).resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Video not found: {source}")
    
    destination = Path(output).resolve()
    pose_dir, face_dir = destination / "pose", destination / "face"
    geometry_dir = destination / "face_geometry"
    pose_dir.mkdir(parents=True, exist_ok=True)
    face_dir.mkdir(parents=True, exist_ok=True)
    geometry_dir.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(source))

    if not capture.isOpened():
        raise OSError(f"Video cannot be opened: {source}")
    source_fps = float(capture.get(cv2.CAP_PROP_FPS))
    if not np.isfinite(source_fps) or source_fps <= 0:
        capture.release()
        raise ValueError(f"Video reports an invalid FPS: {source_fps!r}")

    poses: list[np.ndarray] = []
    faces: list[np.ndarray] = []

    face_geometries: list[np.ndarray] = []

    pose_hits: list[bool] = []
    face_hits: list[bool] = []

    face_box_reused: list[bool] = []
    times: list[float] = []

    next_sample, frame_index = 0.0, 0
    vision = mp.tasks.vision
    base_options = mp.tasks.BaseOptions

    pose_options = vision.PoseLandmarkerOptions(
        base_options=base_options(
            model_asset_path=str(pose_model_path),
        ),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    face_options = vision.FaceLandmarkerOptions(
        base_options=base_options(
            model_asset_path=str(face_model_path),
        ),
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
    )

    pose_api = vision.PoseLandmarker.create_from_options(
        pose_options
    )
    face_api = vision.FaceLandmarker.create_from_options(
        face_options
    )

    smoothed_box: np.ndarray | None = None
    previous_face_gray: np.ndarray | None = None
    missed_face_frames = 0

    try:
        while True:
            ok, bgr = capture.read()
            if not ok:
                break
            time_sec = frame_index / source_fps
            frame_index += 1
            if time_sec + 1e-9 < next_sample:
                continue

            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            rgb = np.ascontiguousarray(rgb)

            media_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb,
            )

            timestamp_ms = int(round(time_sec * 1000))

            pose_result = pose_api.detect_for_video(
                media_image,
                timestamp_ms,
            )

            face_result = face_api.detect_for_video(
                media_image,
                timestamp_ms,
            )

            pose, has_pose = _normalized_pose(pose_result)
            
            geometry, has_face = _face_geometry(face_result)

            detected_box = _face_box(
                face_result, bgr.shape[1], bgr.shape[0], face_margin
            )

            reused = False
            if detected_box is not None:
                smoothed_box = _smooth_box(smoothed_box, detected_box, box_smoothing)
                missed_face_frames = 0
            else:
                missed_face_frames += 1
                reused = smoothed_box is not None and missed_face_frames <= max_box_reuse_frames
                if not reused:
                    smoothed_box = None
            face_input, current_gray = _crop_face_input(
                bgr, smoothed_box, previous_face_gray,
                face_size=face_size, cv2=cv2,
            )
            previous_face_gray = current_gray
            poses.append(pose); faces.append(face_input); face_geometries.append(geometry)
            pose_hits.append(has_pose); face_hits.append(has_face)
            face_box_reused.append(reused); times.append(time_sec)
            next_sample += 1 / fps
    finally:
        capture.release(); pose_api.close(); face_api.close()

    length = max(1, round(fps * window_seconds))
    stride = max(1, round(fps * stride_seconds))
    patient = patient_id or sample_id.split("_", 1)[0]
    assigned_split = split or _patient_split(patient, split_seed)

    if assigned_split not in {"train", "validation", "test", "unassigned"}:
        raise ValueError(f"Invalid split: {assigned_split!r}")
    
    pose_array, face_array = np.asarray(poses, np.float32), np.asarray(faces, np.float32)
    geometry_array = np.asarray(face_geometries, np.float32)
    rows: list[dict[str, Any]] = []

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
            "face_detection_fraction": f"{np.mean(face_hits[start:stop]):.6f}",
            "face_box_reused_fraction": f"{np.mean(face_box_reused[start:stop]):.6f}",
        })
    manifest = destination / "video_manifest.csv"
    _write_rows(manifest, rows, append=append_manifest)
    summary = PreprocessSummary(str(source), source_fps, fps, len(poses), len(rows), str(manifest))
    print(json.dumps(asdict(summary), indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video"); parser.add_argument("--output", required=True)
    parser.add_argument("--sample-id"); parser.add_argument("--patient-id")
    parser.add_argument("--label", default="0"); parser.add_argument("--dataset", default="seizure_video")
    parser.add_argument("--split", choices=("train", "validation", "test", "unassigned"))
    parser.add_argument("--fps", type=float, default=5.0)
    parser.add_argument("--window-seconds", type=float, default=8.0)
    parser.add_argument("--stride-seconds", type=float)

    parser.add_argument("--face-size", type=int, default=224)
    parser.add_argument("--face-margin", type=float, default=.25)

    parser.add_argument("--box-smoothing", type=float, default=.7)
    parser.add_argument("--max-box-reuse-frames", type=int, default=5)

    parser.add_argument(
        "--pose-model",
        default=(
            "checkpoints/pretrained/mediapipe/"
            "pose_landmarker_lite.task"
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

    args = parser.parse_args()
    extract_video(args.video, args.output,
            args.sample_id or Path(args.video).stem,
            args.fps, args.window_seconds,
            pose_model_path=args.pose_model,
            face_model_path=args.face_model,
            stride_seconds=args.stride_seconds,
            patient_id=args.patient_id, label=args.label, dataset=args.dataset, split=args.split,
            append_manifest=args.append_manifest,
            face_size=args.face_size, face_margin=args.face_margin,
            box_smoothing=args.box_smoothing,
            max_box_reuse_frames=args.max_box_reuse_frames)
    


if __name__ == "__main__":
    main()
