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
    "face_region_fraction",
    "face_landmark_fraction",
    "pose_head_fallback_fraction",
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
) -> int | None:
    """Select the most visible pose whose head lies in the configured ROI."""
    if not result.pose_landmarks:
        return None
    x1, y1, x2, y2 = patient_roi
    candidates: list[tuple[float, int]] = []
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
        candidates.append((visibility * len(visible), index))
    return max(candidates)[1] if candidates else None


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
) -> bool:
    """Display the selected region and exact RGB tensor crop; return False on q."""
    source = frame_bgr.copy()
    color = (0, 200, 0) if face_landmarks_found else (0, 165, 255)
    status = "FACE DETECTED" if face_landmarks_found else "NO FACE"
    if box_reused:
        status = "REUSED PREVIOUS BOX"
    if box is not None:
        x1, y1, x2, y2 = np.rint(box).astype(int)
        cv2.rectangle(source, (x1, y1), (x2, y2), color, 3)

    cv2.putText(
        source,
        f"{time_sec:.2f}s | {status} | q: close preview",
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
    box_smoothing: float = 0.7,
    max_box_reuse_frames: int = 5,
    patient_roi: tuple[float, float, float, float] = (0.0, 0.0, 1.0, 1.0),
    max_poses: int = 4,
    head_margin: float = 0.75,
    preview: bool = False,
    preview_delay_ms: int = 1,
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
    if face_margin < 0 or head_margin < 0 or not 0 <= box_smoothing < 1 or max_box_reuse_frames < 0:
        raise ValueError("Invalid face margin, box smoothing, or box reuse setting")
    if max_poses <= 0:
        raise ValueError("max_poses must be positive")
    if not (
        len(patient_roi) == 4
        and 0 <= patient_roi[0] < patient_roi[2] <= 1
        and 0 <= patient_roi[1] < patient_roi[3] <= 1
    ):
        raise ValueError("patient_roi must be normalized x1,y1,x2,y2 coordinates")
    
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
    face_region_hits: list[bool] = []
    pose_head_fallbacks: list[bool] = []

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
        num_poses=max_poses,
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
    preview_active = preview

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

            pose_index = _select_patient_pose(pose_result, patient_roi)
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
            detected_box = None
            has_face = False
            geometry = np.zeros(len(FACE_LANDMARK_PAIRS) * 2, np.float32)
            if head_box is not None:
                head_rgb, actual_head_box = _crop_box(rgb, head_box)
                if head_rgb.size:
                    head_image = mp.Image(
                        image_format=mp.ImageFormat.SRGB,
                        data=np.ascontiguousarray(head_rgb),
                    )
                    face_result = face_api.detect_for_video(head_image, timestamp_ms)
                    geometry, has_face = _face_geometry(face_result)
                    local_box = _face_box(
                        face_result,
                        head_rgb.shape[1],
                        head_rgb.shape[0],
                        face_margin,
                    )
                    if local_box is not None:
                        detected_box = _translate_box(local_box, actual_head_box)
                if detected_box is None:
                    detected_box = head_box

            used_pose_fallback = detected_box is not None and not has_face

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
            if preview_active:
                preview_active = _show_preprocessing_preview(
                    bgr,
                    face_input,
                    smoothed_box,
                    time_sec=time_sec,
                    face_landmarks_found=has_face,
                    box_reused=reused,
                    delay_ms=preview_delay_ms,
                    cv2=cv2,
                )
            previous_face_gray = current_gray
            poses.append(pose); faces.append(face_input); face_geometries.append(geometry)
            pose_hits.append(has_pose); face_hits.append(has_face)
            face_region_hits.append(smoothed_box is not None)
            pose_head_fallbacks.append(used_pose_fallback)
            face_box_reused.append(reused); times.append(time_sec)
            next_sample += 1 / fps
    finally:
        capture.release(); pose_api.close(); face_api.close()
        if preview:
            cv2.destroyAllWindows()

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
            "face_region_fraction": f"{np.mean(face_region_hits[start:stop]):.6f}",
            "face_landmark_fraction": f"{np.mean(face_hits[start:stop]):.6f}",
            "pose_head_fallback_fraction": f"{np.mean(pose_head_fallbacks[start:stop]):.6f}",
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
    parser.add_argument("--max-poses", type=int, default=4)
    parser.add_argument("--head-margin", type=float, default=.75)
    parser.add_argument(
        "--patient-roi",
        default="0,0,1,1",
        help="Normalized patient region x1,y1,x2,y2 used to reject caregivers.",
    )

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
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Show the source selection and exact model crop while preprocessing.",
    )
    parser.add_argument(
        "--preview-delay-ms",
        type=int,
        default=30,
        help="Delay between sampled preview frames; press q or Escape to close it.",
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
            append_manifest=args.append_manifest,
            face_size=args.face_size, face_margin=args.face_margin,
            box_smoothing=args.box_smoothing,
            max_box_reuse_frames=args.max_box_reuse_frames,
            patient_roi=patient_roi,
            max_poses=args.max_poses,
            head_margin=args.head_margin,
            preview=args.preview,
            preview_delay_ms=args.preview_delay_ms)
    


if __name__ == "__main__":
    main()
