"""Convert raw videos to fixed-length, privacy-reduced pose/face windows."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np


FACE_LANDMARK_PAIRS = ((33, 133), (362, 263), (159, 145), (386, 374),
                       (61, 291), (13, 14), (78, 308), (1, 152), (10, 152), (234, 454))
MANIFEST_FIELDS = (
    "window_id", "dataset", "patient_id", "recording_id", "source_video",
    "start_sec", "end_sec", "label", "split", "sampled_fps", "frame_count",
    "pose_path", "face_path", "pose_detection_fraction", "face_detection_fraction",
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
    if result.pose_landmarks is None:
        return np.zeros((33, 4), np.float32), False
    pose = np.asarray([[p.x, p.y, p.z, p.visibility]
                       for p in result.pose_landmarks.landmark], np.float32)
    if pose.shape != (33, 4):
        return np.zeros((33, 4), np.float32), False
    center = (pose[23, :3] + pose[24, :3]) / 2
    scale = float(np.linalg.norm(pose[11, :3] - pose[12, :3]))
    if scale > 1e-6:
        pose[:, :3] = (pose[:, :3] - center) / scale
    return pose, True


def _face_geometry(result: Any) -> tuple[np.ndarray, bool]:
    empty = np.zeros(len(FACE_LANDMARK_PAIRS) * 2, np.float32)
    if not result.multi_face_landmarks:
        return empty, False
    points = np.asarray([[p.x, p.y] for p in result.multi_face_landmarks[0].landmark], np.float32)
    if len(points) <= max(max(pair) for pair in FACE_LANDMARK_PAIRS):
        return empty, False
    scale = float(np.linalg.norm(points[33] - points[263]))
    if scale <= 1e-6:
        return empty, False
    geometry = np.asarray([(points[a] - points[b]) / scale
                           for a, b in FACE_LANDMARK_PAIRS], np.float32).reshape(-1)
    return geometry, True


def extract_video(
    video_path: str | Path,
    output: str | Path,
    sample_id: str,
    fps: float = 5.0,
    window_seconds: float = 8.0,
    *,
    stride_seconds: float | None = None,
    patient_id: str | None = None,
    label: int | str = 0,
    dataset: str = "seizure_video",
    split: str | None = None,
    split_seed: int = 42,
    append_manifest: bool = False,
) -> PreprocessSummary:
    """Extract pose and face geometry; discard incomplete trailing windows."""
    if fps <= 0 or window_seconds <= 0:
        raise ValueError("fps and window_seconds must be positive")
    stride_seconds = window_seconds / 2 if stride_seconds is None else stride_seconds
    if stride_seconds <= 0:
        raise ValueError("stride_seconds must be positive")
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
    pose_dir.mkdir(parents=True, exist_ok=True)
    face_dir.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise OSError(f"Video cannot be opened: {source}")
    source_fps = float(capture.get(cv2.CAP_PROP_FPS))
    if not np.isfinite(source_fps) or source_fps <= 0:
        capture.release()
        raise ValueError(f"Video reports an invalid FPS: {source_fps!r}")

    poses: list[np.ndarray] = []
    faces: list[np.ndarray] = []
    pose_hits: list[bool] = []
    face_hits: list[bool] = []
    times: list[float] = []
    next_sample, frame_index = 0.0, 0
    pose_api = mp.solutions.pose.Pose(static_image_mode=False, model_complexity=1)
    face_api = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False, max_num_faces=1, refine_landmarks=True)
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
            pose, has_pose = _normalized_pose(pose_api.process(rgb))
            face, has_face = _face_geometry(face_api.process(rgb))
            poses.append(pose); faces.append(face)
            pose_hits.append(has_pose); face_hits.append(has_face); times.append(time_sec)
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
    rows: list[dict[str, Any]] = []
    for start in _window_starts(len(poses), length, stride):
        stop, start_sec = start + length, times[start]
        end_sec = start_sec + window_seconds
        window_id = f"{sample_id}__{round(start_sec * 1000):010d}__{round(end_sec * 1000):010d}"
        pose_path, face_path = pose_dir / f"{window_id}.npy", face_dir / f"{window_id}.npy"
        np.save(pose_path, pose_array[start:stop], allow_pickle=False)
        np.save(face_path, face_array[start:stop], allow_pickle=False)
        rows.append({
            "window_id": window_id, "dataset": dataset, "patient_id": patient,
            "recording_id": sample_id, "source_video": str(source),
            "start_sec": f"{start_sec:.6f}", "end_sec": f"{end_sec:.6f}",
            "label": label, "split": assigned_split, "sampled_fps": fps,
            "frame_count": length, "pose_path": str(pose_path.relative_to(destination)),
            "face_path": str(face_path.relative_to(destination)),
            "pose_detection_fraction": f"{np.mean(pose_hits[start:stop]):.6f}",
            "face_detection_fraction": f"{np.mean(face_hits[start:stop]):.6f}",
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
    args = parser.parse_args()
    extract_video(args.video, args.output, args.sample_id or Path(args.video).stem,
                  args.fps, args.window_seconds, stride_seconds=args.stride_seconds,
                  patient_id=args.patient_id, label=args.label, dataset=args.dataset, split=args.split)


if __name__ == "__main__":
    main()
