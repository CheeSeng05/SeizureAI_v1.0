"""Extract normalized patient pose windows from sampled video frames."""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from contextlib import ExitStack
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

import numpy as np

if not __package__:
    # Support the standalone `python video_preprocess.py` entry point.
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "FYPModels.preprocess"

from ..seizure_multimodal.progress import report_progress
from ..seizure_multimodal.reproducibility import build_preprocessing_record, save_run_record

from .io import iter_sampled_frames, open_video
from . import (
    SPEINetConfig, SPEINetEnhancer, add_video_enhancement_arguments,
    iter_enhanced_frames, video_enhancement_kwargs,
)
from .prepare_dataset import patient_split, window_starts, write_rows
from .preprocess import PreprocessSummary


MANIFEST_FIELDS = (
    "window_id", "dataset", "patient_id", "recording_id", "source_video",
    "start_sec", "end_sec", "label", "split", "sampled_fps", "frame_count",
    "pose_path", "pose_detection_fraction",
)

POSE_CONFIDENCE_OPTIONS = (
    "min_pose_detection_confidence",
    "min_pose_presence_confidence",
    "min_pose_tracking_confidence",
)


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


def _show_preprocessing_preview(
    frame_bgr: np.ndarray,
    result: Any,
    pose_index: int | None,
    *,
    time_sec: float,
    delay_ms: int,
    cv2: Any,
) -> bool:
    """Display the selected patient's pose; return False on q or Escape."""
    source = frame_bgr.copy()
    color = (0, 200, 0) if pose_index is not None else (0, 165, 255)
    status = "POSE DETECTED" if pose_index is not None else "NO POSE"
    height, width = source.shape[:2]
    if pose_index is not None:
        for point in result.pose_landmarks[pose_index]:
            if (
                point.x is not None and point.y is not None
                and (point.visibility is None or point.visibility >= 0.2)
                and (point.presence is None or point.presence >= 0.2)
                and 0 <= point.x < 1 and 0 <= point.y < 1
            ):
                cv2.circle(source, (round(point.x * width), round(point.y * height)), 4, color, -1)

    cv2.putText(
        source, f"{time_sec:.2f}s | {status} | q: stop",
        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA,
    )
    if width > 960:
        source = cv2.resize(source, (960, round(height * 960 / width)), interpolation=cv2.INTER_AREA)
    cv2.imshow("Pose preprocessing - source", source)
    key = cv2.waitKey(max(1, delay_ms)) & 0xFF
    if key in (ord("q"), 27):
        cv2.destroyWindow("Pose preprocessing - source")
        return False
    return True


def _write_pose_diagnostics(
    directory: Path,
    sample_id: str,
    rows: list[dict[str, Any]],
    *,
    settings: dict[str, Any],
    processing_seconds: float,
) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    frames_path = directory / f"{sample_id}.frames.csv"
    with frames_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("time_sec", "pose_detected"))
        writer.writeheader()
        writer.writerows(rows)
    path = directory / f"{sample_id}.json"
    path.write_text(json.dumps({
        "settings": settings,
        "metrics": {
            "sampled_frames": len(rows),
            "processing_seconds": processing_seconds,
            "processing_fps": len(rows) / processing_seconds if processing_seconds > 0 else None,
            "pose_detection_fraction": (
                float(np.mean([row["pose_detected"] for row in rows])) if rows else None
            ),
        },
        "frames_path": str(frames_path.resolve()),
        "interpretation": "Coverage only; patient identity and landmark accuracy require annotation.",
    }, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return path


def extract_video(
    video_path: str | Path,
    output: str | Path,
    sample_id: str,
    fps: float = 5.0,
    window_seconds: float = 8.0,
    *,
    pose_model_path: str | Path,
    stride_seconds: float | None = None,
    patient_id: str | None = None,
    label: int | str = 0,
    dataset: str = "seizure_video",
    split: str | None = None,
    split_seed: int = 42,
    append_manifest: bool = False,
    min_pose_detection_confidence: float = 0.5,
    min_pose_presence_confidence: float = 0.5,
    min_pose_tracking_confidence: float = 0.5,
    patient_roi: tuple[float, float, float, float] = (0.0, 0.0, 1.0, 1.0),
    max_poses: int = 4,
    max_head_jump: float = 0.18,
    preview: bool = False,
    preview_delay_ms: int = 1,
    diagnostics_only: bool = False,
    max_duration_seconds: float | None = None,
    progress_config: Mapping[str, Any] | None = None,
    video_enhancement: SPEINetConfig | None = None,
) -> PreprocessSummary:
    """Extract normalized pose windows with shape [frames, 33, 4].

    Discard partial windows and record input hashes, settings and source metadata.
    Optional full-frame video enhancement runs before pose detection. Patient
    selection uses the ROI and temporal continuity of pose head landmarks.
    """
    pose_model_path = Path(pose_model_path).resolve()

    if not pose_model_path.is_file():
        raise FileNotFoundError(
            f"Pose task model not found: {pose_model_path}"
        )

    if not math.isfinite(fps) or not math.isfinite(window_seconds) or fps <= 0 or window_seconds <= 0:
        raise ValueError("fps and window_seconds must be positive")

    stride_seconds = window_seconds / 2 if stride_seconds is None else stride_seconds
    if not math.isfinite(stride_seconds) or stride_seconds <= 0:
        raise ValueError("stride_seconds must be positive")

    tuning = dict(
        min_pose_detection_confidence=min_pose_detection_confidence,
        min_pose_presence_confidence=min_pose_presence_confidence,
        min_pose_tracking_confidence=min_pose_tracking_confidence,
    )
    for name, value in tuning.items():
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"{name} must be finite and in [0, 1]")
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
    assigned_split = split or patient_split(patient, split_seed)
    if assigned_split not in {"train", "validation", "test", "unassigned"}:
        raise ValueError(f"Invalid split: {assigned_split!r}")

    try:
        import cv2
        import mediapipe as mp
    except ImportError as exc:
        raise RuntimeError("Video preprocessing requires opencv-python-headless and mediapipe") from exc
    source = Path(video_path).resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Video not found: {source}")

    video_enhancer = SPEINetEnhancer(video_enhancement) if video_enhancement is not None else None

    destination = Path(output).resolve()
    pose_dir = destination / "pose"
    if not diagnostics_only:
        pose_dir.mkdir(parents=True, exist_ok=True)

    poses: list[np.ndarray] = []
    pose_hits: list[bool] = []
    times: list[float] = []
    diagnostic_rows: list[dict[str, Any]] = []

    base_options = mp.tasks.BaseOptions
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

    with ExitStack() as resources:
        capture, metadata = resources.enter_context(open_video(source, cv2=cv2))
        source_fps = metadata.fps
        settings = {
            **tuning,
            "source_video": str(source), "pose_model_path": str(pose_model_path),
            "fps": fps, "source_fps": source_fps,
            "window_seconds": window_seconds, "stride_seconds": stride_seconds,
            "patient_roi": patient_roi,
            "max_poses": max_poses, "max_head_jump": max_head_jump,
            "running_mode": "VIDEO", "diagnostics_only": diagnostics_only,
            "max_duration_seconds": max_duration_seconds,
            "sample_id": sample_id, "dataset": dataset, "patient_id": patient,
            "patient_id_source": "argument" if patient_id else "sample_id_prefix",
            "label": label, "split": assigned_split, "split_seed": split_seed,
            "split_policy": "explicit" if split else "patient_sha256_70_15_15",
            "append_manifest": append_manifest, "preview": preview,
            "preview_delay_ms": preview_delay_ms,
            "video_enhancement": video_enhancer.record if video_enhancer is not None else None,
        }
        run_record = build_preprocessing_record(
            pipeline="pose", sample_id=sample_id,
            inputs={
                "video": source, "pose_model": pose_model_path,
                **({"speinet_checkpoint": video_enhancer.record["checkpoint"]}
                   if video_enhancer is not None else {}),
            },
            settings=settings, source_metadata=metadata.as_dict(),
            software_versions={
                "python": sys.version.split()[0], "numpy": np.__version__,
                "opencv": getattr(cv2, "__version__", "unknown"),
                "mediapipe": getattr(mp, "__version__", "unknown"),
            },
        )
        pose_api = PoseLandmarker.create_from_options(pose_options)
        resources.callback(pose_api.close)
        if preview:
            resources.callback(cv2.destroyAllWindows)

        duration = metadata.duration_sec
        if max_duration_seconds is not None:
            duration = min(duration, max_duration_seconds) if duration is not None else max_duration_seconds
        estimated_samples = math.ceil(duration * min(fps, source_fps)) if duration is not None else 0
        processing_start = time.perf_counter()
        report_progress(
            progress_config or {}, stage="pose_preprocessing", label="Extracting pose windows",
            completed=0, total=estimated_samples, elapsed_sec=0.0,
        )
        last_progress_time = processing_start
        tracked_head_center: tuple[float, float] | None = None

        frame_options = dict(source_fps=source_fps, fps=fps, max_duration_seconds=max_duration_seconds)
        # Full-frame video restoration precedes pose detection.
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
            if current_head_center is not None:
                tracked_head_center = current_head_center
            pose, has_pose = _normalized_pose(
                pose_result, pose_index if pose_index is not None else -1
            )
            stop_requested = False
            if preview:
                stop_requested = not _show_preprocessing_preview(
                    bgr,
                    pose_result,
                    pose_index,
                    time_sec=time_sec,
                    delay_ms=preview_delay_ms,
                    cv2=cv2,
                )
            diagnostic_rows.append({
                "time_sec": time_sec,
                "pose_detected": int(has_pose),
            })
            if not diagnostics_only:
                poses.append(pose)
            pose_hits.append(has_pose)
            times.append(time_sec)
            now = time.perf_counter()
            if now - last_progress_time >= 1.0:
                report_progress(
                    progress_config or {}, stage="pose_preprocessing", label="Extracting pose windows",
                    completed=len(times), total=estimated_samples,
                    elapsed_sec=now - processing_start,
                )
                last_progress_time = now
            if stop_requested:
                break

    processing_seconds = time.perf_counter() - processing_start
    diagnostics_path = _write_pose_diagnostics(
        destination / "diagnostics", sample_id, diagnostic_rows,
        settings=settings, processing_seconds=processing_seconds,
    )
    rows: list[dict[str, Any]] = []
    manifest = None
    if not diagnostics_only:
        length = max(1, round(fps * window_seconds))
        stride = max(1, round(fps * stride_seconds))
        pose_array = np.asarray(poses, np.float32)

        for start in window_starts(len(poses), length, stride):
            stop, start_sec = start + length, times[start]
            end_sec = start_sec + window_seconds
            window_id = f"{sample_id}__{round(start_sec * 1000):010d}__{round(end_sec * 1000):010d}"
            pose_path = pose_dir / f"{window_id}.npy"
            np.save(pose_path, pose_array[start:stop], allow_pickle=False)
            rows.append({
                "window_id": window_id, "dataset": dataset, "patient_id": patient,
                "recording_id": sample_id, "source_video": str(source),
                "start_sec": f"{start_sec:.6f}", "end_sec": f"{end_sec:.6f}",
                "label": label, "split": assigned_split, "sampled_fps": fps,
                "frame_count": length, "pose_path": str(pose_path.relative_to(destination)),
                "pose_detection_fraction": f"{np.mean(pose_hits[start:stop]):.6f}",
            })
        manifest = destination / "video_manifest.csv"
        write_rows(manifest, rows, append=append_manifest, fieldnames=MANIFEST_FIELDS)
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
        progress_config or {}, stage="pose_preprocessing", label="Pose preprocessing complete",
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

    for name in POSE_CONFIDENCE_OPTIONS:
        parser.add_argument("--" + name.replace("_", "-"), type=float, default=0.5)
    add_video_enhancement_arguments(parser)
    parser.add_argument("--diagnostics-only", action="store_true",
                        help="Save frame diagnostics and metrics without training window arrays.")
    parser.add_argument("--max-duration-seconds", type=float,
                        help="Process only this many seconds from the start of the video.")
    parser.add_argument("--max-poses", type=int, default=4)
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

    parser.add_argument("--append-manifest", action="store_true",)
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Show the selected patient pose while preprocessing.",
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
            stride_seconds=args.stride_seconds,
            patient_id=args.patient_id, label=args.label, dataset=args.dataset, split=args.split,
            split_seed=args.split_seed,
            append_manifest=args.append_manifest,
            **video_enhancement_kwargs(args),
            **{name: getattr(args, name) for name in POSE_CONFIDENCE_OPTIONS},
            patient_roi=patient_roi,
            max_poses=args.max_poses,
            max_head_jump=args.max_head_jump,
            preview=args.preview,
            preview_delay_ms=args.preview_delay_ms,
            diagnostics_only=args.diagnostics_only,
            max_duration_seconds=args.max_duration_seconds,
            progress_config={"testing": {"progress_file": args.progress_file}})


if __name__ == "__main__":
    main()
