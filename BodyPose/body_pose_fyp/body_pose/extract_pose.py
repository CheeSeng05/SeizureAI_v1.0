from __future__ import annotations

import argparse
import logging
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from tqdm import tqdm

from .constants import POSE_LANDMARK_NAMES
from .utils import ensure_dir, load_config, seizeit2_video_message, video_id_from_path

LOGGER = logging.getLogger(__name__)


def _empty_landmark_row() -> list[float]:
    values: list[float] = []
    for _ in POSE_LANDMARK_NAMES:
        values.extend([np.nan, np.nan, np.nan, 0.0])
    return values


def _landmark_columns() -> list[str]:
    columns: list[str] = []
    for name in POSE_LANDMARK_NAMES:
        columns.extend([f"{name}_x", f"{name}_y", f"{name}_z", f"{name}_visibility"])
    return columns


def extract_video(
    video_path: Path,
    output_csv: Path,
    pose_cfg: dict,
    overlay_path: Path | None = None,
) -> None:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if fps <= 0:
        fps = 25.0
        LOGGER.warning("Invalid FPS for %s; using %.1f", video_path.name, fps)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_step = max(1, int(pose_cfg.get("frame_step", 1)))

    writer = None
    if overlay_path is not None:
        ensure_dir(overlay_path.parent)
        writer = cv2.VideoWriter(
            str(overlay_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps / frame_step,
            (width, height),
        )

    mp_holistic = mp.solutions.holistic
    mp_drawing = mp.solutions.drawing_utils
    rows: list[list[float | int]] = []

    with mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=int(pose_cfg.get("model_complexity", 1)),
        smooth_landmarks=bool(pose_cfg.get("smooth_landmarks", True)),
        min_detection_confidence=float(pose_cfg.get("min_detection_confidence", 0.4)),
        min_tracking_confidence=float(pose_cfg.get("min_tracking_confidence", 0.4)),
    ) as holistic:
        progress = tqdm(total=total_frames, desc=video_path.name, unit="frame")
        frame_index = 0
        processed_index = 0

        while True:
            ok, frame = cap.read()
            if not ok:
                break
            progress.update(1)

            if frame_index % frame_step != 0:
                frame_index += 1
                continue

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            result = holistic.process(rgb)

            row: list[float | int] = [
                processed_index,
                frame_index,
                frame_index / fps,
                fps,
                width,
                height,
            ]

            if result.pose_landmarks is None:
                row.extend(_empty_landmark_row())
                detected = 0
                mean_visibility = 0.0
            else:
                detected = 1
                visibilities = []
                for landmark in result.pose_landmarks.landmark:
                    row.extend([
                        float(landmark.x),
                        float(landmark.y),
                        float(landmark.z),
                        float(landmark.visibility),
                    ])
                    visibilities.append(float(landmark.visibility))
                mean_visibility = float(np.mean(visibilities))

                if writer is not None:
                    mp_drawing.draw_landmarks(
                        frame,
                        result.pose_landmarks,
                        mp_holistic.POSE_CONNECTIONS,
                    )

            row.extend([detected, mean_visibility])
            rows.append(row)

            if writer is not None:
                writer.write(frame)

            processed_index += 1
            frame_index += 1

        progress.close()

    cap.release()
    if writer is not None:
        writer.release()

    metadata_columns = [
        "processed_frame", "source_frame", "timestamp_sec", "source_fps",
        "frame_width", "frame_height",
    ]
    columns = metadata_columns + _landmark_columns() + ["pose_detected", "mean_visibility"]
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=columns).to_csv(output_csv, index=False)
    LOGGER.info("Saved %d frames to %s", len(rows), output_csv)


def discover_videos(path: Path) -> list[Path]:
    supported = {".mp4", ".avi", ".mov", ".mkv", ".m4v"}
    if path.is_file():
        return [path]
    return sorted(p for p in path.rglob("*") if p.suffix.lower() in supported)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract 33 MediaPipe Holistic pose landmarks.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--input", default=None, help="Video file or directory. Defaults to config videos path.")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--overlay", action="store_true", help="Save videos with skeleton overlays.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    cfg = load_config(args.config)
    input_path = Path(args.input or cfg["paths"]["videos"])
    output_dir = ensure_dir(args.output_dir or cfg["paths"]["keypoints"])
    videos = discover_videos(input_path)
    if not videos:
        adapter_type = str(cfg.get("annotation_adapter", {}).get("type", "")).lower()
        if adapter_type in {"seizeit2", "ds005873"}:
            raise RuntimeError(seizeit2_video_message())
        raise FileNotFoundError(f"No supported videos found under {input_path}")

    save_overlay = args.overlay or bool(cfg["pose"].get("save_overlay_video", False))
    for video in videos:
        video_id = video_id_from_path(video)
        overlay_path = output_dir / f"{video_id}_overlay.mp4" if save_overlay else None
        extract_video(
            video_path=video,
            output_csv=output_dir / f"{video_id}.csv",
            pose_cfg=cfg["pose"],
            overlay_path=overlay_path,
        )


if __name__ == "__main__":
    main()
