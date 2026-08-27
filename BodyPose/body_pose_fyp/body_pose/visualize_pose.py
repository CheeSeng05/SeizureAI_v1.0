from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from .constants import POSE_CONNECTIONS, POSE_LANDMARK_NAMES


def main() -> None:
    parser = argparse.ArgumentParser(description="Render saved keypoints over the original video.")
    parser.add_argument("--video", required=True)
    parser.add_argument("--keypoints", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--visibility-threshold", type=float, default=0.30)
    args = parser.parse_args()

    df = pd.read_csv(args.keypoints)
    by_source_frame = {int(row.source_frame): row for row in df.itertuples(index=False)}

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )

    frame_index = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        row = by_source_frame.get(frame_index)
        if row is not None:
            points: list[tuple[int, int] | None] = []
            for name in POSE_LANDMARK_NAMES:
                visibility = float(getattr(row, f"{name}_visibility"))
                x = float(getattr(row, f"{name}_x"))
                y = float(getattr(row, f"{name}_y"))
                if visibility >= args.visibility_threshold and np.isfinite(x) and np.isfinite(y):
                    points.append((int(x * width), int(y * height)))
                else:
                    points.append(None)

            for a, b in POSE_CONNECTIONS:
                if points[a] is not None and points[b] is not None:
                    cv2.line(frame, points[a], points[b], (255, 255, 255), 2)
            for point in points:
                if point is not None:
                    cv2.circle(frame, point, 3, (255, 255, 255), -1)

            cv2.putText(
                frame,
                f"Mean visibility: {float(row.mean_visibility):.2f}",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
            )
        writer.write(frame)
        frame_index += 1

    cap.release()
    writer.release()


if __name__ == "__main__":
    main()
