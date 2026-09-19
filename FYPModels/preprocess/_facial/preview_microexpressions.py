"""Diagnostic video preview for head-pose filtering and separate eye/mouth ratios.

Uses independent YuNet face crops and checks refined Face Mesh landmarks against
the detector's anchors. This is a measurement probe, not an emotion classifier.
It writes every source timestamp, including rejected frames, and does not change
the training-window preprocessing pipeline.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import Counter
from contextlib import ExitStack
from pathlib import Path

import cv2
import numpy as np

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    __package__ = "FYPModels.preprocess._facial"

from ...seizure_multimodal.video.io import open_video, iter_sampled_frames
from .face_localization import DEFAULT_YUNET_MODEL, YuNetLocalizer, face_crop_box, landmark_agreement_error

LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
PNP_INDICES = [1, 152, 33, 263, 61, 291]
# Generic face in OpenCV camera axes: x right, y down, z away from camera.
# Neutral, front-facing head has zero rotation. These are not patient measurements.
FACE_3D_POINTS = np.array([
    [0, 0, 0], [0, 330, 65], [-225, -170, 135],
    [225, -170, 135], [-150, 150, 125], [150, 150, 125],
], dtype=np.float64)
FIELDS = ("time_sec", "face_detected", "face_score", "face_x1", "face_y1", "face_x2", "face_y2",
          "landmarks_found", "landmark_error_fraction", "anchor_check_passed", "accepted", "reason", "pitch_deg", "yaw_deg",
          "roll_deg", "reprojection_rmse_px", "left_ear", "right_ear", "mar",
          "eye_state", "mouth_active")


def camera_matrix(width: int, height: int) -> np.ndarray:
    return np.array([[width, 0, width / 2], [0, width, height / 2], [0, 0, 1]], np.float64)


def estimate_pose(points: np.ndarray, width: int, height: int) -> dict | None:
    image_points = np.asarray(points[PNP_INDICES], dtype=np.float64)
    if not np.isfinite(image_points).all():
        return None
    camera = camera_matrix(width, height)
    try:
        success, rotation, translation = cv2.solvePnP(
            FACE_3D_POINTS, image_points, camera, np.zeros((4, 1)), flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not success or not np.isfinite(rotation).all() or not np.isfinite(translation).all():
            return None
        matrix, _ = cv2.Rodrigues(rotation)
        depths = (matrix @ FACE_3D_POINTS.T + translation.reshape(3, 1))[2]
        if np.any(depths <= 0):
            return None
        # RQDecomp3x3 already returns degrees; do not multiply by 360.
        pitch, yaw, roll = cv2.RQDecomp3x3(matrix)[0]
        projected, _ = cv2.projectPoints(FACE_3D_POINTS, rotation, translation, camera, np.zeros((4, 1)))
        rmse = float(np.sqrt(np.mean(np.sum((projected.reshape(-1, 2) - image_points) ** 2, axis=1))))
    except cv2.error:
        return None
    if not np.isfinite([pitch, yaw, roll, rmse]).all():
        return None
    return dict(pitch_deg=float(pitch), yaw_deg=float(yaw), roll_deg=float(roll), reprojection_rmse_px=rmse)


def eye_ratio(points: np.ndarray, indices: list[int]) -> float | None:
    eye = points[indices]
    width = float(np.linalg.norm(eye[0] - eye[3]))
    if not np.isfinite(eye).all() or width < 1e-6:
        return None
    return float((np.linalg.norm(eye[1] - eye[5]) + np.linalg.norm(eye[2] - eye[4])) / (2 * width))


def pose_filter(pose: dict, max_yaw: float, max_pitch: float) -> str:
    reasons = []
    if abs(pose["yaw_deg"]) > max_yaw:
        reasons.append("yaw_limit")
    if abs(pose["pitch_deg"]) > max_pitch:
        reasons.append("pitch_limit")
    return "|".join(reasons)


def expression_ratios(points: np.ndarray, blink_threshold: float, mouth_threshold: float) -> dict | None:
    left, right = eye_ratio(points, LEFT_EYE), eye_ratio(points, RIGHT_EYE)
    mouth_width = float(np.linalg.norm(points[61] - points[291]))
    if left is None or right is None or not np.isfinite(points[[0, 17, 61, 291]]).all() or mouth_width < 1e-6:
        return None
    mar = float(np.linalg.norm(points[0] - points[17]) / mouth_width)
    closed = (left < blink_threshold, right < blink_threshold)
    state = {(False, False): "open", (True, False): "left_closed", (False, True): "right_closed",
             (True, True): "both_closed_candidate"}[closed]
    return dict(left_ear=left, right_ear=right, mar=mar, eye_state=state, mouth_active=int(mar > mouth_threshold))


def measure_face(points, face, width, height, *, max_landmark_error=.2,
                 max_yaw=45, max_pitch=30, blink_threshold=.2, mouth_threshold=.3) -> dict:
    """Require independent detector agreement before pose or expression output."""
    row = dict(accepted=0, anchor_check_passed=0, reason="landmark_mismatch")
    error = landmark_agreement_error(points, face)
    row["landmark_error_fraction"] = error
    if error is None or error > max_landmark_error:
        return row
    row["anchor_check_passed"] = 1
    pose = estimate_pose(points, width, height)
    row["reason"] = "pose_failed"
    if pose is not None:
        row.update(pose)
        row["reason"] = pose_filter(pose, max_yaw, max_pitch)
        if not row["reason"]:
            measures = expression_ratios(points, blink_threshold, mouth_threshold)
            row["reason"] = "invalid_geometry" if measures is None else "checks_passed"
            if measures is not None:
                row.update(measures, accepted=1)
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("--output", type=Path, required=True, help="New directory for CSV, summary, and annotated video.")
    parser.add_argument("--patient-roi", default="0,0,1,1", help="Normalized x1,y1,x2,y2 search region for patient faces.")
    parser.add_argument("--face-detector-model", type=Path, default=DEFAULT_YUNET_MODEL)
    parser.add_argument("--face-confidence", type=float, default=.8, help="YuNet detection score threshold.")
    parser.add_argument("--max-landmark-error", type=float, default=.2,
                        help="Maximum mesh/detector anchor disagreement as a fraction of face size.")
    parser.add_argument("--max-yaw", type=float, default=45)
    parser.add_argument("--max-pitch", type=float, default=30)
    parser.add_argument("--blink-threshold", type=float, default=.20)
    parser.add_argument("--mouth-threshold", type=float, default=.30)
    parser.add_argument("--min-confidence", type=float, default=.5)
    parser.add_argument("--max-duration-seconds", type=float)
    parser.add_argument("--preview", action="store_true", help="Show the annotated video; q or Escape stops the run.")
    args = parser.parse_args()
    try:
        roi = tuple(float(x) for x in args.patient_roi.split(","))
        if not (len(roi) == 4 and 0 <= roi[0] < roi[2] <= 1 and 0 <= roi[1] < roi[3] <= 1):
            raise ValueError
    except ValueError:
        parser.error("--patient-roi must be normalized x1,y1,x2,y2")
    for name in ("max_yaw", "max_pitch"):
        if not math.isfinite(getattr(args, name)) or not 0 <= getattr(args, name) <= 90:
            parser.error(f"--{name.replace('_', '-')} must be between 0 and 90 degrees")
    for name in ("blink_threshold", "mouth_threshold", "min_confidence", "face_confidence", "max_landmark_error"):
        if not math.isfinite(getattr(args, name)) or not 0 <= getattr(args, name) <= 1:
            parser.error(f"--{name.replace('_', '-')} must be in [0, 1]")
    if args.max_duration_seconds is not None and (
        not math.isfinite(args.max_duration_seconds) or args.max_duration_seconds <= 0
    ):
        parser.error("--max-duration-seconds must be positive")

    import mediapipe as mp

    localizer = YuNetLocalizer(args.face_detector_model, args.face_confidence)
    counts = Counter()
    with ExitStack() as stack:
        capture, metadata = stack.enter_context(open_video(args.video, cv2=cv2))
        mesh = stack.enter_context(mp.solutions.face_mesh.FaceMesh(
            # Crops can move/resize between frames. Detect afresh instead of
            # allowing a bad crop acquisition to keep tracking the cap.
            static_image_mode=True, max_num_faces=1, refine_landmarks=True,
            min_detection_confidence=args.min_confidence, min_tracking_confidence=args.min_confidence,
        ))
        args.output.mkdir(parents=True, exist_ok=False)
        csv_file = stack.enter_context((args.output / "frames.csv").open("w", newline="", encoding="utf-8"))
        writer = csv.DictWriter(csv_file, fieldnames=FIELDS)
        writer.writeheader()
        video_writer = None
        previous_center = None
        if args.preview:
            stack.callback(cv2.destroyAllWindows)
        for timestamp, frame in iter_sampled_frames(
            capture, source_fps=metadata.fps, fps=metadata.fps, max_duration_seconds=args.max_duration_seconds,
        ):
            height, width = frame.shape[:2]
            row = dict.fromkeys(FIELDS, None)
            row.update(time_sec=timestamp, face_detected=0, landmarks_found=0, anchor_check_passed=0,
                       accepted=0, reason="no_face_detection")
            face = localizer.detect(frame, roi, previous_center)
            if face is not None:
                previous_center = face.center / np.array([width, height])
                row.update(face_detected=1, face_score=face.score, reason="no_landmarks")
                row.update(zip(("face_x1", "face_y1", "face_x2", "face_y2"), map(float, face.box)))
                x1, y1, x2, y2 = face_crop_box(face, frame.shape)
                crop = frame[y1:y2, x1:x2]
                result = mesh.process(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                if result.multi_face_landmarks:
                    landmarks = result.multi_face_landmarks[0]
                    points = np.array([[point.x * (x2 - x1) + x1, point.y * (y2 - y1) + y1]
                                       for point in landmarks.landmark], dtype=np.float64)
                    row["landmarks_found"] = 1
                    row.update(measure_face(
                        points, face, width, height, max_landmark_error=args.max_landmark_error,
                        max_yaw=args.max_yaw, max_pitch=args.max_pitch,
                        blink_threshold=args.blink_threshold, mouth_threshold=args.mouth_threshold,
                    ))
                    if row["anchor_check_passed"]:
                        mp.solutions.drawing_utils.draw_landmarks(
                            crop, landmarks, mp.solutions.face_mesh.FACEMESH_CONTOURS, None,
                            mp.solutions.drawing_styles.get_default_face_mesh_contours_style(),
                        )
            color = (0, 200, 0) if row["accepted"] else (0, 165, 255)
            # The broad search region is distinct from the actual detected face.
            rx1, ry1, rx2, ry2 = np.rint(np.array(roi) * [width, height, width, height]).astype(int)
            cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), (150, 150, 150), 1)
            if face is not None:
                fx1, fy1, fx2, fy2 = np.rint(face.box).astype(int)
                cv2.rectangle(frame, (fx1, fy1), (fx2, fy2), color, 2)
                for keypoint in face.keypoints:
                    cv2.circle(frame, tuple(np.rint(keypoint).astype(int)), 3, (255, 180, 0), -1)
                cv2.putText(frame, f"face {face.score:.2f}", (fx1, max(20, fy1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, .6, color, 2)
            captions = [f'{timestamp:.2f}s | {row["reason"]}']
            if row["yaw_deg"] is not None:
                captions.append(f'Yaw {row["yaw_deg"]:.1f} deg | Pitch {row["pitch_deg"]:.1f} deg')
            if row["accepted"]:
                captions.append(f'L EAR {row["left_ear"]:.3f} | R EAR {row["right_ear"]:.3f} | MAR {row["mar"]:.3f}')
                captions.append(str(row["eye_state"]))
            for index, caption in enumerate(captions):
                cv2.putText(frame, caption, (20, 35 + index * 32), cv2.FONT_HERSHEY_SIMPLEX, .7, color, 2, cv2.LINE_AA)
            if video_writer is None:
                video_writer = cv2.VideoWriter(str(args.output / "preview.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), metadata.fps, (width, height))
                stack.callback(video_writer.release)
                if not video_writer.isOpened():
                    raise OSError("Cannot create the annotated MP4")
            video_writer.write(frame)
            writer.writerow(row)
            counts["frames"] += 1
            counts["faces_detected"] += row["face_detected"]
            counts["landmarks_found"] += row["landmarks_found"]
            counts["anchor_check_passed"] += row["anchor_check_passed"]
            counts[row["reason"]] += 1
            if args.preview:
                display = cv2.resize(frame, (960, round(height * 960 / width)))
                cv2.imshow("Expression diagnostic - q to stop", display)
                if cv2.waitKey(max(1, round(1000 / metadata.fps))) & 0xFF in (ord("q"), 27):
                    break
    summary = {
        "source_video": str(args.video.resolve()), "source_fps": metadata.fps,
        "counts": dict(counts), "settings": {**vars(args), "patient_roi": roi, "refine_landmarks": True,
            "static_image_mode": True, "face_detector": "opencv_yunet",
            "face_detector_sha256": hashlib.sha256(args.face_detector_model.read_bytes()).hexdigest()},
        "note": "Independent face detection and mesh-anchor agreement are heuristic quality checks, not verified accuracy. Approximate camera and generic face model; inspect pose and landmark quality. Eye closure is a per-frame candidate, not a validated blink or emotion. Filtered frames retain timestamps with blank expression measurements.",
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, default=str, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
