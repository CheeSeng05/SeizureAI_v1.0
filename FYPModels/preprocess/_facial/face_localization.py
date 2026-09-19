"""Independent face localization and landmark agreement checks for diagnostics."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

DEFAULT_YUNET_MODEL = Path(__file__).resolve().parents[2] / "checkpoints/pretrained/yunet/face_detection_yunet_2023mar.onnx"


@dataclass(frozen=True)
class LocalizedFace:
    box: np.ndarray  # x1, y1, x2, y2 in full-frame pixels
    keypoints: np.ndarray  # subject's right eye, left eye, nose, right/left mouth corners
    score: float

    @property
    def center(self) -> np.ndarray:
        return (self.box[:2] + self.box[2:]) / 2


def select_face(faces: np.ndarray | None, offset: tuple[int, int], frame_shape: tuple,
                patient_roi: tuple[float, ...], previous_center: np.ndarray | None = None,
                max_jump: float = .18) -> LocalizedFace | None:
    """Map YuNet's crop coordinates to the frame and select/track one patient."""
    if faces is None:
        return None
    height, width = frame_shape[:2]
    candidates = []
    for index, row in enumerate(faces):
        if len(row) != 15 or not np.isfinite(row).all() or min(row[2:4]) <= 0:
            continue
        origin = row[:2] + offset
        end = origin + row[2:4]
        box = np.array([max(0, origin[0]), max(0, origin[1]), min(width, end[0]), min(height, end[1])])
        if np.any(box[2:] <= box[:2]):
            continue
        center = (box[:2] + box[2:]) / (2 * np.array([width, height]))
        if not (patient_roi[0] <= center[0] <= patient_roi[2]
                and patient_roi[1] <= center[1] <= patient_roi[3]):
            continue
        distance = float(np.linalg.norm(center - previous_center)) if previous_center is not None else 0.0
        if previous_center is not None and distance > max_jump:
            continue
        candidate = LocalizedFace(box, row[4:14].reshape(5, 2).copy() + offset, float(row[14]))
        candidates.append((distance, -candidate.score, index, candidate))
    return min(candidates, key=lambda item: item[:3])[3] if candidates else None


class YuNetLocalizer:
    def __init__(self, model: str | Path = DEFAULT_YUNET_MODEL, confidence: float = .8):
        path = Path(model).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"YuNet model not found: {path}. See preprocess/README.md for the model download.")
        self.detector = cv2.FaceDetectorYN.create(str(path), "", (320, 320), confidence, .3, 5000)

    def detect(self, frame: np.ndarray, patient_roi: tuple[float, ...],
               previous_center: np.ndarray | None = None) -> LocalizedFace | None:
        height, width = frame.shape[:2]
        x1, y1, x2, y2 = np.rint(np.array(patient_roi) * [width, height, width, height]).astype(int)
        if x2 <= x1 or y2 <= y1:
            raise ValueError("Patient ROI is empty at this frame resolution")
        crop = np.ascontiguousarray(frame[y1:y2, x1:x2])
        self.detector.setInputSize((crop.shape[1], crop.shape[0]))
        _, faces = self.detector.detect(crop)
        return select_face(faces, (x1, y1), frame.shape, patient_roi, previous_center)


def face_crop_box(face: LocalizedFace, frame_shape: tuple, margin: float = .15) -> np.ndarray:
    """Pad a detected face to a clipped square; never use the broad patient ROI."""
    height, width = frame_shape[:2]
    side = min(max(float(np.max(face.box[2:] - face.box[:2])) * (1 + 2 * margin), 2), width, height)
    origin = np.clip(face.center - side / 2, [0, 0], [width - side, height - side])
    return np.rint(np.r_[origin, origin + side]).astype(int)


def landmark_agreement_error(points: np.ndarray, face: LocalizedFace) -> float | None:
    """Largest five-anchor displacement as a fraction of detected face size.

    This is an agreement heuristic, not proof that either model is correct.
    In particular, consistent hallucinations and occluded features still require
    visual review. Never accept a mesh only because its derived pose is plausible.
    """
    if points.ndim != 2 or points.shape[0] < 468 or points.shape[1] != 2 or not np.isfinite(points).all():
        return None
    mesh_anchors = np.array([
        (points[33] + points[133]) / 2, (points[362] + points[263]) / 2,
        points[1], points[61], points[291],
    ])
    scale = float(np.max(face.box[2:] - face.box[:2]))
    if not np.isfinite(face.keypoints).all() or scale <= 0:
        return None
    return float(np.linalg.norm(mesh_anchors - face.keypoints, axis=1).max() / scale)
