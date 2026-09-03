from __future__ import annotations

import importlib
import math
import sys
import collections
import time
from functools import lru_cache
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from ..data_inspection import inspect_mp4, resolved_interval_label_records, resolved_label_records
from .vsvig_adapter import VSViGAdapter
from ..progress import report_progress


PATCH_POINT_ORDER = (0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 16, 17)
POSITION_POINT_ORDER = (0, 15, 14, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13)


@dataclass(frozen=True)
class VideoAnnotation:
    patient_id: str | None
    seizure_id: str | None
    seizure_type: str | None
    eeg_onset_sec: float | None
    clinical_onset_sec: float | None
    seizure_end_sec: float | None
    alignment_status: str
    intervals: tuple[dict[str, Any], ...] = ()
    label_source: str = "workbook"
    label_policy: str = "clinical_onset_interval"


def _matches_identity(
    records: list[dict[str, Any]],
    *,
    patient_id: str | None,
    seizure_id: str | None,
    seizure_type: str | None,
) -> list[dict[str, Any]]:
    if patient_id and any(record.get("resolved_PatID") for record in records):
        records = [record for record in records if record.get("resolved_PatID") == patient_id]
    if seizure_id and any(record.get("#Seizure") for record in records):
        records = [record for record in records if str(record.get("#Seizure")) == seizure_id]
    if seizure_type and any(record.get("resolved_Seizure Type") for record in records):
        records = [
            record for record in records if record.get("resolved_Seizure Type") == seizure_type
        ]
    return records


def _load_interval_csv_annotation(
    label_path: str | Path,
    duration_sec: float,
    *,
    patient_id: str | None,
    seizure_id: str | None,
    seizure_type: str | None,
    recording_id: str | None,
) -> VideoAnnotation:
    records = _matches_identity(
        resolved_interval_label_records(label_path),
        patient_id=patient_id,
        seizure_id=seizure_id,
        seizure_type=seizure_type,
    )
    if recording_id and any(record.get("recording_id") for record in records):
        records = [record for record in records if str(record.get("recording_id")) == recording_id]
    if not records:
        return VideoAnnotation(
            None, None, None, None, None, None, "requires_at_least_one_interval",
            label_source="interval_csv",
            label_policy="majority_overlap",
        )
    if max(float(record["end_sec"]) for record in records) > duration_sec:
        status = "out_of_video_range_requires_real_clip_offset"
    else:
        status = "aligned_to_video_time"
    positive_intervals = [record for record in records if int(record["label"]) == 1]
    first_positive = min((float(record["start_sec"]) for record in positive_intervals), default=None)
    last_positive_end = max((float(record["end_sec"]) for record in positive_intervals), default=None)
    first = records[0]
    intervals = tuple(
        {
            "start_sec": float(record["start_sec"]),
            "end_sec": float(record["end_sec"]),
            "label": int(record["label"]),
            "label_name": record.get("label_name") or ("seizure" if int(record["label"]) else "non_seizure"),
            "source": record.get("source") or "csv",
        }
        for record in sorted(records, key=lambda item: (float(item["start_sec"]), float(item["end_sec"])))
    )
    return VideoAnnotation(
        patient_id=str(first.get("resolved_PatID")) if first.get("resolved_PatID") else patient_id,
        seizure_id=str(first.get("#Seizure")) if first.get("#Seizure") else seizure_id,
        seizure_type=(
            str(first.get("resolved_Seizure Type"))
            if first.get("resolved_Seizure Type")
            else seizure_type
        ),
        eeg_onset_sec=first_positive,
        clinical_onset_sec=first_positive,
        seizure_end_sec=last_positive_end,
        alignment_status=status,
        intervals=intervals,
        label_source="interval_csv",
        label_policy="majority_overlap",
    )


def load_video_annotation(
    label_path: str | Path | None,
    duration_sec: float,
    *,
    patient_id: str | None = None,
    seizure_id: str | None = None,
    seizure_type: str | None = None,
    recording_id: str | None = None,
) -> VideoAnnotation:
    if label_path is None:
        return VideoAnnotation(None, None, None, None, None, None, "annotation_file_missing")
    if Path(label_path).suffix.lower() == ".csv":
        return _load_interval_csv_annotation(
            label_path,
            duration_sec,
            patient_id=patient_id,
            seizure_id=seizure_id,
            seizure_type=seizure_type,
            recording_id=recording_id,
        )
    records = resolved_label_records(label_path)
    if patient_id and seizure_id:
        records = _matches_identity(
            records, patient_id=patient_id, seizure_id=seizure_id, seizure_type=seizure_type
        )
    if len(records) != 1:
        return VideoAnnotation(None, None, None, None, None, None, "requires_exactly_one_record")
    record = records[0]
    eeg_onset = record.get("EEG onset seconds")
    clinical_onset = record.get("Clinical Onset seconds")
    end_candidates = ("Seizure End seconds", "Seizure end seconds", "Offset seconds")
    seizure_end = next((record.get(key) for key in end_candidates if record.get(key) is not None), None)
    supplied_times = [value for value in (eeg_onset, clinical_onset, seizure_end) if value is not None]
    status = "aligned_to_video_time"
    if supplied_times and max(float(value) for value in supplied_times) > duration_sec:
        status = "out_of_video_range_requires_real_clip_offset"
    elif eeg_onset is None and clinical_onset is None:
        status = "onset_missing"
    elif seizure_end is None:
        status = "onsets_aligned_but_offset_missing"
    return VideoAnnotation(
        patient_id=str(record.get("resolved_PatID")) if record.get("resolved_PatID") else None,
        seizure_id=str(record.get("#Seizure")) if record.get("#Seizure") else None,
        seizure_type=(
            str(record.get("resolved_Seizure Type"))
            if record.get("resolved_Seizure Type")
            else None
        ),
        eeg_onset_sec=float(eeg_onset) if eeg_onset is not None else None,
        clinical_onset_sec=float(clinical_onset) if clinical_onset is not None else None,
        seizure_end_sec=float(seizure_end) if seizure_end is not None else None,
        alignment_status=status,
    )


def reference_label_for_interval(
    annotation: VideoAnnotation, start_sec: float, end_sec: float
) -> int | None:
    if annotation.alignment_status == "out_of_video_range_requires_real_clip_offset":
        return None
    if annotation.intervals:
        overlaps = {0: 0.0, 1: 0.0}
        for interval in annotation.intervals:
            overlap = max(
                0.0,
                min(end_sec, float(interval["end_sec"])) - max(start_sec, float(interval["start_sec"])),
            )
            if overlap > 0:
                overlaps[int(interval["label"])] += overlap
        if overlaps[0] == overlaps[1]:
            return None
        return 1 if overlaps[1] > overlaps[0] else 0
    onset = annotation.clinical_onset_sec
    eeg_onset = annotation.eeg_onset_sec
    if eeg_onset is not None and end_sec <= eeg_onset:
        return 0
    if onset is not None and start_sec >= onset:
        if annotation.seizure_end_sec is None or end_sec <= annotation.seizure_end_sec:
            return 1
    if annotation.seizure_end_sec is not None and start_sec >= annotation.seizure_end_sec:
        return 0
    return None


class LightweightOpenPose:
    """Inference-only wrapper around the official dependency referenced by VSViG."""

    def __init__(
        self,
        *,
        source_directory: str | Path,
        checkpoint_path: str | Path,
        device: str,
        input_height: int = 256,
    ) -> None:
        try:
            import cv2
            import torch
        except ImportError as exc:
            raise RuntimeError("Raw VSViG video inference requires OpenCV and PyTorch") from exc
        self.cv2 = cv2
        self.torch = torch
        self.device = torch.device(device)
        self.input_height = int(input_height)
        if self.input_height <= 0:
            raise ValueError("Pose input height must be positive")
        source = Path(source_directory).resolve()
        checkpoint = Path(checkpoint_path).resolve()
        if not (source / "models" / "with_mobilenet.py").is_file():
            raise FileNotFoundError(f"Official lightweight OpenPose source is incomplete: {source}")
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Pose checkpoint not found: {checkpoint}")
        sys.path.insert(0, str(source))
        try:
            model_module = importlib.import_module("models.with_mobilenet")
            keypoint_module = importlib.import_module("modules.keypoints")
            pose_module = importlib.import_module("modules.pose")
        finally:
            if sys.path and sys.path[0] == str(source):
                sys.path.pop(0)
        self.extract_keypoints = keypoint_module.extract_keypoints
        self.group_keypoints = keypoint_module.group_keypoints
        self.Pose = pose_module.Pose
        self.model = model_module.PoseEstimationWithMobileNet()
        try:
            with torch.serialization.safe_globals([collections.Counter]):
                payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        except (AttributeError, TypeError):
            payload = torch.load(checkpoint, map_location="cpu")
        state = payload.get("state_dict") if isinstance(payload, dict) else None
        if not isinstance(state, dict):
            raise TypeError("Pose checkpoint does not contain state_dict")
        target = self.model.state_dict()
        missing = [key for key, value in target.items() if key not in state or state[key].shape != value.shape]
        if missing:
            raise ValueError(f"Pose checkpoint is incompatible; first missing key: {missing[0]}")
        self.model.load_state_dict({key: state[key] for key in target}, strict=True)
        self.model.to(self.device).eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    @property
    def is_inference_only(self) -> bool:
        return not self.model.training and all(not p.requires_grad for p in self.model.parameters())

    def _prepare(self, frame_bgr: np.ndarray) -> tuple[np.ndarray, float, list[int]]:
        cv2 = self.cv2
        height = frame_bgr.shape[0]
        scale = self.input_height / height
        scaled = cv2.resize(frame_bgr, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        scaled = _normalize_pose_input(
            scaled, np.array([128, 128, 128], dtype=np.float32), np.float32(1 / 256)
        )
        padded, pad = _pad_pose_input(
            scaled,
            8,
            (0, 0, 0),
            [self.input_height, max(scaled.shape[1], self.input_height)],
        )
        return padded, scale, pad

    def _decode(
        self,
        heatmaps_chw: np.ndarray,
        pafs_chw: np.ndarray,
        *,
        scale: float,
        pad: list[int],
    ) -> tuple[np.ndarray, bool]:
        cv2 = self.cv2
        heatmaps = np.transpose(heatmaps_chw, (1, 2, 0))
        pafs = np.transpose(pafs_chw, (1, 2, 0))
        heatmaps = cv2.resize(heatmaps, (0, 0), fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
        pafs = cv2.resize(pafs, (0, 0), fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
        total = 0
        by_type = []
        for keypoint_index in range(self.Pose.num_kpts):
            total += self.extract_keypoints(heatmaps[:, :, keypoint_index], by_type, total)
        entries, all_keypoints = self.group_keypoints(by_type, pafs)
        if len(entries) == 0:
            return np.column_stack((np.full((18, 2), -1, dtype=np.float32), np.zeros(18))), False
        for index in range(all_keypoints.shape[0]):
            all_keypoints[index, 0] = (all_keypoints[index, 0] * 8 / 4 - pad[1]) / scale
            all_keypoints[index, 1] = (all_keypoints[index, 1] * 8 / 4 - pad[0]) / scale
        candidates: list[tuple[float, float, np.ndarray]] = []
        for entry in entries:
            if len(entry) == 0:
                continue
            points = np.column_stack((np.full((18, 2), -1, dtype=np.float32), np.zeros(18)))
            for keypoint_index in range(18):
                if entry[keypoint_index] != -1.0:
                    detected = all_keypoints[int(entry[keypoint_index])]
                    points[keypoint_index] = detected[:3]
            valid = points[:, 0] >= 0
            if not np.any(valid):
                continue
            xs, ys = points[valid, 0], points[valid, 1]
            area = float((xs.max() - xs.min() + 1) * (ys.max() - ys.min() + 1))
            candidates.append((area, float(entry[18]), points.astype(np.float32)))
        if not candidates:
            return np.column_stack((np.full((18, 2), -1, dtype=np.float32), np.zeros(18))), False
        return max(candidates, key=lambda item: (item[0], item[1]))[2], True

    def predict_many(self, frames_bgr: list[np.ndarray]) -> list[tuple[np.ndarray, bool]]:
        if not frames_bgr:
            return []
        torch = self.torch
        prepared = [self._prepare(frame) for frame in frames_bgr]
        padded_shapes = {item[0].shape for item in prepared}
        if len(padded_shapes) != 1:
            raise ValueError("A pose batch requires frames with a common padded resolution")
        tensor = (
            torch.from_numpy(np.stack([item[0] for item in prepared]))
            .permute(0, 3, 1, 2)
            .float()
            .to(self.device)
        )
        with torch.inference_mode():
            stages = self.model(tensor)
        heatmaps = stages[-2].detach().cpu().numpy()
        pafs = stages[-1].detach().cpu().numpy()
        return [
            self._decode(heatmaps[index], pafs[index], scale=prepared[index][1], pad=prepared[index][2])
            for index in range(len(prepared))
        ]

    def predict(self, frame_bgr: np.ndarray) -> tuple[np.ndarray, bool]:
        return self.predict_many([frame_bgr])[0]


def _normalize_pose_input(
    image: np.ndarray, image_mean: np.ndarray, image_scale: np.float32
) -> np.ndarray:
    """Exact normalization used by the upstream lightweight OpenPose val.py."""

    return (np.asarray(image, dtype=np.float32) - image_mean) * image_scale


def _pad_pose_input(
    image: np.ndarray,
    stride: int,
    pad_value: tuple[int, int, int],
    minimum_dimensions: list[int],
) -> tuple[np.ndarray, list[int]]:
    """Exact padding geometry from upstream val.py, without COCO evaluation imports."""

    cv2 = importlib.import_module("cv2")
    height, width, _ = image.shape
    height = min(minimum_dimensions[0], height)
    minimum_dimensions[0] = math.ceil(minimum_dimensions[0] / float(stride)) * stride
    minimum_dimensions[1] = max(minimum_dimensions[1], width)
    minimum_dimensions[1] = math.ceil(minimum_dimensions[1] / float(stride)) * stride
    pad = [
        int(math.floor((minimum_dimensions[0] - height) / 2.0)),
        int(math.floor((minimum_dimensions[1] - width) / 2.0)),
    ]
    pad.extend(
        [
            int(minimum_dimensions[0] - height - pad[0]),
            int(minimum_dimensions[1] - width - pad[1]),
        ]
    )
    padded = cv2.copyMakeBorder(
        image,
        pad[0],
        pad[2],
        pad[1],
        pad[3],
        cv2.BORDER_CONSTANT,
        value=pad_value,
    )
    return padded, pad


@lru_cache(maxsize=4)
def _gaussian_kernel(size: int = 128, sigma_fraction: float = 0.3) -> np.ndarray:
    sigma = size * sigma_fraction
    kernel = np.fromfunction(
        lambda x, y: (1 / (2 * math.pi * sigma**2))
        * np.exp(-((x - (size - 1) / 2) ** 2 + (y - (size - 1) / 2) ** 2) / (2 * sigma**2)),
        (size, size),
    )
    kernel /= np.sum(kernel)
    kernel = (kernel - np.min(kernel)) / (np.max(kernel) - np.min(kernel))
    return np.repeat(kernel[:, :, None], 3, axis=2).astype(np.float32)


def extract_vsvig_patches(frame_bgr: np.ndarray, raw_keypoints: np.ndarray) -> np.ndarray:
    """Follow VSViG extract_patches.py, with explicit zero patches for missing points."""

    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("OpenCV is required for VSViG patch extraction") from exc
    size = 128
    kernel = _gaussian_kernel(size)
    patches = np.zeros((15, 3, 32, 32), dtype=np.float32)
    for output_index, keypoint_index in enumerate(PATCH_POINT_ORDER):
        x, y, confidence = raw_keypoints[keypoint_index]
        if x < 0 or y < 0 or confidence <= 0:
            continue
        # This is algebraically identical to padding the entire frame by 128
        # pixels and taking padded[int(y+64):+128, int(x+64):+128], as in the
        # source script, but allocates only the keypoint crop.
        frame_x0 = int(x + 0.5 * size) - size
        frame_y0 = int(y + 0.5 * size) - size
        source_x0 = max(0, frame_x0)
        source_y0 = max(0, frame_y0)
        source_x1 = min(frame_bgr.shape[1], frame_x0 + size)
        source_y1 = min(frame_bgr.shape[0], frame_y0 + size)
        if source_x1 <= source_x0 or source_y1 <= source_y0:
            continue
        crop = np.zeros((size, size, 3))
        target_x0 = source_x0 - frame_x0
        target_y0 = source_y0 - frame_y0
        crop[
            target_y0 : target_y0 + (source_y1 - source_y0),
            target_x0 : target_x0 + (source_x1 - source_x0),
        ] = frame_bgr[source_y0:source_y1, source_x0:source_x1]
        weighted = crop * kernel
        minimum, maximum = float(weighted.min()), float(weighted.max())
        if maximum > minimum:
            weighted = (weighted - minimum) / (maximum - minimum) * 255.0
        else:
            weighted = np.zeros_like(weighted)
        resized = cv2.resize(weighted, (32, 32), interpolation=cv2.INTER_LINEAR)
        patches[output_index] = np.transpose(resized, (2, 0, 1))
    return patches


def _flush_clips(
    adapter: VSViGAdapter,
    pending: list[tuple[int, int, np.ndarray, np.ndarray, float]],
    *,
    fps: float,
    recording_id: str,
    threshold: float,
    annotation: VideoAnnotation,
) -> tuple[list[dict[str, Any]], list[np.ndarray]]:
    if not pending:
        return [], []
    patches = np.stack([item[2] for item in pending])
    keypoints = np.stack([item[3] for item in pending])
    output = adapter.predict(patches, keypoints)
    probability = output["probability"].detach().cpu().numpy().reshape(-1)
    logits = output["logits"].detach().cpu().numpy().reshape(-1)
    embeddings = output["embedding"].detach().cpu().numpy()
    rows: list[dict[str, Any]] = []
    for item, score, logit in zip(pending, probability, logits):
        start_frame, stop_frame, _, _, pose_fraction = item
        start_sec, end_sec = start_frame / fps, stop_frame / fps
        rows.append(
            {
                "recording_id": recording_id,
                "start_sec": start_sec,
                "end_sec": end_sec,
                "video_probability": float(score),
                "video_logit": float(logit),
                "video_prediction": int(score >= threshold),
                "reference_label": reference_label_for_interval(annotation, start_sec, end_sec),
                "eeg_onset_sec": annotation.eeg_onset_sec,
                "clinical_onset_sec": annotation.clinical_onset_sec,
                "seizure_end_sec": annotation.seizure_end_sec,
                "annotation_alignment_status": annotation.alignment_status,
                "pose_detection_fraction": pose_fraction,
            }
        )
    return rows, [embedding for embedding in embeddings]


def predict_video_file(
    *,
    video_path: str | Path,
    label_path: str | Path | None,
    vsvig_source: str | Path,
    vsvig_checkpoint: str | Path,
    dynamic_partition: str | Path,
    pose_source: str | Path,
    pose_checkpoint: str | Path,
    annotation_patient_id: str | None = None,
    annotation_seizure_id: str | None = None,
    annotation_seizure_type: str | None = None,
    recording_id: str | None = None,
    device: str,
    threshold: float = 0.5,
    clip_frames: int = 30,
    clip_stride_frames: int = 30,
    batch_size: int = 8,
    pose_batch_size: int = 16,
    pose_input_height: int = 256,
    progress_config: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], np.ndarray, dict[str, Any]]:
    if clip_frames != 30:
        raise ValueError("The pretrained VSViG checkpoint requires exactly 30 frames per clip")
    if clip_stride_frames != clip_frames:
        raise ValueError("Raw inference currently requires deterministic non-overlapping 30-frame clips")
    if pose_batch_size <= 0:
        raise ValueError("Pose batch size must be positive")
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("OpenCV is required to decode Sz2PG.mp4") from exc
    metadata = inspect_mp4(video_path)
    annotation = load_video_annotation(
        label_path,
        metadata.duration_sec,
        patient_id=annotation_patient_id,
        seizure_id=annotation_seizure_id,
        seizure_type=annotation_seizure_type,
        recording_id=recording_id,
    )
    pose = LightweightOpenPose(
        source_directory=pose_source,
        checkpoint_path=pose_checkpoint,
        device=device,
        input_height=pose_input_height,
    )
    rows: list[dict[str, Any]] = []
    embeddings: list[np.ndarray] = []
    pending: list[tuple[int, int, np.ndarray, np.ndarray, float]] = []
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise IOError(f"Video cannot be opened: {video_path}")
    try:
        with VSViGAdapter(
            source_path=vsvig_source,
            checkpoint_path=vsvig_checkpoint,
            dynamic_partition_path=dynamic_partition,
            device=device,
        ) as adapter:
            frame_index = 0
            processed_clips = 0
            started = time.monotonic()
            total_clips = metadata.frame_count // clip_frames
            report_progress(
                progress_config or {}, stage="video_inference",
                label="Processing video clips", completed=0, total=total_clips,
                elapsed_sec=0.0,
            )
            while True:
                clip_start = frame_index
                frames: list[np.ndarray] = []
                for _ in range(clip_frames):
                    ok, frame = capture.read()
                    if not ok:
                        break
                    frames.append(frame)
                    frame_index += 1
                if len(frames) != clip_frames:
                    break
                pose_results: list[tuple[np.ndarray, bool]] = []
                for pose_start in range(0, clip_frames, pose_batch_size):
                    pose_results.extend(
                        pose.predict_many(frames[pose_start : pose_start + pose_batch_size])
                    )
                raw_keypoints = [result[0] for result in pose_results]
                pose_hits = sum(int(result[1]) for result in pose_results)
                pending.append(
                    (
                        clip_start,
                        frame_index,
                        np.stack(
                            [
                                extract_vsvig_patches(frame, points)
                                for frame, points in zip(frames, raw_keypoints)
                            ]
                        ),
                        np.stack([points[list(POSITION_POINT_ORDER)] for points in raw_keypoints]),
                        pose_hits / clip_frames,
                    )
                )
                processed_clips += 1
                elapsed = time.monotonic() - started
                report_progress(
                    progress_config or {}, stage="video_inference",
                    label="Processing video clips", completed=processed_clips,
                    total=total_clips, elapsed_sec=elapsed,
                )
                if len(pending) >= batch_size:
                    new_rows, new_embeddings = _flush_clips(
                        adapter,
                        pending,
                        fps=metadata.fps,
                        recording_id=recording_id or Path(video_path).stem,
                        threshold=threshold,
                        annotation=annotation,
                    )
                    rows.extend(new_rows)
                    embeddings.extend(new_embeddings)
                    pending = []
                if processed_clips % 100 == 0:
                    print(
                        f"video progress: {processed_clips}/{total_clips} clips "
                        f"({processed_clips / max(1, total_clips):.1%}), {elapsed:.1f}s elapsed",
                        flush=True,
                    )
            new_rows, new_embeddings = _flush_clips(
                adapter,
                pending,
                fps=metadata.fps,
                recording_id=recording_id or Path(video_path).stem,
                threshold=threshold,
                annotation=annotation,
            )
            rows.extend(new_rows)
            embeddings.extend(new_embeddings)
            inference_only = adapter.is_inference_only and pose.is_inference_only
    finally:
        capture.release()
    run_metadata = {
        "video": metadata.as_dict(),
        "annotation": annotation.__dict__,
        "clips": len(rows),
        "discarded_trailing_frames": metadata.frame_count - len(rows) * clip_frames,
        "clip_frames": clip_frames,
        "clip_stride_frames": clip_stride_frames,
        "pose_batch_size": pose_batch_size,
        "effective_clip_duration_sec": clip_frames / metadata.fps,
        "pose_point_policy": "largest bounding-box pose per frame; missing points are zero patches with (-1,-1,0) positions",
        "patch_point_order": list(PATCH_POINT_ORDER),
        "position_point_order": list(POSITION_POINT_ORDER),
        "inference_only": inference_only,
    }
    embedding_array = np.stack(embeddings) if embeddings else np.empty((0, 384), dtype=np.float32)
    return rows, embedding_array, run_metadata
