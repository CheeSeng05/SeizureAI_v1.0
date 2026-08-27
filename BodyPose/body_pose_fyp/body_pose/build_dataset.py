from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from .annotation_adapters.base import validate_canonical
from .utils import (
    active_labels,
    ensure_parent,
    label_maps,
    load_config,
    seizeit2_video_message,
    window_manifest_path,
)

LOGGER = logging.getLogger(__name__)
NON_FEATURE_COLUMNS = {
    "processed_frame",
    "source_frame",
    "timestamp_sec",
    "source_fps",
}


def label_frames(
    frame_times: np.ndarray,
    annotations: pd.DataFrame,
    label_to_id: dict[str, int],
) -> np.ndarray:
    labels = np.full(len(frame_times), -1, dtype=np.int64)
    for row in annotations.itertuples(index=False):
        label = str(row.label).strip().lower()
        if label not in label_to_id:
            raise ValueError(f"Unknown label '{label}'. Expected {sorted(label_to_id)}")
        label_id = label_to_id[label]
        mask = (frame_times >= float(row.start_sec)) & (frame_times < float(row.end_sec))
        conflict = mask & (labels >= 0) & (labels != label_id)
        if conflict.any():
            raise ValueError(
                f"Conflicting canonical labels overlap for media_id '{row.media_id}' "
                f"near {frame_times[np.flatnonzero(conflict)[0]]:.3f} seconds."
            )
        labels[mask] = label_id
    return labels


def _split_counts(patient_count: int, fractions: list[float]) -> list[int]:
    raw = np.asarray(fractions, dtype=float) * patient_count
    counts = np.floor(raw).astype(int)
    minimums = np.asarray([1 if fraction > 0 else 0 for fraction in fractions])
    for index, fraction in enumerate(fractions):
        if fraction > 0 and counts[index] == 0:
            counts[index] = 1
    while counts.sum() > patient_count:
        candidates = [
            index for index, count in enumerate(counts) if count > minimums[index]
        ]
        if not candidates:
            break
        index = max(candidates, key=lambda item: (counts[item] - raw[item], counts[item]))
        counts[index] -= 1
    while counts.sum() < patient_count:
        index = max(
            range(len(counts)), key=lambda item: (raw[item] - counts[item], fractions[item])
        )
        counts[index] += 1
    return counts.tolist()


def split_patients(metadata: pd.DataFrame, cfg: dict) -> pd.Series:
    unique_patients = sorted(metadata["patient_id"].astype(str).unique())
    if len(unique_patients) == 1:
        LOGGER.warning(
            "Only one patient is available. Assigning all windows to training; "
            "this is a software check only, not model evaluation."
        )
        mapping = {unique_patients[0]: "train"}
        return metadata["patient_id"].astype(str).map(mapping)
    if len(unique_patients) == 2:
        LOGGER.warning(
            "Only two patients are available. Creating train and validation splits; "
            "there is no independent test patient."
        )
        rng = np.random.default_rng(int(cfg["split_seed"]))
        shuffled = list(np.asarray(unique_patients)[rng.permutation(2)])
        mapping = {shuffled[0]: "train", shuffled[1]: "validation"}
        return metadata["patient_id"].astype(str).map(mapping)

    fractions = [
        float(cfg["train_fraction"]),
        float(cfg["validation_fraction"]),
        float(cfg["test_fraction"]),
    ]
    if any(fraction < 0 for fraction in fractions) or not np.isclose(sum(fractions), 1.0):
        raise ValueError("Train, validation and test fractions must be non-negative and sum to 1.")
    rng = np.random.default_rng(int(cfg["split_seed"]))
    shuffled = list(np.asarray(unique_patients)[rng.permutation(len(unique_patients))])
    train_count, validation_count, _ = _split_counts(len(shuffled), fractions)
    mapping = {
        patient: (
            "train"
            if index < train_count
            else "validation"
            if index < train_count + validation_count
            else "test"
        )
        for index, patient in enumerate(shuffled)
    }
    return metadata["patient_id"].astype(str).map(mapping)


def assert_patient_disjoint(metadata: pd.DataFrame) -> None:
    patient_splits = metadata.groupby("patient_id")["split"].nunique()
    leaked = patient_splits[patient_splits > 1]
    if not leaked.empty:
        raise AssertionError(f"Patient leakage across splits: {leaked.index.tolist()}")


def deterministic_window_id(media_id: str, start_sec: float, end_sec: float) -> str:
    start_ms = int(round(start_sec * 1000.0))
    end_ms = int(round(end_sec * 1000.0))
    return f"{media_id}__{start_ms:010d}__{end_ms:010d}"


def _resample_features(
    frame: pd.DataFrame,
    feature_columns: list[str],
    sample_times: np.ndarray,
) -> np.ndarray:
    source_times = frame["timestamp_sec"].to_numpy(float)
    values = frame[feature_columns].to_numpy(np.float32)
    return np.column_stack(
        [np.interp(sample_times, source_times, values[:, index]) for index in range(values.shape[1])]
    ).astype(np.float32)


def build(config: dict) -> tuple[Path, Path]:
    paths = config["paths"]
    annotation_value = paths.get("canonical_annotations", paths.get("annotations"))
    if annotation_value is None:
        raise ValueError("config.paths.canonical_annotations is required.")
    annotation_path = Path(annotation_value)
    if not annotation_path.exists():
        raise FileNotFoundError(
            f"Canonical annotations not found at {annotation_path}. Run "
            "'python -m body_pose.prepare_annotations --config <config>' first."
        )

    label_to_id, id_to_label = label_maps(config)
    annotations = validate_canonical(
        pd.read_csv(annotation_path), active_labels=active_labels(config)
    )
    feature_dir = Path(paths["features"])
    feature_paths = sorted(feature_dir.glob("*.csv"))
    if not feature_paths:
        adapter_type = str(config.get("annotation_adapter", {}).get("type", "")).lower()
        if adapter_type in {"seizeit2", "ds005873"}:
            raise RuntimeError(seizeit2_video_message())
        raise FileNotFoundError(f"No pose feature CSV files found in {feature_dir}")

    dataset_cfg = config["dataset"]
    window_seconds = float(dataset_cfg["window_seconds"])
    stride_seconds = float(dataset_cfg["stride_seconds"])
    target_fps = float(dataset_cfg.get("target_fps", 10.0))
    if window_seconds <= 0 or stride_seconds <= 0 or target_fps <= 0:
        raise ValueError("window_seconds, stride_seconds and target_fps must be positive.")
    samples_per_window = max(2, round(window_seconds * target_fps))
    coverage_required = float(dataset_cfg["minimum_label_coverage"])
    if not 0 < coverage_required <= 1:
        raise ValueError("minimum_label_coverage must be in (0, 1].")

    windows: list[np.ndarray] = []
    labels: list[int] = []
    metadata_rows: list[dict] = []
    feature_columns: list[str] | None = None

    for feature_path in tqdm(feature_paths, desc="Media"):
        media_id = feature_path.stem
        ann = annotations[annotations["media_id"].astype(str) == media_id]
        if ann.empty:
            LOGGER.warning("Skipping %s: no canonical annotations", media_id)
            continue
        patient_ids = ann["patient_id"].astype(str).unique()
        if len(patient_ids) != 1:
            raise ValueError(f"{media_id} maps to multiple patient IDs.")
        patient_id = patient_ids[0]

        frame = pd.read_csv(feature_path).sort_values("timestamp_sec").reset_index(drop=True)
        if frame["timestamp_sec"].duplicated().any():
            raise ValueError(f"Duplicate timestamps in {feature_path}")
        current_features = [
            column
            for column in frame.select_dtypes(include=[np.number]).columns
            if column not in NON_FEATURE_COLUMNS
        ]
        if feature_columns is None:
            feature_columns = current_features
        elif current_features != feature_columns:
            raise ValueError(f"Feature columns differ in {feature_path}")
        if not current_features:
            raise ValueError(f"No numeric model features found in {feature_path}")

        times = frame["timestamp_sec"].to_numpy(float)
        if len(times) < 2:
            LOGGER.warning("Skipping %s: fewer than two feature frames", media_id)
            continue
        frame_dt = float(np.median(np.diff(times)))
        if not np.isfinite(frame_dt) or frame_dt <= 0:
            raise ValueError(f"Invalid timestamps in {feature_path}")
        available_end = float(times[-1] + frame_dt)
        first_start = float(times[0])
        window_count = int(np.floor((available_end - first_start - window_seconds) / stride_seconds)) + 1
        for window_offset in range(max(window_count, 0)):
            start_sec = round(first_start + window_offset * stride_seconds, 9)
            end_sec = round(start_sec + window_seconds, 9)
            sample_times = start_sec + np.arange(samples_per_window) / target_fps
            sample_labels = label_frames(sample_times, ann, label_to_id)
            valid = sample_labels[sample_labels >= 0]
            if len(valid) / samples_per_window < coverage_required or len(valid) == 0:
                continue
            counts = np.bincount(valid, minlength=len(label_to_id))
            label_id = int(np.argmax(counts))
            class_coverage = float(counts[label_id] / samples_per_window)
            if class_coverage < coverage_required:
                continue

            window_id = deterministic_window_id(media_id, start_sec, end_sec)
            windows.append(_resample_features(frame, current_features, sample_times))
            labels.append(label_id)
            metadata_rows.append({
                "window_index": len(windows) - 1,
                "window_id": window_id,
                "media_id": media_id,
                "patient_id": patient_id,
                "start_sec": start_sec,
                "end_sec": end_sec,
                "label": id_to_label[label_id],
                "label_id": label_id,
                "split": "",
                "label_coverage": class_coverage,
                "source_effective_fps": 1.0 / frame_dt,
                "target_fps": target_fps,
            })

    if not windows or feature_columns is None:
        adapter_type = str(config.get("annotation_adapter", {}).get("type", "")).lower()
        if adapter_type in {"seizeit2", "ds005873"}:
            raise RuntimeError(seizeit2_video_message())
        raise RuntimeError(
            "No windows were created. Check canonical media IDs, annotations, and window settings."
        )

    manifest = pd.DataFrame(metadata_rows)
    if manifest["window_id"].duplicated().any():
        duplicates = manifest.loc[manifest["window_id"].duplicated(), "window_id"].tolist()
        raise ValueError(f"Duplicate deterministic window IDs: {duplicates[:5]}")
    manifest["split"] = split_patients(manifest, dataset_cfg)
    assert_patient_disjoint(manifest)
    x_all = np.stack(windows).astype(np.float32)
    y_all = np.asarray(labels, dtype=np.int64)

    dataset_path = ensure_parent(paths["dataset"])
    manifest_path = ensure_parent(window_manifest_path(config))
    np.savez_compressed(
        dataset_path,
        X=x_all,
        y=y_all,
        feature_names=np.asarray(feature_columns, dtype=str),
        label_names=np.asarray(active_labels(config), dtype=str),
        window_ids=manifest["window_id"].to_numpy(str),
    )
    manifest.to_csv(manifest_path, index=False)
    LOGGER.info("Dataset shape: %s", x_all.shape)
    LOGGER.info("Patient counts by split:\n%s", manifest.groupby("split")["patient_id"].nunique())
    LOGGER.info("Window counts by split:\n%s", manifest["split"].value_counts())
    LOGGER.info("Label counts:\n%s", manifest["label"].value_counts())
    return dataset_path, manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build shared, labelled temporal windows from video pose features."
    )
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    build(load_config(args.config))


if __name__ == "__main__":
    main()
