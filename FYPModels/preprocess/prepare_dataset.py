"""Create a reproducible class-stratified video split and preprocess it."""
from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

from ._facial.face_preprocess import add_enhancement_arguments, enhancement_kwargs, extract_video
from .preprocess_options import add_tuning_arguments, tuning_kwargs, validate_tuning
from . import add_video_enhancement_arguments, video_enhancement_kwargs


PLAN_FIELDS = (
    "source_video",
    "recording_id",
    "patient_id",
    "class_name",
    "label",
    "split",
)


def _patient_id(path: Path) -> str:
    return path.stem.split("_", 1)[0]


def stratified_video_plan(
    dataset_root: str | Path,
    *,
    train_fraction: float = 0.7,
    validation_fraction: float = 0.15,
    seed: int = 42,
    split_unit: str = "patient",
) -> list[dict[str, str]]:
    """Build a class-balanced split by patient (preferred) or by video."""
    if not 0.0 < train_fraction < 1.0 or not 0.0 < validation_fraction < 1.0:
        raise ValueError("train_fraction and validation_fraction must be between 0 and 1")
    if train_fraction + validation_fraction >= 1.0:
        raise ValueError("train_fraction + validation_fraction must be below 1")
    if split_unit not in {"patient", "video"}:
        raise ValueError("split_unit must be 'patient' or 'video'")

    root = Path(dataset_root).resolve()
    class_specs = (("Normal", 0), ("Seizure", 1))
    videos_by_class: dict[str, list[Path]] = {}
    counts_by_class: dict[str, dict[str, int]] = {}
    for class_name, _ in class_specs:
        videos = sorted((root / class_name).glob("*.mp4"))
        if not videos:
            raise FileNotFoundError(f"No MP4 videos found in {root / class_name}")
        videos_by_class[class_name] = videos
        counts: dict[str, int] = {}
        for video in videos:
            patient = _patient_id(video)
            counts[patient] = counts.get(patient, 0) + 1
        counts_by_class[class_name] = counts

    training_patients: set[str] | None = None
    validation_patients: set[str] | None = None
    if split_unit == "patient":
        normal_counts = counts_by_class["Normal"]
        seizure_counts = counts_by_class["Seizure"]
        if normal_counts != seizure_counts:
            raise ValueError(
                "Patient-level splitting currently requires matching per-patient video "
                "counts in Normal and Seizure so both classes retain the same ratio"
            )
        target = round(sum(normal_counts.values()) * train_fraction)
        training_patients = _closest_patient_subset(normal_counts, target, seed)
        remaining_counts = {
            patient: count for patient, count in normal_counts.items()
            if patient not in training_patients
        }
        validation_target = max(1, round(sum(normal_counts.values()) * validation_fraction))
        validation_patients = _closest_patient_subset(
            remaining_counts, validation_target, seed + 1, allow_all=False
        )

    rows: list[dict[str, str]] = []

    for class_name, label in class_specs:
        videos = videos_by_class[class_name]
        if split_unit == "video":
            # A class-specific deterministic shuffle makes the plan reproducible.
            random.Random(f"{seed}:{class_name}").shuffle(videos)
            train_count = round(len(videos) * train_fraction)
            validation_count = round(len(videos) * validation_fraction)

        for index, video in enumerate(videos):
            patient = _patient_id(video)
            if training_patients is not None:
                split = (
                    "train" if patient in training_patients
                    else "validation" if patient in validation_patients
                    else "test"
                )
            else:
                split = (
                    "train" if index < train_count
                    else "validation" if index < train_count + validation_count
                    else "test"
                )
            rows.append(
                {
                    "source_video": str(video),
                    "recording_id": f"{class_name.lower()}_{video.stem}",
                    "patient_id": patient,
                    "class_name": class_name.lower(),
                    "label": str(label),
                    "split": split,
                }
            )
    return rows


def _closest_patient_subset(
    patient_counts: dict[str, int], target_videos: int, seed: int, *, allow_all: bool = False
) -> set[str]:
    """Find a deterministic patient group with video count closest to target."""
    patients = sorted(patient_counts)
    random.Random(seed).shuffle(patients)
    subsets: dict[int, tuple[str, ...]] = {0: ()}
    for patient in patients:
        additions = {
            total + patient_counts[patient]: selected + (patient,)
            for total, selected in list(subsets.items())
        }
        for total, selected in additions.items():
            subsets.setdefault(total, selected)
    upper = sum(patient_counts.values())
    viable = [
        total for total in subsets
        if 0 < total and (allow_all or total < upper)
    ]
    if not viable:
        raise ValueError("At least three patient groups are required for train/validation/test")
    best_total = min(viable, key=lambda total: (abs(total - target_videos), total))
    return set(subsets[best_total])


def write_plan(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=PLAN_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def summarize_plan(rows: list[dict[str, str]]) -> dict:
    counts: dict[str, int] = {}
    patients: dict[str, set[str]] = {
        "train": set(), "validation": set(), "test": set()
    }
    for row in rows:
        key = f"{row['split']}_{row['class_name']}"
        counts[key] = counts.get(key, 0) + 1
        patients[row["split"]].add(row["patient_id"])
    overlap = sorted(
        (patients["train"] & patients["validation"])
        | (patients["train"] & patients["test"])
        | (patients["validation"] & patients["test"])
    )
    return {
        "video_counts": counts,
        "train_patients": len(patients["train"]),
        "validation_patients": len(patients["validation"]),
        "test_patients": len(patients["test"]),
        "patient_overlap": overlap,
        "patient_leakage_warning": bool(overlap),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output", default="results/facial_dataset")
    parser.add_argument("--train-fraction", type=float, default=0.7)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--split-unit",
        choices=("patient", "video"),
        default="patient",
        help="Keep patients disjoint by default; 'video' reproduces a leakage-prone video split.",
    )
    parser.add_argument("--fps", type=float, default=5.0)
    parser.add_argument("--window-seconds", type=float, default=4.0)
    parser.add_argument("--stride-seconds", type=float, default=2.0)
    parser.add_argument("--face-size", type=int, default=224)
    parser.add_argument("--max-poses", type=int, default=4)
    parser.add_argument("--head-margin", type=float, default=0.75)
    parser.add_argument("--max-head-jump", type=float, default=0.18)
    add_tuning_arguments(parser)
    add_enhancement_arguments(parser)
    add_video_enhancement_arguments(parser)
    parser.add_argument(
        "--roi-config",
        help="Optional JSON mapping patient IDs to normalized [x1,y1,x2,y2] regions.",
    )
    parser.add_argument(
        "--pose-model",
        default="checkpoints/pretrained/mediapipe/pose_landmarker_full.task",
    )
    parser.add_argument(
        "--face-model",
        default="checkpoints/pretrained/mediapipe/face_landmarker.task",
    )
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="Write and summarize the 70/30 assignment without preprocessing videos.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Process only the first N planned videos for an integration smoke test.",
    )
    args = parser.parse_args()
    validate_tuning(tuning_kwargs(args))

    output = Path(args.output).resolve()
    rows = stratified_video_plan(
        args.dataset_root,
        train_fraction=args.train_fraction,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
        split_unit=args.split_unit,
    )
    write_plan(output / "video_split.csv", rows)
    summary = summarize_plan(rows)
    (output / "split_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))

    if args.plan_only:
        return

    # Fail before expensive landmark extraction when the requested complete
    # window cannot fit inside any source clip.
    import cv2

    durations: list[float] = []
    for row in rows:
        capture = cv2.VideoCapture(row["source_video"])
        source_fps = float(capture.get(cv2.CAP_PROP_FPS))
        frame_count = float(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        capture.release()
        if source_fps > 0 and frame_count > 0:
            durations.append(frame_count / source_fps)
    eligible_videos = sum(
        duration + 1e-9 >= args.window_seconds for duration in durations
    )
    if not durations:
        raise ValueError("Could not read duration metadata from any source video")
    if eligible_videos != len(rows):
        longest = max(durations, default=0.0)
        shortest = min(durations, default=0.0)
        too_short = len(rows) - eligible_videos
        raise ValueError(
            f"--window-seconds={args.window_seconds:g} cannot fit in "
            f"{too_short} of {len(rows)} source videos (readable duration "
            f"range {shortest:.2f}-{longest:.2f} seconds). Use a window no "
            f"longer than {shortest:.2f} seconds."
        )

    roi_config: dict[str, list[float]] = {}
    if args.roi_config:
        roi_config = json.loads(Path(args.roi_config).read_text(encoding="utf-8"))
        if not isinstance(roi_config, dict):
            raise TypeError("--roi-config must contain a JSON object")

    selected = rows[: args.limit] if args.limit is not None else rows
    manifest = output / "video_manifest.csv"
    if manifest.exists():
        raise FileExistsError(
            f"Refusing to mix a new run with existing manifest: {manifest}. "
            "Choose a new --output directory or move the old results."
        )

    for index, row in enumerate(selected, start=1):
        print(
            f"[{index}/{len(selected)}] {row['split']} {row['class_name']}: "
            f"{row['source_video']}"
        )
        result = extract_video(
            row["source_video"],
            output,
            row["recording_id"],
            args.fps,
            args.window_seconds,
            pose_model_path=args.pose_model,
            face_model_path=args.face_model,
            stride_seconds=args.stride_seconds,
            patient_id=row["patient_id"],
            label=row["label"],
            split=row["split"],
            append_manifest=index > 1,
            face_size=args.face_size,
            patient_roi=tuple(roi_config.get(row["patient_id"], [0, 0, 1, 1])),
            max_poses=args.max_poses,
            head_margin=args.head_margin,
            max_head_jump=args.max_head_jump,
            **tuning_kwargs(args),
            **enhancement_kwargs(args),
            **video_enhancement_kwargs(args),
        )
        if result.windows_written == 0:
            print(f"WARNING: no complete windows written for {row['source_video']}")


if __name__ == "__main__":
    main()
