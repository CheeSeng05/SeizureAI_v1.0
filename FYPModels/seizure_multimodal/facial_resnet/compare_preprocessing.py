"""Compare facial preprocessing settings, changing one setting from baseline per run."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from .face_preprocess import extract_video
from .preprocess_options import BASELINE_TUNING, CONFIDENCE_OPTIONS, validate_tuning


def build_experiments(
    thresholds: tuple[float, ...] = (0.6, 0.7),
    smoothing: tuple[float, ...] = (0.3, 0.5),
    reuse_seconds: tuple[float, ...] = (0.4, 0.6, 1.0),
) -> list[dict[str, Any]]:
    experiments = [{"name": "baseline", "changed_parameter": "none", "value": None,
                    "tuning": dict(BASELINE_TUNING)}]
    candidates = [(name, thresholds) for name in CONFIDENCE_OPTIONS]
    candidates.extend((("box_smoothing", smoothing), ("max_box_reuse_seconds", reuse_seconds)))
    for parameter, values in candidates:
        for value in dict.fromkeys(values):
            if value == BASELINE_TUNING[parameter]:
                continue
            tuning = {**BASELINE_TUNING, parameter: value}
            validate_tuning(tuning)
            experiments.append({
                "name": f"{parameter}_{value:g}".replace(".", "p"),
                "changed_parameter": parameter, "value": value, "tuning": tuning,
            })
    return experiments


def comparison_row(experiment: dict[str, Any], video: Path,
                   metrics: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    row = {"experiment": experiment["name"], "source_video": str(video),
           "changed_parameter": experiment["changed_parameter"], "value": experiment["value"],
           **metrics}
    for field in ("face_landmark_fraction", "face_region_fraction",
                  "face_box_reused_fraction", "mean_box_center_step_fraction",
                  "mean_box_center_offset_fraction"):
        value, reference = metrics[field], baseline[field]
        row[f"delta_{field}"] = value - reference if value is not None and reference is not None else None
    return row


def write_comparison(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_comparison(videos: list[Path], output: Path, experiments: list[dict[str, Any]],
                   common: dict[str, Any], *, plan_only: bool = False) -> list[dict[str, Any]]:
    videos = [path.resolve() for path in videos]
    if not videos or len(set(videos)) != len(videos):
        raise ValueError("Supply at least one video, without duplicate paths")
    for path in videos:
        if not path.is_file():
            raise FileNotFoundError(path)
    for experiment in experiments:
        validate_tuning(experiment["tuning"])
    models = {}
    for name in ("pose_model_path", "face_model_path"):
        path = Path(common[name]).resolve()
        models[name] = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / "plan.json").write_text(json.dumps({
        "videos": [str(path) for path in videos], "models": models,
        "common_settings": common, "experiments": experiments,
        "metrics_note": "Each video frame counted once, not once per overlapping window. Deltas compare the same video to baseline. No automatic accuracy ranking.",
    }, indent=2, default=str, allow_nan=False) + "\n", encoding="utf-8")
    if plan_only:
        print(f"Planned {len(experiments)} configurations × {len(videos)} videos in {output / 'plan.json'}")
        return []

    rows = []
    baselines: dict[Path, dict[str, Any]] = {}
    total = len(experiments) * len(videos)
    for experiment_index, experiment in enumerate(experiments):
        for video_index, video in enumerate(videos):
            number = experiment_index * len(videos) + video_index + 1
            print(f"[{number}/{total}] {experiment['name']}: {video.name}", flush=True)
            summary = extract_video(
                video, output / experiment["name"], f"video_{video_index + 1:03d}",
                split="unassigned", diagnostics_only=True,
                **common, **experiment["tuning"],
            )
            report = json.loads(Path(summary.diagnostics_path).read_text(encoding="utf-8"))
            metrics = report["metrics"]
            if metrics["sampled_frames"] == 0:
                raise ValueError(f"No frames decoded from {video}")
            if experiment["name"] == "baseline":
                baselines[video] = metrics
            rows.append(comparison_row(experiment, video, metrics, baselines[video]))
            rows[-1]["diagnostics_path"] = str(Path(summary.diagnostics_path).resolve())
            write_comparison(output / "comparison.csv", rows)
    print(f"Comparison saved to {output / 'comparison.csv'}", flush=True)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("videos", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path,
                        help="New output directory; existing directories are never overwritten.")
    parser.add_argument("--thresholds", nargs="*", type=float, default=[0.6, 0.7])
    parser.add_argument("--smoothing-values", nargs="*", type=float, default=[0.3, 0.5])
    parser.add_argument("--reuse-seconds", nargs="*", type=float, default=[0.4, 0.6, 1.0])
    parser.add_argument("--fps", type=float, default=5.0)
    parser.add_argument("--max-duration-seconds", type=float,
                        help="Use only the first N seconds of each input video.")
    parser.add_argument("--pose-model", default="checkpoints/pretrained/mediapipe/pose_landmarker_full.task")
    parser.add_argument("--face-model", default="checkpoints/pretrained/mediapipe/face_landmarker.task")
    parser.add_argument("--patient-roi", default="0,0,1,1",
                        help="Common normalized x1,y1,x2,y2 region for these videos.")
    parser.add_argument("--max-poses", type=int, default=4)
    parser.add_argument("--max-head-jump", type=float, default=0.18)
    parser.add_argument("--head-margin", type=float, default=0.75)
    parser.add_argument("--face-margin", type=float, default=0.25)
    parser.add_argument("--face-size", type=int, default=224)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    try:
        roi = tuple(float(value) for value in args.patient_roi.split(","))
        if len(roi) != 4 or not (0 <= roi[0] < roi[2] <= 1 and 0 <= roi[1] < roi[3] <= 1):
            raise ValueError("patient-roi must be normalized x1,y1,x2,y2 coordinates")
        if not math.isfinite(args.fps) or args.fps <= 0:
            raise ValueError("fps must be finite and positive")
        if args.max_duration_seconds is not None and (
            not math.isfinite(args.max_duration_seconds) or args.max_duration_seconds <= 0
        ):
            raise ValueError("max-duration-seconds must be finite and positive")
        experiments = build_experiments(tuple(args.thresholds), tuple(args.smoothing_values), tuple(args.reuse_seconds))
        run_comparison(args.videos, args.output, experiments, common={
            "fps": args.fps, "max_duration_seconds": args.max_duration_seconds,
            "pose_model_path": str(Path(args.pose_model).resolve()),
            "face_model_path": str(Path(args.face_model).resolve()), "patient_roi": roi,
            "max_poses": args.max_poses, "max_head_jump": args.max_head_jump,
            "head_margin": args.head_margin, "face_margin": args.face_margin,
            "face_size": args.face_size,
        }, plan_only=args.plan_only)
    except ValueError as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
