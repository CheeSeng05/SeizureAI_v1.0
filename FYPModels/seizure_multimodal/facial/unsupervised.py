"""Train, score, and independently evaluate a label-free facial movement detector."""
from __future__ import annotations

import argparse
import csv
import json
import uuid
from pathlib import Path
from typing import Any, Mapping, Sequence

import joblib
import numpy as np
import sklearn
from sklearn.ensemble import IsolationForest

from .anomaly_evaluation import DEFAULT_EVENTS, evaluate_anomalies, summarize_anomalies, validate_events
from .movement_features import (
    DEFAULT_QUALITY, FEATURE_NAMES, load_features, read_manifest, validate_manifest, validate_quality,
)


FORMAT_VERSION = 1
SCORE_FIELDS = (
    "detector_id", "window_id", "patient_id", "recording_id", "split", "start_sec", "end_sec",
    "status", "exclusion_reason", "anomaly_score", "candidate_window",
)


def anomaly_scores(model: IsolationForest, features: np.ndarray) -> np.ndarray:
    """Negate sklearn's normality score: higher is more anomalous, in [0, 1]."""
    scores = -model.score_samples(features)
    if not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
        raise ValueError("Unexpected Isolation Forest score range")
    return scores


def assign_scores(records: list[dict[str, Any]], scores: np.ndarray, threshold: float) -> None:
    scored = [row for row in records if row["status"] == "scored"]
    if len(scored) != len(scores):
        raise ValueError("Scores and usable windows must be aligned")
    for row, score in zip(scored, scores):
        row.update(anomaly_score=float(score), candidate_window=int(score >= threshold))


def train_detector(
    manifest_path: str | Path, *, validation_quantile: float = 0.99,
    n_estimators: int = 200, seed: int = 42,
    quality: Mapping[str, float] = DEFAULT_QUALITY,
    event_settings: Mapping[str, float] = DEFAULT_EVENTS,
) -> dict[str, Any]:
    """Fit only train arrays; choose a threshold using unlabelled validation arrays.

    Labels are never read. Test identities are checked for leakage, but test
    arrays are not opened. The quantile controls a validation tail, not a seizure
    prevalence estimate or a guaranteed false alarm rate.
    """
    if not np.isfinite(validation_quantile) or not 0 < validation_quantile < 1:
        raise ValueError("validation_quantile must be in (0, 1)")
    if n_estimators < 1:
        raise ValueError("n_estimators must be positive")
    validate_quality(quality)
    validate_events(event_settings)
    rows = read_manifest(manifest_path)
    partitions = {split: [row for row in rows if row["split"] == split]
                  for split in ("train", "validation")}
    for split, partition in partitions.items():
        if not partition:
            raise ValueError(f"Manifest needs {split} patients; preprocess that split first")
    training_records, training = load_features(manifest_path, partitions["train"], quality=quality)
    validation_records, validation = load_features(manifest_path, partitions["validation"], quality=quality)
    if len(training) < 2 or not len(validation):
        raise ValueError("Need at least two usable training windows and one usable validation window")
    if not np.any(np.ptp(training, axis=0) > 0):
        raise ValueError("Training features are constant; more varied usable recordings are required")
    # Tree splits do not require feature scaling. No imputation: invalid windows
    # remain excluded instead of generating synthetic motion observations.
    model = IsolationForest(n_estimators=n_estimators, contamination="auto", random_state=seed, n_jobs=1)
    model.fit(training)
    validation_scores = anomaly_scores(model, validation)
    # Strictly above the empirical quantile: tied/constant scores should not flag
    # every validation window. Prediction and event utilities use >= consistently.
    threshold = float(np.nextafter(np.quantile(validation_scores, validation_quantile), np.inf))
    assign_scores(training_records, anomaly_scores(model, training), threshold)
    assign_scores(validation_records, validation_scores, threshold)
    reports = {split: summarize_anomalies(records, threshold=threshold, event_settings=event_settings)
               for split, records in (("train", training_records), ("validation", validation_records))}
    for split, records in (("train", training_records), ("validation", validation_records)):
        reports[split]["excluded_records"] = [row for row in records if row["status"] == "excluded"]
    return {
        "format_version": FORMAT_VERSION, "model_type": "facial_movement_isolation_forest",
        "detector_id": uuid.uuid4().hex,
        "estimator": model, "feature_names": list(FEATURE_NAMES),
        "quality": dict(quality), "event_settings": dict(event_settings),
        "anomaly_threshold": threshold, "validation_quantile": validation_quantile,
        "threshold_source": "strictly above unlabelled validation score quantile",
        "training_patient_ids": sorted({row["patient_id"] for row in partitions["train"]}),
        "validation_patient_ids": sorted({row["patient_id"] for row in partitions["validation"]}),
        "training_manifest": str(Path(manifest_path).resolve()),
        "sklearn_version": sklearn.__version__, "seed": seed,
        "reports": reports,
    }


def save_detector(checkpoint: Mapping[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(dict(checkpoint), path)
    report = {key: value for key, value in checkpoint.items() if key != "estimator"}
    path.with_suffix(path.suffix + ".json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8",
    )


def load_detector(path: str | Path) -> dict[str, Any]:
    """Load a trusted local joblib checkpoint created by this command."""
    checkpoint = joblib.load(path)
    if (checkpoint.get("format_version") != FORMAT_VERSION
            or checkpoint.get("model_type") != "facial_movement_isolation_forest"
            or not checkpoint.get("detector_id")
            or checkpoint.get("feature_names") != list(FEATURE_NAMES)
            or not isinstance(checkpoint.get("estimator"), IsolationForest)):
        raise ValueError("Incompatible facial anomaly checkpoint")
    validate_quality(checkpoint["quality"])
    validate_events(checkpoint["event_settings"])
    return checkpoint


def check_unseen_patients(records: Sequence[Mapping[str, Any]], checkpoint: Mapping[str, Any]) -> None:
    seen = set(checkpoint["training_patient_ids"]) | set(checkpoint["validation_patient_ids"])
    if seen & {row["patient_id"] for row in records}:
        raise ValueError("Prediction/evaluation patients overlap training or validation patients")


def predict_manifest(
    manifest_path: str | Path, checkpoint: Mapping[str, Any], *, split: str = "test",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if split not in {"test", "unassigned"}:
        raise ValueError("Prediction supports test or unassigned patients")
    rows = [row for row in read_manifest(manifest_path) if row["split"] == split]
    if not rows:
        raise ValueError(f"No windows for split={split!r}")
    check_unseen_patients(rows, checkpoint)
    records, features = load_features(manifest_path, rows, quality=checkpoint["quality"])
    for row in records:
        row["detector_id"] = checkpoint["detector_id"]
    scores = anomaly_scores(checkpoint["estimator"], features) if len(features) else np.empty(0)
    assign_scores(records, scores, checkpoint["anomaly_threshold"])
    summary = summarize_anomalies(records, threshold=checkpoint["anomaly_threshold"],
                                  event_settings=checkpoint["event_settings"])
    summary.update(split=split, detector_id=checkpoint["detector_id"], threshold_source=checkpoint["threshold_source"])
    return records, summary


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train", help="Fit without labels; select threshold on validation patients")
    train.add_argument("--manifest", required=True)
    train.add_argument("--checkpoint", required=True)
    train.add_argument("--validation-quantile", type=float, default=.99)
    train.add_argument("--trees", type=int, default=200)
    train.add_argument("--seed", type=int, default=42)
    for name, default in {**DEFAULT_QUALITY, **DEFAULT_EVENTS}.items():
        train.add_argument("--" + name.replace("_", "-"), type=float, default=default)
    predict = commands.add_parser("predict", help="Score unseen patients without labels")
    predict.add_argument("--manifest", required=True)
    predict.add_argument("--checkpoint", required=True)
    predict.add_argument("--split", choices=("test", "unassigned"), default="test")
    predict.add_argument("--output", required=True)
    evaluate = commands.add_parser("evaluate", help="Evaluate frozen test predictions against independent annotations")
    evaluate.add_argument("--predictions", required=True, help="anomaly_scores.csv from predict")
    evaluate.add_argument("--checkpoint", required=True)
    evaluate.add_argument("--output", required=True)
    references = evaluate.add_mutually_exclusive_group(required=True)
    references.add_argument("--labels", help="CSV: window_id,label (blank, 0, or 1)")
    references.add_argument("--annotations", help="Fully reviewed event/monitoring CSV; see facial README")
    args = parser.parse_args()
    if args.command == "train":
        checkpoint = train_detector(
            args.manifest, validation_quantile=args.validation_quantile, n_estimators=args.trees,
            seed=args.seed, quality={key: getattr(args, key) for key in DEFAULT_QUALITY},
            event_settings={key: getattr(args, key) for key in DEFAULT_EVENTS},
        )
        save_detector(checkpoint, args.checkpoint)
        print(f"Saved label-free detector: {args.checkpoint}")
        print(f"Validation-selected anomaly threshold: {checkpoint['anomaly_threshold']:.6f}")
    elif args.command == "predict":
        checkpoint = load_detector(args.checkpoint)
        records, summary = predict_manifest(args.manifest, checkpoint, split=args.split)
        output = Path(args.output)
        output.mkdir(parents=True, exist_ok=True)
        write_csv(output / "anomaly_scores.csv", records, SCORE_FIELDS)
        write_csv(output / "candidate_events.csv", summary["candidate_events"],
                  ("patient_id", "recording_id", "start_sec", "end_sec", "score"))
        (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(f"Scored {summary['scored_windows']} windows; excluded {summary['excluded_windows']}. Results: {output}")
    else:
        checkpoint = load_detector(args.checkpoint)
        records = read_csv(args.predictions)
        if not records or any(row.get("detector_id") != checkpoint["detector_id"] for row in records):
            raise ValueError("Predictions were not generated by this detector checkpoint")
        validate_manifest(records)
        check_unseen_patients(records, checkpoint)
        result = evaluate_anomalies(
            records, threshold=checkpoint["anomaly_threshold"], event_settings=checkpoint["event_settings"],
            labels=read_csv(args.labels) if args.labels else None,
            annotations=read_csv(args.annotations) if args.annotations else None,
        )
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(f"Saved independent reference evaluation: {output}")


if __name__ == "__main__":
    main()
