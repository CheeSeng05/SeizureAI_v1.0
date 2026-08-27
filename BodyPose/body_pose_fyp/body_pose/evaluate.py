from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader

from .build_dataset import assert_patient_disjoint
from .models.dataset import PoseWindowDataset
from .models.lstm import PoseLSTMClassifier
from .utils import get_device, load_config, save_json, window_manifest_path

LOGGER = logging.getLogger(__name__)


@torch.no_grad()
def predict(model, loader, device) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    labels, probabilities = [], []
    for x_batch, y_batch in loader:
        logits = model(x_batch.to(device))
        probabilities.append(torch.softmax(logits, dim=1).cpu().numpy())
        labels.append(y_batch.numpy())
    y_true = np.concatenate(labels)
    y_prob = np.concatenate(probabilities)
    return y_true, y_prob.argmax(axis=1), y_prob


def estimate_false_positives_per_hour(
    metadata: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    positive_classes: set[int],
) -> float:
    """Window approximation; adjacent false-positive windows form one alarm."""
    false_alarm_events = 0
    total_hours = 0.0
    evaluation = metadata.copy().reset_index(drop=True)
    evaluation["y_true"] = y_true
    evaluation["y_pred"] = y_pred

    for _, group in evaluation.sort_values(["media_id", "start_sec"]).groupby("media_id"):
        duration = max(float(group["end_sec"].max() - group["start_sec"].min()), 0.0)
        total_hours += duration / 3600.0
        false_positive = (
            group["y_pred"].isin(positive_classes)
            & ~group["y_true"].isin(positive_classes)
        ).to_numpy()
        starts = false_positive & ~np.r_[False, false_positive[:-1]]
        false_alarm_events += int(starts.sum())
    return false_alarm_events / total_hours if total_hours > 0 else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the trained body-pose classifier.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument(
        "--split", choices=["auto", "train", "validation", "test"], default="auto"
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    cfg = load_config(args.config)
    device = get_device()
    data = np.load(cfg["paths"]["dataset"], allow_pickle=False)
    x_all, y_all = data["X"], data["y"]
    metadata = pd.read_csv(window_manifest_path(cfg))
    if len(metadata) != len(x_all):
        raise ValueError("Window manifest and array lengths differ.")
    assert_patient_disjoint(metadata)

    split = args.split
    if split == "auto":
        split = "test" if (metadata["split"] == "test").any() else "validation"
        LOGGER.warning("Evaluation split selected automatically: %s", split)
    indices = metadata.index[metadata["split"] == split].to_numpy()
    if len(indices) == 0:
        raise ValueError(f"No windows exist in split '{split}'.")

    scaler = joblib.load(cfg["paths"]["scaler"])
    x = x_all[indices]
    x = scaler.transform(x.reshape(-1, x.shape[-1])).reshape(x.shape)
    y = y_all[indices]

    checkpoint = torch.load(cfg["paths"]["checkpoint"], map_location=device, weights_only=False)
    label_names = [str(name) for name in checkpoint.get("label_names", [])]
    if not label_names:
        raise ValueError("Checkpoint has no label_names; retrain with the current pipeline.")
    model = PoseLSTMClassifier(
        input_size=int(checkpoint["input_size"]),
        num_classes=int(checkpoint["num_classes"]),
        **checkpoint["model_config"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])

    loader = DataLoader(
        PoseWindowDataset(x, y),
        batch_size=int(cfg["training"]["batch_size"]),
        shuffle=False,
    )
    y_true, y_pred, y_prob = predict(model, loader, device)
    labels = list(range(len(label_names)))
    positive_classes = {index for index, name in enumerate(label_names) if name != "normal"}

    metrics = {
        "split": split,
        "n_windows": int(len(y_true)),
        "n_patients": int(metadata.iloc[indices]["patient_id"].nunique()),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "macro_precision": float(
            precision_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "false_positives_per_hour_approx": float(
            estimate_false_positives_per_hour(
                metadata.iloc[indices], y_true, y_pred, positive_classes
            )
        ),
    }
    try:
        if len(labels) == 2:
            metrics["macro_ovr_auroc"] = float(roc_auc_score(y_true, y_prob[:, 1]))
        else:
            metrics["macro_ovr_auroc"] = float(
                roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro", labels=labels)
            )
    except ValueError as exc:
        metrics["macro_ovr_auroc"] = None
        LOGGER.warning("AUROC unavailable: %s", exc)

    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=label_names,
        output_dict=True,
        zero_division=0,
    )
    results_dir = Path(cfg["paths"]["results"])
    results_dir.mkdir(parents=True, exist_ok=True)
    save_json(metrics, results_dir / f"{split}_metrics.json")
    with (results_dir / f"{split}_classification_report.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(report, handle, indent=2)

    predictions = metadata.iloc[indices].copy().reset_index(drop=True)
    predictions["true_label"] = [label_names[int(index)] for index in y_true]
    predictions["predicted_label"] = [label_names[int(index)] for index in y_pred]
    for class_id, class_name in enumerate(label_names):
        predictions[f"probability_{class_name}"] = y_prob[:, class_id]
    predictions.to_csv(results_dir / f"{split}_predictions.csv", index=False)

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    ConfusionMatrixDisplay(cm, display_labels=label_names).plot(values_format="d")
    plt.title(f"Confusion Matrix - {split}")
    plt.tight_layout()
    plt.savefig(results_dir / f"{split}_confusion_matrix.png", dpi=180)
    plt.close()

    for class_id, class_name in enumerate(label_names):
        binary_truth = (y_true == class_id).astype(int)
        if len(np.unique(binary_truth)) < 2:
            continue
        RocCurveDisplay.from_predictions(
            binary_truth, y_prob[:, class_id], name=class_name
        )
        plt.title(f"One-vs-Rest ROC - {class_name}")
        plt.tight_layout()
        plt.savefig(results_dir / f"{split}_roc_{class_name}.png", dpi=180)
        plt.close()

    LOGGER.info("Metrics:\n%s", json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
