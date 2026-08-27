from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Mapping


MODELS = ("EEG only", "VSViG only", "Late Fusion", "Feature Fusion")
METRICS = (
    "Sensitivity",
    "Specificity",
    "AUROC",
    "F1",
    "FP/h",
    "Onset MAE",
    "Offset MAE",
    "Cohen κ",
)


def comparison_rows(
    results: Mapping[str, Mapping[str, Any]], *, paired_real_veeg_verified: bool
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model in MODELS:
        if model in {"Late Fusion", "Feature Fusion"} and not paired_real_veeg_verified:
            values: Mapping[str, Any] = {}
        else:
            values = results.get(model, {})
        row = {"Model": model}
        row.update({metric: values.get(metric, "N/A") for metric in METRICS})
        rows.append(row)
    return rows


def markdown_table(rows: list[Mapping[str, Any]]) -> str:
    headers = ("Model",) + METRICS
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(header, "N/A")) for header in headers) + " |")
    return "\n".join(lines)


def write_report_bundle(
    *,
    output_directory: str | Path,
    results: Mapping[str, Mapping[str, Any]],
    metadata: Mapping[str, Any],
    paired_real_veeg_verified: bool,
) -> None:
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    rows = comparison_rows(results, paired_real_veeg_verified=paired_real_veeg_verified)
    with (output / "comparison.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["Model", *METRICS])
        writer.writeheader()
        writer.writerows(rows)
    (output / "comparison.md").write_text(markdown_table(rows) + "\n", encoding="utf-8")
    payload = {
        "paired_real_veeg_verified": paired_real_veeg_verified,
        "real_multimodal_notice": None
        if paired_real_veeg_verified
        else (
            "Real multimodal fusion evaluation was not performed because synchronized paired "
            "video and EEG from the same recordings were not available."
        ),
        "metadata": dict(metadata),
        "results": rows,
    }
    (output / "report.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
