"""Working out where a run has got to.

The model CLI prints a startup warning and then stays silent until it finishes,
so there is no progress stream to read. What follows infers a stage from three
signals - the log lines that do appear, files landing in the results directory,
and elapsed time against previous runs.

Everything here is explicitly labelled as measured or estimated. A progress bar
that invents certainty is worse than no progress bar, because a clinician
waiting on it has no way to tell a slow run from a stuck one.
"""
from __future__ import annotations

import re
import statistics
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# Ordered pipeline stages. `key` is matched against log lines.
STAGES: list[tuple[str, str, tuple[str, ...]]] = [
    ("staged", "Recording staged", ()),
    ("starting", "Starting inference", ()),
    ("loading", "Loading model checkpoints", (
        "timm", "registry", "loading", "checkpoint", "state_dict", "cuda", "device",
    )),
    ("inspecting", "Inspecting data and labels", (
        "label.xlsx", "workbook", "inspect", "provenance", "sha256", "discovery",
    )),
    ("pose", "Extracting pose keypoints", (
        "pose", "openpose", "keypoint", "frame", "decoding",
    )),
    ("inference", "Running VSViG inference", (
        "clip", "batch", "window", "inference", "forward", "predict",
    )),
    ("writing", "Writing results", (
        "writing", "wrote", "saved", "results/", "metrics.json", "timeline",
    )),
    ("done", "Complete", ()),
]

STAGE_ORDER = {key: i for i, (key, _, _) in enumerate(STAGES)}
STAGE_LABEL = {key: label for key, label, _ in STAGES}

# "45%", "120/836", "frame 3000 of 25081" — whatever the CLI happens to emit.
PERCENT = re.compile(r"(\d{1,3})\s*%")
FRACTION = re.compile(r"\b(\d+)\s*/\s*(\d+)\b")
OF_FORM = re.compile(r"\b(\d+)\s+of\s+(\d+)\b", re.I)


@dataclass
class Progress:
    stage: str
    stage_label: str
    stage_index: int
    stage_count: int
    fraction: Optional[float]        # None when genuinely unknown
    fraction_is_estimate: bool
    elapsed_sec: float
    eta_sec: Optional[float]
    seconds_since_output: Optional[float]
    detail: Optional[str]
    stalled: bool


def _parse_at(line: str) -> Optional[float]:
    """Pull an explicit completion fraction out of a log line, if there is one."""
    m = PERCENT.search(line)
    if m:
        v = int(m.group(1))
        if 0 <= v <= 100:
            return v / 100.0
    for pattern in (FRACTION, OF_FORM):
        m = pattern.search(line)
        if m:
            done, total = int(m.group(1)), int(m.group(2))
            if 0 < total and done <= total:
                return done / total
    return None


def infer_stage(log: list[str], finished: bool) -> tuple[str, Optional[str]]:
    """Latest stage any log line indicates. Never goes backwards."""
    if finished:
        return "done", None
    best, detail = "starting", None
    for line in log:
        low = line.lower()
        for key, _, keywords in STAGES:
            if keywords and any(k in low for k in keywords):
                if STAGE_ORDER[key] >= STAGE_ORDER[best]:
                    best, detail = key, line.strip()
    return best, detail


def historical_duration(job_dir: Path) -> Optional[float]:
    """Median wall-clock of previous successful runs, for an ETA.

    Median rather than mean: one interrupted or unusually short run should not
    drag the estimate.
    """
    import json
    durations = []
    for p in job_dir.glob("*.json"):
        try:
            d = json.loads(p.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        if d.get("status") != "succeeded" or not d.get("started_at") or not d.get("finished_at"):
            continue
        try:
            start = datetime.fromisoformat(d["started_at"])
            end = datetime.fromisoformat(d["finished_at"])
        except ValueError:
            continue
        seconds = (end - start).total_seconds()
        if seconds > 5:                      # ignore instant failures
            durations.append(seconds)
    return statistics.median(durations) if durations else None


def compute(
    *,
    status: str,
    started_at: Optional[str],
    finished_at: Optional[str],
    log: list[str],
    last_output_at: Optional[str],
    job_dir: Path,
    stall_after_sec: float = 3600.0,
) -> Progress:
    """stall_after_sec is deliberately generous. This model emits nothing
    between loading and completion, so silence is the normal state - flagging
    it early would train the user to ignore a warning that occasionally
    matters."""
    now = datetime.now(timezone.utc)
    finished = status in ("succeeded", "failed", "cancelled")

    def _elapsed(from_iso: Optional[str], to_iso: Optional[str]) -> float:
        if not from_iso:
            return 0.0
        start = datetime.fromisoformat(from_iso)
        end = datetime.fromisoformat(to_iso) if to_iso else now
        return max(0.0, (end - start).total_seconds())

    elapsed = _elapsed(started_at, finished_at)
    since_output = _elapsed(last_output_at, None) if last_output_at and not finished else None

    stage, detail = infer_stage(log, finished)

    # Prefer a fraction the model actually reported.
    reported = None
    for line in reversed(log[-40:]):
        reported = _parse_at(line)
        if reported is not None:
            break

    if finished:
        fraction, estimate = (1.0, False)
        eta = 0.0
    elif reported is not None:
        fraction, estimate = reported, False
        eta = (elapsed / reported - elapsed) if reported > 0.02 else None
    else:
        typical = historical_duration(job_dir)
        if typical:
            fraction = min(0.97, elapsed / typical)   # never claim 100% early
            estimate = True
            eta = max(0.0, typical - elapsed)
        else:
            fraction, estimate, eta = None, True, None

    return Progress(
        stage=stage,
        stage_label=STAGE_LABEL[stage],
        stage_index=STAGE_ORDER[stage],
        stage_count=len(STAGES) - 1,
        fraction=fraction,
        fraction_is_estimate=estimate,
        elapsed_sec=round(elapsed, 1),
        eta_sec=round(eta, 1) if eta is not None else None,
        seconds_since_output=round(since_output, 1) if since_output is not None else None,
        detail=detail,
        stalled=bool(since_output and since_output > stall_after_sec),
    )