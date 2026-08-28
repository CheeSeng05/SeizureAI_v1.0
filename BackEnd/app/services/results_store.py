"""Reads the model's results/ directory. Nothing else touches the filesystem.

The model writes files; this module is the only thing that knows their layout.
When your teammate reruns inference, the API serves the new numbers with no
code change and nothing to re-embed.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Optional

import pandas as pd

# results/
#   summary.csv
#   data_inspection.json
#   video_only/
#     <rec>_probability_timeline.csv
#     <rec>_metrics.json
#   eeg_only/ | strict_multimodal/ | simulated_multimodal/


class ResultsStore:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.video_dir = self.root / "video_only"

    # -- discovery ---------------------------------------------------------
    def recording_ids(self) -> list[str]:
        if not self.video_dir.is_dir():
            return []
        return sorted(p.name.replace("_probability_timeline.csv", "")
                      for p in self.video_dir.glob("*_probability_timeline.csv"))

    def _timeline_path(self, rec: str) -> Path:
        return self.video_dir / f"{rec}_probability_timeline.csv"

    def _metrics_path(self, rec: str) -> Path:
        return self.video_dir / f"{rec}_metrics.json"

    # -- reads -------------------------------------------------------------
    @lru_cache(maxsize=32)
    def timeline(self, rec: str) -> pd.DataFrame:
        path = self._timeline_path(rec)
        if not path.exists():
            raise FileNotFoundError(f"no timeline for {rec}")
        return pd.read_csv(path)

    @lru_cache(maxsize=32)
    def metrics_json(self, rec: str) -> Optional[dict]:
        path = self._metrics_path(rec)
        return json.loads(path.read_text()) if path.exists() else None

    def summary(self) -> list[dict]:
        path = self.root / "summary.csv"
        if not path.exists():
            return []
        return pd.read_csv(path).to_dict("records")

    def data_inspection(self) -> Optional[dict]:
        path = self.root / "data_inspection.json"
        return json.loads(path.read_text()) if path.exists() else None

    # -- assembled views ---------------------------------------------------
    def recording(self, rec: str) -> dict:
        """Recording metadata, preferring metrics.json and falling back to the
        timeline's own columns so a run without metrics still displays."""
        df = self.timeline(rec)
        first = df.iloc[0]
        mj = self.metrics_json(rec) or {}
        run = mj.get("run", {})
        vid = run.get("video", {})
        ann = run.get("annotation", {})

        window_sec = float(first.end_sec - first.start_sec)
        return {
            "recording_id": rec,
            "patient_id": ann.get("patient_id") or str(first.get("recording_id", rec)).split("_")[0],
            "seizure_id": ann.get("seizure_id", ""),
            "seizure_type": ann.get("seizure_type", ""),
            "source_file": Path(vid.get("path", rec)).name,
            "duration_sec": float(vid.get("duration_sec", df.end_sec.max())),
            "fps": float(vid.get("fps", 0.0)),
            "frame_count": int(vid.get("frame_count", 0)),
            "width": int(vid.get("width", 0)),
            "height": int(vid.get("height", 0)),
            "eeg_onset_sec": _f(first.get("eeg_onset_sec")),
            "clinical_onset_sec": _f(first.get("clinical_onset_sec")),
            "seizure_offset_sec": _f(ann.get("seizure_end_sec")),
            "alignment_status": str(first.get("annotation_alignment_status", "unknown")),
            "window_count": int(len(df)),
            "window_sec": window_sec,
        }

    def provenance(self, rec: str) -> dict:
        df = self.timeline(rec)
        first = df.iloc[0]
        mj = self.metrics_json(rec) or {}
        return {
            "test_mode": str(first.get("test_mode", "unknown")),
            "video_model_used": bool(first.get("video_model_used", False)),
            "eeg_model_used": bool(first.get("EEG_model_used", False)),
            "fusion_used": bool(first.get("fusion_used", False)),
            "pairing_type": str(first.get("pairing_type", "none")),
            "training_performed": bool(first.get("training_performed", False)),
            "video_checkpoint": _s(first.get("video_checkpoint")),
            "eeg_checkpoint": _s(first.get("EEG_checkpoint")),
            "label_policy": (mj.get("metrics") or {}).get("classification_label_policy"),
        }


def _f(v):
    try:
        return None if v is None or pd.isna(v) else float(v)
    except (TypeError, ValueError):
        return None


def _s(v):
    return None if v is None or (isinstance(v, float) and pd.isna(v)) else str(v)
