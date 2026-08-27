from __future__ import annotations

import logging
import re
from pathlib import Path

import pandas as pd

from .base import AnnotationAdapter

LOGGER = logging.getLogger(__name__)
EXPECTED_COLUMNS = {
    "onset",
    "duration",
    "eventType",
    "lateralization",
    "localization",
    "vigilance",
    "confidence",
    "channels",
    "dateTime",
    "recordingDuration",
}


def _bids_entity(path: Path, prefix: str) -> str | None:
    match = re.search(rf"(?:^|_){re.escape(prefix)}-([^_]+)", path.name)
    return f"{prefix}-{match.group(1)}" if match else None


class SeizeIT2Adapter(AnnotationAdapter):
    """Read ds005873 v1.1.0 BIDS-SCORE event tables, never sensor samples."""

    def read(self) -> pd.DataFrame:
        if not self.source.exists():
            raise FileNotFoundError(
                f"SeizeIT2 root not found: {self.source}. Downloading the full dataset is not required; "
                "a metadata-only/DataLad checkout containing *_events.tsv files is sufficient."
            )
        pattern = str(self.config.get("events_glob", "**/eeg/*_events.tsv"))
        paths = sorted(self.source.glob(pattern))
        subjects = {
            str(value) if str(value).startswith("sub-") else f"sub-{value}"
            for value in self.config.get("subjects", [])
        }
        sessions = {
            str(value) if str(value).startswith("ses-") else f"ses-{value}"
            for value in self.config.get("sessions", [])
        }
        if subjects:
            paths = [path for path in paths if _bids_entity(path, "sub") in subjects]
        if sessions:
            paths = [path for path in paths if _bids_entity(path, "ses") in sessions]
        if not paths:
            raise FileNotFoundError(
                f"No SeizeIT2 *_events.tsv annotation files matched '{pattern}' under {self.source}."
            )

        outputs: list[pd.DataFrame] = []
        excluded_counts: dict[str, int] = {}
        for path in paths:
            raw = pd.read_csv(path, sep="\t", na_values=["n/a", "N/A", "NA", ""])
            missing = EXPECTED_COLUMNS - set(raw.columns)
            if missing:
                raise ValueError(f"{path} is missing SeizeIT2 columns: {sorted(missing)}")

            event_type = raw["eventType"].astype("string").str.strip().str.lower()
            labels = pd.Series(pd.NA, index=raw.index, dtype="string")
            labels.loc[event_type.eq("bckg")] = "normal"
            labels.loc[event_type.str.startswith("sz", na=False)] = "ictal"

            excluded = labels.isna()
            for value, count in event_type[excluded].value_counts(dropna=False).items():
                key = "missing" if pd.isna(value) else str(value)
                excluded_counts[key] = excluded_counts.get(key, 0) + int(count)
            keep = ~excluded
            if not keep.any():
                continue

            selected = raw.loc[keep].reset_index(drop=True)
            selected_labels = labels.loc[keep].reset_index(drop=True)
            onset = pd.to_numeric(selected["onset"], errors="raise")
            duration = pd.to_numeric(selected["duration"], errors="raise")
            recording_id = path.stem.removesuffix("_events")
            patient_id = _bids_entity(path, "sub")
            if patient_id is None:
                raise ValueError(f"Could not parse BIDS subject entity from {path.name}")

            relative_source = str(path.relative_to(self.source))
            frame = pd.DataFrame({
                "dataset": "ds005873",
                "media_id": recording_id,
                "recording_id": recording_id,
                "patient_id": patient_id,
                "start_sec": onset,
                "end_sec": onset + duration,
                "label": selected_labels,
                "source_event_type": selected["eventType"],
                "lateralization": selected["lateralization"],
                "localization": selected["localization"],
                "confidence": pd.to_numeric(selected["confidence"], errors="coerce"),
                "annotation_source": relative_source,
                "derived_label": selected_labels,
                "session_id": _bids_entity(path, "ses"),
                "vigilance": selected["vigilance"],
                "channels": selected["channels"],
                "date_time": selected["dateTime"],
                "recording_duration_sec": pd.to_numeric(
                    selected["recordingDuration"], errors="coerce"
                ),
                "source_tsv_path": relative_source,
            })
            outputs.append(frame)

        if excluded_counts:
            LOGGER.info("Excluded SeizeIT2 event types: %s", excluded_counts)
        self.excluded_counts = excluded_counts
        if not outputs:
            raise RuntimeError(
                "SeizeIT2 event files were found, but no bckg or sz* annotation rows were retained."
            )
        return pd.concat(outputs, ignore_index=True)
