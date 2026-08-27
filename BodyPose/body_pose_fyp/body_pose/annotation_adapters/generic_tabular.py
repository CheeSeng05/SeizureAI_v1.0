from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .base import REQUIRED_CANONICAL_COLUMNS, AnnotationAdapter


def _read_table(path: Path, config: dict[str, Any]) -> pd.DataFrame:
    extension = path.suffix.lower()
    if extension == ".csv":
        return pd.read_csv(path, **config.get("read_options", {}))
    if extension in {".tsv", ".txt"}:
        options = {"sep": "\t", **config.get("read_options", {})}
        return pd.read_csv(path, **options)
    raise ValueError(f"Unsupported annotation table '{path}'. Use CSV or TSV.")


class GenericTabularAdapter(AnnotationAdapter):
    """Configurable CSV/TSV adapter for future HUKM annotation exports."""

    def _paths(self) -> list[Path]:
        if self.source.is_file():
            return [self.source]
        if self.source.is_dir():
            paths = sorted(
                path
                for path in self.source.rglob("*")
                if path.is_file() and path.suffix.lower() in {".csv", ".tsv", ".txt"}
            )
            if paths:
                return paths
        raise FileNotFoundError(f"No CSV/TSV annotations found at {self.source}")

    def _fps(self, raw: pd.DataFrame) -> float | pd.Series:
        fps_config = self.config.get("fps")
        if fps_config is None:
            raise ValueError("time_unit 'frames' requires annotation_adapter.fps.")
        if isinstance(fps_config, str) and fps_config in raw.columns:
            fps = pd.to_numeric(raw[fps_config], errors="raise")
            if (fps <= 0).any():
                raise ValueError("All per-row FPS values must be positive.")
            return fps
        fps = float(fps_config)
        if fps <= 0:
            raise ValueError("annotation_adapter.fps must be positive.")
        return fps

    def _convert_times(self, frame: pd.DataFrame, raw: pd.DataFrame) -> None:
        unit = str(self.config.get("time_unit", "seconds")).strip().lower()
        start = pd.to_numeric(frame["start_sec"], errors="raise")
        end = pd.to_numeric(frame["end_sec"], errors="raise")
        if unit in {"second", "seconds", "sec", "s"}:
            divisor: float | pd.Series = 1.0
        elif unit in {"millisecond", "milliseconds", "ms"}:
            divisor = 1000.0
        elif unit in {"frame", "frames"}:
            divisor = self._fps(raw)
        else:
            raise ValueError("time_unit must be seconds, milliseconds, or frames.")
        frame["start_sec"] = start / divisor
        frame["end_sec"] = end / divisor

    def _map_labels(self, labels: pd.Series) -> pd.Series:
        mapping = {
            str(source).strip().casefold(): str(target).strip().lower()
            for source, target in self.config.get("label_map", {}).items()
        }
        raw_labels = labels.astype("string").str.strip()
        if not mapping:
            return raw_labels.str.lower()
        mapped = raw_labels.map(lambda value: mapping.get(str(value).casefold(), pd.NA))
        if mapped.isna().any():
            unknown = sorted(set(raw_labels[mapped.isna()].dropna().astype(str)))
            raise ValueError(f"No label_map entry exists for raw labels: {unknown}")
        return mapped.astype("string")

    def read(self) -> pd.DataFrame:
        column_map = self.config.get("column_map", {})
        missing_targets = REQUIRED_CANONICAL_COLUMNS - set(column_map)
        if missing_targets:
            raise ValueError(
                "annotation_adapter.column_map must define canonical targets "
                f"{sorted(missing_targets)}."
            )

        outputs: list[pd.DataFrame] = []
        for path in self._paths():
            raw = _read_table(path, self.config)
            missing_sources = sorted(set(column_map.values()) - set(raw.columns))
            if missing_sources:
                raise ValueError(f"{path} is missing configured columns: {missing_sources}")

            frame = pd.DataFrame({target: raw[source] for target, source in column_map.items()})
            raw_label = frame["label"].copy()
            self._convert_times(frame, raw)
            frame["label"] = self._map_labels(raw_label)
            frame["dataset"] = str(self.config.get("dataset", "generic"))
            if "recording_id" not in frame:
                frame["recording_id"] = frame["media_id"]
            frame["source_event_type"] = raw_label.astype("string")
            frame["annotation_source"] = str(path)
            frame["derived_label"] = frame["label"]

            mapped_sources = set(column_map.values())
            for source_column in raw.columns:
                if source_column not in mapped_sources:
                    destination = f"source_{source_column}"
                    if destination not in frame:
                        frame[destination] = raw[source_column]
            outputs.append(frame)

        return pd.concat(outputs, ignore_index=True)
