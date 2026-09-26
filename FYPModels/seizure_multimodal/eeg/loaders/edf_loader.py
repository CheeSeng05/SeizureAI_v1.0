from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from .base import (
    EEGAnnotation,
    EEGLoadError,
    EEGMetadata,
    EEGRecording,
    private_recording_id,
    provisional_annotation_type,
)


class EDFRecording(EEGRecording):
    """Read-only EDF/EDF+ adapter backed by PyEDFlib."""

    def __init__(
        self,
        path: str | Path,
        *,
        annotation_overrides: Mapping[str, str] | None = None,
    ) -> None:
        try:
            import pyedflib
        except ImportError as exc:
            raise EEGLoadError("EDF support requires pyedflib.") from exc

        self.path = Path(path).resolve()
        try:
            self._reader = pyedflib.EdfReader(str(self.path))
        except Exception as exc:
            raise EEGLoadError("The file is not a readable EDF/EDF+ recording.") from exc

        recording_id = private_recording_id(self.path)
        channel_count = int(self._reader.signals_in_file)
        channel_names = tuple(str(value).strip() for value in self._reader.getSignalLabels())
        sampling_rates = tuple(float(self._reader.getSampleFrequency(index)) for index in range(channel_count))
        signal_units = tuple(
            str(self._reader.getPhysicalDimension(index)).strip() or "unknown"
            for index in range(channel_count)
        )
        sample_counts = tuple(int(value) for value in self._reader.getNSamples())
        duration = max(
            (count / rate for count, rate in zip(sample_counts, sampling_rates) if rate > 0),
            default=0.0,
        )
        try:
            start_time = self._reader.getStartdatetime()
        except Exception:
            start_time = None

        self.metadata = EEGMetadata(
            recording_id=recording_id,
            source_format="edf",
            channel_names=channel_names,
            sampling_rates=sampling_rates,
            signal_units=signal_units,
            duration_sec=float(duration),
            sample_count=max(sample_counts, default=0),
            montage_reference=None,
            recording_start_time=start_time,
            reader="pyedflib",
        )

        annotations: list[EEGAnnotation] = []
        try:
            onsets, durations, descriptions = self._reader.readAnnotations()
        except Exception:
            onsets, durations, descriptions = (), (), ()
        for index, (onset, duration_value, description) in enumerate(
            zip(onsets, durations, descriptions), start=1
        ):
            raw_name = str(description).strip() or "Annotation"
            duration_sec = max(float(duration_value or 0.0), 0.0)
            annotations.append(
                EEGAnnotation(
                    recording_id=recording_id,
                    event_id=f"event-{index:05d}",
                    time_sec=float(onset),
                    duration_sec=duration_sec,
                    raw_name=raw_name,
                    normalized_type=provisional_annotation_type(
                        raw_name, "annotation", overrides=annotation_overrides
                    ),
                    annotation_source="embedded_edfplus_annotation",
                    event_type="annotation",
                )
            )
        self.annotations = tuple(annotations)

    def read_samples(
        self,
        *,
        channel_indices: Sequence[int],
        start_sample: int,
        stop_sample: int,
    ) -> np.ndarray:
        if start_sample < 0 or stop_sample <= start_sample:
            raise ValueError("Invalid half-open EEG sample interval.")
        indices = [int(value) for value in channel_indices]
        if not indices or any(value < 0 or value >= len(self.metadata.channel_names) for value in indices):
            raise ValueError("Invalid EDF channel selection.")
        rates = {round(self.metadata.sampling_rates[value], 9) for value in indices}
        if len(rates) != 1:
            raise EEGLoadError("Selected EDF channels have mixed sampling rates.")
        count = stop_sample - start_sample
        rows = [
            np.asarray(self._reader.readSignal(index, start=start_sample, n=count), dtype=np.float32)
            for index in indices
        ]
        if any(len(row) != count for row in rows):
            raise EEGLoadError("EDF sample request extends beyond the available recording.")
        return np.stack(rows)

    def close(self) -> None:
        reader = getattr(self, "_reader", None)
        if reader is not None:
            reader.close()
            self._reader = None

