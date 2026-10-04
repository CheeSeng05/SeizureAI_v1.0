from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .base import (
    EEGAnnotation,
    EEGLoadError,
    EEGMetadata,
    EEGRecording,
    private_recording_id,
    provisional_annotation_type,
)


def nicolet_events_to_annotations(
    events: Sequence[Any],
    *,
    recording_id: str,
    recording_start: Any,
    annotation_overrides: Mapping[str, str] | None = None,
) -> tuple[EEGAnnotation, ...]:
    annotations: list[EEGAnnotation] = []
    for index, event in enumerate(events, start=1):
        event_type = str(getattr(event, "IDStr", "") or "").strip()
        raw_name = str(
            getattr(event, "annotation", None)
            or getattr(event, "label", None)
            or event_type
            or "Annotation"
        ).strip()
        event_time = getattr(event, "date", None)
        if recording_start is None or event_time is None:
            raise EEGLoadError("Nicolet annotation time cannot be aligned to recording start.")
        time_sec = float((event_time - recording_start).total_seconds())
        if time_sec < -1e-3:
            raise EEGLoadError("Nicolet annotation precedes the recording start.")
        duration_sec = max(float(getattr(event, "duration", 0.0) or 0.0), 0.0)
        annotations.append(
            EEGAnnotation(
                recording_id=recording_id,
                event_id=f"event-{index:05d}",
                time_sec=max(time_sec, 0.0),
                duration_sec=duration_sec,
                raw_name=raw_name,
                normalized_type=provisional_annotation_type(
                    raw_name, event_type, overrides=annotation_overrides
                ),
                annotation_source="embedded_nicolet_event",
                event_type=event_type or None,
                event_guid=str(getattr(event, "GUID", "") or "") or None,
                source_user_available=bool(getattr(event, "user", "")),
            )
        )
    return tuple(annotations)


class NicoletERecording(EEGRecording):
    """Read-only Natus/Nicolet/Nervus `.e` adapter."""

    def __init__(
        self,
        path: str | Path,
        *,
        annotation_overrides: Mapping[str, str] | None = None,
    ) -> None:
        try:
            from nicolet_e2edf.nicolet.data import read_nervus_data
            from nicolet_e2edf.nicolet.header import read_nervus_header
        except ImportError as exc:
            raise EEGLoadError(
                "Nicolet .e support requires nicolet-e2edf>=0.4.2."
            ) from exc

        self.path = Path(path).resolve()
        self._read_nervus_data = read_nervus_data
        try:
            public, raw = read_nervus_header(self.path)
        except Exception as exc:
            raise EEGLoadError(
                "The .e file is not a supported Natus/Nicolet/Nervus recording."
            ) from exc
        if str(getattr(raw, "format", "")) not in {"nicolet-e", "nervus-legacy-e"}:
            raise EEGLoadError("The .e reader did not identify a supported Nicolet container.")
        if not getattr(raw, "matchingChannels", None):
            raise EEGLoadError("The Nicolet recording has no readable dominant-rate channels.")

        self._raw_header = raw
        self._raw_channels = tuple(int(value) for value in raw.matchingChannels)
        recording_id = private_recording_id(self.path)
        channel_names = tuple(str(value).strip() for value in public["label"])
        sampling_rate = float(public["Fs"])
        start_time = getattr(raw, "startDateTime", None)
        timeline_segments = []
        source_sample_cursor = 0
        dominant_channel = int(raw.matchingChannels[0]) - 1
        raw_segments = tuple(getattr(raw, "Segments", ()) or ())
        complete_segment_timing = bool(raw_segments) and start_time is not None and all(
            hasattr(segment, "sampleCount")
            and hasattr(segment, "date")
            and hasattr(segment, "duration")
            for segment in raw_segments
        )
        if complete_segment_timing:
            for index, segment in enumerate(raw_segments):
                segment_samples = int(segment.sampleCount[dominant_channel])
                global_start_sec = float((segment.date - start_time).total_seconds())
                timeline_segments.append({
                    "segment_index": index,
                    "source_sample_start": source_sample_cursor,
                    "source_sample_stop": source_sample_cursor + segment_samples,
                    "global_start_sec": global_start_sec,
                    "global_end_sec": global_start_sec + float(segment.duration),
                    "duration_sec": float(segment.duration),
                })
                source_sample_cursor += segment_samples
        else:
            duration_sec = float(public["nSamples"]) / sampling_rate
            timeline_segments.append({
                "segment_index": 0,
                "source_sample_start": 0,
                "source_sample_stop": int(public["nSamples"]),
                "global_start_sec": 0.0,
                "global_end_sec": duration_sec,
                "duration_sec": duration_sec,
            })
        wall_duration_sec = max(
            (float(item["global_end_sec"]) for item in timeline_segments),
            default=float(public["nSamples"]) / sampling_rate,
        )
        self.metadata = EEGMetadata(
            recording_id=recording_id,
            source_format="nicolet-e",
            channel_names=channel_names,
            sampling_rates=tuple(sampling_rate for _ in channel_names),
            signal_units=tuple("uV" for _ in channel_names),
            duration_sec=float(public["nSamples"]) / sampling_rate,
            sample_count=int(public["nSamples"]),
            montage_reference=getattr(raw, "reference", None),
            recording_start_time=start_time,
            vendor="Natus/Nicolet/Nervus",
            reader="nicolet-e2edf",
            extra={
                "segment_count": len(timeline_segments),
                "excluded_mixed_rate_channel_count": len(getattr(raw, "excludedChannels", ())),
                "timeline": "recording_wall_clock_elapsed",
                "timeline_segments": timeline_segments,
                "wall_duration_sec": wall_duration_sec,
            },
        )
        self.annotations = nicolet_events_to_annotations(
            getattr(raw, "Events", ()),
            recording_id=recording_id,
            recording_start=start_time,
            annotation_overrides=annotation_overrides,
        )

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
        if not indices or any(value < 0 or value >= len(self._raw_channels) for value in indices):
            raise ValueError("Invalid Nicolet channel selection.")
        if stop_sample > self.metadata.sample_count:
            raise EEGLoadError("Nicolet sample request extends beyond the recording.")
        source_channels = [self._raw_channels[index] for index in indices]
        try:
            data = self._read_nervus_data(
                self.path,
                self._raw_header,
                channels=source_channels,
                begsample=start_sample + 1,
                endsample=stop_sample,
            )
        except Exception as exc:
            raise EEGLoadError("Nicolet waveform extraction failed.") from exc
        result = np.asarray(data, dtype=np.float32)
        expected = (len(indices), stop_sample - start_sample)
        if result.shape != expected or not np.isfinite(result).all():
            raise EEGLoadError(f"Nicolet reader returned invalid waveform shape {result.shape}.")
        return result

