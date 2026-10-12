from __future__ import annotations

import struct
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path, PureWindowsPath
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np

from ..data_inspection import inspect_video
from ..fusion.late_fusion import LateFusion


NICOLET_VIDEO_SYNC_HEADER_BYTES = 752
NICOLET_VIDEO_SYNC_RECORD_BYTES = 600
NICOLET_VIDEO_SYNC_FILENAME_START = 24
NICOLET_VIDEO_SYNC_FILENAME_END = 536
OLE_EPOCH = datetime(1899, 12, 30)


@dataclass(frozen=True)
class NicoletVideoSyncRecord:
    filename: str
    global_time_sec: float


@dataclass(frozen=True)
class VideoSegment:
    segment_id: str
    video_path: str
    filename: str
    global_start_sec: float
    global_end_sec: float
    duration_sec: float
    fps: float
    sync_source: str = "nicolet_videosyncguid+video_container_duration"
    sync_verified: bool = True

    def public_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value.pop("video_path", None)
        return value


def _naive(value: datetime) -> datetime:
    return value.replace(tzinfo=None)


def parse_nicolet_video_sync_payload(
    payload: bytes,
    *,
    recording_start: datetime,
) -> list[NicoletVideoSyncRecord]:
    """Decode the fixed-size records stored in Nicolet ``VIDEOSYNCGUID``.

    Each record contains an OLE timestamp, a sub-second remainder, and the
    source video filename. The timestamp is kept on the EEG recording's
    wall-clock timeline; acquisition gaps are therefore preserved.
    """

    if len(payload) < NICOLET_VIDEO_SYNC_HEADER_BYTES:
        raise ValueError("Nicolet VIDEOSYNCGUID payload is shorter than its header.")
    body = payload[NICOLET_VIDEO_SYNC_HEADER_BYTES:]
    if len(body) % NICOLET_VIDEO_SYNC_RECORD_BYTES:
        raise ValueError("Nicolet VIDEOSYNCGUID payload has an unsupported record layout.")
    start = _naive(recording_start)
    records: list[NicoletVideoSyncRecord] = []
    for offset in range(0, len(body), NICOLET_VIDEO_SYNC_RECORD_BYTES):
        raw = body[offset:offset + NICOLET_VIDEO_SYNC_RECORD_BYTES]
        ole_days = struct.unpack_from("<d", raw, 8)[0]
        fractional_seconds = struct.unpack_from("<d", raw, 16)[0]
        filename = (
            raw[NICOLET_VIDEO_SYNC_FILENAME_START:NICOLET_VIDEO_SYNC_FILENAME_END]
            .decode("utf-16-le", errors="ignore")
            .split("\x00", 1)[0]
            .strip()
        )
        if not filename:
            continue
        timestamp = OLE_EPOCH + timedelta(days=ole_days, seconds=fractional_seconds)
        global_time = (timestamp - start).total_seconds()
        if not np.isfinite(global_time) or global_time < -1e-3:
            raise ValueError("Nicolet video synchronization timestamp is outside the EEG timeline.")
        records.append(NicoletVideoSyncRecord(filename, max(0.0, float(global_time))))
    if not records:
        raise ValueError("Nicolet VIDEOSYNCGUID contains no video synchronization records.")
    if any(later.global_time_sec <= earlier.global_time_sec for earlier, later in zip(records, records[1:])):
        raise ValueError("Nicolet video synchronization timestamps are not strictly increasing.")
    return records


def _nicolet_video_sync_payload(path: Path, raw_header: Any) -> bytes:
    packet = next(
        (item for item in (getattr(raw_header, "StaticPackets", ()) or ()) if item.IDStr == "VIDEOSYNCGUID"),
        None,
    )
    if packet is None:
        raise ValueError("The Nicolet recording does not contain VIDEOSYNCGUID metadata.")
    sections = [
        item
        for item in (getattr(raw_header, "MainIndex", ()) or ())
        if int(item.sectionIdx) == int(packet.index)
    ]
    if not sections:
        raise ValueError("The Nicolet VIDEOSYNCGUID metadata has no readable payload sections.")
    chunks: list[bytes] = []
    with path.open("rb") as stream:
        for section in sections:
            stream.seek(int(section.offset))
            chunk = stream.read(int(section.sectionL))
            if len(chunk) != int(section.sectionL):
                raise ValueError("The Nicolet VIDEOSYNCGUID payload is truncated.")
            chunks.append(chunk)
    return b"".join(chunks)


def read_nicolet_video_sync(path: str | Path) -> tuple[list[NicoletVideoSyncRecord], Any]:
    try:
        from nicolet_e2edf.nicolet.header import read_nervus_header
    except ImportError as exc:
        raise RuntimeError("Nicolet video synchronization requires nicolet-e2edf.") from exc

    eeg_path = Path(path).resolve()
    _, raw_header = read_nervus_header(eeg_path)
    recording_start = getattr(raw_header, "startDateTime", None)
    if recording_start is None:
        raise ValueError("The Nicolet recording start time is unavailable.")
    payload = _nicolet_video_sync_payload(eeg_path, raw_header)
    return parse_nicolet_video_sync_payload(payload, recording_start=recording_start), raw_header


def _first_records(records: Sequence[NicoletVideoSyncRecord]) -> list[NicoletVideoSyncRecord]:
    first: list[NicoletVideoSyncRecord] = []
    previous: str | None = None
    for record in records:
        key = PureWindowsPath(record.filename).name.casefold()
        if key != previous:
            first.append(record)
            previous = key
    names = [PureWindowsPath(item.filename).name.casefold() for item in first]
    if len(names) != len(set(names)):
        raise ValueError("Nicolet synchronization reuses a video filename in non-contiguous intervals.")
    return first


def _video_start_offsets(raw_header: Any) -> list[float]:
    recording_start = getattr(raw_header, "startDateTime", None)
    if recording_start is None:
        return []
    return [
        float((event.date - recording_start).total_seconds())
        for event in (getattr(raw_header, "Events", ()) or ())
        if getattr(event, "IDStr", None) == "Video Start" and getattr(event, "date", None) is not None
    ]


def build_video_segment_manifest(
    *,
    eeg_path: str | Path,
    video_paths: Sequence[str | Path],
    recording_id: str,
    video_inspector: Callable[[str | Path], Any] = inspect_video,
    event_tolerance_sec: float = 0.25,
    video_start_offsets: Mapping[str, float] | None = None,
    eeg_duration_sec: float | None = None,
) -> list[VideoSegment]:
    if not video_paths:
        raise ValueError("At least one video segment is required for multimodal fusion.")
    if Path(eeg_path).suffix.casefold() == ".edf":
        offsets = dict(video_start_offsets or {})
        names = [Path(value).name for value in video_paths]
        if len({name.casefold() for name in names}) != len(names):
            raise ValueError("Duplicate video segment filename.")
        if set(offsets) != set(names):
            raise ValueError("EDF synchronization requires a start time in EEG seconds for every video segment.")
        if eeg_duration_sec is None:
            from ..eeg.loaders import load_eeg
            with load_eeg(eeg_path) as recording:
                eeg_duration_sec = recording.metadata.duration_sec
        segments = []
        for index, value in enumerate(video_paths, start=1):
            path = Path(value).resolve()
            start = float(offsets[path.name])
            inspected = video_inspector(path)
            duration, fps = float(inspected.duration_sec), float(inspected.fps)
            if not all(np.isfinite(item) for item in (start, duration, fps, eeg_duration_sec)) or start < 0 or duration <= 0 or fps <= 0:
                raise ValueError(f"Invalid EDF synchronization timing for {path.name}.")
            if start + duration > eeg_duration_sec + event_tolerance_sec:
                raise ValueError(f"Video segment extends beyond the EEG recording: {path.name}")
            segments.append(VideoSegment(
                segment_id=f"segment-{index:04d}", video_path=str(path), filename=path.name,
                global_start_sec=start, global_end_sec=start + duration,
                duration_sec=duration, fps=fps,
                sync_source="human_declared_offsets+video_container_duration",
            ))
        return sorted(segments, key=lambda item: (item.global_start_sec, item.segment_id))
    if Path(eeg_path).suffix.casefold() != ".e":
        raise ValueError("Automatic multimodal synchronization currently requires a Nicolet .e recording.")
    records, raw_header = read_nicolet_video_sync(eeg_path)
    starts = _first_records(records)
    sync_by_name = {PureWindowsPath(item.filename).name.casefold(): item for item in starts}
    supplied: dict[str, Path] = {}
    for value in video_paths:
        path = Path(value).resolve()
        key = path.name.casefold()
        if key in supplied:
            raise ValueError(f"Duplicate video segment filename: {path.name}")
        supplied[key] = path
    missing_metadata = sorted(path.name for key, path in supplied.items() if key not in sync_by_name)
    if missing_metadata:
        raise ValueError(
            "Video/EEG synchronization could not be verified for: " + ", ".join(missing_metadata)
        )
    event_starts = _video_start_offsets(raw_header)
    segments: list[VideoSegment] = []
    for index, record in enumerate(starts, start=1):
        key = PureWindowsPath(record.filename).name.casefold()
        path = supplied.get(key)
        if path is None:
            continue
        if not any(abs(record.global_time_sec - value) <= event_tolerance_sec for value in event_starts):
            raise ValueError(
                f"Nicolet synchronization for {path.name} is not confirmed by a Video Start event."
            )
        inspected = video_inspector(path)
        duration = float(inspected.duration_sec)
        fps = float(inspected.fps)
        if duration <= 0 or fps <= 0:
            raise ValueError(f"Video segment has invalid duration or frame rate: {path.name}")
        segments.append(
            VideoSegment(
                segment_id=f"segment-{index:04d}",
                video_path=str(path),
                filename=path.name,
                global_start_sec=float(record.global_time_sec),
                global_end_sec=float(record.global_time_sec + duration),
                duration_sec=duration,
                fps=fps,
            )
        )
    if len(segments) != len(supplied):
        raise ValueError("Not every uploaded video segment could be associated with the EEG recording.")
    return sorted(segments, key=lambda item: (item.global_start_sec, item.segment_id))


def video_rows_on_global_timeline(
    rows: Iterable[Mapping[str, Any]],
    segment: VideoSegment,
    *,
    recording_id: str,
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for row in rows:
        local_start = float(row["start_sec"])
        local_end = float(row["end_sec"])
        if local_start < 0 or local_end <= local_start or local_end > segment.duration_sec + 0.25:
            raise ValueError(f"Video prediction is outside segment bounds: {segment.segment_id}")
        result.append(
            {
                **dict(row),
                "recording_id": recording_id,
                "video_segment_id": segment.segment_id,
                "video_filename": segment.filename,
                "local_start_sec": local_start,
                "local_end_sec": local_end,
                "start_sec": segment.global_start_sec + local_start,
                "end_sec": segment.global_start_sec + local_end,
                "global_start_sec": segment.global_start_sec + local_start,
                "global_end_sec": segment.global_start_sec + local_end,
            }
        )
    return result


def _covered_duration(intervals: Sequence[tuple[float, float]]) -> float:
    if not intervals:
        return 0.0
    ordered = sorted(intervals)
    start, end = ordered[0]
    total = 0.0
    for next_start, next_end in ordered[1:]:
        if next_start <= end:
            end = max(end, next_end)
        else:
            total += end - start
            start, end = next_start, next_end
    return total + end - start


def validate_fusion_settings(config: Mapping[str, Any]) -> tuple[float, float, float, float]:
    fusion = config.get("fusion", {}) if isinstance(config, Mapping) else {}
    eeg_weight = float(fusion.get("eeg_weight", fusion.get("late_eeg_weight", 0.8)))
    video_weight = float(fusion.get("video_weight", fusion.get("late_video_weight", 0.2)))
    threshold = float(fusion.get("threshold", 0.5))
    minimum_coverage = float(fusion.get("minimum_video_coverage", 1.0))
    LateFusion(video_weight=video_weight, eeg_weight=eeg_weight)
    if not 0 <= threshold <= 1:
        raise ValueError("Fusion threshold must be in [0, 1].")
    if not 0 <= minimum_coverage <= 1:
        raise ValueError("Minimum video coverage must be in [0, 1].")
    return eeg_weight, video_weight, threshold, minimum_coverage


def fuse_prediction_timelines(
    *,
    eeg_rows: Sequence[Mapping[str, Any]],
    video_rows: Sequence[Mapping[str, Any]],
    recording_id: str,
    config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Align video by temporal overlap onto EEG windows and apply late fusion."""

    eeg_weight, video_weight, threshold, minimum_coverage = validate_fusion_settings(config)
    fusion = LateFusion(video_weight=video_weight, eeg_weight=eeg_weight)
    result: list[dict[str, Any]] = []
    for index, eeg in enumerate(eeg_rows, start=1):
        start = float(eeg["start_sec"])
        end = float(eeg["end_sec"])
        if end <= start:
            raise ValueError("EEG fusion window end must be after its start.")
        contributions: list[tuple[Mapping[str, Any], float, float, float]] = []
        for video in video_rows:
            video_start = float(video["start_sec"])
            video_end = float(video["end_sec"])
            overlap_start = max(start, video_start)
            overlap_end = min(end, video_end)
            if overlap_end > overlap_start:
                contributions.append((video, overlap_end - overlap_start, overlap_start, overlap_end))
        overlap_total = sum(item[1] for item in contributions)
        aligned_video = (
            sum(float(item[0]["video_probability"]) * item[1] for item in contributions) / overlap_total
            if overlap_total > 0 else None
        )
        covered = _covered_duration([(item[2], item[3]) for item in contributions])
        coverage = min(1.0, covered / (end - start))
        available = aligned_video is not None and coverage + 1e-9 >= minimum_coverage
        eeg_probability = float(eeg.get("probability", eeg.get("eeg_probability")))
        fused_probability = (
            float(fusion.predict(
                video_probability=np.asarray([aligned_video]),
                eeg_probability=np.asarray([eeg_probability]),
            )[0])
            if available else None
        )
        segment_ids = sorted({str(item[0].get("video_segment_id", "")) for item in contributions if item[0].get("video_segment_id")})
        contributor_ids = [
            str(item[0].get("video_window_id") or f"{item[0].get('video_segment_id', 'video')}:{item[0].get('local_start_sec', item[0].get('start_sec'))}")
            for item in contributions
        ]
        result.append(
            {
                "recording_id": recording_id,
                "window_id": str(eeg.get("window_id") or f"window-{index:06d}"),
                "start_sec": start,
                "end_sec": end,
                "eeg_probability": eeg_probability,
                "video_probability": aligned_video,
                "fused_probability": fused_probability,
                "eeg_prediction": int(eeg_probability >= 0.5),
                "video_prediction": int(aligned_video >= 0.5) if aligned_video is not None else None,
                "fused_prediction": int(fused_probability >= threshold) if fused_probability is not None else None,
                "fusion_available": available,
                "video_coverage": coverage,
                "video_segment_ids": "|".join(segment_ids),
                "video_window_ids": "|".join(contributor_ids),
                "eeg_weight": eeg_weight,
                "video_weight": video_weight,
                "fusion_threshold": threshold,
            }
        )
    return result
