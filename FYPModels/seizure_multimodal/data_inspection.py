from __future__ import annotations

import csv
import json
import math
import struct
from dataclasses import asdict, dataclass
from datetime import time
from pathlib import Path
from typing import Any, BinaryIO, Iterable


@dataclass(frozen=True)
class MP4VideoMetadata:
    path: str
    duration_sec: float
    fps: float
    frame_count: int
    width: int
    height: int
    codec: str
    variable_frame_rate: bool | None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class _Box:
    kind: str
    start: int
    data_start: int
    end: int


_CONTAINERS = {"moov", "trak", "mdia", "minf", "stbl"}


def _u32(stream: BinaryIO) -> int:
    raw = stream.read(4)
    if len(raw) != 4:
        raise ValueError("Unexpected end of MP4 data")
    return struct.unpack(">I", raw)[0]


def _u64(stream: BinaryIO) -> int:
    raw = stream.read(8)
    if len(raw) != 8:
        raise ValueError("Unexpected end of MP4 data")
    return struct.unpack(">Q", raw)[0]


def _boxes(stream: BinaryIO, start: int, end: int) -> Iterable[_Box]:
    position = start
    while position + 8 <= end:
        stream.seek(position)
        size = _u32(stream)
        kind_bytes = stream.read(4)
        if len(kind_bytes) != 4:
            break
        kind = kind_bytes.decode("latin-1")
        header = 8
        if size == 1:
            size = _u64(stream)
            header = 16
        elif size == 0:
            size = end - position
        if size < header or position + size > end:
            raise ValueError(f"Invalid MP4 box {kind!r} at byte {position}")
        yield _Box(kind, position, position + header, position + size)
        position += size


def _child(stream: BinaryIO, parent: _Box, kind: str) -> _Box | None:
    return next((box for box in _boxes(stream, parent.data_start, parent.end) if box.kind == kind), None)


def _read_mdhd(stream: BinaryIO, box: _Box) -> tuple[int, int]:
    stream.seek(box.data_start)
    version = stream.read(1)
    stream.read(3)
    if version == b"\x01":
        stream.read(16)
        timescale, duration = _u32(stream), _u64(stream)
    else:
        stream.read(8)
        timescale, duration = _u32(stream), _u32(stream)
    if timescale <= 0:
        raise ValueError("MP4 video track has an invalid time scale")
    return timescale, duration


def _read_handler(stream: BinaryIO, box: _Box) -> str:
    stream.seek(box.data_start + 8)
    return stream.read(4).decode("latin-1")


def _read_tkhd_size(stream: BinaryIO, box: _Box) -> tuple[int, int]:
    stream.seek(box.end - 8)
    width_fixed, height_fixed = _u32(stream), _u32(stream)
    return round(width_fixed / 65536), round(height_fixed / 65536)


def _read_stsd(stream: BinaryIO, box: _Box) -> tuple[str, int | None, int | None]:
    stream.seek(box.data_start + 4)
    entry_count = _u32(stream)
    if entry_count < 1:
        raise ValueError("MP4 video track has no sample description")
    entry_start = stream.tell()
    entry_size = _u32(stream)
    codec = stream.read(4).decode("latin-1")
    width = height = None
    if entry_size >= 36:
        stream.seek(entry_start + 32)
        raw = stream.read(4)
        if len(raw) == 4:
            width, height = struct.unpack(">HH", raw)
    return codec, width, height


def _read_stts(stream: BinaryIO, box: _Box) -> tuple[int, int, bool]:
    stream.seek(box.data_start + 4)
    entry_count = _u32(stream)
    sample_count = 0
    total_ticks = 0
    deltas: set[int] = set()
    for _ in range(entry_count):
        count, delta = _u32(stream), _u32(stream)
        sample_count += count
        total_ticks += count * delta
        deltas.add(delta)
    return sample_count, total_ticks, len(deltas) > 1


def inspect_mp4(path: str | Path) -> MP4VideoMetadata:
    """Inspect an ISO-BMFF MP4 without requiring FFmpeg or OpenCV.

    Frame count and average FPS come from the video track's decoding time table,
    not from the file name or a guessed nominal frame rate.
    """

    video_path = Path(path).resolve()
    file_size = video_path.stat().st_size
    with video_path.open("rb") as stream:
        moov = next((box for box in _boxes(stream, 0, file_size) if box.kind == "moov"), None)
        if moov is None:
            raise ValueError(f"No moov box found in MP4: {video_path}")
        for trak in (box for box in _boxes(stream, moov.data_start, moov.end) if box.kind == "trak"):
            mdia = _child(stream, trak, "mdia")
            if mdia is None:
                continue
            handler = _child(stream, mdia, "hdlr")
            if handler is None or _read_handler(stream, handler) != "vide":
                continue
            mdhd = _child(stream, mdia, "mdhd")
            minf = _child(stream, mdia, "minf")
            stbl = _child(stream, minf, "stbl") if minf else None
            stsd = _child(stream, stbl, "stsd") if stbl else None
            stts = _child(stream, stbl, "stts") if stbl else None
            tkhd = _child(stream, trak, "tkhd")
            if not all((mdhd, stsd, stts, tkhd)):
                raise ValueError("MP4 video track is missing mdhd/tkhd/stsd/stts metadata")
            timescale, track_duration = _read_mdhd(stream, mdhd)
            frame_count, decoding_ticks, variable = _read_stts(stream, stts)
            codec, stsd_width, stsd_height = _read_stsd(stream, stsd)
            tkhd_width, tkhd_height = _read_tkhd_size(stream, tkhd)
            duration_sec = (decoding_ticks or track_duration) / timescale
            if duration_sec <= 0 or frame_count <= 0:
                raise ValueError("MP4 video track has invalid duration or frame count")
            return MP4VideoMetadata(
                path=str(video_path),
                duration_sec=duration_sec,
                fps=frame_count / duration_sec,
                frame_count=frame_count,
                width=stsd_width or tkhd_width,
                height=stsd_height or tkhd_height,
                codec=codec,
                variable_frame_rate=variable,
            )
    raise ValueError(f"No video track found in MP4: {video_path}")


def inspect_video(path: str | Path) -> MP4VideoMetadata:
    """Inspect supported video containers without treating RIFF/Matroska as MP4.

    MP4 retains its exact sample-table timing. Other containers use the same
    OpenCV backend as inference, with nominal FPS and frame-count duration.
    Their variable-frame-rate status is unknown, rather than assumed false.
    """
    video_path = Path(path).resolve()
    if video_path.suffix.lower() == ".mp4":
        return inspect_mp4(video_path)
    if not video_path.is_file():
        raise FileNotFoundError(video_path)
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("OpenCV is required to inspect non-MP4 videos") from exc
    capture = cv2.VideoCapture(str(video_path))
    try:
        if not capture.isOpened():
            raise ValueError(f"Video cannot be opened; unsupported codec or damaged file: {video_path.name}")
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        count = float(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if not math.isfinite(fps) or fps <= 0 or not math.isfinite(count) or count < 1:
            raise ValueError(f"Video has invalid FPS or frame count: {video_path.name}")
        ok, frame = capture.read()
        if not ok or frame is None:
            raise ValueError(f"Video contains no decodable frames: {video_path.name}")
        height, width = frame.shape[:2]
        fourcc = int(capture.get(cv2.CAP_PROP_FOURCC))
        codec = "".join(chr((fourcc >> (8 * index)) & 0xff) for index in range(4)).rstrip("\x00")
        frame_count = int(round(count))
        return MP4VideoMetadata(
            path=str(video_path), duration_sec=frame_count / fps, fps=fps,
            frame_count=frame_count, width=width, height=height,
            codec=codec or "unknown", variable_frame_rate=None,
        )
    finally:
        capture.release()


def seconds_from_excel_time(value: Any) -> float | None:
    if value is None or value == "":
        return None
    if isinstance(value, time):
        return value.hour * 3600 + value.minute * 60 + value.second + value.microsecond / 1e6
    if isinstance(value, (int, float)):
        return float(value) * 86400.0 if 0 <= float(value) < 1 else float(value)
    parts = str(value).strip().split(":")
    if len(parts) == 3:
        hours, minutes, seconds = int(parts[0]), int(parts[1]), float(parts[2])
        return hours * 3600 + minutes * 60 + seconds
    raise ValueError(f"Unsupported timestamp value: {value!r}")


def inspect_label_workbook(path: str | Path) -> dict[str, Any]:
    try:
        from openpyxl import load_workbook
    except ImportError as exc:
        raise RuntimeError("openpyxl is required to inspect Label.xlsx") from exc
    workbook_path = Path(path).resolve()
    workbook = load_workbook(workbook_path, data_only=False, read_only=True)
    sheets: list[dict[str, Any]] = []
    for worksheet in workbook.worksheets:
        rows = [list(row) for row in worksheet.iter_rows(values_only=True)]
        while rows and not any(value is not None for value in rows[-1]):
            rows.pop()
        columns = [str(value) if value is not None else "" for value in (rows[0] if rows else [])]
        records = [dict(zip(columns, row)) for row in rows[1:] if any(value is not None for value in row)]
        serializable_records = []
        for record in records:
            serialized = dict(record)
            for column in ("EEG onset", "Clinical Onset"):
                if column in serialized:
                    serialized[column] = (
                        None if serialized[column] is None else str(serialized[column])
                    )
                    serialized[f"{column} seconds"] = seconds_from_excel_time(record[column])
            serializable_records.append(serialized)
        sheets.append(
            {
                "name": worksheet.title,
                "columns": columns,
                "data_rows": len(records),
                "records": serializable_records,
            }
        )
    sheet_names = list(workbook.sheetnames)
    workbook.close()
    return {"path": str(workbook_path), "sheet_names": sheet_names, "sheets": sheets}


def resolved_label_records(path: str | Path) -> list[dict[str, Any]]:
    """Resolve the workbook's blank continuation cells without altering raw data."""

    workbook = inspect_label_workbook(path)
    resolved: list[dict[str, Any]] = []
    current_patient: str | None = None
    current_type: str | None = None
    for sheet in workbook["sheets"]:
        for raw_record in sheet["records"]:
            record = dict(raw_record)
            if record.get("PatID"):
                current_patient = str(record["PatID"])
            if record.get("Seizure Type"):
                current_type = str(record["Seizure Type"])
            record["resolved_PatID"] = current_patient
            record["resolved_Seizure Type"] = current_type
            record["resolution_policy"] = "forward-fill PatID and Seizure Type within the catalog"
            resolved.append(record)
    return resolved


def _required_csv_value(row: dict[str, Any], column: str, row_number: int) -> str:
    value = row.get(column)
    if value is None or str(value).strip() == "":
        raise ValueError(f"Label CSV row {row_number} is missing required column {column!r}")
    return str(value).strip()


def _binary_label(value: Any, row_number: int) -> int:
    normalized = str(value).strip().casefold()
    if normalized in {"1", "s", "seizure", "ictal", "true", "yes"}:
        return 1
    if normalized in {"0", "n", "non_seizure", "non-seizure", "nonictal", "non_ictal", "false", "no"}:
        return 0
    raise ValueError(
        f"Label CSV row {row_number} has unsupported label {value!r}; use 0/1 or N/S."
    )


def inspect_interval_label_csv(path: str | Path) -> dict[str, Any]:
    csv_path = Path(path).resolve()
    with csv_path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        columns = reader.fieldnames or []
        required = {"start_sec", "end_sec", "label"}
        missing = sorted(required - set(columns))
        if missing:
            raise ValueError(
                "Label CSV must contain required columns: "
                + ", ".join(sorted(required))
            )
        records: list[dict[str, Any]] = []
        label_counts = {0: 0, 1: 0}
        for row_number, row in enumerate(reader, start=2):
            start = float(_required_csv_value(row, "start_sec", row_number))
            end = float(_required_csv_value(row, "end_sec", row_number))
            if end <= start:
                raise ValueError(
                    f"Label CSV row {row_number} must have end_sec greater than start_sec."
                )
            label = _binary_label(_required_csv_value(row, "label", row_number), row_number)
            label_counts[label] += 1
            records.append(
                {
                    **row,
                    "start_sec": start,
                    "end_sec": end,
                    "label": label,
                    "row_number": row_number,
                }
            )
    if not records:
        raise ValueError("Label CSV does not contain any data rows.")
    return {
        "path": str(csv_path),
        "columns": columns,
        "data_rows": len(records),
        "records": records,
        "label_values": {str(key): value for key, value in label_counts.items() if value},
        "time_range_sec": [min(row["start_sec"] for row in records), max(row["end_sec"] for row in records)],
    }


def resolved_interval_label_records(path: str | Path) -> list[dict[str, Any]]:
    """Read timestamped binary video labels from CSV.

    Required columns are start_sec, end_sec, and label. Optional identity columns
    such as patient_id, recording_id, seizure_id, and seizure_type are preserved.
    """

    inspection = inspect_interval_label_csv(path)
    return [
        {
            **record,
            "resolved_PatID": record.get("patient_id") or record.get("PatID"),
            "resolved_Seizure Type": record.get("seizure_type") or record.get("Seizure Type"),
            "#Seizure": record.get("seizure_id") or record.get("#Seizure"),
            "resolution_policy": "timestamped interval CSV; no workbook forward-fill required",
        }
        for record in inspection["records"]
    ]


def inspect_eeg_csv(path: str | Path) -> dict[str, Any]:
    csv_path = Path(path).resolve()
    with csv_path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        columns = reader.fieldnames or []
        feature_columns = [column for column in columns if column.startswith("X") and column[1:].isdigit()]
        label_counts: dict[str, int] = {}
        row_count = 0
        identifiers_unique = True
        identifiers: set[str] = set()
        for row in reader:
            row_count += 1
            label = str(row.get("y", ""))
            label_counts[label] = label_counts.get(label, 0) + 1
            identifier = row.get("Unnamed")
            if identifier:
                if identifier in identifiers:
                    identifiers_unique = False
                identifiers.add(identifier)
    timestamp_columns = [column for column in columns if "time" in column.lower() or "date" in column.lower()]
    patient_columns = [column for column in columns if "patient" in column.lower() or column.lower() in {"patid", "subject_id"}]
    recording_columns = [column for column in columns if "recording" in column.lower() or "record" in column.lower()]
    return {
        "path": str(csv_path),
        "rows": row_count,
        "columns": len(columns),
        "column_names": columns,
        "feature_columns": feature_columns,
        "feature_count": len(feature_columns),
        "target_column": "y" if "y" in columns else None,
        "label_values": {key: label_counts[key] for key in sorted(label_counts, key=lambda value: int(value))},
        "sample_identifier_column": "Unnamed" if "Unnamed" in columns else None,
        "sample_identifiers_unique": identifiers_unique if identifiers else None,
        "timestamp_columns": timestamp_columns,
        "patient_id_columns": patient_columns,
        "recording_id_columns": recording_columns,
        "row_represents_complete_window": len(feature_columns) > 0,
        "sampling_assumption": (
            "178 sequential samples from a single EEG segment; the CSV itself has no sampling-rate metadata"
            if len(feature_columns) == 178
            else "not established from the file"
        ),
    }


def write_inspection_json(payload: dict[str, Any], path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
