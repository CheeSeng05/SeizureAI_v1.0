from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

from .checkpoints.provenance import sha256_file
from .data_inspection import inspect_eeg_csv, inspect_interval_label_csv, inspect_label_workbook, inspect_mp4


@dataclass(frozen=True)
class ModalitySource:
    source_id: str
    has_video: bool
    has_eeg: bool
    is_verified_pair: bool
    video_path: str | None = None
    label_path: str | None = None
    eeg_path: str | None = None
    patient_id: str | None = None
    recording_id: str | None = None
    start_sec: float | None = None
    end_sec: float | None = None
    pairing_note: str = ""

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _resolve(root: Path, configured: str | None, candidates: tuple[str, ...]) -> Path | None:
    if configured:
        candidate = root / configured
        if candidate.is_file():
            return candidate.resolve()
    lower_candidates = {name.lower() for name in candidates}
    for path in root.iterdir():
        if path.is_file() and path.name.lower() in lower_candidates:
            return path.resolve()
    return None


def discover_sources(root: str | Path, config: Mapping[str, Any]) -> list[ModalitySource]:
    """Discover current inputs as independent sources unless pairing is verified.

    Co-location and simultaneous availability are deliberately not evidence of a
    same-patient synchronized pair.
    """

    project_root = Path(root).resolve()
    data = config.get("data", {})
    data = data if isinstance(data, Mapping) else {}
    video = _resolve(project_root, data.get("video"), ("Sz2PG.mp4",))
    labels = (
        None
        if "video_labels" in data and data.get("video_labels") is None
        else _resolve(project_root, data.get("video_labels"), ("Label.xlsx", "label.xlsx"))
    )
    eeg = _resolve(
        project_root,
        data.get("eeg_csv"),
        ("Epileptic Seizure Recognition.csv",),
    )
    sources: list[ModalitySource] = []
    if video:
        provenance = data.get("video_provenance", {})
        provenance = provenance if isinstance(provenance, Mapping) else {}
        expected_hash = str(provenance.get("expected_sha256", "")).lower()
        actual_hash = sha256_file(video)
        patient_id = provenance.get("patient_id") if expected_hash == actual_hash else None
        metadata = inspect_mp4(video)
        sources.append(
            ModalitySource(
                source_id=video.stem,
                has_video=True,
                has_eeg=False,
                is_verified_pair=False,
                video_path=str(video),
                label_path=str(labels) if labels else None,
                patient_id=str(patient_id) if patient_id else None,
                recording_id=(
                    str(provenance['recording_id'])
                    if expected_hash == actual_hash and provenance.get('recording_id')
                    else f"{patient_id}_{provenance.get('seizure_id')}_{provenance.get('seizure_type')}"
                    if patient_id
                    else video.stem
                ),
                start_sec=0.0,
                end_sec=metadata.duration_sec,
                pairing_note=(
                    "Video identity is hash-verified against the configured upstream dataset path; "
                    "no synchronized EEG is attached."
                    if patient_id
                    else "Video provenance hash did not verify; no label row or synchronized EEG is attached."
                ),
            )
        )
    if eeg:
        sources.append(
            ModalitySource(
                source_id="epileptic_seizure_recognition_csv",
                has_video=False,
                has_eeg=True,
                is_verified_pair=False,
                eeg_path=str(eeg),
                patient_id=None,
                recording_id=None,
                pairing_note=(
                    "Static EEG benchmark rows have no patient, recording, or timestamp fields and are "
                    "not paired with Sz2PG."
                ),
            )
        )
    return sources


def inspection_bundle(root: str | Path, config: Mapping[str, Any]) -> dict[str, Any]:
    project_root = Path(root).resolve()
    sources = discover_sources(project_root, config)
    payload: dict[str, Any] = {"sources": [source.as_dict() for source in sources]}
    video_source = next((source for source in sources if source.has_video), None)
    eeg_source = next((source for source in sources if source.has_eeg), None)
    if video_source and video_source.video_path:
        payload["video"] = inspect_mp4(video_source.video_path).as_dict()
        provenance = config.get("data", {}).get("video_provenance", {})
        provenance = provenance if isinstance(provenance, Mapping) else {}
        actual_hash = sha256_file(video_source.video_path)
        payload["video_provenance"] = {
            **dict(provenance),
            "actual_sha256": actual_hash,
            "hash_verified": actual_hash == str(provenance.get("expected_sha256", "")).lower(),
        }
        payload["video_labels"] = (
            (inspect_interval_label_csv(video_source.label_path)
             if Path(video_source.label_path).suffix.lower() == '.csv'
             else inspect_label_workbook(video_source.label_path))
            if video_source.label_path else None
        )
    if eeg_source and eeg_source.eeg_path:
        payload["eeg_csv"] = inspect_eeg_csv(eeg_source.eeg_path)
    return payload


def validate_verified_pair(video: ModalitySource, eeg: ModalitySource) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if not video.has_video or not eeg.has_eeg:
        reasons.append("both modalities are required")
    if not video.is_verified_pair or not eeg.is_verified_pair:
        reasons.append("is_verified_pair is false")
    if not video.patient_id or not eeg.patient_id or video.patient_id != eeg.patient_id:
        reasons.append("patient_id is missing or does not match")
    if not video.recording_id or not eeg.recording_id or video.recording_id != eeg.recording_id:
        reasons.append("recording_id is missing or does not match")
    intervals_available = all(
        value is not None for value in (video.start_sec, video.end_sec, eeg.start_sec, eeg.end_sec)
    )
    if not intervals_available:
        reasons.append("timestamp interval is missing")
    elif min(float(video.end_sec), float(eeg.end_sec)) <= max(
        float(video.start_sec), float(eeg.start_sec)
    ):
        reasons.append("timestamp intervals do not overlap")
    return not reasons, reasons
