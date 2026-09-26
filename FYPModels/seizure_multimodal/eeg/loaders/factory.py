from __future__ import annotations

import struct
from pathlib import Path
from typing import Mapping

from .base import EEGRecording, UnsupportedEEGFormatError
from .edf_loader import EDFRecording
from .nicolet_e_loader import NicoletERecording


def detect_eeg_format(path: str | Path) -> str:
    """Conservatively identify EDF or a structurally plausible Nicolet `.e`."""

    candidate = Path(path)
    if not candidate.is_file():
        raise FileNotFoundError(candidate)
    with candidate.open("rb") as stream:
        header = stream.read(176)
    if len(header) >= 8 and header[:8] == b"0       ":
        return "edf"
    if candidate.suffix.casefold() == ".e" and len(header) >= 176:
        index_offset = struct.unpack_from("<I", header, 24)[0]
        static_packet_count = struct.unpack_from("<I", header, 172)[0]
        if 0 < static_packet_count < 100_000 and (
            index_offset == 0 or index_offset < candidate.stat().st_size
        ):
            return "nicolet-e"
    raise UnsupportedEEGFormatError(
        "Unsupported EEG container; .e is accepted only when its Nicolet structure is verified."
    )


def load_eeg(
    path: str | Path,
    *,
    annotation_overrides: Mapping[str, str] | None = None,
) -> EEGRecording:
    detected = detect_eeg_format(path)
    if detected == "edf":
        return EDFRecording(path, annotation_overrides=annotation_overrides)
    if detected == "nicolet-e":
        return NicoletERecording(path, annotation_overrides=annotation_overrides)
    raise UnsupportedEEGFormatError(f"No loader registered for {detected}.")

