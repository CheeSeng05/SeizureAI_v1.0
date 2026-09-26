from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np


BIPOLAR_CHANNELS = (
    "FP1-F7",
    "F7-T7",
    "T7-P7",
    "P7-O1",
    "FP1-F3",
    "F3-C3",
    "C3-P3",
    "P3-O1",
    "FP2-F4",
    "F4-C4",
    "C4-P4",
    "P4-O2",
    "FP2-F8",
    "F8-T8",
    "T8-P8",
    "P8-O2",
    "FZ-CZ",
    "CZ-PZ",
)

BIPOLAR_PAIRS = tuple(tuple(name.split("-", 1)) for name in BIPOLAR_CHANNELS)

# Modern 10-10 names are preferred. These older 10-20 labels are used only
# when the corresponding modern label is absent.
STANDARD_ELECTRODE_ALIASES: Mapping[str, tuple[str, ...]] = {
    "T7": ("T3",),
    "T8": ("T4",),
    "P7": ("T5",),
    "P8": ("T6",),
}


class BipolarMontageError(ValueError):
    """Raised when the required CHB-MIT montage cannot be constructed."""


def channel_key(value: str) -> str:
    return " ".join(str(value).strip().casefold().split())


@dataclass(frozen=True)
class BipolarMontagePlan:
    channel_order: tuple[str, ...]
    source_pairs: tuple[tuple[int, int] | None, ...]
    direct_indices: tuple[int | None, ...]
    resolved_source_names: tuple[tuple[str, ...], ...]
    aliases_used: Mapping[str, str]

    @property
    def uses_existing_bipolar_channels(self) -> bool:
        return all(index is not None for index in self.direct_indices)

    @property
    def source_indices(self) -> tuple[int, ...]:
        values: list[int] = []
        for direct, pair in zip(self.direct_indices, self.source_pairs):
            if direct is not None:
                values.append(direct)
            elif pair is not None:
                values.extend(pair)
        return tuple(dict.fromkeys(values))


def _unique_match(available_by_key: Mapping[str, list[int]], candidates: Sequence[str]) -> int | None:
    for candidate in candidates:
        matches = available_by_key.get(channel_key(candidate), ())
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise BipolarMontageError(f"Ambiguous EEG electrode/channel: {candidate}")
    return None


def build_bipolar_montage_plan(
    available_channels: Sequence[str],
    *,
    aliases: Mapping[str, Sequence[str]] | None = None,
) -> BipolarMontagePlan:
    """Resolve the exact frozen-model montage without inventing electrodes."""

    available = tuple(str(name).strip() for name in available_channels)
    by_key: dict[str, list[int]] = {}
    for index, name in enumerate(available):
        by_key.setdefault(channel_key(name), []).append(index)

    # CHB-MIT EDF files may already contain the exact bipolar derivations.
    direct = tuple(_unique_match(by_key, (name,)) for name in BIPOLAR_CHANNELS)
    if all(index is not None for index in direct):
        indices = tuple(int(index) for index in direct)
        if len(set(indices)) != len(BIPOLAR_CHANNELS):
            raise BipolarMontageError("Exact bipolar EEG channels are duplicated or ambiguous.")
        return BipolarMontagePlan(
            channel_order=BIPOLAR_CHANNELS,
            source_pairs=tuple(None for _ in BIPOLAR_CHANNELS),
            direct_indices=indices,
            resolved_source_names=tuple((available[index],) for index in indices),
            aliases_used={},
        )

    configured_aliases = {
        str(name).upper(): tuple(str(value) for value in values)
        for name, values in (aliases or STANDARD_ELECTRODE_ALIASES).items()
    }
    electrodes = tuple(dict.fromkeys(value for pair in BIPOLAR_PAIRS for value in pair))
    resolved: dict[str, int] = {}
    aliases_used: dict[str, str] = {}
    missing: list[str] = []
    for electrode in electrodes:
        candidates = (electrode, *configured_aliases.get(electrode, ()))
        index = _unique_match(by_key, candidates)
        if index is None:
            missing.append(electrode)
            continue
        resolved[electrode] = index
        if channel_key(available[index]) != channel_key(electrode):
            aliases_used[electrode] = available[index]
    if missing:
        raise BipolarMontageError(
            "Missing required EEG electrode(s): " + ", ".join(missing)
        )

    pairs = tuple((resolved[first], resolved[second]) for first, second in BIPOLAR_PAIRS)
    names = tuple((available[first], available[second]) for first, second in pairs)
    return BipolarMontagePlan(
        channel_order=BIPOLAR_CHANNELS,
        source_pairs=pairs,
        direct_indices=tuple(None for _ in BIPOLAR_CHANNELS),
        resolved_source_names=names,
        aliases_used=aliases_used,
    )


def apply_bipolar_montage(
    source_data: np.ndarray,
    plan: BipolarMontagePlan,
) -> np.ndarray:
    """Return `[18, samples]` in the explicit frozen-model channel order."""

    data = np.asarray(source_data)
    if data.ndim != 2:
        raise ValueError("Bipolar montage input must have shape [channels, samples].")
    rows: list[np.ndarray] = []
    for direct, pair in zip(plan.direct_indices, plan.source_pairs):
        if direct is not None:
            rows.append(data[direct])
        elif pair is not None:
            rows.append(data[pair[0]] - data[pair[1]])
        else:  # pragma: no cover - protected by plan construction
            raise RuntimeError("Invalid bipolar montage plan.")
    result = np.asarray(rows, dtype=np.float32)
    if result.shape[0] != len(BIPOLAR_CHANNELS) or plan.channel_order != BIPOLAR_CHANNELS:
        raise RuntimeError("Bipolar montage order/count verification failed.")
    return result
