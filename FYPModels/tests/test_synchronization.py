from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

from seizure_multimodal.synchronization.alignment import (
    ClockMapping,
    assert_paired_windows,
    eeg_sample_bounds,
    overlapping_interval,
    validate_extracted_duration,
    video_frame_indices,
)
from seizure_multimodal.synchronization.manifest import read_manifest, validate_manifest, write_manifest
from seizure_multimodal.synchronization.splits import assert_patient_disjoint, assign_patient_splits
from seizure_multimodal.synchronization.windows import generate_windows, make_window_id
from seizure_multimodal.types import WindowRecord


def window(patient: str = "P001", split: str = "train", has_eeg: bool = True) -> WindowRecord:
    return WindowRecord(
        window_id=make_window_id("REC001", 30.0, 35.0),
        dataset="HUKM",
        patient_id=patient,
        recording_id="REC001",
        start_sec=30.0,
        end_sec=35.0,
        label="ictal",
        split=split,
        has_video=True,
        has_eeg=has_eeg,
        video_fps=30.0,
        eeg_sampling_rate=256.0,
    )


class SynchronizationTests(unittest.TestCase):
    def test_deterministic_window_id(self) -> None:
        self.assertEqual(make_window_id("REC001", 30, 35), "REC001__0000030000__0000035000")

    def test_windows_stay_inside_cross_modal_overlap(self) -> None:
        rows = list(
            generate_windows(
                "REC001",
                overlap_start_sec=2.0,
                overlap_end_sec=13.0,
                window_duration_sec=5.0,
                stride_sec=5.0,
            )
        )
        self.assertEqual([(start, end) for _, start, end in rows], [(2.0, 7.0), (7.0, 12.0)])

    def test_video_explicit_timestamps_handle_dropped_frames(self) -> None:
        timestamps = [0.0, 0.04, 0.08, 0.16, 0.20]
        np.testing.assert_array_equal(
            video_frame_indices(0.05, 0.19, frame_timestamps_sec=timestamps), [2, 3]
        )

    def test_video_and_eeg_use_timestamp_mapping_not_shared_indices(self) -> None:
        clock = ClockMapping(offset_sec=1.0, drift_ppm=100.0)
        frames = video_frame_indices(30.0, 35.0, fps=30.0, clock=clock)
        samples = eeg_sample_bounds(30.0, 35.0, sampling_rate=256.0, clock=clock)
        self.assertEqual(frames[0], 931)
        self.assertEqual(samples[0], 7937)
        self.assertGreater(len(frames), 0)
        self.assertGreater(samples[1] - samples[0], 0)

    def test_non_overlapping_recordings_fail(self) -> None:
        with self.assertRaisesRegex(ValueError, "do not overlap"):
            overlapping_interval(0, 10, 11, 20)

    def test_duration_mismatch_fails(self) -> None:
        with self.assertRaisesRegex(ValueError, "Video duration mismatch"):
            validate_extracted_duration(
                expected_duration_sec=5.0,
                video_timestamps_sec=[0.0, 4.0],
                eeg_sample_count=1280,
                eeg_sampling_rate=256,
                tolerance_sec=0.05,
            )

    def test_missing_modality_is_not_zero_filled(self) -> None:
        with self.assertRaisesRegex(ValueError, "both video and EEG"):
            validate_manifest([window(has_eeg=False)], require_paired=True)

    def test_patient_leakage_is_detected(self) -> None:
        rows = [window(split="train"), replace(window(split="test"), window_id=make_window_id("REC002", 30, 35), recording_id="REC002")]
        with self.assertRaisesRegex(ValueError, "Patient leakage"):
            assert_patient_disjoint(rows)

    def test_split_assignment_keeps_every_patient_in_one_split(self) -> None:
        rows = [
            replace(window(patient=f"P{i:03d}"), recording_id=f"REC{i:03d}", window_id=make_window_id(f"REC{i:03d}", 30, 35), split="unassigned")
            for i in range(20)
        ]
        assigned = assign_patient_splits(rows, seed=42)
        assert_patient_disjoint(assigned)

    def test_manifest_round_trip_and_pair_assertion(self) -> None:
        rows = [window()]
        assert_paired_windows(rows)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.csv"
            write_manifest(rows, path)
            loaded = read_manifest(path)
        self.assertEqual(loaded, rows)

    def test_preictal_is_never_invented(self) -> None:
        with self.assertRaisesRegex(ValueError, "explicit clinical label policy"):
            validate_manifest([replace(window(), label="pre_ictal")])


if __name__ == "__main__":
    unittest.main()
