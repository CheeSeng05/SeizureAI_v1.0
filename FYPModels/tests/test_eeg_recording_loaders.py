from __future__ import annotations

import importlib.util
import csv
import json
import struct
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

from seizure_multimodal.eeg.eeg_adapter import EEGAdapter, EEGPreprocessor
from seizure_multimodal.eeg.loaders.base import EEGAnnotation, EEGMetadata, EEGRecording
from seizure_multimodal.eeg.loaders.discovery import discover_eeg_files
from seizure_multimodal.eeg.loaders.factory import (
    UnsupportedEEGFormatError,
    detect_eeg_format,
    load_eeg,
)
from seizure_multimodal.eeg.loaders.nicolet_e_loader import (
    NicoletERecording,
    nicolet_events_to_annotations,
)
from seizure_multimodal.eeg.recording_inference import (
    EEGCompatibilityError,
    ModelEEGRequirements,
    assess_compatibility,
    convert_signal_units,
    prepare_model_windows,
    require_compatible,
    resolve_channel_indices,
    run_recording_inference,
)
from seizure_multimodal.eeg.montage import (
    BIPOLAR_CHANNELS,
    BipolarMontageError,
    apply_bipolar_montage,
    build_bipolar_montage_plan,
)


ROOT = Path(__file__).resolve().parents[1]


class SyntheticRecording(EEGRecording):
    def __init__(
        self,
        data: np.ndarray,
        *,
        names: tuple[str, ...],
        sampling_rate: float,
        units: tuple[str, ...],
        montage: str = "common",
    ) -> None:
        self.path = Path("synthetic.edf")
        self._data = np.asarray(data, dtype=np.float32)
        self.metadata = EEGMetadata(
            recording_id="recording-synthetic",
            source_format="synthetic",
            channel_names=names,
            sampling_rates=tuple(sampling_rate for _ in names),
            signal_units=units,
            duration_sec=self._data.shape[1] / sampling_rate,
            sample_count=self._data.shape[1],
            montage_reference=montage,
        )
        self.annotations = ()

    def read_samples(self, *, channel_indices, start_sample, stop_sample):
        return self._data[list(channel_indices), start_sample:stop_sample]


def _requirements(**changes) -> ModelEEGRequirements:
    values = {
        "sampling_rate": 256.0,
        "channel_count": 18,
        "required_channels": BIPOLAR_CHANNELS,
        "expected_units": "uV",
        "expected_montage": "CHB-MIT bipolar",
        "window_duration_sec": 4.0,
        "window_stride_sec": 2.0,
        "input_samples": 1024,
        "channel_aliases": {"T7": ("T3",), "T8": ("T4",), "P7": ("T5",), "P8": ("T6",)},
    }
    values.update(changes)
    return ModelEEGRequirements(**values)


LEGACY_COMPONENTS = (
    "Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2",
    "F7", "F8", "T3", "T4", "T5", "T6", "Fz", "Cz", "Pz",
)


class DiscoveryAndDetectionTests(unittest.TestCase):
    def test_recursive_e_discovery_does_not_assume_directory_names(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "arbitrary" / "nested" / "recording.e"
            second = root / "another-layout" / "recording.edf"
            first.parent.mkdir(parents=True)
            second.parent.mkdir(parents=True)
            first.write_bytes(b"private-placeholder")
            second.write_bytes(b"private-placeholder")
            (root / "ignore.txt").write_text("not EEG", encoding="utf-8")
            discovered = discover_eeg_files(root)
        self.assertEqual([path.suffix for path in discovered], [".edf", ".e"])

    def test_format_detection_uses_structure_not_e_extension_alone(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            invalid = root / "invalid.e"
            invalid.write_bytes(b"not a Nicolet file")
            with self.assertRaisesRegex(UnsupportedEEGFormatError, "Unsupported EEG"):
                detect_eeg_format(invalid)

            plausible = root / "plausible.e"
            header = bytearray(176)
            struct.pack_into("<I", header, 24, 0)
            struct.pack_into("<I", header, 172, 1)
            plausible.write_bytes(header)
            self.assertEqual(detect_eeg_format(plausible), "nicolet-e")

    def test_loader_factory_rejects_structural_candidate_when_reader_fails(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            candidate = Path(directory) / "candidate.e"
            header = bytearray(176)
            struct.pack_into("<I", header, 24, 0)
            struct.pack_into("<I", header, 172, 1)
            candidate.write_bytes(header)
            with self.assertRaisesRegex(RuntimeError, "not a supported Natus"):
                load_eeg(candidate)

    def test_factory_dispatches_edf_and_nicolet_readers(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            edf = root / "synthetic.edf"
            edf.write_bytes(b"0       " + bytes(168))
            nicolet = root / "synthetic.e"
            header = bytearray(176)
            struct.pack_into("<I", header, 24, 0)
            struct.pack_into("<I", header, 172, 1)
            nicolet.write_bytes(header)
            with mock.patch(
                "seizure_multimodal.eeg.loaders.factory.EDFRecording", return_value="edf-reader"
            ) as edf_reader, mock.patch(
                "seizure_multimodal.eeg.loaders.factory.NicoletERecording", return_value="nicolet-reader"
            ) as nicolet_reader:
                self.assertEqual(load_eeg(edf), "edf-reader")
                self.assertEqual(load_eeg(nicolet), "nicolet-reader")
            edf_reader.assert_called_once()
            nicolet_reader.assert_called_once()


class ChannelAndPreprocessingTests(unittest.TestCase):
    def test_standard_temporal_aliases_and_exact_order(self) -> None:
        plan = build_bipolar_montage_plan(LEGACY_COMPONENTS)
        self.assertEqual(plan.channel_order, BIPOLAR_CHANNELS)
        self.assertEqual(
            plan.aliases_used,
            {"T7": "T3", "T8": "T4", "P7": "T5", "P8": "T6"},
        )

    def test_modern_temporal_names_are_preferred_over_aliases(self) -> None:
        plan = build_bipolar_montage_plan((*LEGACY_COMPONENTS, "T7", "T8", "P7", "P8"))
        self.assertEqual(plan.aliases_used, {})

    def test_bipolar_subtraction_is_mathematically_correct(self) -> None:
        plan = build_bipolar_montage_plan(LEGACY_COMPONENTS)
        data = np.stack([
            np.full(5, index + 1, dtype=np.float32) for index in range(len(LEGACY_COMPONENTS))
        ])
        result = apply_bipolar_montage(data, plan)
        self.assertEqual(result.shape, (18, 5))
        fp1 = LEGACY_COMPONENTS.index("Fp1") + 1
        f7 = LEGACY_COMPONENTS.index("F7") + 1
        np.testing.assert_allclose(result[0], fp1 - f7)
        self.assertEqual(plan.channel_order[0], "FP1-F7")
        self.assertEqual(plan.channel_order[-1], "CZ-PZ")

    def test_missing_required_electrode_is_clear(self) -> None:
        with self.assertRaisesRegex(BipolarMontageError, "Missing required EEG electrode.*PZ"):
            build_bipolar_montage_plan(tuple(name for name in LEGACY_COMPONENTS if name != "Pz"))

    def test_physical_unit_conversion(self) -> None:
        data = np.array([[1.0], [2.0], [3.0]], dtype=np.float32)
        converted = convert_signal_units(data, ("V", "mV", "uV"), "uV")
        np.testing.assert_allclose(converted[:, 0], [1_000_000.0, 2_000.0, 3.0])
        with self.assertRaisesRegex(EEGCompatibilityError, "cannot be guessed"):
            convert_signal_units(data[:1], ("ADC",), "uV")

    @unittest.skipUnless(importlib.util.find_spec("scipy"), "SciPy unavailable")
    def test_resampling_and_model_tensor_dimensions(self) -> None:
        source_rate = 500.0
        seconds = 8
        samples = int(source_rate * seconds)
        time = np.arange(samples) / source_rate
        data = np.stack([
            np.sin(2 * np.pi * (index % 8 + 1) * time) + index
            for index in range(len(LEGACY_COMPONENTS))
        ])
        recording = SyntheticRecording(
            data,
            names=LEGACY_COMPONENTS,
            sampling_rate=source_rate,
            units=tuple("uV" for _ in LEGACY_COMPONENTS),
        )
        requirements = _requirements()
        prepared = prepare_model_windows(recording, requirements)
        self.assertEqual(prepared.windows.shape, (3, 1024, 18))
        np.testing.assert_allclose(prepared.start_seconds, [0.0, 2.0, 4.0])
        self.assertTrue(prepared.resampling_performed)
        self.assertEqual(tuple(prepared.selected_channels), BIPOLAR_CHANNELS)
        self.assertEqual(prepared.aliases_used["T7"], "T3")

    def test_incorrect_required_order_blocks_inference(self) -> None:
        recording = SyntheticRecording(
            np.zeros((len(LEGACY_COMPONENTS), 1024)),
            names=LEGACY_COMPONENTS,
            sampling_rate=256.0,
            units=tuple("uV" for _ in LEGACY_COMPONENTS),
        )
        requirements = _requirements(required_channels=tuple(reversed(BIPOLAR_CHANNELS)))
        compatibility = assess_compatibility(recording.metadata, requirements)
        with self.assertRaisesRegex(EEGCompatibilityError, "Clinical EEG inference blocked"):
            require_compatible(compatibility)

    @unittest.skipUnless(importlib.util.find_spec("scipy"), "SciPy unavailable")
    def test_existing_preprocessor_shape_is_preserved(self) -> None:
        rng = np.random.default_rng(4)
        transformed = EEGPreprocessor().transform(rng.normal(size=(2, 18, 1024)))
        self.assertEqual(transformed.shape, (2, 18, 1024))
        self.assertEqual(transformed.dtype, np.float32)


class AnnotationTests(unittest.TestCase):
    def test_point_events_and_elapsed_time_are_preserved(self) -> None:
        start = datetime(2026, 1, 1, tzinfo=timezone.utc)
        events = [
            SimpleNamespace(
                IDStr="Annotation",
                annotation="FIT 1",
                label="",
                date=start + timedelta(seconds=12.5),
                duration=0.0,
                GUID="synthetic-guid",
                user="reviewer",
            ),
            SimpleNamespace(
                IDStr="Spike",
                annotation=None,
                label="Spike",
                date=start + timedelta(seconds=18),
                duration=2.0,
                GUID="synthetic-guid-2",
                user="",
            ),
        ]
        annotations = nicolet_events_to_annotations(
            events,
            recording_id="recording-synthetic",
            recording_start=start,
        )
        self.assertEqual(annotations[0].time_sec, 12.5)
        self.assertTrue(annotations[0].is_point_event)
        self.assertEqual(annotations[0].normalized_type, "candidate_seizure_event")
        self.assertEqual(annotations[1].normalized_type, "eeg_event")
        self.assertFalse(annotations[1].is_point_event)


@unittest.skipUnless(importlib.util.find_spec("scipy"), "SciPy unavailable")
class RecordingInferenceTests(unittest.TestCase):
    def test_batched_probability_timeline_uses_no_annotations_as_input(self) -> None:
        rate = 500.0
        seconds = 8
        time = np.arange(int(rate * seconds)) / rate
        data = np.stack([
            np.sin(2 * np.pi * (index % 6 + 1) * time) + index
            for index in range(len(LEGACY_COMPONENTS))
        ])
        recording = SyntheticRecording(
            data,
            names=LEGACY_COMPONENTS,
            sampling_rate=rate,
            units=tuple("uV" for _ in LEGACY_COMPONENTS),
        )
        recording.annotations = (
            EEGAnnotation(
                recording_id=recording.metadata.recording_id,
                event_id="event-00001",
                time_sec=1.0,
                duration_sec=0.0,
                raw_name="FIT TEST",
                normalized_type="candidate_seizure_event",
                annotation_source="synthetic",
            ),
        )
        captured_shapes = []

        class FakeAdapter:
            def __init__(self, **_kwargs):
                self.is_inference_only = True

            def predict_probabilities(self, model_input, *, preprocessed):
                self.assert_preprocessed = preprocessed
                captured_shapes.append(model_input.shape)
                probability = np.linspace(0.25, 0.75, len(model_input), dtype=np.float32)
                return {"probability": probability, "logits": probability - 0.5}

        config = {
            "training": {"enabled": False},
            "testing": {"threshold": 0.5},
            "eeg_encoder": {
                "checkpoint": "checkpoint.keras",
                "sampling_rate": 256,
                "channels": 18,
                "window_duration_sec": 4,
                "window_stride_sec": 2,
                "required_channels": list(BIPOLAR_CHANNELS),
                "channel_aliases": {"T7": ["T3"], "T8": ["T4"], "P7": ["T5"], "P8": ["T6"]},
                "signal_units": "uV",
                "montage_reference": "CHB-MIT bipolar",
                "inference_batch_size": 2,
            },
            "eeg_annotations": {"overrides": {}},
        }
        with tempfile.TemporaryDirectory() as directory, mock.patch(
            "seizure_multimodal.eeg.recording_inference.load_eeg", return_value=recording
        ), mock.patch(
            "seizure_multimodal.eeg.recording_inference.EEGAdapter", FakeAdapter
        ):
            output = Path(directory) / "output"
            payload = run_recording_inference(
                input_path="synthetic.e", project_root=directory,
                config=config, output_root=output, batch_size=2,
            )
            with (output / "predictions.csv").open(encoding="utf-8", newline="") as stream:
                rows = list(csv.DictReader(stream))
            annotations = (output / "annotations.csv").read_text(encoding="utf-8")

        self.assertEqual(captured_shapes, [(2, 1024, 18), (1, 1024, 18)])
        self.assertEqual([int(row["predicted_label"]) for row in rows], [0, 1, 0])
        self.assertTrue(all(row["probability"] for row in rows))
        self.assertIn("FIT TEST", annotations)
        self.assertFalse(payload["metadata"]["annotations_passed_to_model"])
        self.assertEqual(payload["metadata"]["model_input_shape"], [3, 1024, 18])


@unittest.skipUnless(importlib.util.find_spec("nicolet_e2edf"), "Nicolet reader unavailable")
class NicoletAdapterTests(unittest.TestCase):
    def test_synthetic_reader_contract_without_private_data(self) -> None:
        start = datetime(2026, 1, 1, tzinfo=timezone.utc)
        public = {
            "label": ["A", "B"],
            "Fs": 500.0,
            "nSamples": 2_000,
        }
        raw = SimpleNamespace(
            format="nicolet-e",
            matchingChannels=[4, 7],
            excludedChannels=[],
            startDateTime=start,
            reference="common",
            Segments=[object()],
            Events=[],
        )

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "synthetic.e"
            path.write_bytes(bytes(176))
            with mock.patch(
                "nicolet_e2edf.nicolet.header.read_nervus_header",
                return_value=(public, raw),
            ), mock.patch(
                "nicolet_e2edf.nicolet.data.read_nervus_data",
                return_value=np.ones((2, 500), dtype=np.float32),
            ):
                recording = NicoletERecording(path)
                samples = recording.read_samples(
                    channel_indices=(0, 1), start_sample=0, stop_sample=500
                )

        self.assertEqual(recording.metadata.source_format, "nicolet-e")
        self.assertEqual(recording.metadata.channel_names, ("A", "B"))
        self.assertEqual(recording.metadata.sampling_rate, 500.0)
        self.assertEqual(recording.metadata.signal_units, ("uV", "uV"))
        self.assertEqual(samples.shape, (2, 500))


@unittest.skipUnless(importlib.util.find_spec("pyedflib"), "PyEDFlib unavailable")
class EDFRegressionTests(unittest.TestCase):
    def test_synthetic_edf_loading_and_annotation(self) -> None:
        import pyedflib.highlevel

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "synthetic.edf"
            rate = 256
            signals = np.vstack((np.linspace(-10, 10, rate * 4), np.linspace(5, -5, rate * 4)))
            headers = pyedflib.highlevel.make_signal_headers(
                ["A", "B"],
                sample_frequency=rate,
                physical_min=-100,
                physical_max=100,
                dimension="uV",
            )
            header = pyedflib.highlevel.make_header()
            header["annotations"] = [[1.5, 0.0, "Synthetic event"]]
            self.assertTrue(pyedflib.highlevel.write_edf(str(path), signals, headers, header))
            with load_eeg(path) as recording:
                samples = recording.read_samples(
                    channel_indices=(1, 0), start_sample=0, stop_sample=1024
                )
                self.assertEqual(recording.metadata.channel_names, ("A", "B"))
                self.assertEqual(samples.shape, (2, 1024))
                self.assertEqual(len(recording.annotations), 1)
                self.assertTrue(recording.annotations[0].is_point_event)


@unittest.skipUnless(importlib.util.find_spec("tensorflow"), "TensorFlow unavailable")
class CheckpointRegressionTests(unittest.TestCase):
    def test_pretrained_checkpoint_loads_frozen_and_predicts(self) -> None:
        checkpoint = ROOT / "checkpoints" / "pretrained" / "eeg" / "seizure_detector_model.keras"
        adapter = EEGAdapter(checkpoint_path=checkpoint)
        output = adapter.predict(np.zeros((1, 1024, 18), dtype=np.float32), preprocessed=True)
        self.assertTrue(adapter.is_inference_only)
        self.assertEqual(np.asarray(output["probability"]).shape, (1,))
        self.assertEqual(np.asarray(output["embedding"]).shape, (1, 32))


if __name__ == "__main__":
    unittest.main()
