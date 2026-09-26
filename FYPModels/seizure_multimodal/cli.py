from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .checkpoints.inspection import inspect_checkpoint_archive, serializable_state_dict_metadata
from .config import load_config, validate_training_policy
from .eeg.eeg_adapter import EEGAdapter
from .eeg.recording_inference import run_first_recording_inference, run_recording_inference
from .smoke import run_synthetic_smoke_test
from .synchronization.manifest import read_manifest, validate_manifest
from .testing import run_test_mode
from .video.vsvig_adapter import VSViGAdapter


def _video_inference(args: argparse.Namespace) -> None:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required to read VSViG .pt inputs") from exc
    with VSViGAdapter(
        source_path=args.source,
        checkpoint_path=args.checkpoint,
        dynamic_partition_path=args.dynamic_partition,
        device=args.device,
    ) as adapter:
        output = adapter.predict(
            torch.load(args.patches, map_location="cpu", weights_only=True),
            torch.load(args.keypoints, map_location="cpu", weights_only=True),
        )
        print(
            json.dumps(
                {
                    "embedding_shape": list(output["embedding"].shape),
                    "logits": output["logits"].cpu().tolist(),
                    "probability": output["probability"].cpu().tolist(),
                    "inference_only": adapter.is_inference_only,
                },
                indent=2,
            )
        )


def _eeg_inference(args: argparse.Namespace) -> None:
    adapter = EEGAdapter(checkpoint_path=args.checkpoint)
    output = adapter.predict(np.load(args.input), preprocessed=args.preprocessed)
    print(
        json.dumps(
            {
                "embedding_shape": list(output["embedding"].shape),
                "logits": output["logits"].tolist(),
                "probability": output["probability"].tolist(),
                "inference_only": adapter.is_inference_only,
            },
            indent=2,
        )
    )


def _recording_eeg_inference(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    if args.input:
        result = run_recording_inference(
            input_path=args.input, project_root=args.root, config=config,
            output_root=args.output, batch_size=args.batch_size,
        )
    else:
        result = run_first_recording_inference(
            dataset_root=args.dataset_root, project_root=args.root, config=config,
            output_root=args.output, batch_size=args.batch_size,
        )
    print(json.dumps(result, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="FYP multimodal seizure integration")
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect = subparsers.add_parser("inspect-checkpoint")
    inspect.add_argument("path")
    inspect.add_argument("--state-shapes", action="store_true")

    manifest = subparsers.add_parser("validate-manifest")
    manifest.add_argument("path")
    manifest.add_argument("--require-paired", action="store_true")
    manifest.add_argument("--require-files", action="store_true")

    smoke = subparsers.add_parser("synthetic-smoke")
    smoke.add_argument("--output", default="results/synthetic_smoke")

    guard = subparsers.add_parser("check-training-policy")
    guard.add_argument("config")
    guard.add_argument("--paired-data-verified", action="store_true")

    video = subparsers.add_parser("video-infer")
    video.add_argument("--source", default="VSViG-main/VSViG.py")
    video.add_argument("--checkpoint", default="VSViG-main/VSViG-base.pth")
    video.add_argument("--dynamic-partition", default="VSViG-main/dy_point_order.pt")
    video.add_argument("--patches", required=True)
    video.add_argument("--keypoints", required=True)
    video.add_argument("--device", default="cpu")

    eeg = subparsers.add_parser("eeg-infer")
    eeg.add_argument("--checkpoint", required=True)
    eeg.add_argument("--input", required=True)
    eeg.add_argument("--preprocessed", action="store_true")

    recording_eeg = subparsers.add_parser(
        "eeg-recording-infer",
        help="Run one compatible EDF or Nicolet .e recording",
    )
    recording_eeg.add_argument("--input", help="Specific .edf or .e recording")
    recording_eeg.add_argument(
        "--dataset-root",
        help="Private dataset root; defaults to EEG_DATA_ROOT",
    )
    recording_eeg.add_argument("--config", default="configs/inference.yaml")
    recording_eeg.add_argument("--root", default=".")
    recording_eeg.add_argument("--output", default="results/private")
    recording_eeg.add_argument("--batch-size", type=int, default=64)

    test = subparsers.add_parser("test", help="Run separated unimodal/strict/simulated testing")
    test.add_argument(
        "--mode",
        choices=("unimodal", "video", "eeg", "strict_multimodal", "simulated_multimodal"),
        default="unimodal",
    )
    test.add_argument("--config", default="configs/inference.yaml")
    test.add_argument("--root", default=".")
    test.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    test.add_argument("--allow-simulated-pairing", action="store_true")
    test.add_argument("--calculate-artificial-metrics", action="store_true")
    test.add_argument("--reference-labels", choices=("video", "eeg"))

    args = parser.parse_args()
    if args.command == "inspect-checkpoint":
        payload = inspect_checkpoint_archive(args.path)
        if args.state_shapes:
            payload["state_dict"] = serializable_state_dict_metadata(args.path)
        print(json.dumps(payload, indent=2))
    elif args.command == "validate-manifest":
        rows = validate_manifest(
            read_manifest(args.path),
            require_paired=args.require_paired,
            require_files=args.require_files,
        )
        print(f"Validated {len(rows)} timestamp-aligned windows with no patient leakage")
    elif args.command == "synthetic-smoke":
        print(json.dumps(run_synthetic_smoke_test(args.output), indent=2))
    elif args.command == "check-training-policy":
        validate_training_policy(
            load_config(args.config), paired_data_verified=args.paired_data_verified
        )
        print("Training policy permits future fusion training for verified paired data")
    elif args.command == "video-infer":
        _video_inference(args)
    elif args.command == "eeg-infer":
        _eeg_inference(args)
    elif args.command == "eeg-recording-infer":
        _recording_eeg_inference(args)
    elif args.command == "test":
        config = load_config(args.config)
        config.setdefault("testing", {})["device"] = args.device
        try:
            results = run_test_mode(
                root=args.root,
                config=config,
                mode=args.mode,
                allow_simulated_pairing=args.allow_simulated_pairing,
                calculate_artificial_metrics=(
                    True if args.calculate_artificial_metrics else None
                ),
                reference_labels=args.reference_labels,
            )
        except PermissionError as exc:
            parser.error(str(exc))
        summary = {
            name: {"status": result.status, "error": result.error}
            for name, result in results.items()
        }
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
