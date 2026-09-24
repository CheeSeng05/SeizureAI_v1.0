"""FaceMap 3DMM landmarks and coefficients from an already-extracted face crop.

No face detection, ROI selection, padding, or recropping happens here. Pass the
unannotated uint8 crop produced by preview_microexpressions or face_preprocess.
The optional crop_box uses this project's (x1, y1, x2, y2) EXCLUSIVE-end pixel
coordinates, unlike the inclusive bounds in Qualcomm's demo.

The network and projection are adapted from Qualcomm AI Hub Models v0.62.2:
https://github.com/qualcomm/ai-hub-models/tree/v0.62.2/src/qai_hub_models/models/facemap_3dmm
Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
SPDX-License-Identifier: BSD-3-Clause. See facemap_LICENSE.

FaceMap returns 68 landmarks, not MediaPipe's 478-point mesh. Its indices and
pose convention must not be passed to the existing MediaPipe geometry helpers.
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn

DEFAULT_MODEL_DIR = Path(__file__).resolve().parents[2] / "checkpoints/pretrained/facemap_3dmm"
CHECKPOINT_NAME = "resnet_wd2_weak_score_1202_3ch.pth.tar"
ASSET_NAMES = (CHECKPOINT_NAME, "meanFace.npy", "shapeBasis.npy", "blendShape.npy")
ASSET_BASE_URL = "https://qaihub-public-assets.s3.us-west-2.amazonaws.com/qai-hub-models/models/facemap_3dmm/v1"
INPUT_SIZE = 128


class _ConvBlock(nn.Module):
    def __init__(self, inputs, outputs, kernel, *, stride=1, padding=0,
                 use_bn=True, activate=True, bias=False):
        super().__init__()
        self.conv = nn.Conv2d(inputs, outputs, kernel, stride, padding, bias=bias)
        if use_bn:
            self.bn = nn.BatchNorm2d(outputs)
        if activate:
            self.activ = nn.ReLU(inplace=True)

    def forward(self, value):
        value = self.conv(value)
        if hasattr(self, "bn"):
            value = self.bn(value)
        return self.activ(value) if hasattr(self, "activ") else value


class _ResBlock(nn.Module):
    def __init__(self, inputs, outputs, stride):
        super().__init__()
        self.conv1 = _ConvBlock(inputs, outputs, 3, stride=stride, padding=1)
        self.conv2 = _ConvBlock(outputs, outputs, 3, padding=1, activate=False)

    def forward(self, value):
        return self.conv2(self.conv1(value))


class _ResUnit(nn.Module):
    def __init__(self, inputs, outputs, stride):
        super().__init__()
        self.body = _ResBlock(inputs, outputs, stride)
        if inputs != outputs or stride != 1:
            self.identity_conv = _ConvBlock(inputs, outputs, 1, stride=stride, activate=False)
        self.activ = nn.ReLU(inplace=True)

    def forward(self, value):
        identity = self.identity_conv(value) if hasattr(self, "identity_conv") else value
        return self.activ(self.body(value) + identity)


class _ResInitBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = _ConvBlock(1, 32, 7, stride=2, padding=3)
        self.pool = nn.Sequential(nn.ZeroPad2d((0, 1, 0, 1)), nn.MaxPool2d(3, 2))

    def forward(self, value):
        return self.pool(self.conv(value))


class _FaceMapNetwork(nn.Module):
    """Checkpoint-compatible ResNet18 width-0.5, including its final 512 unit."""

    def __init__(self):
        super().__init__()
        self.rgb2gray_block = _ConvBlock(3, 1, 1, use_bn=False, activate=False, bias=True)
        self.features = nn.Sequential()
        self.features.add_module("init_block", _ResInitBlock())
        inputs = 32
        for index, widths in enumerate(((32, 32), (64, 64), (128, 128), (256, 512))):
            stage = nn.Sequential()
            for unit, outputs in enumerate(widths):
                stride = 2 if index > 0 and unit == 0 else 1
                stage.add_module(f"unit{unit + 1}", _ResUnit(inputs, outputs, stride))
                inputs = outputs
            self.features.add_module(f"stage{index + 1}", stage)
        self.features.add_module("final_pool", nn.AvgPool2d(4, 1))
        self.output = nn.Linear(512, 265)

    def forward(self, image):
        # Match model.py: callers supply RGB floats in [0, 1], the original
        # resnet_score_rgb network consumes RGB values in [0, 255].
        value = self.features(self.rgb2gray_block(image * 255))
        return self.output(value.flatten(1))


def prepare_face_crop(face_crop: np.ndarray, *, color_space: str = "bgr") -> torch.Tensor:
    """Resize the entire existing crop to [1,3,128,128] RGB float32 in [0,1]."""
    if (not isinstance(face_crop, np.ndarray) or face_crop.dtype != np.uint8
            or face_crop.ndim != 3 or face_crop.shape[2] != 3
            or min(face_crop.shape[:2]) == 0):
        raise ValueError("face_crop must be a nonempty uint8 H×W×3 image")
    if color_space not in ("bgr", "rgb"):
        raise ValueError("color_space must be 'bgr' or 'rgb'")
    rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB) if color_space == "bgr" else face_crop
    resized = cv2.resize(rgb, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_LINEAR)
    return torch.from_numpy(np.ascontiguousarray(resized.transpose(2, 0, 1))).float().unsqueeze(0) / 255


def project_landmarks(parameters: np.ndarray, mean_face: np.ndarray,
                      shape_basis: np.ndarray, expression_basis: np.ndarray) -> np.ndarray:
    """Reconstruct 68 landmarks in FaceMap's centered 128×128 image space.

    Match the upstream weak-perspective projection with fixed tZ=500. The
    265th output is preserved separately; upstream does not use it for geometry.
    """
    values = np.asarray(parameters, dtype=np.float32)
    if values.shape != (265,) or not np.isfinite(values).all():
        raise ValueError("FaceMap parameters must be 265 finite values")
    pitch, yaw, roll = values[258:261] * (np.pi / 2)
    cp, sp = math.cos(-pitch), math.sin(-pitch)
    cy, sy = math.cos(-yaw), math.sin(-yaw)
    cr, sr = math.cos(-roll), math.sin(-roll)
    # The reference applies a -pi X rotation between pitch and roll.
    rotation = (
        np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], np.float32)
        @ np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]], np.float32)
        @ np.diag(np.array([1, -1, -1], np.float32))
        @ np.array([[cr, -sr, 0], [sr, cr, 0], [0, 0, 1]], np.float32)
    )
    vertices = (
        mean_face.reshape(204)
        + shape_basis.reshape(204, 219) @ (values[:219] * 3)
        + expression_basis.reshape(204, 39) @ (values[219:258] * .5 + .5)
    ).reshape(68, 3) @ rotation.T
    vertices[:, :2] += values[261:263] * 60
    points = vertices[:, :2] * (values[263] * 150 + 450) / 500
    if not np.isfinite(points).all():
        raise ValueError("FaceMap projection produced nonfinite landmarks")
    return points.astype(np.float32)


def map_landmarks(points: np.ndarray, crop_box) -> np.ndarray:
    """Map centered model coordinates into an exclusive-end xyxy pixel box.

    crop_box may describe the native crop before upscaling: the mapping uses
    its original width and height, not the enhanced crop's pixel dimensions.
    """
    box = np.asarray(crop_box, dtype=np.float32)
    if box.shape != (4,) or not np.isfinite(box).all() or np.any(box[2:] <= box[:2]):
        raise ValueError("crop_box must be finite (x1,y1,x2,y2) with positive width and height")
    points = np.asarray(points, dtype=np.float32)
    if points.shape != (68, 2) or not np.isfinite(points).all():
        raise ValueError("points must have shape (68,2) and contain finite coordinates")
    return (points + INPUT_SIZE / 2) * ((box[2:] - box[:2]) / INPUT_SIZE) + box[:2]


@dataclass(frozen=True)
class FaceMapFeatures:
    landmarks_crop: np.ndarray  # [68,2], pixels in the supplied crop
    landmarks_frame: np.ndarray | None  # [68,2], original frame pixels when a box was supplied
    parameters: np.ndarray  # [265], raw model output; the last value is not a validated confidence
    identity_coefficients: np.ndarray  # [219], de-normalized
    expression_coefficients: np.ndarray  # [39], de-normalized blendshape coefficients
    pose_degrees: np.ndarray  # [3], FaceMap pitch/yaw/roll, not the preview's solvePnP convention

    def as_dict(self) -> dict:
        return {name: value.tolist() if value is not None else None
                for name, value in vars(self).items()}


class FaceMapFeatureExtractor:
    """Load the model once, then extract features from each existing face crop."""

    def __init__(self, model_dir: str | Path = DEFAULT_MODEL_DIR, *, device: str = "cpu"):
        directory = Path(model_dir).resolve()
        missing = [name for name in ASSET_NAMES if not (directory / name).is_file()]
        if missing:
            raise FileNotFoundError(
                f"Missing FaceMap assets in {directory}: {', '.join(missing)}. "
                f"Download these filenames from {ASSET_BASE_URL}/; see preprocess/README.md."
            )
        self.device = torch.device(device)
        self.model = _FaceMapNetwork()
        checkpoint = torch.load(directory / CHECKPOINT_NAME, map_location="cpu", weights_only=True)
        self.model.load_state_dict(checkpoint["state_dict"], strict=True)
        self.model.to(self.device).eval()
        arrays = []
        for name, shape in (("meanFace.npy", (204,)), ("shapeBasis.npy", (204, 219)),
                            ("blendShape.npy", (204, 39))):
            array = np.load(directory / name, allow_pickle=False).astype(np.float32).reshape(shape)
            if not np.isfinite(array).all():
                raise ValueError(f"Nonfinite values in FaceMap asset {name}")
            arrays.append(array)
        self.mean_face, self.shape_basis, self.expression_basis = arrays

    def extract(self, face_crop: np.ndarray, *, crop_box=None,
                color_space: str = "bgr") -> FaceMapFeatures:
        """Extract from unannotated crop pixels; never substitute a whole frame.

        crop_box describes the region in the original frame in xyxy order, with
        x2/y2 exclusive. Omit it to return crop-local landmarks only. No quality
        acceptance is implied by a model prediction; inspect landmark alignment.
        """
        image = prepare_face_crop(face_crop, color_space=color_space)
        if crop_box is not None:
            # Validate coordinates before model inference.
            map_landmarks(np.zeros((68, 2), np.float32), crop_box)
        with torch.inference_mode():
            output = self.model(image.to(self.device))
        if output.shape != (1, 265):
            raise ValueError(f"Expected FaceMap output [1,265], got {tuple(output.shape)}")
        parameters = output[0].detach().cpu().numpy().astype(np.float32, copy=True)
        points = project_landmarks(parameters, self.mean_face, self.shape_basis, self.expression_basis)
        height, width = face_crop.shape[:2]
        return FaceMapFeatures(
            landmarks_crop=map_landmarks(points, (0, 0, width, height)),
            landmarks_frame=map_landmarks(points, crop_box) if crop_box is not None else None,
            parameters=parameters,
            identity_coefficients=parameters[:219] * 3,
            expression_coefficients=parameters[219:258] * .5 + .5,
            pose_degrees=parameters[258:261] * 90,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("crop", type=Path, help="Already-cropped face image; this script does not detect or crop faces.")
    parser.add_argument("--output", type=Path, required=True, help="Output feature JSON.")
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--crop-box", help="Original-frame x1,y1,x2,y2 pixel bounds, with x2/y2 exclusive.")
    args = parser.parse_args()
    box = None
    if args.crop_box:
        try:
            box = tuple(float(value) for value in args.crop_box.split(","))
            map_landmarks(np.zeros((68, 2), np.float32), box)
        except ValueError as exc:
            parser.error(str(exc))
    if args.output.resolve() == args.crop.resolve():
        parser.error("--output must not overwrite the input crop")
    crop = cv2.imread(str(args.crop))
    if crop is None:
        parser.error(f"Cannot read crop: {args.crop}")
    extractor = FaceMapFeatureExtractor(args.model_dir, device=args.device)
    result = extractor.extract(crop, crop_box=box)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result.as_dict(), indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Saved 68 FaceMap landmarks and 3DMM coefficients to {args.output}")


if __name__ == "__main__":
    main()
