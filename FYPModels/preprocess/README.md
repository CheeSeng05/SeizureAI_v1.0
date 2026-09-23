# Video Enhancement and Facial Detection

This module integrates [yangt1013/SPEINet](https://github.com/yangt1013/SPEINet)
before facial preprocessing:

```text
Decoded video at source FPS
  -> SPEINet full-frame enhancement using neighbouring frames
  -> sampled enhanced frames with original timestamps and dimensions
  -> full-frame MediaPipe Face Detection / patient selection
  -> pose detection, using the selected face to guide patient matching
  -> face landmarks, stabilized face crops, RGB + optical-flow tensors
```

`FYPModels.preprocess._facial.face_preprocess.extract_video()` calls this module directly.
The enhanced frames pass in memory to the facial pipeline. The original video
path and hash remain in the output manifest/run record. SPEINet settings,
checkpoint hash, source revision and software versions are also recorded.

## Setup and invocation

From `FYPModels`, install the additional dependencies into your facial environment:

```bash
python -m pip install -e '.[enhancement]'
```

The author's README links to [GoProRS model weights](https://drive.google.com/drive/folders/1AfAH4Fmj1DE0tcxOCMssCqCWuIcptkBI).
The SPEINet file is named `speinet_model_best`; save its PyTorch state dictionary
as `checkpoints/pretrained/speinet/model_best.pt`, or pass another path with
`--speinet-checkpoint`. The cached local copy's hash and source information are
in `checkpoints/pretrained/speinet/source.json`. Weights are excluded from Git.

Enable the stage through the existing entry point:

```bash
python preprocess/_facial/face_preprocess.py \
  "datasets/Seizure videos of epilepsy patients/Seizure/S1_5_136.mp4" \
  --output results/facial_speinet \
  --patient-id S1 --label 1 --split unassigned \
  --fps 5 --window-seconds 1 --max-duration-seconds 1 \
  --video-enhancement speinet \
  --speinet-checkpoint checkpoints/pretrained/speinet/model_best.pt \
  --speinet-device cpu
```

The enhancement flags also work with `FYPModels.preprocess.prepare_dataset`. Omit
`--video-enhancement` for the original pipeline. The earlier `--upscale-model`
and `--sharpen-amount` flags independently control optional face-crop operations
later in the pipeline; leave them at defaults when evaluating SPEINet alone.

For Python callers:

```python
from FYPModels.preprocess import SPEINetConfig
from FYPModels.preprocess._facial.face_preprocess import extract_video

summary = extract_video(
    video_path="input.mp4", output="results/facial_speinet", sample_id="sample",
    pose_model_path="checkpoints/pretrained/mediapipe/pose_landmarker_full.task",
    face_model_path="checkpoints/pretrained/mediapipe/face_landmarker.task",
    video_enhancement=SPEINetConfig(checkpoint="path/to/model_best.pt", device="cpu"),
)
```

`SPEINetEnhancer` and `iter_enhanced_frames` are also exported for other video
consumers. The iterator receives an open OpenCV capture and yields
`(source_time_seconds, enhanced_uint8_bgr_frame)` pairs.

## Face detection and preview

The facial pipeline uses `mp.solutions.face_detection.FaceDetection` from the
project's pinned `mediapipe==0.10.21`. Each sampled frame is converted from BGR
to RGB after optional SPEINet enhancement. The detector defaults to
`model_selection=1` (full range) and `min_detection_confidence=0.5`.
Use `--face-detection-model-selection 0` for the short-range model, or
`--min-face-detection-confidence` to change the detector and landmark detector
threshold. A confidence threshold controls which results are accepted; it does
not guarantee detection of profiles, occluded faces, or motion-blurred faces.
See the [MediaPipe API source](https://github.com/google-ai-edge/mediapipe/blob/v0.10.21/mediapipe/python/solutions/face_detection.py).

Only detections with their centers in `--patient-roi` can start a face track.
Tracking prefers the nearest face and rejects movement beyond `--max-head-jump`.
Bounding boxes are padded with `--face-margin`, made square, and clipped to the
image before landmark extraction. If landmarks are missing, the detected face
box still supplies the RGB/flow crop. Pose head boxes and limited reuse remain
fallbacks when face detection fails. Geometry remains zero when landmarks are
unavailable.

From `FYPModels`, inspect the selected patient and crop with:

```bash
python preprocess/_facial/face_preprocess.py "datasets/Seizure videos of epilepsy patients/Seizure/S1_5_136.mp4" --output results/facial_detection --patient-roi "0.15,0.05,0.45,0.65" --preview --window-seconds 2 --stride-seconds 1 --label 1
```

Add `--video-enhancement speinet --speinet-device cpu` to restore frames first.
The ROI above is an initial estimate for this clip; verify it in the preview.
The preview draws the selected detector result and the final crop box. Its text
distinguishes face boxes, landmarks, pose fallbacks, and reused boxes. Press
`q` or Escape to stop processing and save complete windows processed so far.

Diagnostics report `face_detector_fraction` separately from
`face_landmark_fraction` and `pose_head_fallback_fraction`. In training manifests,
`face_detection_fraction` counts frames with either a selected detector box or
valid facial geometry; `face_landmark_fraction` continues to count only geometry.

## Test head pose and expression measurements on a video

`_facial/preview_microexpressions.py` is a standalone diagnostic for the supplied
Face Mesh example. At the video's native frame rate, it detects the patient's
face with OpenCV YuNet, crops that face, and runs Face Mesh with
`refine_landmarks=True`. The patient ROI is only a search region. Fresh mesh
detection on each crop (`static_image_mode=True`) avoids carrying a previous
mesh across moving or resized crops. It does not run SPEINet or change training
windows.

The YuNet checkpoint is already cached in this workspace. On a fresh checkout,
download the OpenCV 4.x-compatible model from [OpenCV Zoo](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet).
Run these commands from `FYPModels`:

```bash
mkdir -p checkpoints/pretrained/yunet
curl --fail --location --output checkpoints/pretrained/yunet/face_detection_yunet_2023mar.onnx https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx
shasum -a 256 checkpoints/pretrained/yunet/face_detection_yunet_2023mar.onnx
```

Expected SHA-256:
`8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4`.
The model's MIT license and source information are stored alongside the cache.
Use `--face-detector-model` for another location. Then run:

```bash
python preprocess/_facial/preview_microexpressions.py "datasets/Seizure videos of epilepsy patients/Seizure/S1_5_136.mp4" --output results/microexpression_preview --patient-roi "0.15,0.05,0.45,0.65" --max-yaw 45 --max-pitch 30 --blink-threshold 0.20 --mouth-threshold 0.30 --preview
```

Choose a new output directory for each run. Omit `--preview` to process without
opening a window. Press `q` or Escape to stop an interactive run. Outputs are:

- `preview.mp4`: annotated video, including missing/filtered frames. Gray marks
  the search ROI; the smaller colored box marks a detected face. Cyan points
  are YuNet's five landmark estimates. Dense contours appear only when Face Mesh
  agrees with those estimates. Green means all configured checks passed;
  orange means measurements were rejected.
- `frames.csv`: original timestamps, face boxes and scores, mesh/detector
  agreement, pose estimates, reprojection error, separate left/right eye aspect
  ratios (EAR), mouth aspect ratio (MAR), and filter reason.
- `summary.json`: frame counts, input path, frame rate, and settings.

`--face-confidence` defaults to `0.8`. Before calculating pose or expressions,
the mesh's eye centers, nose and mouth corners must agree with YuNet's estimates.
`--max-landmark-error` defaults to `0.2`: the largest anchor displacement may be
at most 20% of the detected face's largest dimension. A missing face or mesh is
recorded as `no_face_detection` or `no_landmarks`; excessive disagreement is
`landmark_mismatch`. Missing detections never reuse an old mesh or substitute the
whole patient ROI. This agreement check is a heuristic, not verified accuracy.

`--max-yaw` and `--max-pitch` are absolute angle limits in degrees. Values exactly
on a limit pass; values beyond it suppress expression measurements. Missing
landmarks, failed pose estimation, and degenerate geometry are recorded explicitly.
Rejected frames keep their timestamps and have blank expression fields.
`--blink-threshold` compares each eye independently; both closed eyes produce a
per-frame blink candidate, not a verified blink event. `--mouth-threshold` applies
to MAR and does not assign an emotion. The example annotation CSV is not needed
for this diagnostic.

Pose uses a generic six-point face model with x right, y down, z away, approximate
focal length equal to full-frame width, and zero lens distortion. Crop landmarks
are mapped back to full-frame coordinates before solving pose. Angles returned by
`RQDecomp3x3` are already degrees. Inspect the overlay and reprojection error;
real head-pose accuracy requires camera/person calibration and ground truth.
See [OpenCV pose conventions](https://docs.opencv.org/4.x/d5/d1f/calib3d_solvePnP.html).

### Local face-localization check (2026-09-19)

On `S1_5_136.mp4` with the ROI and thresholds above, the corrected diagnostic
found a face box in 96 of 127 frames. At 2.84 seconds the box covered the actual
face; the earlier cap mesh and its expression measurements were absent.
Face Mesh returned landmarks in only one frame (4.32 seconds), which exceeded
the pitch limit. **No expression frames passed all checks.** This clip's face
boxes are more useful than its dense landmark estimates; do not interpret the
previous cap-based measurements as patient expressions. All 127 timestamps
were preserved, with blank expression fields for rejected frames.

The review artifacts are under `results/microexpression_face_fixed_s1_20260919/`.
The 46 focused diagnostic and face-preprocessing tests include rejection of a
cap-positioned mesh before pose estimation and correct ROI coordinate mapping.

## Frame and memory behaviour

- Neighbours are adjacent frames at the source FPS, even when facial extraction
  samples at 5 FPS. The network restores only the output frames the consumer needs.
- Up to 15 decoded frames are buffered for sharp-reference selection. The stage
  uses future frames and is intended for offline preprocessing.
- Output dimensions and channel order match the decoded BGR frames. No frame is
  enlarged or silently downscaled by this stage. All face-pipeline operations,
  including optical flow, see the SPEINet-enhanced full frame.
- `--speinet-tile-size` defaults to 200 and must be a multiple of 20.
  `--speinet-tile-overlap` defaults to 40 and must be smaller than the tile size.
  Overlaps are blended; border padding is removed after inference.
- `auto` selects CUDA when available and CPU otherwise. MPS can be explicitly
  requested, but has not been validated. CPU inference can be slow on full videos.
- Timing follows the existing constant-FPS decoder policy, `frame_index / FPS`;
  this does not add variable-frame-rate presentation-timestamp support.

## Validation status and numerical limitation

54 focused automated tests pass, covering temporal sampling, endpoint handling, duration limits,
reference selection, tiling, padding, color order, failure handling and the
handoff to both pose and face detection.

The locally cached 128,419,101-byte `speinet_model_best` checkpoint, SHA-256
`2c762f9f0e0ec3c2ac995e7f5ae872a64d80747074708ee724821d31ed5da0bd`, loads all
1,020 state-dictionary entries strictly. A full 1920×1080 frame from
`S1_5_136.mp4` was restored successfully using its native temporal neighbours,
default 200-pixel tiles and CPU inference (two PyTorch threads, about 155 seconds).
This validates one restored frame, not a complete-video quality benchmark.

However, other inputs, including a natural-image test and a random image,
produced non-finite values with the pinned upstream source. A diagnostic
traced the first invalid activation to
`recons_net.encoder_second.1.te.cw` on a random test input. The attention-block
computation matches the upstream file. The exact reason for this instability
has not been established.

The integration rejects invalid output with a clear error. Reliability over the
full dataset and improvements in face-landmark accuracy remain unverified.
If this error occurs on your clips, investigate the checkpoint/source pairing
before using it for training. The network's attention computations are preserved.

Run focused checks:

```bash
python -m unittest tests.test_video_enhancement tests.test_preprocessing_comparison -v
```

See [_speinet/UPSTREAM.md](_speinet/UPSTREAM.md) for attribution and the adapter's
temporal/spatial differences from the research inference script.
