# SPEINet video enhancement

This module integrates [yangt1013/SPEINet](https://github.com/yangt1013/SPEINet)
before facial preprocessing:

```text
Decoded video at source FPS
  -> SPEINet full-frame enhancement using neighbouring frames
  -> sampled enhanced frames with original timestamps and dimensions
  -> pose detection / patient selection
  -> face landmarks, stabilized face crops, RGB + optical-flow tensors
```

`facial_resnet.face_preprocess.extract_video()` calls this module directly.
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
python -m seizure_multimodal.facial_resnet.face_preprocess \
  "datasets/Seizure videos of epilepsy patients/Seizure/S1_5_136.mp4" \
  --output results/facial_speinet \
  --patient-id S1 --label 1 --split unassigned \
  --fps 5 --window-seconds 1 --max-duration-seconds 1 \
  --video-enhancement speinet \
  --speinet-checkpoint checkpoints/pretrained/speinet/model_best.pt \
  --speinet-device cpu
```

The same flags work with `facial_resnet.prepare_dataset`. Omit
`--video-enhancement` for the original pipeline. The earlier `--upscale-model`
and `--sharpen-amount` flags independently control optional face-crop operations
later in the pipeline; leave them at defaults when evaluating SPEINet alone.

For Python callers:

```python
from seizure_multimodal.preprocess import SPEINetConfig
from seizure_multimodal.facial_resnet.face_preprocess import extract_video

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
