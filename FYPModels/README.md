# FYP pretrained video + EEG testing

Inference-only test and integration code for **Automated Video Analysis in Temporal
Lobe Epileptic Seizures Using Multimodal Deep Learning**.

The repository keeps the two supplied sources scientifically separate:

- Local `test_video/Sz2PG.mp4` + `test_video/Label.xlsx` are one real **video-only** test source.
- `Epileptic Seizure Recognition.csv` is an independent **EEG-only** benchmark.
- They are never fused in normal or strict mode. Their only permitted pairing is
  the explicit, prominently labelled `simulated_multimodal` software test.
- Training, fine-tuning, and checkpoint replacement are disabled by default.

## Current evidence and status

### Video and labels

| Item | Inspected value |
|---|---|
| File | `test_video/Sz2PG.mp4` |
| SHA-256 | `819634aac6b8a6fe5779b2da786b6b5697524eee761acccd270c4968af135c67` |
| Duration | 836.8693667 s |
| FPS | 29.97002997 average/constant |
| Frames | 25,081 |
| Resolution | 1920×1080 |
| Codec | `avc1` (H.264/AVC sample entry) |
| Upstream identity | hash/size match `dataset/pat01/Sz2PG.mp4` in WU-SAHZU-EMU-Video |

`Label.xlsx` contains one sheet, `Sheet1`, with 33 seizure rows and columns
`PatID`, `Seizure Type`, `#Seizure`, `EEG onset`, and `Clinical Onset`.
Blank patient/type cells continue the preceding patient's catalog group; the
code forward-fills only those two identity fields in a separate resolved view
and never modifies the workbook.

The upstream hash identifies the relevant row unambiguously:

| Patient | Type | Seizure | EEG onset | Clinical onset | Offset |
|---|---|---|---:|---:|---|
| Pat01 | PG | Sz2 | 624 s | 640 s | not supplied |

`P` means focal/partial seizure and `PG` means partial-to-generalized. EEG and
clinical onsets remain separate. No seizure offset is inferred.

### Facial preprocessing smoke test

Run this command from the `FYPModels` directory to test the facial preprocessing
pipeline on one seizure video before preparing the complete dataset:

```bash
MPLCONFIGDIR=/tmp/mpl-facial \
python -m seizure_multimodal.facial_resnet.face_preprocess \
  "datasets/Seizure videos of epilepsy patients/Seizure/S1_5_136.mp4" \
  --output results/facial_head_smoke \
  --sample-id seizure_S1_5_136 \
  --patient-id S1 \
  --label 1 \
  --split train \
  --fps 5 \
  --window-seconds 2 \
  --stride-seconds 1 \
  --max-poses 4 \
  --head-margin 0.75 \
  --patient-roi "0.28,0.05,0.70,0.90" \
  --preview \
  --preview-delay-ms 150
```

The preview displays the selected head region and the exact `224×224` crop
used to construct the model input. Press `q` or Escape to close the preview;
preprocessing continues after the window closes. The patient ROI uses
normalized `x1,y1,x2,y2` coordinates and should include the patient's possible
head movement while excluding caretakers where possible.

The command writes overlapping two-second windows to
`results/facial_head_smoke/face`, together with pose and facial-geometry arrays
and manifest metadata. The default identity-tracking settings allow a maximum
normalized head jump of `0.18` and reuse the previous crop for at most 15
sampled frames during short detection failures. Use a new output directory, or
remove the previous smoke-test output, before repeating the same sample because
manifest rows are not intended to be duplicated.

### Testing six image upscaling algorithms

`_upscaling(image, model, scale=4, model_dir=...)` accepts a non-empty **uint8 BGR**
image (as returned by `cv2.imread`). The algorithms are `bilinear`, `bicubic`,
`edsr`, `espcn`, `fsrcnn`, and `lapsrn`. Neural models are cached for sequential
calls; interpolation uses `cv2.resize` and needs no weights.

`preprocess/_facial/frame_enhancement.py` owns the reusable upscaling functions,
crop sharpening and enhancement CLI options. `face_preprocess.py` imports these
helpers and keeps them available for existing callers.
`tests/test_upscaling.py` owns the comparison workflow and automated tests.
From `FYPModels`, with the facial Python environment active, compare all six on
the same small face crop:

```bash
python -m tests.test_upscaling path/to/face_crop.png \
  --output results/upscaling_x4 --scale 4 --download-models
```

`path/to/face_crop.png` is a placeholder: replace it with an existing image.
If you only have the source video, use the video command below instead.

The download flag fetches missing weights from the model authors into
`checkpoints/pretrained/super_resolution` (roughly 40 MB total for x4). Later
runs reuse the files, so the flag can be omitted. `--model-dir` overrides the
location. The installed OpenCV must expose `cv2.dnn_superres`; it is included in
the contrib distributions. Use one OpenCV distribution in an environment, rather
than installing conflicting base/contrib or GUI/headless builds together.

Alternatively, select a frame and normalized face ROI from a video:

```bash
python -m tests.test_upscaling \
  "datasets/Seizure videos of epilepsy patients/Seizure/S1_5_136.mp4" \
  --video-time 0.6 --roi 0.23 0.13 0.33 0.31 \
  --output results/upscaling_frame --scale 4 --download-models
```

This ROI selects the face of the seated person wearing the EEG cap at 0.6 seconds
in this video; choose suitable coordinates for other recordings or timestamps.
Video seeking uses the nearest frame index at the reported FPS. Begin
with a small crop (for example, 64x64 pixels), since EDSR on a full video frame
can be expensive. The runner preserves the supplied crop's resolution and does
not silently shrink it before inference.

Each run saves `input.png`, six full-resolution algorithm PNGs, a labeled
`comparison.png`, `detail.png`, `timings.csv`, and `run.json` with input/model
hashes and settings. The overview labels input/output pixel dimensions but
shrinks previews to fit. `detail.png` compares the same central region at native
output pixels, alongside an input reference enlarged with nearest-neighbor
pixel replication. View it or the individual PNGs at **100% zoom**; fitting
images to the same viewer size can hide enlargement. All six outputs deliberately
have the same dimensions: x4 means four times the width and height (16 times
the pixel count), not a guarantee of recovered detail or greater sharpness.
Timings separate the first call (including model loading) from repeated warm
calls (`--repeats`, default 3). Choose a new output directory for each run.

Use the same scale for a meaningful comparison: the command supports x2 and x4,
which all four neural models provide. The function also supports x3 for
EDSR/ESPCN/FSRCNN and x8 for LapSRN, with matching checkpoint files.
`setModel` requires the actual checkpoint's integer scale; see the
[OpenCV super-resolution tutorial](https://docs.opencv.org/4.x/d5/d29/tutorial_dnn_superres_upscale_image_single.html).

Run automated checks with `python -m unittest tests.test_upscaling -v`.
The real-model test skips until all four x4 checkpoints are available. This
comparison runs independently of the video extraction entry point and reports
image outputs and speed. Assess landmark availability and landmark placement
separately on the same frames before adopting an upscaler for face detection;
visual sharpness alone does not establish an improvement. PSNR/SSIM require an
aligned reference image and are not computed against the low-resolution input.

### SPEINet video enhancement before facial preprocessing

The requested model is integrated under `seizure_multimodal/preprocess`:

```text
Video -> preprocess.SPEINetEnhancer -> face_preprocess -> face windows
```

Add `--video-enhancement speinet --speinet-checkpoint path/to/model_best.pt` to
`face_preprocess` or `prepare_dataset`. SPEINet receives full frames and native
temporal neighbours before pose and face detection. Source dimensions, sampling
timestamps and original-video provenance are preserved. Enhancement defaults off.

See [the preprocess module guide](seizure_multimodal/preprocess/README.md) for
setup, invocation and validation details. The cached checkpoint restored a full
1920×1080 frame from the sample video. **Some other test inputs produced non-finite
values inside the upstream model.** The adapter detects this failure and stops
before passing invalid frames onward; broad dataset reliability is unverified.

### Sharpening face crops during video preprocessing

`enhance_face_crop()` in `preprocess/_facial/frame_enhancement.py` reuses `_upscaling()`
and optionally applies an unsharp mask (adds a scaled difference between the
crop and its Gaussian-blurred copy). Enable it on the video extraction command:

```bash
python -m seizure_multimodal.facial_resnet.face_preprocess \
  "datasets/Seizure videos of epilepsy patients/Seizure/S1_5_136.mp4" \
  --output results/facial_sharpened \
  --patient-id S1 --label 1 --split unassigned \
  --window-seconds 2 --stride-seconds 1 \
  --upscale-model bicubic --upscale-scale 2 \
  --sharpen-amount 0.5 --sharpen-sigma 1.0 \
  --max-duration-seconds 10 --preview
```

Run from `FYPModels` with the facial Python environment active. The same
enhancement flags work with `prepare_dataset`. Omit `--preview` when running
without a display. The pose and face task model flags/defaults are unchanged.

The pipeline crops from the original frame, enhances the head crop before
Face Landmarker, and maps normalized landmarks back using the original crop
dimensions. It also enhances each stabilized face crop before resizing the RGB
input to `--face-size`. Optical flow uses the crop before these extra crop
operations (including any upstream SPEINet restoration); the output remains
`[4, face_size, face_size]` (RGB plus flow magnitude). Enhancement does not affect
the full-frame pose detector.

Omit `--upscale-model` for sharpening without enlargement. Set
`--sharpen-amount 0` for upscaling alone. Both are disabled by default. Neural
upscalers (`edsr`, `espcn`, `fsrcnn`, `lapsrn`) use the same cached checkpoints
described above, with `--sr-model-dir` to select their directory. Settings and
the selected neural checkpoint hash are recorded with each preprocessing run.

To reference the code directly from another module:

```python
from FYPModels.preprocess._facial.frame_enhancement import enhance_face_crop

# face_bgr is a non-empty uint8 BGR crop from a decoded video frame.
enhanced_bgr = enhance_face_crop(
    face_bgr, upscale_model="bicubic", upscale_scale=2,
    sharpen_amount=0.5, sharpen_sigma=1.0,
)
```

Sharpening increases existing edge contrast and can amplify noise or halos.
It does not recover missing detail. Compare the preview and landmark diagnostics
on the same video before using the settings for training; the code does not
assume that visually sharper crops improve detection accuracy.

Run checks with `python -m unittest tests.test_face_enhancement -v`.

### Comparing MediaPipe preprocessing settings

Facial preprocessing uses `video/io.py` for decoder lifetime and frame sampling.
The decoder supplements its OpenCV metadata with `data_inspection.inspect_mp4`
when the input is an MP4-compatible container. Inspection failures are recorded
and OpenCV decoding remains available for unsupported MP4 layouts or other
containers. Sampling still uses `frame_index / source_fps`; MP4 average FPS and
variable-frame-rate status are recorded separately, without claiming support
for presentation timestamps.

Every successful extraction also writes `runs/<sample_id>.json`, including
SHA-256 hashes of the video and both task models, software versions, source
metadata, resolved patient/split information, preprocessing settings and output
paths. `PreprocessSummary.run_record_path` points to this record. Input hashing
adds file reads before inference and is excluded from `processing_fps`. Hashes
identify file contents; the record distinguishes caller-supplied patient IDs
from the legacy sample-ID prefix fallback.

Add `--progress-file results/facial_head_smoke/progress.json` to the command
above for optional status snapshots. Python callers can pass
`progress_config={"testing": {"progress_file": "results/progress.json"}}`, matching
the raw-video workflow. Updates are throttled to once per second during inference;
the final snapshot is written after output files and the run record are saved.
The CLI also accepts `--split-seed` (default 42) for automatic patient splits.

Keep these module responsibilities when extending preprocessing:

| Module | Responsibility and recommended boundary |
|---|---|
| `data_inspection.py` | Read and validate metadata/label schemas. If it grows, separate MP4, EEG and label inspectors behind the existing public functions. Keep frame decoding and model inference out. |
| `discovery.py` | Resolve configured sources and verify provenance before calling preprocessing. Pass resolved paths and identity to `extract_video`; avoid rediscovering inputs inside its frame loop. Share inspected metadata/hash results with `inspection_bundle` to avoid its repeated reads. |
| `progress.py` | Serialize progress and ETA only. Keep update frequency and work-unit counts in the caller; use a separate progress file per concurrent worker. A future path/writer interface could remove the dependency on `testing.progress_file`. |
| `reproducibility.py` | Build and save stage-specific records. Use `build_preprocessing_record` for preprocessing and retain the stricter `build_run_record` schema for experiments. Reuse the shared provenance hash helper. |
| `video/io.py` | Own capture cleanup and sampling. `raw_pipeline.py` and dataset duration checks can adopt `open_video` next while keeping their own clip/annotation policies. Implement presentation-timestamp sampling here if variable-frame-rate support is required. |
| `facial_resnet/face_preprocess.py` | Own patient selection, face/pose extraction, crop stabilization and window outputs. Accept explicit inputs from discovery or dataset planning. |

Both `face_preprocess` and `prepare_dataset` accept independent pose and face
thresholds:

```text
--min-pose-detection-confidence 0.5
--min-pose-presence-confidence 0.5
--min-pose-tracking-confidence 0.5
--min-face-detection-confidence 0.5
--min-face-presence-confidence 0.5
--min-face-tracking-confidence 0.5
--box-smoothing 0.7
```

The baseline still reuses crop coordinates for up to 15 sampled frames. Use
`--max-box-reuse-seconds 0.6` to expire reuse based on elapsed video timestamps
since the last detected face **or pose-derived head box**. Choose either the
seconds flag or `--max-box-reuse-frames`, not both. In the Python API, a supplied
`max_box_reuse_seconds` overrides the frame limit. This controls reuse of an old
box; pose-head fallback remains available while pose detection succeeds.

Run the comparison from `FYPModels` with the facial environment active:

```bash
python -m seizure_multimodal.facial_resnet.compare_preprocessing \
  "datasets/Seizure videos of epilepsy patients/Seizure/S1_5_136.mp4" \
  "datasets/Seizure videos of epilepsy patients/Normal/S1_5_259.mp4" \
  --patient-roi "0.28,0.05,0.70,0.90" \
  --output results/preprocessing_comparison_new
```

Use a new output directory for each comparison. The default 18 configurations
are the baseline, each of the six confidence settings individually changed to
0.6 and 0.7, smoothing individually changed to 0.3 and 0.5, and reuse individually
changed to 0.4, 0.6, and 1.0 seconds. All other settings stay at baseline for each
experiment. Pose Full and `VIDEO` mode are used throughout. This is an initial
screening experiment; combinations of chosen settings need a subsequent check.

Use `--thresholds`, `--smoothing-values`, and `--reuse-seconds` to customize the
candidate lists; an empty list disables that group. Put positional video paths
before these list options. `--plan-only` writes the experiment plan without
inference. `--max-duration-seconds 30` limits each run to its first 30 seconds;
use clips containing the movement/occlusion events you want to compare. The
common `--patient-roi` applies to every input; use separate comparisons when
recordings require different patient regions.

The output contains:

- `plan.json`: input videos, fixed settings, model paths/checksums, and experiments.
- `comparison.csv`: one row per configuration/video, with deltas from that video's
  baseline. Fractions are measured on distinct sampled frames, so overlapping
  training windows do not duplicate observations.
- `<experiment>/diagnostics/video_001.json`: exact settings and summary measurements.
- `<experiment>/diagnostics/video_001.frames.csv`: timestamps, detection/fallback
  flags, raw boxes, smoothed crop boxes, and crop-motion measurements.

`face_landmark_fraction` measures actual landmark availability;
`face_region_fraction` also counts pose-head fallbacks and reused coordinates.
High region coverage alone does not demonstrate successful face detection or
correct patient identity. The comparison saves diagnostics without generating
large training arrays; decoding, landmark inference, cropping and optical flow
still run. `processing_fps` excludes model initialization and output serialization
and is only a rough timing measurement from one pass.

`mean_box_center_step_fraction` measures movement of consecutive available crop
centers, divided by the frame diagonal. `mean_box_center_offset_fraction` measures
the distance between a smoothed crop center and its current raw face/head box
center, also divided by the diagonal. Missing measurements are blank, not zero.
These are motion and smoothing-offset proxies, not annotated jitter/accuracy
scores: excessive smoothing can produce a low step value while the crop lags.
Smoothing is applied after landmark extraction, so changing it need not alter
landmark availability. Reuse is also applied after detection.

Inspect the source video against the frame boxes and review candidate crops:

```bash
python -m seizure_multimodal.facial_resnet.face_preprocess \
  "datasets/Seizure videos of epilepsy patients/Seizure/S1_5_136.mp4" \
  --output results/preprocessing_candidate_preview \
  --patient-roi "0.28,0.05,0.70,0.90" \
  --min-face-detection-confidence 0.6 \
  --box-smoothing 0.5 \
  --max-box-reuse-seconds 0.6 \
  --diagnostics-only --preview
```

This last command is an illustrative combination, not an automatically selected
winner. Once settings are chosen on validation patients, pass the same flags to
`prepare_dataset` and regenerate inputs consistently for downstream evaluation.
Use annotated patient identities/face boxes and visible landmark locations to
measure wrong-person selection and geometric accuracy; keep test patients out
of parameter selection. No preset is promoted automatically from coverage alone.

### EEG CSV

`Epileptic Seizure Recognition.csv` has 11,500 rows and 180 columns:

- unique source-provided string key `Unnamed`;
- 178 ordered integer samples `X1..X178`;
- target `y` with exactly 2,300 rows in each class 1–5;
- no patient ID, recording ID, timestamp, duration, or channel columns.

Each row is a complete 178-sample, single-channel benchmark segment. The
underlying Bonn recordings were sampled at about 173.61 Hz and divided into 23
chunks; the CSV itself does not store sampling metadata or permit event-time
reconstruction.

The checked notebook uses this exact binary mapping:

| CSV `y` | Meaning | Binary target |
|---:|---|---:|
| 1 | ictal/seizure activity | 1 |
| 2 | non-ictal recording from the epileptogenic/tumour area | 0 |
| 3 | non-ictal recording from a healthy area distant from that area | 0 |
| 4 | healthy subject, eyes closed | 0 |
| 5 | healthy subject, eyes open | 0 |

The mapping is explicit in `configs/inference.yaml` and is printed before EEG
evaluation.

### Checkpoints

| Model | Status |
|---|---|
| VSViG base | present: `VSViG-main/VSViG-base.pth`, 1,283 tensors, SHA-256 `0a95e3e0093730dc535e21a84f04943c73549a2b6a19a85e33aebc7c64f965a9` |
| Dynamic partition | present: `VSViG-main/dy_point_order.pt` |
| Custom pose model | present: `VSViG-main/pose.pth`, 250 tensors, SHA-256 `7c61f38c17d4b54d7f429717544be1de0905f901aa6f7913f7868ae5c5b83827` |
| EEG CNN-LSTM-attention model | **missing**: the notebook names `best_seizure_model.h5`, `seizure_detector_model.h5`, and `seizure_detector.weights.h5`, but none was supplied |

The EEG adapter fails closed when the original artifact is absent. It never
initializes, saves, or evaluates random replacement weights.

### Executed result (2026-08-18)

The supplied video was processed end to end on the pretrained VSViG and pose
checkpoints: 25,080 frames formed 836 complete non-overlapping clips, with one
trailing frame discarded. No training or fine-tuning occurred.

| Test | Video | EEG | Fusion | Status |
|---|---:|---:|---:|---|
| Video-only | yes | no | no | completed, 836 prediction windows |
| EEG-only | no | no | no | blocked: original trained `.h5` artifact is absent |
| Strict multimodal | no | no | no | skipped: sources are not a verified synchronized pair |
| Simulated multimodal | yes | no | no | blocked: no EEG model output; no substitute scores generated |

Valid video-only metrics at threshold 0.5 are sensitivity `0.7398`, specificity
`0.5040`, AUROC `0.7378`, F1 `0.4462`, precision `0.3194`, and Cohen's kappa
`0.1680`. There were 61 merged false-alarm events in the 624-second
pre-EEG-onset interval (`351.9231` events/hour). The first post-EEG-onset alarm
started at 631.631 s, giving signed clinical-onset error `-8.369` s and absolute
error `8.369` s. Offset error is `N/A` because no reference offset exists.

These figures evaluate this one supplied recording under the documented source
label policy; they are not a patient-level clinical validation study. See
`results/video_only/Sz2PG_metrics.json` and `results/summary.md` for the
machine-readable record.

| Test | Video used | EEG used | Fusion | Data relationship | Result validity |
|---|---:|---:|---:|---|---|
| Video-only | yes | no | no | real video/label source | valid one-recording unimodal evaluation |
| EEG-only | no | no | no | real independent CSV | not evaluated; original EEG checkpoint missing |
| Strict multimodal | no | no | no | unverified and unsynchronized | correctly rejected |
| Simulated multimodal | yes | no | no | explicitly artificial | blocked; software fusion did not run without EEG output |

## Test modes

### Default: independent unimodal testing

```powershell
python -m seizure_multimodal test --mode unimodal
```

Available modalities run independently. A missing or blocked modality does not
erase the other result and does not trigger fusion.

```powershell
python -m seizure_multimodal test --mode video
python -m seizure_multimodal test --mode eeg
```

### Strict multimodal

```powershell
python -m seizure_multimodal test --mode strict_multimodal
```

Fusion requires matching non-empty patient and recording IDs, overlapping
timestamps, both modalities, and `is_verified_pair=true`. The current sources
fail those checks, so strict mode writes
`results/strict_multimodal/SKIPPED.txt` and performs no fusion.

### Simulated multimodal

```powershell
python -m seizure_multimodal test `
  --mode simulated_multimodal `
  --allow-simulated-pairing
```

The acknowledgement flag is mandatory and is checked before model execution.
The deterministic policy pairs the first `N` predictions in original order,
where `N=min(video windows, EEG rows)`, under IDs `sim_window_0001`, ... . It
does not search, select, or reorder samples by labels or predictions.

By default the run reports probabilities, predictions, and software
success/failure only. Artificial metrics require an extra flag and explicit
label source:

```powershell
python -m seizure_multimodal test `
  --mode simulated_multimodal `
  --allow-simulated-pairing `
  --calculate-artificial-metrics `
  --reference-labels eeg
```

Those metrics are stored with `validity=simulated` and
`clinical_interpretation=prohibited` and are not performance evidence.

## Model input pipelines



### Facial InceptionResnetV1–BiGRU–multi-head-attention model

The experimental facial classifier is informed by the computer-vision framing
in [Deep facial analysis: A new phase I epilepsy evaluation using computer
vision](https://doi.org/10.1016/j.yebeh.2018.02.010) and by the linked
[Pattern Recognition article](https://www.sciencedirect.com/science/article/pii/S0031320325006752).
It is an adaptation to this project's RGB-plus-optical-flow windows and binary
normal/seizure labels, not a claim of reproducing either paper's experiments,
dataset, results, or released model.

The implemented forward path is:

```text
[B,T,4,224,224] RGB + optical-flow magnitude
→ resize to 160×160 and standardize RGB as (255*x - 127.5)/128
→ VGGFace2-pretrained InceptionResnetV1 applied to every frame
→ [B,T,512] frame features
→ bidirectional GRU (256 units per direction)
→ [B,T,512] temporal features
→ 8-head temporal self-attention + residual connection + LayerNorm
→ temporal mean pooling
→ LayerNorm + dropout + binary linear classifier
```

The InceptionResnetV1 RGB filters retain their VGGFace2 initialization. Its fourth input
channel is initialized from the mean pretrained RGB filter. During the current
training policy, the InceptionResnetV1 backbone is frozen except for this adapted input
layer; the BiGRU, multi-head attention, and classifier are trainable.

Train it on a patient-disjoint preprocessed manifest with:

```bash
python -m seizure_multimodal.facial_resnet.train \
  --manifest results/facial_patient_70_30/video_manifest.csv \
  --checkpoint checkpoints/fyp/facial/inception_resnet_v1_gru_attention_final.pth \
  --epochs 20 \
  --batch-size 2 \
  --attention-heads 8 \
  --device auto
```

The held-out test split is evaluated once after the fixed training epochs. The
attention model requires a newly trained checkpoint: checkpoints from the old
ResNet50 model use an incompatible backbone and feature dimension and inference rejects
them explicitly rather than silently using partially initialized weights.

### VSViG

Raw frames are decoded at their measured 29.97 FPS. The custom pose checkpoint
is loaded with the official lightweight OpenPose implementation pinned under
`third_party/lightweight_openpose` at commit
`d23c284b09acf27a163e1febd511e7482cac25ed`.

For each frame the deterministic largest-bounding-box pose is selected. The
provided Gaussian-patch procedure is reproduced with 128-pixel kernels, sigma
`0.3×128`, per-patch 0–255 scaling, and resize to 32×32. Undetected keypoints
are explicit zero patches with `(-1,-1,0)` position triples.

The pretrained input is:

```text
patches:   [B, 30, 15, 3, 32, 32]
positions: [B, 30, 15, 3]
```

Thirty consecutive frames form a non-overlapping ~1.001 s clip. The adapter
strictly loads the original state dictionary, freezes every parameter, runs
`eval()` plus inference/no-grad mode, and returns a 384-D embedding, logit,
sigmoid seizure probability, and thresholded prediction.

For classification, the source model's reference policy is used: windows
through EEG onset are 0, windows from clinical onset onward are 1, and the
EEG-to-clinical transition is excluded from hard-label metrics. Missing seizure
offset remains `N/A` and is never invented.

### EEG notebook adapter

The notebook's training source is CHB-MIT, not this CSV. It trained a 4-second,
256-Hz, 18-channel network with input `[B,1024,18]`:

```text
Conv1D(32/64/128)
→ bidirectional LSTM(64/32)
→ 4-head self-attention
→ global average + max pooling
→ dense 64/32
→ sigmoid
```

Its CSV cross-dataset cell linearly expands each 178-sample row to 1,024 time
steps, repeats the signal across 18 channels, and globally z-scores the table.
The implementation reproduces that adaptation in batches and exposes the 32-D
pre-output embedding, logit, probability, and prediction. It runs only after
the original `.h5` checkpoint is supplied.

## Fusion and training

Late fusion is fixed and configurable:

```text
fusion_probability = 0.5 × video_probability + 0.5 × eeg_probability
```

Weights are never optimized on the artificial pair. Feature fusion defines the
shape-only architecture `384→128` and `32→128`, concatenation to 256, and an
MLP head. With no verified paired checkpoint/data, that head is not trained or
reported as a clinical model.

Defaults remain:

```yaml
training:
  enabled: false
video_encoder:
  pretrained: true
  freeze: true
eeg_encoder:
  pretrained: true
  freeze: true
fusion:
  trainable: false
```

Future HUKM stage 1 may freeze both encoders and train only a fusion head on
verified, synchronized, patient-disjoint vEEG. Controlled fine-tuning is a
later stage and is never activated by testing commands.

## Metrics and outputs

Classification metrics are sensitivity, specificity, continuous-score AUROC,
F1, and model-vs-reference Cohen's kappa. FP/h is merged false alarm events
divided by known non-seizure monitoring hours. Onset error is signed and
absolute against the real onset. Offset error is calculated only when a real
offset exists.

Static EEG rows support classification metrics only; FP/h and temporal metrics
are `N/A`. Video offset metrics are `N/A` because the workbook has no offset.

```text
results/
  data_inspection.json
  summary.csv
  summary.md
  video_only/
    predictions.csv
    Sz2PG_predictions.csv
    Sz2PG_metrics.json
    Sz2PG_probability_timeline.csv
    Sz2PG_timeline.png
  eeg_only/
    eeg_predictions.csv
    eeg_metrics.json
    label_mapping.json
    eeg_confusion_matrix.png  # only after successful EEG inference
    eeg_roc.png               # only after successful EEG inference
  strict_multimodal/
    metrics.json
    SKIPPED.txt
  simulated_multimodal/
    predictions.csv
    metrics.json
    pairing_manifest.csv
    WARNING.txt
```

Every result JSON and populated prediction row records test mode, models used,
fusion used, pairing type, training status, and both checkpoints.

## Facial model clinical-evaluation workflow

The facial workflow uses disjoint patient groups: training fits weights,
validation selects the seizure-probability threshold, and the standalone
evaluator opens the test cohort only for the final report. Regenerate existing
two-way manifests before using this workflow.

1. Create and preprocess a 70/15/15 patient split:

```bash
python -m seizure_multimodal.facial_resnet.prepare_dataset \
  --dataset-root "datasets/Seizure videos of epilepsy patients" \
  --output results/facial_patient_70_15_15 \
  --split-unit patient --train-fraction 0.70 --validation-fraction 0.15
```

2. Train on `train`; threshold selection uses only `validation`. Training never
loads `test`:

```bash
python -m seizure_multimodal.facial_resnet.train \
  --manifest results/facial_patient_70_15_15/video_manifest.csv \
  --checkpoint checkpoints/fyp/facial/inception_resnet_v1_gru_attention_final.pth
```

3. Copy `configs/facial_annotations.example.csv`, replace the example rows with
clinician-verified recording bounds and seizure onset/offset timestamps, and
freeze `configs/facial_evaluation.yaml` before final testing. A blank onset and
offset represents a monitored recording with no seizure; never use clip bounds
as invented seizure offsets.

4. Evaluate isolated labelled clips without inventing temporal annotations:

```bash
python -m seizure_multimodal.facial_resnet.evaluation \
  --mode clip \
  --manifest results/facial_patient_70_15_15/video_manifest.csv \
  --checkpoint checkpoints/fyp/facial/inception_resnet_v1_gru_attention_final.pth \
  --config configs/facial_evaluation.yaml \
  --output results/facial_final_evaluation
```

Clip mode reports classification metrics, patient-bootstrap confidence
intervals, and biomarkers. Event sensitivity, FP/hour, and onset/offset errors
are explicitly unavailable because short curated clips are not continuous
monitoring records.

5. When clinician-verified continuous event annotations are available, use
event mode:

```bash
python -m seizure_multimodal.facial_resnet.evaluation \
  --mode event \
  --manifest results/facial_patient_70_15_15/video_manifest.csv \
  --checkpoint checkpoints/fyp/facial/inception_resnet_v1_gru_attention_final.pth \
  --annotations configs/facial_annotations.csv \
  --config configs/facial_evaluation.yaml \
  --output results/facial_final_evaluation
```

This writes timestamped `predictions.csv`, transparent motor/facial-expression
proxies in `biomarkers.csv`, and `metrics.json` containing window sensitivity,
specificity, AUROC, F1, event sensitivity, false positives/hour, onset/offset
errors, and patient-bootstrap 95% confidence intervals. Biomarkers are
measurement proxies rather than emotion diagnoses. Validate them against
independent clinician scores with `evaluation/biomarker_validation.py` before
making clinical claims.

## MHAN-BiGRU micro-expression semiology model

Two independent multi-task architectures are available. The ResNet model uses
an VGGFace2-pretrained InceptionResnetV1 plus a BiGRU. The separate MHAN model uses a
compact RGB-flow/geometry encoder, BiGRU, and multi-head attention pooling that
exposes the frames attended by each head. Both predict visible expression,
facial semiology, and seizure status.

The existing 5 fps binary manifest cannot train this model. Re-preprocess at a
minimum of 25 fps with short overlapping windows:

```bash
python -m seizure_multimodal.facial_resnet.prepare_dataset \
  --dataset-root "datasets/Seizure videos of epilepsy patients" \
  --output results/microexpression_patient_70_15_15 \
  --split-unit patient --train-fraction 0.70 --validation-fraction 0.15 \
  --fps 25 --window-seconds 1.0 --stride-seconds 0.25
```

Create a blinded annotation CSV from
`configs/microexpression_annotations.example.csv`. `emotion` is one of
`neutral`, `fear`, `anger`, `sadness`, `disgust`, `surprise`, or `happiness`.
`semiologies` is `none` or a pipe-separated selection from `fear`, `crying`,
`laughter`, `grimacing`, `oral_automatism`, and `motor_asymmetry`. A blank field
means missing—not a negative label.

```bash
python -m seizure_multimodal.facial_mhan.train \
  --manifest results/microexpression_patient_70_15_15/video_manifest.csv \
  --annotations configs/microexpression_annotations.csv \
  --checkpoint checkpoints/fyp/facial/mhan_bigru_best.pth \
  --epochs 40 --patience 8
```

Training selects the lowest validation-loss checkpoint, writes a recoverable
`mhan_bigru_best_latest.pth` every epoch, and stops early after the configured
patience. The labels must be clinician-annotated; the seizure folder name is
not treated as an emotion or semiology label.

Resume an interrupted run with `--resume checkpoints/fyp/facial/mhan_bigru_best_latest.pth`.

Train the independent InceptionResnetV1-BiGRU model with the same annotations:

```bash
python -m seizure_multimodal.facial_resnet.train_emotions \
  --manifest results/microexpression_patient_70_15_15/video_manifest.csv \
  --annotations configs/microexpression_annotations.csv \
  --checkpoint checkpoints/fyp/facial/inception_resnet_v1_emotions_best.pth
```

## Installation and verification

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

For GPU video inference, install a CUDA-enabled PyTorch build appropriate to
the machine rather than a CPU-only wheel. Tests cover real-file inspection,
data discovery, missing modalities, strict rejection, simulated acknowledgement
and pairing, checkpoints, training safeguards, probability/label handling,
late fusion, embedding shapes, metrics, and result metadata.

The executed run used Python 3.12, PyTorch `2.11.0+cu128`, CUDA runtime 12.8,
and an NVIDIA GeForce RTX 4070 Laptop GPU. Its checkpoint/integration suite
passes all 50 tests:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

## Non-negotiable interpretation

- Video-only evaluation is a real unimodal experiment.
- EEG-only evaluation is real only when the original pretrained EEG checkpoint
  is present.
- Real multimodal evaluation requires the same patient, recording, and time.
- `Sz2PG.mp4` plus this EEG CSV is **never** real vEEG.
- Simulated pairing is a software test only and prohibits clinical interpretation.
- VSViG retrained/fine-tuned: **NO**.
- EEG model retrained/fine-tuned: **NO**.
- Real fusion model trained: **NO**.

The facial backbone requires `facenet-pytorch` (included in `pip install -e ".[facial]"`). VGGFace2 weights download automatically on first training initialization. They provide face identity features; the expression and seizure heads still require task-specific training. Existing 224×224 RGB+flow windows are resized inside the model. The `facial_resnet` module path is retained.
