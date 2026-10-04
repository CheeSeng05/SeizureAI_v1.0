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
| EEG CNN-LSTM-attention model | present: `checkpoints/pretrained/eeg/seizure_detector_model.keras`; frozen inference input `[B,1024,18]` |

The EEG adapter fails closed when the configured artifact is absent. It never
initializes, saves, or evaluates random replacement weights. The recovered
Kaggle implementation supplies the exact CHB-MIT montage, physical units,
256 Hz sampling, four-second windows, and 50% overlap now encoded in the
inference configuration.

### Clinical `.e` compatibility check (2026-09-26)

One private recording was inspected without copying identifying metadata into
the repository. It was identified as a directly readable Natus/Nicolet/Nervus
`.e` container. The adapter exposed 33 dominant-rate physical channels at
500 Hz from a two-hour recording and read 34 embedded events (31 point events
and 3 intervals). Raw annotation text remains only in the ignored private
result directory.

The recording pipeline now resolves modern electrode names first, applies the
standard T3/T4/T5/T6 fallbacks where needed, constructs the exact 18 bipolar
channels by physical-µV subtraction, and resamples 500 Hz to 256 Hz with
polyphase resampling. It preprocesses channels into a disk-backed working
array and runs model windows in configurable batches, so all windows are never
held on the GPU at once. No model training, fine-tuning, checkpoint write, or
annotation-derived model input occurs.

### Executed result (2026-08-18)

The supplied video was processed end to end on the pretrained VSViG and pose
checkpoints: 25,080 frames formed 836 complete non-overlapping clips, with one
trailing frame discarded. No training or fine-tuning occurred. This table is
the historical August run and predates the recovered EEG checkpoint/montage
provenance described above.

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

For backward compatibility, strict mode still rejects ordinary sources that do
not form a verified pair. When its configuration contains one Nicolet `.e`
recording and synchronized `video_segments`, it routes to the same real fusion
workflow described below. The independent sample video and EEG CSV still fail
the strict checks and are never presented as multimodal evidence.

### Verified synchronized fusion

The platform normally creates the required configuration after a **Video + EEG**
upload. The equivalent model command is:

```powershell
python -m seizure_multimodal test --mode fusion --config configs/inference.yaml --root .
```

`data.eeg_file` must identify one Nicolet `.e` file, `data.video_segments` must
list one or more matching AVI/MP4 files, and `data.recording_id` identifies the
shared recording. Fusion does not trust list order. It decodes the embedded
Nicolet `VIDEOSYNCGUID` timestamps, confirms them against `Video Start` events,
matches the embedded filenames exactly, and uses each video container's measured
duration/FPS to build `fusion/video_segment_manifest.json`.

EEG wall-clock time is the master timeline, so acquisition gaps remain gaps.
Each video's local prediction timestamps are converted to this global time.
For every four-second EEG window, overlapping video probabilities are aggregated
with an overlap-duration weighted mean. The same calculation works across video
segment boundaries. With the default full-coverage requirement, a window without
four seconds of synchronized video receives no fused value; missing video is
never converted to probability zero.

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
the original checkpoint is supplied.

### EDF and Natus/Nicolet `.e` recordings

Clinical recording paths are never committed. Set the private root locally,
or pass one EDF/`.e` file on the command line:

```powershell
$env:EEG_DATA_ROOT = "C:\path\to\private\eeg_dataset"
python -m seizure_multimodal eeg-recording-infer `
  --input "C:\path\to\recording.e" `
  --config configs/inference.yaml `
  --root .
```

Discovery is recursive and does not depend on patient, session, or directory
names. `load_eeg(path)` selects one of two read-only adapters:

- EDF/EDF+ via PyEDFlib, including embedded EDF+ annotations;
- structurally verified Natus/Nicolet/Nervus `.e` via `nicolet-e2edf`,
  including embedded event records and physical waveform samples.

The extension alone is not trusted. Unsupported `.e` files are rejected, and
files are never renamed to EDF. Both adapters expose the same metadata,
physical-sample, and annotation interface. Raw events remain separate from
model inputs and retain their original text; provisional categories do not
assert that a candidate marker or spike is a confirmed seizure. Zero-duration
annotations remain point events.

The configured recording transformation is:

```text
physical µV common-reference electrodes
→ exact CHB-MIT bipolar A-B derivations
→ 500 Hz to 256 Hz scipy.signal.resample_poly (when required)
→ fourth-order 0.5–100 Hz bandpass
→ 60 Hz notch, Q=30
→ per-channel z-score and ±5 clipping
→ four-second/1024-sample windows, two-second stride
→ [B, 1024, 18]
→ frozen checkpoint, probability and threshold-0.5 prediction
```

The order is explicitly `FP1-F7`, `F7-T7`, `T7-P7`, `P7-O1`, `FP1-F3`,
`F3-C3`, `C3-P3`, `P3-O1`, `FP2-F4`, `F4-C4`, `C4-P4`, `P4-O2`, `FP2-F8`,
`F8-T8`, `T8-P8`, `P8-O2`, `FZ-CZ`, `CZ-PZ`. Missing components stop the run;
the loader never invents, duplicates, zero-fills, interpolates, or silently
substitutes electrodes. Private annotations, compatibility reports, and
predictions are written under `results/private/<non-identifying-recording-id>/`,
which is ignored by Git.

Successful execution is experimental compatibility testing, not clinical
validation. The model was trained on CHB-MIT, not the new clinical source;
montage reconstruction, resampling, and acquisition-domain differences may
affect predictions. Clinician event semantics remain unconfirmed and are not
used as labels. The original notebook also used overlapping windows with a
random window-level split from a small patient set, creating leakage and
generalisation concerns; its reported 97% is not an expected accuracy here.

## Fusion and training

Real multimodal inference uses configurable weighted probability-level late
fusion:

```text
fusion_probability = 0.20 × video_probability + 0.80 × eeg_probability
```

The default 80% EEG / 20% video split is an engineering baseline and has not
been clinically optimized. Both weights must be non-negative and sum to one.
The weights, `0.50` fused decision threshold, and minimum video coverage are
configuration values; they must later be selected on patient-disjoint validation
data, never the final test recording. Feature fusion defines the shape-only
architecture `384→128` and `32→128`, concatenation to 256, and an MLP head. It
has no trained paired checkpoint and is not invoked by real fusion.

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
  method: weighted_late_fusion
  trainable: false
  eeg_weight: 0.80
  video_weight: 0.20
  threshold: 0.50
  timeline: eeg
  alignment: overlap_weighted_mean
  require_verified_sync: true
  minimum_video_coverage: 1.0
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
  fusion/
    video_segment_manifest.json
    video_timeline.csv
    predictions.csv
    metrics.json
    video_segments/           # per-segment frozen video-model artifacts
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

The executed video run used Python 3.12, PyTorch `2.11.0+cu128`, CUDA runtime
12.8, and an NVIDIA GeForce RTX 4070 Laptop GPU. Run the current suite locally
for the authoritative pass/skip count because optional media and system
dependencies change which integration checks are available:

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
