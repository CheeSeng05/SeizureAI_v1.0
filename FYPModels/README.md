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
python -m seizure_multimodal.facial_region_hybrid.face_preprocess \
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



### Facial ResNet50–BiGRU–multi-head-attention model

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
→ ImageNet-pretrained ResNet50 applied to every frame
→ [B,T,2048] frame features
→ bidirectional GRU (256 units per direction)
→ [B,T,512] temporal features
→ 8-head temporal self-attention + residual connection + LayerNorm
→ temporal mean pooling
→ LayerNorm + dropout + binary linear classifier
```

The ResNet50 RGB filters retain their ImageNet initialization. Its fourth input
channel is initialized from the mean pretrained RGB filter. During the current
training policy, the ResNet50 backbone is frozen except for this adapted input
layer; the BiGRU, multi-head attention, and classifier are trainable.

Train it on a patient-disjoint preprocessed manifest with:

```bash
python -m seizure_multimodal.facial_region_hybrid.train \
  --manifest results/facial_patient_70_30/video_manifest.csv \
  --checkpoint checkpoints/fyp/facial/resnet50_gru_attention_final.pth \
  --epochs 20 \
  --batch-size 2 \
  --attention-heads 8 \
  --device auto
```

The held-out test split is evaluated once after the fixed training epochs. The
attention model requires a newly trained checkpoint: checkpoints from the old
ResNet50–GRU model do not contain attention parameters and inference rejects
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
