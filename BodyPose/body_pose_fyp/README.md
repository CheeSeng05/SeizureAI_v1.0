# Dataset-Adapted Body-Pose Stream for vEEG Video Analysis

This repository is the body-pose baseline for the FYP **Automated Video Analysis
in Temporal Lobe Epileptic Seizures Using Multimodal Deep Learning**. It is a
research prototype, not a medical device or diagnostic system.

The intended HUKM input is clinical vEEG: synchronized hospital video and
clinical scalp EEG. The current multimodal plan has three distinct video-derived
streams:

1. facial behaviour;
2. body pose;
3. video motion (for example optical flow or frame differences).

This repository implements the video-only body-pose stream. It does not process
ECG, EMG, accelerometers, or gyroscopes. Wearable movement is not used as video
motion. Clinical EEG remains a future, optional synchronization/embedding
interface; no EEG model is implemented here.

For cloud-integrated runs, the separate sibling project
[`../../fyp_data_platform`](../../fyp_data_platform/README.md) owns S3 and
PostgreSQL access. It stages an explicitly selected video to a local working
directory, calls this project's existing explicit input/output CLIs, then stores
and catalogs the generated pose/features. This project remains fully usable with
local files and never needs AWS credentials.

## Data boundary

Raw dataset columns are isolated from model code:

```text
Raw annotations
    -> dataset-specific adapter
    -> canonical_annotations.csv
    -> timestamp-aligned shared window_manifest.csv
    -> body / face / video-motion streams
```

Downstream code requires only `media_id`, `patient_id`, `start_sec`, `end_sec`,
and `label`. The complete canonical core is:

```text
dataset, media_id, recording_id, patient_id, start_sec, end_sec, label,
source_event_type, lateralization, localization, confidence,
annotation_source, derived_label
```

Adapters are under `body_pose/annotation_adapters/`:

- `base.py`: schema validation;
- `factory.py`: adapter selection;
- `generic_tabular.py`: configurable future HUKM CSV/TSV reader;
- `seizeit2.py`: OpenNeuro `ds005873` BIDS-SCORE event reader.

## SeizeIT2 (`ds005873`, version 1.1.0)

The public [OpenNeuro dataset](https://openneuro.org/datasets/ds005873/versions/1.1.0)
contains wearable behind-the-ear EEG, ECG, EMG, accelerometer, and gyroscope
recordings. It does not provide the patient video required by MediaPipe. It is
therefore used only for annotation semantics, BIDS structure, identifiers,
onset/duration handling, and adapter tests.

The adapter was matched to the published `events.json` and representative real
`*_events.tsv` files. It reads only `**/eeg/*_events.tsv`; it never opens EDF or
wearable sensor data. Mappings are:

- `eventType` beginning with `sz` -> `ictal`;
- `bckg` -> `normal` when that event is explicitly present;
- `impd` and unknown event types -> excluded.

It calculates `end_sec = onset + duration` and preserves event type,
lateralization, localization, vigilance, confidence, channels, recording
duration, and the source TSV path. It does not infer background from gaps and
does not invent `pre_ictal`.

A full dataset download is unnecessary for annotation validation. A local
metadata/DataLad checkout containing the TSV files is sufficient. Set its path
in `configs/seizeit2_annotations.example.yaml`, then run:

```powershell
python -m body_pose.prepare_annotations --config configs/seizeit2_annotations.example.yaml
```

Expected output: `annotations/seizeit2_canonical.csv` with active classes
`normal` and `ictal` when those source event types occur.

Trying to use that configuration without real patient video gives a deliberate
error explaining that SeizeIT2 can validate annotations but cannot train the
MediaPipe body-pose stream.

## Future HUKM adapter

The HUKM source columns do not need to match the canonical names. The example
`configs/hukm_adapter.example.yaml` maps a possible frame-based export:

```yaml
column_map:
  media_id: VideoName
  patient_id: Patient
  start_sec: BeginFrame
  end_sec: FinishFrame
  label: ClinicalState
label_map:
  Normal: normal
  PreIctal: pre_ictal
  Seizure: ictal
time_unit: frames
fps: 25
```

`time_unit` supports `seconds`, `milliseconds`, and `frames`. For frames, `fps`
may be a number or the name of a per-row source column. Once the real annotation
protocol is known, change only this adapter configuration and label mapping;
the body model remains unchanged.

Never define `pre_ictal` as an arbitrary interval before onset. Enable it only
when the HUKM annotation protocol provides a clinically agreed definition.

## Body-pose pipeline

```text
Video
  -> MediaPipe Holistic (33 pose landmarks per frame)
  -> keypoints/<media_id>.csv
  -> normalized coordinates, visibility, velocity, acceleration,
     speed, joint angles, and trajectory features
  -> features/<media_id>.csv
  -> shared temporal windows
  -> attention-pooled bidirectional LSTM
  -> body embedding and baseline logits
```

Keypoint CSVs include timestamps, source frame/FPS, `x/y/z/visibility` for all
33 landmarks, `pose_detected`, and `mean_visibility`. Low-visibility coordinates
are masked before limited interpolation and smoothing. The MediaPipe `z` value
is relative depth, not calibrated metric 3D.

The LSTM exposes both interfaces required for fusion:

```python
embedding = model.encode(x)
logits = model(x)
```

## Installation

Python 3.10 or 3.11 is recommended.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Do not commit identifiable clinical video, annotations, keypoints, or features
to a public repository.

## Standard run

The included `config.yaml` demonstrates a generic adapter over
`annotations/annotations.csv`.

```powershell
python -m body_pose.prepare_annotations --config config.yaml
python -m body_pose.extract_pose --config config.yaml
python -m body_pose.extract_features --config config.yaml
python -m body_pose.build_dataset --config config.yaml
python -m body_pose.train_lstm --config config.yaml
python -m body_pose.evaluate --config config.yaml --split auto
python -m body_pose.export_embeddings --config config.yaml --split all
```

Or run through evaluation:

```powershell
python run_pipeline.py --config config.yaml --through evaluate
```

`--split auto` selects `test` when a test patient exists, otherwise
`validation`. A final clinical evaluation must always retain an independent test
cohort.

## Shared temporal window manifest

`body_pose.build_dataset` creates `datasets/window_manifest.csv` with at least:

```text
window_index, window_id, media_id, patient_id, start_sec, end_sec,
label, label_id, split
```

IDs use exact millisecond boundaries, for example:

```text
Patient001__0000000000__0000003000
```

Source features are resampled by timestamp to `dataset.target_fps`, producing
fixed-length windows even when videos have different source FPS. These window
IDs and boundaries are the contract shared by the face, body, and video-motion
streams and later by synchronized clinical EEG if that stream is added.

All windows from one `patient_id` are assigned to exactly one of `train`,
`validation`, or `test`. The builder, trainer, evaluator, exporter, and synthetic
test all check this invariant.

## Configurable classes

Class order and output width come from configuration:

```yaml
labels:
  active: [normal, ictal]                 # SeizeIT2 annotation work
```

or:

```yaml
labels:
  active: [normal, pre_ictal, ictal]      # only with HUKM protocol support
```

The ordered names are stored in both the window NPZ and model checkpoint, so
evaluation, inference, and probability columns use the same mapping.

## Fusion-ready embeddings

```powershell
python -m body_pose.export_embeddings --config config.yaml --split all
```

Outputs:

- `results/body_embeddings.csv`: window metadata plus `body_emb_000`,
  `body_emb_001`, ...;
- `results/body_embeddings.npz`: the embedding matrix and aligned metadata
  arrays.

Join other stream embeddings by `window_id`, never by row position alone.

## Synthetic mechanical test

Because SeizeIT2 has no suitable public patient video, use the non-clinical
software test:

```powershell
python run_synthetic_test.py
```

It generates pose-like numeric features for nine synthetic patients, converts a
frame-based raw annotation table through the generic adapter, builds shared
windows, verifies patient splitting, trains a small LSTM, saves a checkpoint,
evaluates a held-out synthetic split, and exports body embeddings.

Outputs are under `synthetic_test_output/`. Synthetic accuracy and other metrics
have **no clinical meaning**.

For fast adapter/window/model unit checks:

```powershell
python -m unittest discover -s tests -v
```

## Expected outputs

```text
annotations/canonical_annotations.csv
keypoints/<media_id>.csv
features/<media_id>.csv
datasets/windows.npz
datasets/window_manifest.csv
checkpoints/best_lstm.pt
checkpoints/standard_scaler.joblib
results/training_history.csv
results/<split>_metrics.json
results/<split>_predictions.csv
results/<split>_confusion_matrix.png
results/body_embeddings.csv
results/body_embeddings.npz
```

## What changes when HUKM vEEG arrives

1. Confirm raw annotation column names, time origin/unit, per-video FPS, patient
   and recording identifiers, and the clinical definition of every class.
2. Update only the generic adapter configuration (or add a thin HUKM-specific
   adapter if one export has richer semantics).
3. Confirm video filenames map uniquely to canonical `media_id` values and
   verify video/EEG clock synchronization and drift.
4. Run MediaPipe on de-identified HUKM video, review pose visibility and failure
   patterns under blankets/occlusion, and choose window duration/stride with the
   clinical team.
5. Build face and optical-flow/frame-difference streams against the same
   `window_manifest.csv`, then join embeddings by `window_id`.
6. If clinical scalp EEG is later included, implement a separate clinical EEG
   adapter/preprocessor/encoder. Do not transfer assumptions from SeizeIT2
   behind-the-ear wearable EEG without a dedicated validation study.

Report patient/event/hour counts per split, per-patient metrics, confidence
intervals, simple baselines, ablations, and clinician-reviewed failure cases.
