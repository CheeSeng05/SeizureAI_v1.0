# Unsupervised facial movement detection

This workflow learns movement patterns without seizure, emotion, or eye-state
labels. It uses an Isolation Forest on 21 window features: left/right eye and
mouth aperture, eye asymmetry, variation and speed, geometry change, and optical
flow. It is a separate baseline alongside the existing supervised neural model.

The neural model now uses InceptionResnetV1, learned frame positions, and temporal
multi-head attention, without a BiGRU. `train` and `train_emotions` remain
supervised; `unsupervised` below remains the separate label-free baseline.

Inputs are the existing `video_manifest.csv`, `[T,4,H,W]` face arrays and `[T,20]`
geometry arrays created by `preprocess.prepare_dataset`. The geometry uses the
MediaPipe pair layout in `face_preprocess.FACE_LANDMARK_PAIRS`; the standalone
68-point `facial_landmark_dataset.py` export is a different format.

## Prepare and train

Run these commands from `FYPModels` using `.venv-facial/bin/python`, or activate
that environment and use `python`. The base project dependencies include
scikit-learn and joblib; neural checkpoints are unnecessary for this detector.

First prepare an unlabelled patient-folder dataset following
[the preprocessing instructions](../../preprocess/README.md). The full manifest
must include both training and validation patients. A preview containing only
training windows is not enough to select an independent threshold. Keep a third,
untouched patient group for testing.

```bash
python -m seizure_multimodal.facial.unsupervised train \
  --manifest results/HUKM_dataset/video_manifest.csv \
  --checkpoint checkpoints/fyp/facial/movement_anomaly.joblib
```

Training ignores `label` and `class_name`, including any labels that happen to
exist in the manifest. Test arrays are never loaded during training. Patient
overlap between splits is rejected before loading arrays.

The forest learns from training windows only. Its threshold is set strictly
above the 99th percentile of unlabelled validation scores by default. Ties are
excluded at the cutoff. `--validation-quantile 0.99` is a chosen review workload
setting, not a learned seizure prevalence or a guaranteed 1% false alarm rate.
The forest does not need feature scaling; there is no imputation of failed
tracking. All features and settings are saved with the model.

The checkpoint's `.joblib.json` companion records patient partitions, feature
names, settings, threshold, package version, score summaries and exclusions.
Joblib checkpoints should come from a trusted source and be loaded with a
compatible scikit-learn environment.

## Score unseen videos

```bash
python -m seizure_multimodal.facial.unsupervised predict \
  --manifest results/HUKM_dataset/video_manifest.csv \
  --checkpoint checkpoints/fyp/facial/movement_anomaly.joblib \
  --output results/facial_anomalies
```

The default prediction split is `test`. Use `--split unassigned` for new patient
recordings prepared with that split. Patients used for training or threshold
selection cannot be used for this prediction/evaluation path.

Outputs:

- `anomaly_scores.csv`: one row for every selected window, with timestamps,
  quality status, exclusion reason, anomaly score and candidate-window flag.
  Excluded windows have blank scores and flags, never a negative prediction.
  A detector ID prevents evaluating these scores with a different checkpoint.
- `candidate_events.csv`: merged intervals requiring review; headers are still
  written when no events are found.
- `summary.json`: score quantiles, exclusions, scored duration and candidate
  events per scored hour. Overlapping windows are counted once in duration;
  unknown gaps are excluded. No seizure metrics are inferred without references.

The score is the negative of scikit-learn's `IsolationForest.score_samples`:
higher values indicate more unusual observations. Although bounded by 0 and 1,
it is **not a seizure probability**. See the
[Isolation Forest documentation](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html).

By default a window needs at least 80% face-region coverage and 80% numerically
valid geometry frames. All-zero landmark rows are failed detections. Aperture
and geometry speeds use only adjacent valid frames; optical flow excludes the
first frame and invalid transitions. These checks cannot guarantee correct
anatomical landmark placement. Missing files or malformed arrays fail explicitly.

Candidate events must last at least 2 seconds by default. Adjust
`--minimum-event-duration-sec`, `--gap-tolerance-sec` and
`--refractory-period-sec` when training, before examining test results. The
settings are frozen in the checkpoint. Even a nonzero gap tolerance cannot
merge events across time with no scored-window coverage. Event timestamps have
the resolution of the preprocessing windows, not frame-level seizure onsets.

## Evaluate against independent labels

Collect labels separately; do not use the held-out test results to select the
threshold or settings. A window-label CSV contains:

```csv
window_id,label
example_window_0001,0
example_window_0002,1
example_window_0003,
```

Use the actual window IDs from `anomaly_scores.csv`. Blank or absent labels are
unknown; partially labelled sets are evaluated only on labelled, scored windows.

```bash
python -m seizure_multimodal.facial.unsupervised evaluate \
  --predictions results/facial_anomalies/anomaly_scores.csv \
  --checkpoint checkpoints/fyp/facial/movement_anomaly.joblib \
  --labels annotations/test_window_labels.csv \
  --output results/facial_anomalies/reference_evaluation.json
```

Alternatively replace `--labels` with `--annotations annotations/test_events.csv`
for continuous recordings using these columns:

```csv
patient_id,recording_id,monitor_start_sec,monitor_end_sec,seizure_onset_sec,seizure_offset_sec
P1,R1,0,600,120,135
P1,R1,0,600,420,445
P2,R2,0,600,,
```

Every monitoring interval must have been fully reviewed for seizures. A blank
onset/offset means a reviewed seizure-free recording, not an unlabelled recording.
Repeat the same monitoring bounds for multiple seizures in one recording.
Annotations must cover exactly the predicted recordings and contain all their
window timestamps. Onsets are treated as clinical onsets.

Window metrics use the continuous anomaly score and frozen threshold. Event
sensitivity, false positives per hour and onset/offset errors are available only
when scored windows fully cover the reviewed monitoring intervals. Tracking
exclusions or gaps otherwise leave those temporal metrics unavailable with a
reason. Quality exclusions are reported and never counted as true negatives.

The detector assumes relevant unusual movements differ from common training
patterns. If seizures dominate training, or a seizure resembles common movement,
anomalies may not identify seizures. Lighting, viewpoint and tracking changes can
also be unusual. Independent reference evaluation is what establishes whether
the learned patterns are useful for seizure detection.

## Verification

```bash
python -m unittest tests.test_facial_unsupervised tests.test_facial_evaluation
```

## Neural attention model

From `FYPModels`, with `facenet-pytorch` available:

```bash
python -m seizure_multimodal.facial.train \
  --manifest results/facial_patient_70_15_15/video_manifest.csv \
  --checkpoint checkpoints/fyp/facial/inception_resnet_v1_attention_final.pth \
  --attention-heads 8 --max-frames 512

python -m seizure_multimodal.facial.inference \
  --input path/to/face_window.npy \
  --checkpoint checkpoints/fyp/facial/inception_resnet_v1_attention_final.pth
```

The trainer includes position embeddings in its optimizer and stores their
capacity in the checkpoint for inference. Existing BiGRU checkpoints need
retraining. Windows must contain 1 to `max_frames` frames, without padding.

For the expression/semiology model, preprocess at 25 fps or above and supply
`window_id,emotion,semiologies,annotator_id` annotations:

```bash
python -m seizure_multimodal.facial.train_emotions \
  --manifest results/microexpression_patient_70_15_15/video_manifest.csv \
  --annotations configs/microexpression_annotations.csv \
  --checkpoint checkpoints/fyp/facial/inception_resnet_v1_attention_emotions_best.pth
```

Blank targets are unknown and excluded from the loss. `semiologies=none` is an
explicit negative; `fear|motor_asymmetry` lists observed signs. Seizure targets
come from the manifest's explicit 0/1 labels. Windows without any observed task
label are excluded from this supervised trainer. The training manifest enforces
patient separation, and test arrays are not loaded during training.

```bash
python -m unittest tests.test_facial_model tests.test_emotion_training
```
