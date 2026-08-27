# Multimodal Fusion Interface

The EEG notebook produces two outputs intended to make the EEG branch reusable
with face, body-pose, motion, or other modality pipelines.

## 1. Late fusion

`eeg_predictions.csv`

Columns:

- `session_id`
- `window_id`
- `start_sec`
- `end_sec`
- `eeg_seizure_probability`
- `eeg_predicted_class`
- `eeg_predicted_label`

A video/pose/motion pipeline should emit the same timing keys, for example:

- `session_id`
- `window_id`
- `start_sec`
- `end_sec`
- `video_probability`
- `pose_probability`
- `motion_probability`

Then join by `session_id + window_id`, or use time-overlap matching when the
modalities use different window sizes.

## 2. Intermediate fusion

`eeg_fusion_features.npz`

Arrays:

- `session_id`
- `window_id`
- `start_sec`
- `end_sec`
- `eeg_seizure_probability`
- `eeg_embedding` with shape `(N, 32)`

The `eeg_embedding` is taken from the model's penultimate 32-unit Dense layer.
It can be concatenated with video/face/pose/motion embeddings before a fusion
classifier.

Example conceptual fusion vector:

`[EEG 32-D] + [face embedding] + [pose embedding] + [motion embedding]`

## 3. Cross-laptop rule

Do not use machine-specific paths such as `/home/jameslim/...`.
Use:

- project-relative file paths;
- environment variables such as `EEG_EDF_PATH`, `SEIZURE_MODEL_PATH`;
- shared `SESSION_ID`;
- project-local `.venv`;
- Docker volumes for data/output folders.

## 4. Recommended container architecture

For a larger multimodal project, use one container per modality when dependency
stacks conflict:

- `eeg-service` — TensorFlow EEG model
- `face-service` — facial model
- `pose-service` — pose model
- `motion-service` — optical flow/motion model
- `fusion-service` — reads timestamped outputs/embeddings and fuses them

All services should write to a shared output directory and use a common
`session_id`, `window_id`, `start_sec`, and `end_sec` schema.

This prevents TensorFlow/PyTorch/MediaPipe/OpenCV dependency conflicts and makes
each modality independently replaceable.
