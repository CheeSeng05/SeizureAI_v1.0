# EEG checkpoint

The active inference configuration uses the Keras full-model artifact:

- `seizure_detector_model.keras`

Verified artifact metadata:

- SHA-256: `048e57a6125970011d13c3edea9e38bf9a7bea027c67dddf8bb0372e83a32eae`
- input: `[batch, 1024, 18]` (time-major, 4 seconds at 256 Hz)
- output: one sigmoid seizure probability
- pre-output embedding: 32 values

The original Kaggle implementation establishes 256 Hz, four-second windows,
50% overlap, physical microvolts, and the exact 18-channel CHB-MIT bipolar
order now recorded in `configs/inference.yaml`. Common-reference recordings
are converted by explicit A-B subtraction; T3/T4/T5/T6 are accepted only as
the standard fallbacks for T7/T8/P7/P8. The checkpoint remains frozen and is
never overwritten by recording inference.

The Kaggle notebook's overlapping windows were randomly split at window level
from a small patient set. Possible overlap/patient leakage and domain shift mean
its reported accuracy is not evidence of clinical validity on new recordings.

The local Kaggle-derived notebook also references these compatible generated
artifacts:

- `best_seizure_model.h5`
- `seizure_detector_model.h5`
- `seizure_detector.weights.h5`

If you switch between formats, place the original downloaded model artifact in
this directory and update `configs/inference.yaml`. Do not create or save
randomly initialized replacement weights.
