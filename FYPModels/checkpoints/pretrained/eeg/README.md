# EEG checkpoint

The active inference configuration uses the Keras full-model artifact:

- `seizure_detector_model.keras`

The local Kaggle-derived notebook also references these compatible generated
artifacts:

- `best_seizure_model.h5`
- `seizure_detector_model.h5`
- `seizure_detector.weights.h5`

If you switch between formats, place the original downloaded model artifact in
this directory and update `configs/inference.yaml`. Do not create or save
randomly initialized replacement weights.
