# Catalog data model

| Table | Purpose | Important uniqueness/provenance |
|---|---|---|
| `datasets` | Public, HUKM, and future dataset registry | Primary `id`; version/source/license metadata |
| `patients` | Dataset-scoped pseudonymous subjects | `(dataset_id, source_patient_id)` |
| `recordings` | Session/task/modality recording unit | `(dataset_id, source_recording_id)`; video fields nullable |
| `files` | Object/reference catalog for raw and derived artifacts | `(dataset_id, source_relative_path)` and `(storage_backend, storage_key)`; SHA-256 |
| `annotations` | Canonical timestamp intervals | Stable SHA-256 `annotation_key`; explicit source event/provenance |
| `temporal_windows` | Shared multimodal/fusion intervals | Exact deterministic external `window_id` from BodyPose |
| `model_runs` | Reproducible `seizure_multimodal` execution metadata | Status, config, metrics, command, timestamps, artifacts |
| `predictions` | Compact per-window outputs | `(model_run_id, window_id)` via the window FK; JSON probabilities |

Database primary keys may be internal UUID strings, but `temporal_windows.window_id`
is the cross-stream fusion contract and must not be replaced by row order or an
incompatible identifier.
