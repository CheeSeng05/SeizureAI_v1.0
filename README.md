# SeizureAI v1.0

SeizureAI is a local research platform for running video and EEG seizure inference.
It provides one browser dashboard for uploading recordings, cataloguing inputs,
starting the authoritative model pipeline, and reviewing stored results.

This is a research prototype. It is not a medical device and must not be used
for clinical diagnosis or patient care.

## How it works

```text
Browser dashboard (React)
          |
          v
Web API and orchestration (Flask)
          |
          +---- PostgreSQL: datasets, patients, recordings, model runs
          +---- MinIO: video/EEG inputs, prediction files, plots, metrics
          +---- FYPModels: authoritative seizure inference pipeline
```

The browser never recalculates model metrics. It displays the prediction and
metric artifacts produced by `FYPModels` and stored by `fyp_data_platform`.

## Repository structure

```text
FYPModels/                    Model code and checkpoints
fyp_data_platform/
  frontend/                   React dashboard source
  fyp_data_platform/          Flask API, catalog, storage, orchestration
  migrations/                 PostgreSQL migrations
  docker/                     Container startup and MinIO build
  tests/                      Platform and API tests
  docker-compose.yml          Universal CPU application
  docker-compose.gpu.yml      NVIDIA CUDA override
```

## Local addresses

| Service | Address |
|---|---|
| SeizureAI dashboard | http://localhost:8080 |
| MinIO console | http://localhost:9001 |
| MinIO API | http://localhost:9000 |
| PostgreSQL | localhost:5432 |

Ports can be changed in `fyp_data_platform/.env`.

## Requirements

- Docker Desktop
- Git
- Enough disk space for container images, checkpoints, videos and results

Python, Node.js, PostgreSQL and MinIO do not need to be installed on the host.
The default image runs inference on CPU so the same setup works on Windows,
macOS and Linux. NVIDIA users can build the optional CUDA image. The dashboard's
**System** page reports the processing mode, device name, memory, PyTorch version
and CUDA runtime that the application can actually access.

## Windows first-time setup

1. Install and open Docker Desktop. Wait until Docker is running.
2. Open PowerShell in the repository and create the local configuration:

```powershell
cd fyp_data_platform
Copy-Item .env.example .env
```

3. Open `.env` and replace the example PostgreSQL and MinIO passwords. Keep the
   MinIO password identical in `MINIO_ROOT_PASSWORD` and `AWS_SECRET_ACCESS_KEY`.
4. Build and start everything:

```powershell
docker compose up --build
```

The first build can take a while because it installs both the video/PyTorch and
EEG/TensorFlow model dependencies. When `app` is healthy, open
http://localhost:8080. Database migrations and bucket creation are automatic.

### Windows with an NVIDIA GPU

CUDA mode requires a current NVIDIA Windows driver and Docker Desktop using its
WSL2 backend. Confirm the driver can see the GPU first:

```powershell
nvidia-smi
```

Build and start the CUDA application with:

```powershell
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build
```

The CUDA image is larger and takes longer to download the first time. Startup
prints the detected GPU and refuses to continue if PyTorch cannot access CUDA.
Open http://localhost:8080/system to verify the active processing device.

## macOS first-time setup

1. Install and open Docker Desktop. Wait until Docker is running.
2. Open Terminal in the repository and create the local configuration:

```bash
cd fyp_data_platform
cp .env.example .env
```

3. Open `.env` and replace the example PostgreSQL and MinIO passwords. Keep the
   MinIO password identical in `MINIO_ROOT_PASSWORD` and `AWS_SECRET_ACCESS_KEY`.
4. Build and start everything:

```bash
docker compose up --build
```

When `app` is healthy, open http://localhost:8080. Apple Silicon uses the
CPU-oriented image, so inference may be slower than on a CUDA-capable machine.
CUDA is an NVIDIA technology and is not available through Linux Docker on macOS.

## Windows daily run

Open Docker Desktop, then run:

```powershell
cd fyp_data_platform
docker compose up
```

On a configured NVIDIA computer, use this instead:

```powershell
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up
```

Open http://localhost:8080. Stop with `Ctrl+C`, then optionally run:

```powershell
docker compose stop
```

## Choosing CPU or CUDA

Use universal CPU mode on macOS, computers without an NVIDIA GPU, or whenever
maximum portability matters:

```text
docker compose up
```

Use CUDA mode on Windows/Linux computers where Docker can access an NVIDIA GPU:

```text
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up
```

The application workflow and address are identical in both modes. Upload a
recording and click **Run** as usual; the model's existing `device=auto` setting
selects CUDA when the GPU container exposes it. Moving between different
supported NVIDIA models does not require changing `.env` or resetting volumes.

To switch modes, stop the current stack with `Ctrl+C` and `docker compose down`,
then start the other mode. Do not add `-v`; the same database, uploads and results
are retained.

AMD GPUs currently fall back to the universal CPU setup. Apple Metal/MPS is not
available to this Linux Docker container, so Macs also use CPU mode.

### Verify NVIDIA Docker access

If CUDA startup fails, update the NVIDIA driver and Docker Desktop, confirm WSL2
is enabled, and verify Docker can expose the GPU:

```powershell
docker run --rm --gpus all nvidia/cuda:12.8.1-base-ubuntu24.04 nvidia-smi
```

After startup, the **System** page is the authoritative check. It must show
`NVIDIA CUDA` and the GPU name; simply owning an NVIDIA GPU is not sufficient.

## macOS daily run

Open Docker Desktop, then run:

```bash
cd fyp_data_platform
docker compose up
```

Open http://localhost:8080. Stop with `Ctrl+C`, then optionally run:

```bash
docker compose stop
```

## Data safety

PostgreSQL, MinIO and model working data use named Docker volumes. Both of these
commands keep stored data:

```text
docker compose stop
docker compose down
```

Do not run `docker compose down -v` unless you intentionally want to delete the
database, uploaded videos and stored results.

## Basic use

1. Open the dashboard.
2. Choose **Upload**, select Video, EEG, or **Both independently**, and register the required files.
3. Open **Recordings** and select the matching Video, EEG, or Both run button.
4. Wait for inference to finish. Only one inference run is allowed at a time.
5. Open **Model runs** to view probabilities, events, metrics and artifacts.

While a run is active, the Recordings page shows the current pipeline stage,
processed video clips or EEG batches, percentage, elapsed time and an approximate
remaining time. Progress uses the existing job polling endpoint and requires no
additional service or different Docker command.

Completed video runs include a synchronized result viewer. The original MP4 is
streamed from private storage with byte-range requests, detected events can be
clicked to seek three seconds before the event, and the probability graph shows
the current playback position. Workbook onset markers are shown when available.
The event list can be filtered to show all entries, matched detections,
model-only detections or workbook timestamps. Light red means model-only, blue
means workbook ground truth, and green means a model event matched the labelled
timing. Every entry can be clicked to play from three seconds before it.

Matching does not change the model output. When the workbook includes a seizure
end, a model event is matched when the two time ranges overlap. When the workbook
only provides onset timestamps, a model event is matched when it contains the
clinical onset (falling back to EEG onset). No seizure duration is invented.
The result page also has an analysis threshold slider from `0.00` to `1.00`,
defaulting to `0.50`. It instantly rebuilds detected events and recalculates
threshold-dependent label metrics from saved probabilities; it does not rerun
the model or overwrite the stored run. AUROC stays unchanged because it measures
probability ranking across thresholds. Use this for exploration, not for tuning
against the final test dataset.

The threshold can also be typed precisely. The result page shows a moving
threshold line, default-versus-selected metrics, a plain-language summary,
high-confidence and label-aware event filters, event details, playback/zoom
controls, and a downloadable JSON report. Metric names include hover help.
These tools work for prediction-only and label-backed video; ground-truth
comparisons are shown only when a workbook was supplied.

The **Compare runs** page applies separate temporary thresholds to two stored
probability timelines, recalculates their event and label-backed metrics, and
shows Run A minus Run B. Differences use green only when the direction is an
improvement, red for deterioration and grey for neutral measures such as event
count. It warns when recordings, models, modalities or label modes differ.

The result Export button creates a human-readable PDF containing the selected
threshold, run/device information, a plain-language summary, evaluation metrics,
detected event timestamps and the research-use disclaimer. When workbook labels
are present, it also includes EEG onset, clinical onset, seizure end availability
and the number of model events matched to the workbook timing.
The Compare Runs page can export its own formatted PDF with both configurations,
independent thresholds, the colour-coded A-minus-B table, fair-comparison
warnings and a plain-language interpretation.

The **Settings** page provides persistent light, dark or system appearance,
standard/high contrast, and reduced-motion preferences. Preferences are stored
only in the current browser and do not change the pipeline or database.
AVI, MKV and MOV uploads remain stored and downloadable, but browser preview is
currently guaranteed only for browser-compatible MP4 files.

### Prediction-only video

Upload one or more videos without a workbook, then choose **Predict** on the
Recordings page. The result contains probabilities and detected events but no
ground-truth accuracy metrics.

### Label-backed video evaluation

Upload exactly one video and optionally attach an `.xlsx` label workbook. Enter
the Patient ID, Seizure ID and Seizure Type that identify exactly one workbook
row. For the supplied VSViG workbook these look like `Pat01`, `Sz2` and `PG`.

On the Upload page, first enable **Compare video with ground truth**. The workbook
selector and matching identity fields remain hidden when comparison is disabled.

The platform validates the match before storage. On the Recordings page choose
**Evaluate** to run with ground truth. The result reports confusion counts,
accuracy, balanced accuracy, sensitivity/recall, specificity, precision, F1,
AUROC, evaluated/excluded windows, false-positive alarms per hour and onset
error when those quantities are supported by the labels.

The supplied `Label.xlsx` contains EEG and clinical onset but no seizure end.
The transition between those onsets is excluded from classification, and offset
accuracy remains unavailable; the platform does not invent a missing end time.

### EEG evaluation and recording prediction

Choose **EEG** and upload one `.csv`, `.edf`, or Nicolet/Natus `.e` file.
Benchmark CSV files must contain the exact feature columns `X1` through `X178`
and a ground-truth `y` column; that route retains the existing label-backed
classification metrics. EDF and `.e` files are validated by their recording
reader and do not require CSV columns or labels.

Raw recordings are converted to the frozen model's exact 18-channel CHB-MIT
bipolar montage. Common-reference electrodes are subtracted in physical
microvolts, with standard T3/T4/T5/T6 fallbacks for T7/T8/P7/P8. Signals are
resampled to 256 Hz, filtered and normalized, then processed as overlapping
four-second windows with a two-second stride. The result page shows recording
duration, window count, windows at or above the fixed 0.50 threshold, maximum
probability, and a downloadable complete probability timeline. Embedded
clinician events are retained separately and never supplied to the model.

Raw-recording results are experimental compatibility predictions, not a
clinical diagnosis. The model was trained on CHB-MIT rather than this clinical
dataset, the montage is reconstructed for common-reference recordings, and
500 Hz recordings are resampled to 256 Hz. Domain shift and the original
notebook's overlapping-window/random-window split limit generalisation; the
reported “97%” must not be treated as expected accuracy on a new recording.

### Video and EEG independently

Choose **Both independently**, provide one video, one supported EEG file, and a Recording
ID. Both files are registered under the same catalog entry. Selecting **Both
independently** on Recordings runs the existing video and EEG models separately
and displays separate result sections. It does not fuse their predictions and
does not claim that unrelated inputs are synchronized.

The Docker startup commands are unchanged. Both model dependency sets are in
the same CPU or CUDA application image. The CUDA setup accelerates the PyTorch
video workflow when available; the System page and each stored run report the
device actually detected.

### CPU and CUDA runtime comparison

Every new run stores its actual processing device and elapsed time for input
staging, configuration, the model pipeline, artifact persistence and the total
orchestration run. The Model runs page shows total time and CPU/CUDA mode. Open a
result for the detailed phase timings.

For a fair CPU/CUDA benchmark, run the same stored recording in each mode with
the same checkpoint and configuration. Existing historical runs created before
runtime tracking show an unavailable duration.

## Skeleton overlay and archived items

New video runs save `video_only/pose_keypoints.json`. On a run result, enable
**Show skeleton** above the playback caption. It follows playback and seeking,
and **Fullscreen with overlay** keeps the skeleton visible in fullscreen.
Green/yellow/pink indicate stronger/weaker OpenPose confidence. These
colours are not seizure probability or model explanations. Undetected joints
and unprocessed trailing frames are hidden. OpenPose currently selects the
largest detected person per frame; it can track the wrong person when others
enter the scene. Old runs need to be rerun to produce pose data. Labels are
optional, and CPU and CUDA use the same overlay format.

Use **Archive** on Recordings or Model runs to hide an item. Archiving a
recording also hides its runs. In **Settings → Archived items**, restore the
recording to show those runs again; individually archived runs stay archived.
Archived recordings cannot receive uploads or start new runs until restored.

Permanent deletion is only available for archived items in Settings. The
confirmation shows file count, recorded file size, and affected run count;
type the exact ID to confirm. Deleting a run removes its results and working
copies but keeps its source recording. Deleting a recording also removes its
managed video, EEG, labels, runs, annotations, windows and results. Original
files selected from your computer are untouched. Dataset/patient catalog rows
are retained. Active processing prevents archive/delete operations.

If cleanup fails, the item stays archived and marked as deletion started.
Use **Retry deletion** after fixing the storage problem. Restoration is blocked
once deletion begins because some objects may already have been removed.
The supported Docker deployment uses one app worker to coordinate processing
and data changes; do not run parallel CLI ingestion/deletion during cleanup.

From `fyp_data_platform`, apply this update using the usual CPU command
`docker compose up --build`, or the NVIDIA command:

```text
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build
```

Startup automatically adds the archive fields through a database migration.
Daily startup uses the same command without `--build`. Do not use `down -v`
to delete individual items: it removes the application's named data volumes.

## Development commands

Backend tests:

```text
cd fyp_data_platform
python -m pip install -e ".[dev,ui]"
python -m pytest -q
```

Frontend production build:

```text
cd fyp_data_platform/frontend
pnpm install
pnpm run build
```

For UI development, run Flask on port 8080 and Vite on port 5173. Vite proxies
`/api` to Flask.
