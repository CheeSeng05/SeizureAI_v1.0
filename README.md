# SeizureAI v1.0

SeizureAI is a local research platform for running video-based seizure inference.
It provides one browser dashboard for uploading videos, cataloguing recordings,
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
          +---- MinIO: videos, prediction files, plots, metrics
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

The first build can take a while. When `app` is healthy, open
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
2. Choose **Upload** and register videos.
3. Open **Recordings** and select **Run**.
4. Wait for inference to finish. Only one inference run is allowed at a time.
5. Open **Model runs** to view probabilities, events, metrics and artifacts.

An upload without a registered label workbook is prediction-only. Accuracy
metrics require matching labels registered under the same dataset and recording.

## Development and tests

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
