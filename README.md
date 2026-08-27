# MDS10-FYP

Automated Video Analysis in Temporal Lobe Epileptic Seizures Using Multimodal Deep Learning.

This repository keeps the model, data platform, database setup, and web UI in one Git repo.
Raw datasets and local test videos are not stored in Git.

## What Is Inside

```text
FYPModels/           Current seizure model code and pretrained video checkpoints
fyp_data_platform/   PostgreSQL + MinIO database/storage/orchestration commands
frontend/            Laravel web UI prototype
BodyPose/            Legacy body-pose baseline kept for reference only
```

`FYPModels/test_video/` is ignored by Git. Put local test videos there when you want to run the model.

## What Not To Push

Do not commit raw data, local environments, or generated results:

```text
FYPModels/test_video/
FYPModels/results/
FYPModels/.venv/
FYPModels/datasets/
FYPModels/Epileptic Seizure Recognition.csv
*.mp4
*.zip
.DS_Store
```

Each person creates their own Python environment locally. That environment is not part of the model and should not be pushed.

## Simple Explanation

The model and database are separate:

- The model can read only an MP4 and create timestamped prediction files.
- A label workbook is optional and is only needed when you want accuracy/evaluation metrics.
- MinIO stores large files such as videos and model outputs.
- PostgreSQL stores searchable records such as dataset ID, patient ID, recording ID, file locations, checksums, model-run status, and metrics.
- PostgreSQL does not store the actual video. It stores the MinIO path to the video.
- The local browser UI lives in `fyp_data_platform`. Dragging a video into that UI uploads it to MinIO and records it in PostgreSQL.

For the original sample evaluation run, put these files on your own computer:

```text
FYPModels/test_video/Sz2PG.mp4
FYPModels/test_video/Label.xlsx
```

For deployment-style prediction runs, only the MP4 is required. The terminal
workflow is documented in `fyp_data_platform/README.md` under
`Authoritative model integration`.

## CLI, Docker, And Environment

These three things are different:

```text
CLI command          A command you type in Terminal or PowerShell
Docker              Runs PostgreSQL and MinIO for you
Python environment  Installs the Python packages needed by the commands
```

The CLI does not replace Docker. The CLI talks to Docker services.

The CLI also does not set up Python automatically. You set up the Python environment first, then the CLI command can run.

Example CLI command:

```bash
python -m fyp_data_platform --help
```

If this command fails with `ModuleNotFoundError`, the Python environment is not set up yet.

The usual order is:

```text
1. Create Python environment
2. Install project packages
3. Start Docker
4. Run CLI commands
```

## Run Model Only On Mac

Use this when you only want to test the model and do not need the database upload.

Install Python 3.12 if you do not have it:

```bash
brew install python@3.12
```

Set up the model environment:

```bash
cd "/Users/lychee/Documents/Y3S2_FIT3162_FYP/MDS10-FYP/FYPModels"
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[video]"
```

Run the video model:

```bash
python -m seizure_multimodal test --mode video --config configs/inference.yaml --root . --device auto
```

The output will be created locally in:

```text
FYPModels/results/video_only/
```

## Run Model Only On Windows

Use PowerShell.

Set up the model environment:

```powershell
cd "C:\path\to\MDS10-FYP\FYPModels"
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[video]"
```

Run the video model:

```powershell
python -m seizure_multimodal test --mode video --config configs/inference.yaml --root . --device auto
```

The output will be created locally in:

```text
FYPModels\results\video_only\
```

## Run With Database On Mac

Use this when you want to upload the sample video and label file into MinIO, record metadata in PostgreSQL, run the model, then upload the model outputs.

First open Docker Desktop and wait until it is running.

For the simple drag-and-drop browser UI, follow the `Local browser UI` section
in `fyp_data_platform/README.md`. The UI runs locally at
`http://127.0.0.1:8080` and uses the same Docker PostgreSQL and MinIO services
as the terminal commands below.

### First-Time Setup On Mac

Run this once on your computer.

Set up the database CLI environment:

```bash
cd "/Users/lychee/Documents/Y3S2_FIT3162_FYP/MDS10-FYP/fyp_data_platform"
cp .env.example .env
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev,ui]"
python -m pip install -e "../FYPModels[video]"
```

Check that the CLI works:

```bash
python -m fyp_data_platform --help
```

Start PostgreSQL and MinIO:

```bash
docker compose up -d --build
docker compose ps -a
```

Create the database tables:

```bash
python -m fyp_data_platform --config configs/docker.example.yaml db-upgrade
```

### Daily Use On Mac

Run this when the project has already been set up before.

Turn on the database CLI environment:

```bash
cd "/Users/lychee/Documents/Y3S2_FIT3162_FYP/MDS10-FYP/fyp_data_platform"
source .venv/bin/activate
```

Start PostgreSQL and MinIO:

```bash
docker compose up -d
docker compose ps -a
```

Upload the local test video into MinIO and register it in PostgreSQL:

```bash
python -m fyp_data_platform --config configs/docker.example.yaml register-media \
  --file ../FYPModels/test_video/Sz2PG.mp4 \
  --dataset-id development \
  --dataset-name "Development recordings" \
  --patient-id Pat01 \
  --recording-id Sz2PG \
  --modality video
```

Run prediction-only mode:

```bash
python -m fyp_data_platform --config configs/docker.example.yaml model-run \
  --dataset-id development \
  --recording-id Sz2PG \
  --mode video
```

To calculate evaluation metrics, also register the matching label file:

```bash
python -m fyp_data_platform --config configs/docker.example.yaml register-media \
  --file ../FYPModels/test_video/Label.xlsx \
  --dataset-id development \
  --dataset-name "Development recordings" \
  --patient-id Pat01 \
  --recording-id Sz2PG \
  --modality labels
```

Then run development/evaluation mode:

```bash
python -m fyp_data_platform --config configs/docker.example.yaml model-run \
  --dataset-id development \
  --recording-id Sz2PG \
  --mode video \
  --require-labels
```

Check the database counts:

```bash
python -m fyp_data_platform --config configs/docker.example.yaml catalog-summary
```

Open MinIO in a browser:

```text
http://localhost:9001
```

Default local login from `.env.example`:

```text
username: fyp_minio
password: change_me_minio
```

## Run With Database On Windows

Use PowerShell. First open Docker Desktop and wait until it is running.

For the simple drag-and-drop browser UI, follow the `Local browser UI` section
in `fyp_data_platform\README.md`. The UI runs locally at
`http://127.0.0.1:8080` and uses the same Docker PostgreSQL and MinIO services
as the terminal commands below.

### First-Time Setup On Windows

Run this once on your computer.

Set up the database CLI environment:

```powershell
cd "C:\path\to\MDS10-FYP\fyp_data_platform"
Copy-Item .env.example .env
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev,ui]"
python -m pip install -e "..\FYPModels[video]"
```

Check that the CLI works:

```powershell
python -m fyp_data_platform --help
```

Start PostgreSQL and MinIO:

```powershell
docker compose up -d --build
docker compose ps -a
```

Create the database tables:

```powershell
python -m fyp_data_platform --config configs/docker.example.yaml db-upgrade
```

### Daily Use On Windows

Run this when the project has already been set up before.

Turn on the database CLI environment:

```powershell
cd "C:\path\to\MDS10-FYP\fyp_data_platform"
.\.venv\Scripts\Activate.ps1
```

Start PostgreSQL and MinIO:

```powershell
docker compose up -d
docker compose ps -a
```

Upload the local test video into MinIO and register it in PostgreSQL:

```powershell
python -m fyp_data_platform --config configs/docker.example.yaml register-media `
  --file ..\FYPModels\test_video\Sz2PG.mp4 `
  --dataset-id development `
  --dataset-name "Development recordings" `
  --patient-id Pat01 `
  --recording-id Sz2PG `
  --modality video
```

Run prediction-only mode:

```powershell
python -m fyp_data_platform --config configs/docker.example.yaml model-run `
  --dataset-id development `
  --recording-id Sz2PG `
  --mode video
```

To calculate evaluation metrics, also register the matching label file:

```powershell
python -m fyp_data_platform --config configs/docker.example.yaml register-media `
  --file ..\FYPModels\test_video\Label.xlsx `
  --dataset-id development `
  --dataset-name "Development recordings" `
  --patient-id Pat01 `
  --recording-id Sz2PG `
  --modality labels
```

Then run development/evaluation mode:

```powershell
python -m fyp_data_platform --config configs/docker.example.yaml model-run `
  --dataset-id development `
  --recording-id Sz2PG `
  --mode video `
  --require-labels
```

Check the database counts:

```powershell
python -m fyp_data_platform --config configs/docker.example.yaml catalog-summary
```

Open MinIO in a browser:

```text
http://localhost:9001
```

## What The Commands Mean

```bash
python -m fyp_data_platform --help
```

Checks whether the database CLI is installed and can run.

```bash
python3.12 -m venv .venv
```

Creates a private Python folder for this project.

```bash
source .venv/bin/activate
```

Turns on that private Python folder in the current terminal.

```bash
python -m pip install -e ".[video]"
```

Installs the model code and video dependencies.

```bash
python -m pip install -e ".[dev]"
```

Installs the database CLI and its dependencies.

```bash
docker compose up -d --build
```

Builds and starts the database services in Docker. Use this for first-time setup.

```bash
docker compose up -d
```

Starts the database services after they have already been built before. Use this for daily use.

```bash
db-upgrade
```

Creates or updates PostgreSQL tables.

```bash
register-media
```

Uploads a file to MinIO and saves its record in PostgreSQL.

```bash
model-run --mode video
```

Runs video-only inference and stores the outputs. Add `--require-labels` when
you intentionally want a development/evaluation run that fails if the label
workbook is missing.
