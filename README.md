# SeizureAI v1.0

SeizureAI v1.0 is a local research prototype for video-based seizure prediction.
It lets you upload MP4 videos in a browser, stores the videos in MinIO, stores
metadata in PostgreSQL, runs the seizure model, then shows whether seizure-like
segments were detected with timestamps and probabilities.

This is not a clinical system or medical device.

## Project Overview

```text
FYPModels/           Model code, video pipeline, EEG adapter, checkpoints
fyp_data_platform/   Local UI, Docker setup, MinIO/PostgreSQL catalog, model runner
```

The local website is a Flask UI inside `fyp_data_platform`.

```text
Browser UI      http://127.0.0.1:8080
MinIO console   http://localhost:9001
PostgreSQL      localhost:5432
```

MinIO stores uploaded videos and model output files. PostgreSQL stores searchable
metadata such as dataset ID, patient ID, recording ID, file paths, run status,
and result summaries. Videos are not stored inside PostgreSQL.

## Mac First-Time Setup

Open Docker Desktop first and wait until it is running.

```bash
cd "/path/to/SeizureAI_v1.0/fyp_data_platform"
cp .env.example .env
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev,ui]"
python -m pip install -e "../FYPModels[video]"
docker compose up -d --build
python -m fyp_data_platform --config configs/docker.example.yaml db-upgrade
python -m fyp_data_platform --config configs/docker.example.yaml ui --host 127.0.0.1 --port 8080
```

Open:

```text
http://127.0.0.1:8080
```

## Mac Daily Run

Open Docker Desktop first and wait until it is running.

```bash
cd "/path/to/SeizureAI_v1.0/fyp_data_platform"
source .venv/bin/activate
docker compose up -d
python -m fyp_data_platform --config configs/docker.example.yaml ui --host 127.0.0.1 --port 8080
```

Open:

```text
http://127.0.0.1:8080
```

## Mac Safe Shutdown

Stop the website:

```bash
Ctrl+C
```

Stop Docker services but keep all uploaded videos, results, and database rows:

```bash
docker compose stop
```

Use this only when you want to remove containers but keep stored data:

```bash
docker compose down
```

Do not run `docker compose down -v` unless you want to delete the local
PostgreSQL database and MinIO files.

## Windows First-Time Setup

Use PowerShell. Open Docker Desktop first and wait until it is running.

```powershell
cd "C:\path\to\SeizureAI_v1.0\fyp_data_platform"
Copy-Item .env.example .env
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev,ui]"
python -m pip install -e "..\FYPModels[video]"
docker compose up -d --build
python -m fyp_data_platform --config configs/docker.example.yaml db-upgrade
python -m fyp_data_platform --config configs/docker.example.yaml ui --host 127.0.0.1 --port 8080
```

Open:

```text
http://127.0.0.1:8080
```

## Windows Daily Run

Use PowerShell. Open Docker Desktop first and wait until it is running.

```powershell
cd "C:\path\to\SeizureAI_v1.0\fyp_data_platform"
.\.venv\Scripts\Activate.ps1
docker compose up -d
python -m fyp_data_platform --config configs/docker.example.yaml ui --host 127.0.0.1 --port 8080
```

Open:

```text
http://127.0.0.1:8080
```

## Windows Safe Shutdown

Stop the website:

```text
Ctrl+C
```

Stop Docker services but keep all uploaded videos, results, and database rows:

```powershell
docker compose stop
```

Use this only when you want to remove containers but keep stored data:

```powershell
docker compose down
```

Do not run `docker compose down -v` unless you want to delete the local
PostgreSQL database and MinIO files.

## Basic Use

1. Start Docker Desktop.
2. Run the daily command for your operating system.
3. Open `http://127.0.0.1:8080`.
4. Drag and drop an MP4 video into the upload area.
5. Click `Run` beside the uploaded video.
6. Click the completed model run to view detection result, timestamps, and probabilities.

## Local Files Not Pushed

Do not commit local videos, generated results, database files, or environment
folders.

```text
.env
.venv/
fyp_data_platform/work/
fyp_data_platform/data/
FYPModels/test_video/
FYPModels/results/
*.mp4
*.zip
```
