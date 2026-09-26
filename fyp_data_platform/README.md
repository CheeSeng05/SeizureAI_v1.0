# FYP Data Platform

The complete setup and usage guide is in the repository [README](../README.md).

This directory contains the React dashboard, Flask API, PostgreSQL catalog,
MinIO/S3 storage integration, model orchestration, migrations, tests and Docker
Compose definition.

Quick start from PowerShell:

```powershell
Copy-Item .env.example .env
docker compose up --build
```

If the prompt is Command Prompt (`cmd.exe`) instead of PowerShell, use
`copy .env.example .env`. The dashboard accepts EEG `.csv`, `.edf`, and
Nicolet/Natus `.e` uploads.

Then open http://localhost:8080.

NVIDIA CUDA quick start (Windows/Linux with NVIDIA Docker support):

```text
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build
```

Open http://localhost:8080/system to confirm whether PyTorch is using CUDA or
CPU. See the repository README for first-time requirements, daily commands and
troubleshooting.
