# Portable Seizure Detector Setup Guide

This package contains a verified Keras EEG seizure model plus a portable
inference notebook. It supports:

1. native Ubuntu/WSL2 development;
2. Docker Desktop on Windows using the WSL2 backend;
3. CPU fallback when no NVIDIA GPU is available;
4. fusion-ready timestamped EEG outputs for a larger multimodal system.

---

## Package contents

- `seizure_detector_model.keras` — trained CNN–BiLSTM–Attention EEG model
- `seizure_model_inference.ipynb` — EDF inference notebook
- `requirements.txt` — Python dependencies
- `setup_wsl.sh` — reproducible WSL project environment setup
- `Dockerfile` — portable Docker image
- `verify_setup.py` — environment/model verification
- `MULTIMODAL_FUSION.md` — interface for EEG + video/face/pose/motion fusion
- `SHA256SUMS.txt` — integrity hashes

---

# A. Windows + WSL2 Setup

## 1. Install or check WSL2

Open Windows PowerShell:

```powershell
wsl --status
wsl -l -v
```

If Ubuntu is not installed:

```powershell
wsl --install -d Ubuntu
```

Set Ubuntu as the default development distribution:

```powershell
wsl --set-default Ubuntu
```

Update WSL:

```powershell
wsl --update
```

Enter Ubuntu:

```powershell
wsl -d Ubuntu
```

Do **not** use the `docker-desktop` distribution as your development shell.

Inside Ubuntu, verify:

```bash
cat /etc/os-release
```

For NVIDIA GPU users:

```bash
nvidia-smi
```

The GPU should appear before proceeding with GPU setup.

## 2. Project location

For best Linux filesystem performance, place the project inside WSL, e.g.:

```bash
mkdir -p ~/projects
cd ~/projects
```

A project stored under `/mnt/c/...` also works, but large EEG/video datasets can
be slower because they are accessed through the Windows filesystem.

## 3. Recreate the Python environment

From the extracted package/project folder:

```bash
chmod +x setup_wsl.sh
./setup_wsl.sh
```

This creates a user-independent project environment:

```text
<project>/.venv/bin/python
```

Activate it:

```bash
source .venv/bin/activate
```

Verify:

```bash
python verify_setup.py
```

Expected model shapes:

```text
Model input:  (None, 1024, 18)
Model output: (None, 1)
```

For a configured NVIDIA system, the TensorFlow GPU list should not be empty.

## 4. VS Code through WSL

Install Visual Studio Code on Windows and install the Microsoft WSL extension.

From the Ubuntu project directory:

```bash
code .
```

The bottom-left of VS Code should show:

```text
WSL: Ubuntu
```

For the notebook, select the kernel:

```text
EEG Project (WSL/Docker)
```

or the Python interpreter:

```text
<project>/.venv/bin/python
```

The notebook must not use a Windows interpreter such as:

```text
C:\Users\...\python.exe
```

---

# B. Docker Desktop Setup

Docker is the recommended portability layer when different laptops have
different Python installations.

## 1. Windows host

Install Docker Desktop and use its WSL2-based engine.

In Docker Desktop:

- enable the WSL2 engine;
- enable WSL integration for Ubuntu;
- use Linux containers.

For NVIDIA GPU acceleration, the Windows host must have an NVIDIA GPU and a
current NVIDIA Windows driver supporting WSL2.

## 2. Verify Docker itself

From PowerShell or Ubuntu:

```bash
docker version
```

For NVIDIA GPU hosts, verify GPU passthrough with the GPU validation command in
Docker Desktop's current GPU documentation before running the EEG image.

## 3. Build the image

From the package directory:

```bash
docker build -t eeg-seizure:2.20 .
```

## 4. Run with NVIDIA GPU

From WSL/Linux:

```bash
docker run --rm -it \
  --gpus=all \
  -p 8888:8888 \
  -v "$PWD:/workspace" \
  eeg-seizure:2.20
```

From Windows PowerShell:

```powershell
docker run --rm -it `
  --gpus=all `
  -p 8888:8888 `
  -v "${PWD}:/workspace" `
  eeg-seizure:2.20
```

Jupyter prints a local URL containing a token. Open that URL in the browser.

## 5. Run without a GPU

The same image can be run in CPU mode by omitting `--gpus=all`:

```bash
docker run --rm -it \
  -p 8888:8888 \
  -v "$PWD:/workspace" \
  eeg-seizure:2.20
```

Inference will be slower but the model remains usable.

---

# C. Running the EEG Inference Notebook

Open:

```text
seizure_model_inference.ipynb
```

In Cell 2, change `EDF_PATH`, or set it before starting Jupyter:

```bash
export EEG_EDF_PATH="/path/to/file.edf"
export SESSION_ID="patient01_session01"
```

The notebook loads:

```text
seizure_detector_model.keras
```

by default.

It reproduces the original training preprocessing:

- 256 Hz
- first 18 EDF channels
- 0.5–100 Hz band-pass
- 60 Hz notch
- per-channel z-score
- ±5 clipping
- 4-second windows
- 50% overlap / 2-second stride
- model input `(1024, 18)`

---

# D. Multimodal Fusion

The notebook generates:

```text
eeg_predictions.csv
eeg_fusion_features.npz
```

Use `eeg_predictions.csv` for late fusion.

Use `eeg_fusion_features.npz` for intermediate/embedding fusion.

Every modality should share:

```text
session_id
window_id
start_sec
end_sec
```

For the same patient/session, set the same `SESSION_ID` in all modality
pipelines.

See `MULTIMODAL_FUSION.md` for the detailed interface.

For a large multimodal system, separate containers per modality are preferable
when TensorFlow, PyTorch, MediaPipe, or OpenCV dependency versions conflict.
Share only timestamped outputs/embeddings through a common mounted folder or
database.

---

# E. Important Model Limitations

1. The EEG model was trained using the first 18 channels rather than a saved,
   explicit channel-name montage. Cross-dataset use must verify channel identity
   and order.
2. The training pipeline used a window-level split. Research evaluation should
   be repeated with patient/session-independent train/validation/test splits to
   reduce leakage from overlapping windows.
3. The output is a model score/probability used with a 0.5 decision threshold;
   it should not automatically be interpreted as a calibrated clinical
   probability.
4. This package is for model development/research and is not a clinical medical
   device.

---

# F. Recommended Repository Layout

```text
project/
├── seizure_detector_model.keras
├── seizure_model_inference.ipynb
├── requirements.txt
├── setup_wsl.sh
├── Dockerfile
├── verify_setup.py
├── MULTIMODAL_FUSION.md
├── Input/
│   └── ...
├── outputs/
│   ├── eeg/
│   ├── face/
│   ├── pose/
│   └── motion/
└── fusion/
```

For multimodal development, keep raw patient/session identifiers outside
model-specific code and pass a shared `SESSION_ID` into each branch.
