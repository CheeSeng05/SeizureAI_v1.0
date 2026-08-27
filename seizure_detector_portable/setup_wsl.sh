#!/usr/bin/env bash
set -euo pipefail

echo "============================================================"
echo "Portable EEG Project - WSL2 Environment Setup"
echo "============================================================"

if ! grep -qi microsoft /proc/version 2>/dev/null; then
  echo "ERROR: Run this script inside Ubuntu on WSL2."
  echo "From Windows PowerShell: wsl -d Ubuntu"
  exit 1
fi

sudo apt update
sudo apt install -y build-essential curl ca-certificates

if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

uv python install 3.13

if [ ! -d ".venv" ]; then
  uv venv .venv --python 3.13 --seed
else
  echo "Reusing existing project-local .venv"
fi

uv pip install --python .venv/bin/python -r requirements.txt

.venv/bin/python -m ipykernel install \
  --user \
  --name eeg-project \
  --display-name "EEG Project (WSL/Docker)"

echo
echo "Checking environment..."
.venv/bin/python verify_setup.py || true

echo
echo "Setup complete."
echo "Activate with:"
echo "  source .venv/bin/activate"
echo
echo "Then open VS Code from this WSL project directory:"
echo "  code ."
