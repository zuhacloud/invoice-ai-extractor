#!/usr/bin/env bash
# One-time setup on macOS / Linux. Usage: bash setup.sh
# Creates the virtual environment, installs PyTorch (CUDA if an NVIDIA GPU is present),
# installs the dependencies, creates .env, downloads the OCR models and pulls the Ollama models.
set -euo pipefail
cd "$(dirname "$0")"
say() { printf '\033[36m[docextract]\033[0m %s\n' "$1"; }

# 1. Virtual environment (prefers Python 3.12)
if [ ! -x .venv/bin/python ]; then
  PY=$(command -v python3.12 || command -v python3.11 || command -v python3)
  say "Creating virtual environment with $PY"
  "$PY" -m venv .venv
fi
PYTHON=.venv/bin/python
"$PYTHON" -m pip install --upgrade pip --quiet

# 2. PyTorch: CUDA build on Linux with an NVIDIA GPU, default build otherwise (Apple Silicon included)
if command -v nvidia-smi >/dev/null 2>&1; then
  say "NVIDIA GPU detected, installing PyTorch with CUDA"
  "$PYTHON" -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128 --quiet
else
  say "Installing PyTorch"
  "$PYTHON" -m pip install torch torchvision --quiet
fi

# 3. Everything else
say "Installing dependencies"
"$PYTHON" -m pip install -r requirements-dev.txt --quiet

# 4. Settings
if [ ! -f .env ]; then
  cp .env.example .env
  say "Created .env (defaults to local Ollama models)"
fi

# 5. Download the OCR models now, so the first run is quick
say "Downloading OCR models (docTR)"
"$PYTHON" -c "from docextract.config import get_settings as g; from docextract.ocr import OCR; s = g(); OCR(s.ocr_det_arch, s.ocr_reco_arch)"

# 6. Language models (optional: any OpenAI-compatible API also works)
if command -v ollama >/dev/null 2>&1; then
  for KEY in LLM_MODEL VLM_MODEL; do
    MODEL=$(grep -E "^\s*$KEY=" .env | head -1 | cut -d= -f2- | awk '{print $1}')
    if [ -n "$MODEL" ]; then say "Pulling $MODEL with Ollama"; ollama pull "$MODEL"; fi
  done
else
  say "Ollama not found. Install it from https://ollama.com, then run:"
  say "  ollama pull qwen2.5:3b && ollama pull qwen2.5vl:3b   (or point .env at OpenAI instead)"
fi

say "Setup complete. Start the app with: bash run.sh"
