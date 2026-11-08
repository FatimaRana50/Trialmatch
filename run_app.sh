#!/usr/bin/env bash
# One-step launcher (Linux/macOS): on first run creates .venv and installs dependencies,
# then starts the TrialMatch app and opens it in the browser at http://127.0.0.1:7860
set -e
cd "$(dirname "$0")"
if [ ! -x .venv/bin/python ]; then
  echo "First run: creating .venv and installing dependencies (a few minutes)..."
  python3 -m venv .venv
  .venv/bin/pip install --no-cache-dir --upgrade pip
  if command -v nvidia-smi >/dev/null 2>&1; then
    .venv/bin/pip install --no-cache-dir torch
  else
    .venv/bin/pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
  fi
  .venv/bin/pip install --no-cache-dir "transformers>=4.56" "peft>=0.13" accelerate numpy "gradio>=5.0"
fi
echo "Starting TrialMatch (first start downloads ~4 GB of model weights)..."
HOST="${HOST:-127.0.0.1}" TRIALMATCH_INBROWSER=1 exec .venv/bin/python app/app.py
