@echo off
REM One-step launcher (Windows): creates .venv on first run, installs dependencies, starts the app.
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (
  echo First run: creating .venv and installing dependencies...
  python -m venv .venv
  .venv\Scripts\python -m pip install --no-cache-dir --upgrade pip
  .venv\Scripts\pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
  .venv\Scripts\pip install --no-cache-dir "transformers>=4.56" "peft>=0.13" accelerate numpy "gradio>=5.0"
)
echo Starting TrialMatch (first start downloads about 4 GB of model weights)...
set HOST=127.0.0.1
set TRIALMATCH_INBROWSER=1
.venv\Scripts\python app\app.py
