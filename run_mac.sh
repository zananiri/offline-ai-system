#!/usr/bin/env bash
# run_mac.sh — starts Ollama, the FastAPI backend, and the Gradio UI.
# Run this from the project root after setup_mac.sh has completed once.
#
# Usage:
#     chmod +x run_mac.sh
#     ./run_mac.sh
#
# Mirrors run.ps1. Instead of three separate PowerShell windows, this opens
# three tabs in Terminal.app (each stays open so you can watch its logs,
# same as run.ps1's separate windows) via osascript.

set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo -e "\033[36mStarting offline AI system...\033[0m"

# ---------------------------------------------------------------------------
# 1. Ollama — skip if already running
# ---------------------------------------------------------------------------
if curl -s -o /dev/null -m 2 "http://localhost:11434"; then
    echo -e "\033[32mOllama already running, skipping.\033[0m"
else
    echo -e "\033[33mStarting Ollama...\033[0m"
    osascript -e "tell application \"Terminal\" to do script \"ollama serve\""
    sleep 5
fi

# ---------------------------------------------------------------------------
# 2. FastAPI backend
# ---------------------------------------------------------------------------
echo -e "\033[33mStarting FastAPI backend on :8000...\033[0m"
osascript -e "tell application \"Terminal\" to do script \"cd '$PROJECT_ROOT'; source .venv/bin/activate; uvicorn app.main:app --host 0.0.0.0 --port 8000\""

sleep 3

# ---------------------------------------------------------------------------
# 3. Gradio UI
# ---------------------------------------------------------------------------
echo -e "\033[33mStarting Gradio UI on :7860...\033[0m"
osascript -e "tell application \"Terminal\" to do script \"cd '$PROJECT_ROOT'; source .venv/bin/activate; python -m app.ui\""

echo ""
echo -e "\033[36m=== All services starting in separate Terminal tabs ===\033[0m"
echo "FastAPI docs: http://localhost:8000/docs"
echo "Gradio UI:    http://localhost:7860"
echo "Give it 10-20 seconds for both to finish loading their models."
