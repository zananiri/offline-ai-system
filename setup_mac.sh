#!/usr/bin/env bash
# setup_mac.sh — one-time setup for the offline translator / document-OCR
# system on macOS (Intel or Apple Silicon).
#
# Run from the project root:
#     chmod +x setup_mac.sh
#     ./setup_mac.sh
#
# Mirrors setup.ps1's steps. Differences from Windows are called out inline.

set -euo pipefail
echo -e "\033[36m=== Offline AI System Setup (macOS) ===\033[0m"

# ---------------------------------------------------------------------------
# 0. Sanity checks
# ---------------------------------------------------------------------------
if ! command -v brew >/dev/null 2>&1; then
    echo "Homebrew not found. Install it first: https://brew.sh"
    exit 1
fi

# ---------------------------------------------------------------------------
# 1. Python 3.11 (Docling/ctranslate2 wheels are built and tested against 3.11)
# ---------------------------------------------------------------------------
if ! command -v python3.11 >/dev/null 2>&1; then
    echo -e "\033[33mInstalling Python 3.11...\033[0m"
    brew install python@3.11
fi

# tkinter is NOT bundled with Homebrew's python@3.11 (unlike the python.org
# Windows installer, which bundles it). gui_run.py needs it, so install
# separately even if you only plan to use run_mac.sh.
echo -e "\033[33mInstalling Tk (needed if you ever run gui_run.py)...\033[0m"
brew install python-tk@3.11 || true

# ---------------------------------------------------------------------------
# 2. System tools: Ollama, pandoc, Tesseract (+ Hebrew language pack)
# ---------------------------------------------------------------------------
echo -e "\033[33mInstalling Ollama...\033[0m"
brew install ollama

echo -e "\033[33mInstalling pandoc (for markdown -> docx/pdf conversion)...\033[0m"
brew install pandoc

echo -e "\033[33mInstalling Tesseract OCR...\033[0m"
brew install tesseract

# Homebrew's plain "tesseract" formula only ships eng.traineddata. Unlike
# setup.ps1 (which curls heb.traineddata into place by hand because winget's
# Tesseract installer has no language-selection step at all), macOS has a
# proper all-languages formula, so use that instead of a manual download.
# RapidOCR (the default engine everywhere else in this app) has no Hebrew
# support at all, so this is REQUIRED, not optional, for the Hebrew OCR
# path in document.py to work.
echo -e "\033[33mInstalling Tesseract language packs (includes Hebrew)...\033[0m"
brew install tesseract-lang

echo -e "\033[32mTesseract + Hebrew language data ready.\033[0m"

# ---------------------------------------------------------------------------
# 3. Python virtual environment + pinned dependencies
# ---------------------------------------------------------------------------
echo -e "\033[33mCreating virtual environment...\033[0m"
python3.11 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
# Uses requirements.txt (the floor-pinned file), NOT requirements.lock.txt --
# that lock file was frozen on Windows and includes Windows-only packages
# (pywin32) plus exact wheel builds that don't exist for macOS. Installing
# from requirements.txt lets pip resolve mac-appropriate versions itself.
pip install -r requirements.txt

# Freeze the EXACT resolved versions for THIS machine -- same idea as
# setup.ps1, but kept in its own file so it never overwrites (or gets
# confused with) the Windows lock file if both are ever in the same repo.
pip freeze > requirements.lock-mac.txt
echo -e "\033[32mExact resolved versions written to requirements.lock-mac.txt\033[0m"

# ---------------------------------------------------------------------------
# 4. Pull the Ollama models (needs internet once; fully offline after this)
# ---------------------------------------------------------------------------
echo -e "\033[33mStarting Ollama service...\033[0m"
brew services start ollama
sleep 5

echo -e "\033[33mPulling qwen2.5:7b-instruct-q4_K_M (~4.7GB)...\033[0m"
ollama pull qwen2.5:7b-instruct-q4_K_M

# NOTE: this pulls the 24B model to match what app/main.py and app/ui.py
# actually reference (LEGAL_MODEL). setup.ps1 currently pulls the 1.7B
# variant instead, which no longer matches the code -- see the note in the
# chat for that discrepancy. If your Mac has less than ~24GB RAM, swap the
# line below for the smaller model and change LEGAL_MODEL in both
# app/main.py and app/ui.py to match:
#   ollama pull hf.co/dicta-il/DictaLM-3.0-1.7B-Thinking-GGUF:Q4_K_M
echo -e "\033[33mPulling hf.co/dicta-il/DictaLM-3.0-24B-Thinking-GGUF:Q4_K_M (Legal tab model, ~14.3GB)...\033[0m"
ollama pull hf.co/dicta-il/DictaLM-3.0-24B-Thinking-GGUF:Q4_K_M

# ---------------------------------------------------------------------------
# 5. Download + convert MADLAD-400 to CTranslate2 format (one-time, needs internet)
# ---------------------------------------------------------------------------
echo -e "\033[33mConverting MADLAD-400-3B to CTranslate2 format...\033[0m"
echo "(this downloads several GB from Hugging Face once, then quantizes to int8)"
python scripts/convert_translation_model.py

echo ""
echo -e "\033[36m=== Setup complete ===\033[0m"
echo "Everything below this point runs with NO internet connection required."
echo "Run ./run_mac.sh to start all three services."
