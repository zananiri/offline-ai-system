#!/usr/bin/env bash
#
# setup_mac.sh — one-time setup for the offline translator / document-OCR
# system on macOS. Port of setup.ps1 (Windows).
#
# Run from Terminal, from the project folder:
#   chmod +x setup_mac.sh
#   ./setup_mac.sh
#
set -euo pipefail

CYAN='\033[0;36m'; YELLOW='\033[0;33m'; GREEN='\033[0;32m'; RED='\033[0;31m'; NC='\033[0m'
info()  { echo -e "${YELLOW}$1${NC}"; }
ok()    { echo -e "${GREEN}$1${NC}"; }
head()  { echo -e "${CYAN}$1${NC}"; }
die()   { echo -e "${RED}$1${NC}" >&2; exit 1; }

head "=== Offline AI System Setup (macOS) ==="

# ---------------------------------------------------------------------------
# 0. Sanity checks
# ---------------------------------------------------------------------------
if ! command -v brew >/dev/null 2>&1; then
    die "Homebrew not found. Install it first from https://brew.sh, then re-run this script:
  /bin/bash -c \"\$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\""
fi

ARCH="$(uname -m)"
if [ "$ARCH" = "arm64" ]; then
    info "Apple Silicon (M-series) detected — Ollama will use Metal/unified memory automatically."
else
    info "Intel Mac detected — same caveat as Intel-integrated-graphics Windows machines: Ollama"
    info "has no GPU acceleration path here, so everything runs on CPU. The 24B legal model (step 4"
    info "below) will be slow on this hardware; see the note there for the lighter fallback."
fi

TOTAL_RAM_GB=$(( $(sysctl -n hw.memsize) / 1024 / 1024 / 1024 ))
info "Total system RAM: ${TOTAL_RAM_GB}GB (setup.ps1's original target was 32GB)."

# ---------------------------------------------------------------------------
# 1. Python 3.11 (Docling/ctranslate2 wheels are built and tested against 3.11)
# ---------------------------------------------------------------------------
if ! brew list python@3.11 >/dev/null 2>&1; then
    info "Installing Python 3.11..."
    brew install python@3.11
else
    ok "Python 3.11 already installed."
fi
PY311="$(brew --prefix python@3.11)/bin/python3.11"

# ---------------------------------------------------------------------------
# 2. System tools: Ollama, pandoc, Tesseract (Hebrew OCR only)
# ---------------------------------------------------------------------------
info "Installing Ollama..."
brew list ollama >/dev/null 2>&1 || brew install ollama

info "Installing pandoc (for markdown -> docx/pdf conversion)..."
brew list pandoc >/dev/null 2>&1 || brew install pandoc

info "Installing Tesseract OCR..."
brew list tesseract >/dev/null 2>&1 || brew install tesseract

# The base 'tesseract' formula only bundles English -- same situation as the
# Windows winget install. RapidOCR (the default engine everywhere else in
# this app) has no Hebrew support at all, so this is REQUIRED, not optional,
# for the Hebrew OCR path in document.py to work.
TESSDATA_DIR="$(brew --prefix tesseract)/share/tessdata"
HEB_PATH="$TESSDATA_DIR/heb.traineddata"
if [ -d "$TESSDATA_DIR" ]; then
    if [ -f "$HEB_PATH" ]; then
        ok "Hebrew language data already present."
    else
        info "Downloading Hebrew language data for Tesseract..."
        curl -fsSL "https://github.com/tesseract-ocr/tessdata/raw/main/heb.traineddata" -o "$HEB_PATH"
    fi
else
    echo -e "${RED}WARNING: expected tessdata folder not found at $TESSDATA_DIR -${NC}"
    echo -e "${RED}  Tesseract may have installed to a different prefix. Download${NC}"
    echo -e "${RED}  https://github.com/tesseract-ocr/tessdata/raw/main/heb.traineddata${NC}"
    echo -e "${RED}  into its tessdata folder manually.${NC}"
fi
ok "Tesseract + Hebrew language data ready."

# ---------------------------------------------------------------------------
# 3. Python virtual environment + pinned dependencies
# ---------------------------------------------------------------------------
info "Creating virtual environment..."
"$PY311" -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate

python -m pip install --upgrade pip
pip install -r requirements.txt

# Freeze the EXACT resolved versions -- this is your real compatibility record
pip freeze > requirements.lock.txt
ok "Exact resolved versions written to requirements.lock.txt"

# ---------------------------------------------------------------------------
# 4. Pull the Ollama models (needs internet once; fully offline after this)
# ---------------------------------------------------------------------------
info "Starting Ollama service..."
# 'brew services start ollama' also works and survives reboots/relogin; a
# background 'ollama serve' here matches the Windows script's one-shot style.
if ! curl -s http://localhost:11434 >/dev/null 2>&1; then
    ollama serve >/tmp/ollama_serve.log 2>&1 &
    sleep 5
else
    ok "Ollama already running."
fi

info "Pulling gpt-oss:20b (~13GB)..."
ollama pull gpt-oss:20b

# Backs ONLY the Translate tab's optional summarizer step (app/main.py's
# TRANSLATE_SUMMARY_MODEL / app/ui.py's TRANSLATE_SUMMARY_MODEL) -- kept as
# a separate, smaller pull since qwen2.5's multilingual output is more
# reliable than gpt-oss:20b's for non-English summaries.
info "Pulling qwen2.5:7b-instruct-q4_K_M (~4.7GB)..."
ollama pull qwen2.5:7b-instruct-q4_K_M

# Matches the LEGAL_MODEL constant currently in app/main.py and app/ui.py --
# Dicta's 24B flagship, Q4_K_M quant, ~14.3GB download. On an Intel Mac (no
# GPU offload) or any machine under ~32GB RAM, this will be slow -- swap to
# the smaller line below (and update LEGAL_MODEL in both main.py and ui.py
# to match) if that's the case here:
#   ollama pull hf.co/dicta-il/DictaLM-3.0-1.7B-Thinking-GGUF:Q4_K_M
info "Pulling hf.co/dicta-il/DictaLM-3.0-24B-Thinking-GGUF:Q4_K_M (Legal tab model, ~14.3GB)..."
ollama pull hf.co/dicta-il/DictaLM-3.0-24B-Thinking-GGUF:Q4_K_M

# ---------------------------------------------------------------------------
# 5. Download + convert MADLAD-400 to CTranslate2 format (one-time, needs internet)
# ---------------------------------------------------------------------------
info "Converting MADLAD-400-3B to CTranslate2 format..."
echo "(this downloads several GB from Hugging Face once, then quantizes to int8)"
python scripts/convert_translation_model.py

# ---------------------------------------------------------------------------
# 6. Build the Canon AI vector store (needs internet once, for chromadb + scraping)
# ---------------------------------------------------------------------------
info "Installing chromadb (not in requirements.txt's Ollama/torch pins -- kept"
info "separate since it's only used by the Canon AI tab)..."
pip install chromadb

info "Scraping CIC IT source content..."
python scripts/scrape_cic_it.py

info "Embedding into ChromaDB (app/chroma_db, collection 'cic_it')..."
python scripts/embed_to_chroma.py

echo ""
head "=== Setup complete ==="
echo "Everything below this point runs with NO internet connection required."
echo "To start the app later:"
echo "  source .venv/bin/activate"
echo "  ollama serve &"
echo "  uvicorn app.main:app --host 0.0.0.0 --port 8000 &"
echo "  python -m app.ui"