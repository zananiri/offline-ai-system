<#
  setup.ps1 — one-time setup for the offline translator / document-OCR system
  Target: Windows 10/11, 32GB RAM, 16GB shared-memory GPU
  Run from an elevated PowerShell prompt (Run as Administrator):
      Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
      .\setup.ps1
#>

$ErrorActionPreference = "Stop"
Write-Host "=== Offline AI System Setup ===" -ForegroundColor Cyan

# ---------------------------------------------------------------------------
# 0. Sanity checks
# ---------------------------------------------------------------------------
if (-not (Get-Command winget -ErrorAction SilentlyContinue)) {
    Write-Error "winget not found. Install 'App Installer' from the Microsoft Store first."
}

# ---------------------------------------------------------------------------
# 1. Python 3.11 (Docling wheels are built and tested against 3.11)
# ---------------------------------------------------------------------------
$pyOk = $false
try {
    $v = (py -3.11 --version) 2>$null
    if ($v -match "3\.11") { $pyOk = $true }
} catch {}

if (-not $pyOk) {
    Write-Host "Installing Python 3.11..." -ForegroundColor Yellow
    winget install --id Python.Python.3.11 -e --source winget
}

# ---------------------------------------------------------------------------
# 2. System tools: Ollama, pandoc, Tesseract (Hebrew OCR only)
# ---------------------------------------------------------------------------
Write-Host "Installing Ollama..." -ForegroundColor Yellow
winget install --id Ollama.Ollama -e --source winget

Write-Host "Installing pandoc (for markdown -> docx/pdf conversion)..." -ForegroundColor Yellow
winget install --id JohnMacFarlane.Pandoc -e --source winget

Write-Host "Installing Tesseract OCR..." -ForegroundColor Yellow
$tesseractDir = "C:\Program Files\Tesseract-OCR"
$tesseractExe = Join-Path $tesseractDir "tesseract.exe"
if (-not (Test-Path $tesseractExe)) {
    winget install --id UB-Mannheim.TesseractOCR -e --source winget
}

# The default winget install is silent and only bundles English -- it does
# NOT show the graphical installer's language-selection screen, so Hebrew
# is never included unless we fetch it separately. RapidOCR (the default
# engine everywhere else in this app) has no Hebrew support at all, so this
# is REQUIRED, not optional, for the Hebrew OCR path in document.py to work.
$tessdataDir = Join-Path $tesseractDir "tessdata"
$hebPath = Join-Path $tessdataDir "heb.traineddata"
if (Test-Path $tessdataDir) {
    if (Test-Path $hebPath) {
        Write-Host "Hebrew language data already present." -ForegroundColor Green
    } else {
        Write-Host "Downloading Hebrew language data for Tesseract..." -ForegroundColor Yellow
        Invoke-WebRequest `
            -Uri "https://github.com/tesseract-ocr/tessdata/raw/main/heb.traineddata" `
            -OutFile $hebPath
    }
} else {
    Write-Host "WARNING: expected tessdata folder not found at $tessdataDir -" -ForegroundColor Red
    Write-Host "  Tesseract may have installed to a different location. Download" -ForegroundColor Red
    Write-Host "  https://github.com/tesseract-ocr/tessdata/raw/main/heb.traineddata" -ForegroundColor Red
    Write-Host "  into its tessdata folder manually." -ForegroundColor Red
}

# Confirm PATH includes it (winget usually handles this automatically, but
# a currently-open terminal won't see the change until it's reopened).
if ($env:Path -notlike "*$tesseractDir*") {
    Write-Host "Adding Tesseract to PATH..." -ForegroundColor Yellow
    [Environment]::SetEnvironmentVariable(
        "Path",
        [Environment]::GetEnvironmentVariable("Path", "Machine") + ";$tesseractDir",
        "Machine"
    )
    $env:Path += ";$tesseractDir"
}
Write-Host "Tesseract + Hebrew language data ready." -ForegroundColor Green

# ---------------------------------------------------------------------------
# 3. Python virtual environment + pinned dependencies
# ---------------------------------------------------------------------------
Write-Host "Creating virtual environment..." -ForegroundColor Yellow
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
pip install -r requirements.txt

# Freeze the EXACT resolved versions — this is your real compatibility record
pip freeze > requirements.lock.txt
Write-Host "Exact resolved versions written to requirements.lock.txt" -ForegroundColor Green

# ---------------------------------------------------------------------------
# 4. Pull the Ollama models (needs internet once; fully offline after this)
# ---------------------------------------------------------------------------
Write-Host "Starting Ollama service (iGPU offload + RAM tuning enabled)..." -ForegroundColor Yellow
# See run.ps1's matching comment for why each of these is set -- this
# machine has no usable discrete GPU and sits close to its RAM ceiling with
# this app's larger models (esp. the Legal tab's 24B model), so these are
# set from first run onward, not just added later in run.ps1.
$env:OLLAMA_IGPU_ENABLE = "1"
$env:OLLAMA_MAX_LOADED_MODELS = "1"
$env:OLLAMA_KEEP_ALIVE = "2m"
$env:OLLAMA_KV_CACHE_TYPE = "q8_0"
Start-Process -NoNewWindow ollama serve
Start-Sleep -Seconds 5

Write-Host "Pulling gpt-oss:20b (~13GB)..." -ForegroundColor Yellow
ollama pull gpt-oss:20b

# qwen2.5:32b backs the Chat tab (chat, document rewrite, and the two-step
# LLM document-translation flow -- see app/ui.py's CHAT_MODEL/
# TRANSLATE_MODEL and translate_document_via_llm). This replaces the old
# dedicated Translate tab's offline MADLAD-400/CTranslate2 pipeline, which
# has been removed entirely -- translation is now just a chat request like
# any other ("translate this to French"), not a separate model/tab.
Write-Host "Pulling qwen2.5:32b (Chat tab: chat, document rewrite, and document translation)..." -ForegroundColor Yellow
ollama pull qwen2.5:32b

# Matches the LEGAL_MODEL constant in app/main.py and app/ui.py -- keep all
# three in sync if that ever changes. Dicta's 24B flagship, Q4_K_M quant,
# ~14.3GB download. Needs ~32GB RAM alongside this app's other models (see
# ui.py's LEGAL_MODEL comment); if that's tight, switch this line (and both
# LEGAL_MODEL constants) to the IQ4_XS quant (~12.8GB) or the smaller 1.7B
# line kept there for an easy revert.
Write-Host "Pulling hf.co/dicta-il/DictaLM-3.0-24B-Thinking-GGUF:Q4_K_M (Legal tab model, ~14.3GB)..." -ForegroundColor Yellow
ollama pull hf.co/dicta-il/DictaLM-3.0-24B-Thinking-GGUF:Q4_K_M

# Attorney tabs' pipeline (app/legal_pipeline_v2.py -- see that module's
# docstring for the full architecture: Dicta query understanding -> hybrid
# BGE-M3/BM25 retrieval -> Dicta answers directly from retrieved sources ->
# an independent Dicta verification pass + a deterministic citation-
# existence check). Every reasoning/drafting/verification step runs on
# DictaLM above -- there is no separate reasoning model to pull here, just:
#   bge-m3 -- embedding model for the hybrid vector+BM25 retrieval. MUST
#             match the model scripts/embed_local_law_pdfs_bgem3.py used to
#             build the 'israeli_legal_db' collection -- mixing embedding
#             models between indexing and querying silently breaks vector
#             search (see that script's module docstring).
Write-Host "Pulling bge-m3 (embedding model for the Attorney tabs' hybrid retrieval)..." -ForegroundColor Yellow
ollama pull bge-m3

# ---------------------------------------------------------------------------
# 5. Build the Canon AI vector store (needs internet once, for chromadb + scraping)
# ---------------------------------------------------------------------------
# rank_bm25 is installed alongside chromadb here (not in requirements.txt,
# same reasoning as chromadb's own comment below) -- it's the keyword-search
# half of the Attorney tabs' hybrid retrieval (app/legal_pipeline_v2.py),
# a pure-Python package with no extra system dependency, so there's no real
# cost to always installing it here rather than gating it behind a flag.
Write-Host "Installing chromadb + rank_bm25 (not in requirements.txt's Ollama/torch pins -- kept" -ForegroundColor Yellow
Write-Host "separate since they're only used by the RAG tabs)..." -ForegroundColor Yellow
pip install chromadb rank_bm25

Write-Host "Scraping CIC IT source content..." -ForegroundColor Yellow
python scripts\scrape_cic_it.py

Write-Host "Embedding into ChromaDB (app\chroma_db, collection 'cic_it')..." -ForegroundColor Yellow
python scripts\embed_to_chroma.py

# ---------------------------------------------------------------------------
# 6. Build the Attorney tabs' legal vector store (BGE-M3 + hybrid retrieval)
# ---------------------------------------------------------------------------
# Safe to run even with an empty/nonexistent uploads/ folder -- the script
# just creates it and prints a reminder to drop law PDFs in before re-running
# (same graceful-empty-run behavior as scripts/embed_local_law_pdfs.py). Not
# something this step can meaningfully verify succeeded beyond "it ran": the
# real corpus depends on what law PDFs you actually have -- see that script's
# own docstring for the uploads/ layout and optional sidecar-metadata format.
Write-Host "Building the Attorney tabs' legal vector store (collection 'israeli_legal_db')..." -ForegroundColor Yellow
Write-Host "(embeds any PDFs already in uploads\ with bge-m3 -- re-run this script anytime" -ForegroundColor DarkYellow
Write-Host " after adding more law PDFs to uploads\)" -ForegroundColor DarkYellow
python scripts\embed_local_law_pdfs_bgem3.py

Write-Host ""
Write-Host "=== Setup complete ===" -ForegroundColor Cyan
Write-Host "Everything below this point runs with NO internet connection required."