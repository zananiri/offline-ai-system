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
# 1. Python 3.11 (Docling/ctranslate2 wheels are built and tested against 3.11)
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
Write-Host "Starting Ollama service..." -ForegroundColor Yellow
Start-Process -NoNewWindow ollama serve
Start-Sleep -Seconds 5

Write-Host "Pulling qwen2.5:7b-instruct-q4_K_M (~4.7GB)..." -ForegroundColor Yellow
ollama pull qwen2.5:7b-instruct-q4_K_M

Write-Host "Pulling hf.co/dicta-il/DictaLM-3.0-24B-Thinking-GGUF:Q4_K_M (Legal tab model, several GB)..." -ForegroundColor Yellow
ollama pull hf.co/dicta-il/DictaLM-3.0-24B-Thinking-GGUF:Q4_K_M

# ---------------------------------------------------------------------------
# 5. Download + convert MADLAD-400 to CTranslate2 format (one-time, needs internet)
# ---------------------------------------------------------------------------
Write-Host "Converting MADLAD-400-3B to CTranslate2 format..." -ForegroundColor Yellow
Write-Host "(this downloads several GB from Hugging Face once, then quantizes to int8)" -ForegroundColor DarkYellow
python scripts\convert_translation_model.py

Write-Host ""
Write-Host "=== Setup complete ===" -ForegroundColor Cyan
Write-Host "Everything below this point runs with NO internet connection required."
