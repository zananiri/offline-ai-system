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
winget install --id UB-Mannheim.TesseractOCR -e --source winget
Write-Host "IMPORTANT: the default OCR engine (RapidOCR) has no Hebrew support at all -" -ForegroundColor Red
Write-Host "  Tesseract is the ONLY engine here that can read Hebrew, so its Hebrew" -ForegroundColor Red
Write-Host "  language pack is REQUIRED, not optional, if you'll process Hebrew documents." -ForegroundColor Red
Write-Host "  Re-run the Tesseract installer once more and, on the language selection" -ForegroundColor DarkYellow
Write-Host "  page, tick 'Hebrew' (and any of eng/ara/chi_sim/rus/fra/deu/spa you want" -ForegroundColor DarkYellow
Write-Host "  as extra fallback coverage)." -ForegroundColor DarkYellow
Write-Host "  Also confirm tesseract.exe was added to PATH (winget usually handles this" -ForegroundColor DarkYellow
Write-Host "  automatically) - a new PowerShell window may be needed to pick it up." -ForegroundColor DarkYellow

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
# 4. Pull the Ollama model (needs internet once; fully offline after this)
# ---------------------------------------------------------------------------
Write-Host "Starting Ollama service..." -ForegroundColor Yellow
Start-Process -NoNewWindow ollama serve
Start-Sleep -Seconds 5

Write-Host "Pulling qwen2.5:7b-instruct-q4_K_M (~4.7GB)..." -ForegroundColor Yellow
ollama pull qwen2.5:7b-instruct-q4_K_M

# ---------------------------------------------------------------------------
# 5. Download + convert MADLAD-400 to CTranslate2 format (one-time, needs internet)
# ---------------------------------------------------------------------------
Write-Host "Converting MADLAD-400-3B to CTranslate2 format..." -ForegroundColor Yellow
Write-Host "(this downloads several GB from Hugging Face once, then quantizes to int8)" -ForegroundColor DarkYellow
python scripts\convert_translation_model.py

Write-Host ""
Write-Host "=== Setup complete ===" -ForegroundColor Cyan
Write-Host "Everything below this point runs with NO internet connection required."
