<#
  install_tesseract_hebrew.ps1 — one-off fix: installs Tesseract (if not
  already present) and downloads the Hebrew language pack directly, so no
  manual re-run of the graphical installer / checkbox-ticking is needed.

  Run as Administrator, then close and reopen your terminal.
#>

$ErrorActionPreference = "Stop"

$tesseractDir = "C:\Program Files\Tesseract-OCR"
$tessdataDir  = Join-Path $tesseractDir "tessdata"
$tesseractExe = Join-Path $tesseractDir "tesseract.exe"

# 1. Install Tesseract itself if it isn't already there
if (Test-Path $tesseractExe) {
    Write-Host "Tesseract already installed at $tesseractExe" -ForegroundColor Green
} else {
    Write-Host "Installing Tesseract OCR via winget..." -ForegroundColor Yellow
    winget install --id UB-Mannheim.TesseractOCR -e --source winget
}

if (-not (Test-Path $tesseractExe)) {
    Write-Error "tesseract.exe still not found at $tesseractExe after install. If you installed to a custom location, edit `$tesseractDir` at the top of this script and re-run."
}

# 2. Download the Hebrew language pack directly (skips the interactive
#    installer's language-selection screen entirely)
$hebPath = Join-Path $tessdataDir "heb.traineddata"
if (Test-Path $hebPath) {
    Write-Host "Hebrew language data already present." -ForegroundColor Green
} else {
    Write-Host "Downloading Hebrew language data..." -ForegroundColor Yellow
    Invoke-WebRequest `
        -Uri "https://github.com/tesseract-ocr/tessdata/raw/main/heb.traineddata" `
        -OutFile $hebPath
    Write-Host "Saved to $hebPath" -ForegroundColor Green
}

# 3. Make sure tesseract.exe is on PATH (winget usually does this, but
#    double-check and add it for both this session and future ones)
if ($env:Path -notlike "*$tesseractDir*") {
    Write-Host "Adding Tesseract to PATH..." -ForegroundColor Yellow
    [Environment]::SetEnvironmentVariable(
        "Path",
        [Environment]::GetEnvironmentVariable("Path", "Machine") + ";$tesseractDir",
        "Machine"
    )
    $env:Path += ";$tesseractDir"  # also apply to this current session
}

Write-Host ""
Write-Host "=== Done ===" -ForegroundColor Cyan
Write-Host "Verify with: tesseract --version"
Write-Host "Verify Hebrew is available with: tesseract --list-langs"
Write-Host "IMPORTANT: close this terminal and open a new one (and restart run.ps1 /"
Write-Host "gui_run.py) so the updated PATH is picked up by newly started processes."