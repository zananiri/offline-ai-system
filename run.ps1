<#
  run.ps1 — starts Ollama, the FastAPI backend, and the Gradio UI,
  each in its own window. Run this from the project root after setup.ps1
  has already been completed once.

  Usage:
      .\run.ps1
#>

$ErrorActionPreference = "Stop"
$projectRoot = $PSScriptRoot

Write-Host "Starting offline AI system..." -ForegroundColor Cyan

# ---------------------------------------------------------------------------
# 1. Ollama — skip if already running as a background service
# ---------------------------------------------------------------------------
$ollamaRunning = $false
try {
    Invoke-WebRequest -Uri "http://localhost:11434" -UseBasicParsing -TimeoutSec 2 | Out-Null
    $ollamaRunning = $true
} catch {}

if ($ollamaRunning) {
    Write-Host "Ollama already running, skipping." -ForegroundColor Green
} else {
    Write-Host "Starting Ollama..." -ForegroundColor Yellow
    Start-Process powershell -ArgumentList "-NoExit", "-Command", "ollama serve"
    Start-Sleep -Seconds 5
}

# ---------------------------------------------------------------------------
# 2. FastAPI backend
# ---------------------------------------------------------------------------
Write-Host "Starting FastAPI backend on :8000..." -ForegroundColor Yellow
Start-Process powershell -ArgumentList "-NoExit", "-Command", `
    "cd '$projectRoot'; .\.venv\Scripts\Activate.ps1; uvicorn app.main:app --host 0.0.0.0 --port 8000"

Start-Sleep -Seconds 3

# ---------------------------------------------------------------------------
# 3. Gradio UI
# ---------------------------------------------------------------------------
Write-Host "Starting Gradio UI on :7860..." -ForegroundColor Yellow
Start-Process powershell -ArgumentList "-NoExit", "-Command", `
    "cd '$projectRoot'; .\.venv\Scripts\Activate.ps1; python -m app.ui"

Write-Host ""
Write-Host "=== All services starting in separate windows ===" -ForegroundColor Cyan
Write-Host "FastAPI docs: http://localhost:8000/docs"
Write-Host "Gradio UI:    http://localhost:7860"
Write-Host "Give it 10-20 seconds for both to finish loading their models."
