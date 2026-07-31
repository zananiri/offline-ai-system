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
#
# Tuning env vars, set for the Ollama process specifically (this app runs on
# a 32GB-RAM machine with no usable discrete GPU -- Ollama's own log drops
# the Intel Iris Xe iGPU by default: "dropping integrated GPU; to enable,
# set OLLAMA_IGPU_ENABLE=1"):
#
#   OLLAMA_IGPU_ENABLE=1      -- lets Ollama offload some layers to the Iris
#                                 Xe iGPU instead of pure CPU. Iris Xe shares
#                                 system RAM and is not fast, so this is a
#                                 modest win, not a transformative one --
#                                 worth having on regardless.
#   OLLAMA_MAX_LOADED_MODELS=1 -- was 0 (=unlimited). With multiple tabs in
#                                 this app using different models (MADLAD-400
#                                 translation, gpt-oss:20b, qwen2.5, DictaLM,
#                                 nomic-embed-text), leaving this unlimited
#                                 lets them all stay resident in RAM at once,
#                                 which is what pushed a 24B model's load
#                                 into a machine that was already ~85% RAM
#                                 used at idle -- forcing pagefile-swapped
#                                 (very slow) inference instead of a real
#                                 compute bottleneck. Capping at 1 forces
#                                 Ollama to unload the previous model before
#                                 loading the next one requested.
#   OLLAMA_KEEP_ALIVE=2m       -- was the 5m default. Unloads an idle model
#                                 sooner, freeing RAM faster between tab
#                                 switches (e.g. Translate -> Legal).
#   OLLAMA_KV_CACHE_TYPE=q8_0 -- quantizes the KV cache (was full-precision
#                                 by default). Roughly halves KV-cache memory
#                                 with minimal quality loss -- directly
#                                 reduces the RAM pressure causing the swap
#                                 above, and is what makes ui.py's
#                                 _LEGAL_NUM_CTX safe to keep as high as it is.
$env:OLLAMA_IGPU_ENABLE = "1"
$env:OLLAMA_MAX_LOADED_MODELS = "1"
$env:OLLAMA_KEEP_ALIVE = "2m"
$env:OLLAMA_KV_CACHE_TYPE = "q8_0"

$ollamaRunning = $false
try {
    Invoke-WebRequest -Uri "http://localhost:11434" -UseBasicParsing -TimeoutSec 2 | Out-Null
    $ollamaRunning = $true
} catch {}

if ($ollamaRunning) {
    Write-Host "Ollama already running, skipping (NOTE: it was started without" -ForegroundColor Yellow
    Write-Host "this script's tuning env vars -- restart it via this script for" -ForegroundColor Yellow
    Write-Host "the iGPU/RAM optimizations below to take effect)." -ForegroundColor Yellow
} else {
    Write-Host "Starting Ollama (iGPU offload + RAM tuning enabled)..." -ForegroundColor Yellow
    Start-Process powershell -ArgumentList "-NoExit", "-Command", `
        "`$env:OLLAMA_IGPU_ENABLE='1'; `$env:OLLAMA_MAX_LOADED_MODELS='1'; `$env:OLLAMA_KEEP_ALIVE='2m'; `$env:OLLAMA_KV_CACHE_TYPE='q8_0'; ollama serve"
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
