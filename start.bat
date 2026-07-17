@echo off
REM Double-click this file to start Ollama, the FastAPI backend, and the Gradio UI.
REM It just calls run.ps1 with the execution policy bypassed for this run only
REM (doesn't change your system-wide PowerShell policy).

cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0run.ps1"

echo.
echo If three windows didn't open above, scroll up for the error.
pause
