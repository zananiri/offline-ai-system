@echo off
REM Double-click this to launch the GUI status window (no console window
REM stays open alongside it — uses pythonw instead of python).

cd /d "%~dp0"
".venv\Scripts\pythonw.exe" gui_run.py

if errorlevel 1 (
    echo Something went wrong starting the GUI. Try running it manually instead:
    echo   .venv\Scripts\Activate.ps1
    echo   python gui_run.py
    pause
)
