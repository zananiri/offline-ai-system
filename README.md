# Offline Translator + Document OCR System

Windows / 32GB RAM / 16GB shared-memory GPU. Ollama for orchestration,
NLLB-200 (CTranslate2) for translation, Docling for document conversion + OCR.

## Folder layout
```
offline-ai-system/
├── requirements.txt          # pinned Python deps
├── setup.ps1                 # one-time installer (run once, needs internet)
├── scripts/
│   └── convert_nllb_model.py # downloads + quantizes NLLB-200 (run by setup.ps1)
├── models/
│   └── nllb-ct2/             # created by setup.ps1, ~1.3GB, fully offline after
└── app/
    ├── document.py           # Docling: convert + OCR -> markdown
    ├── translate.py          # NLLB-200 via CTranslate2
    ├── main.py                # FastAPI orchestrator
    └── ui.py                  # Gradio front end
```

## One-time setup
```powershell
cd offline-ai-system
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\setup.ps1
```
This installs Python 3.11, Ollama, pandoc, Tesseract (OCR fallback), creates a
venv, installs pinned packages, pulls the Ollama model, and downloads +
quantizes NLLB-200 to CTranslate2 format. Needs internet. Takes a while
(~10GB total download between the two models). Run once.

After this finishes, check `requirements.lock.txt` — that's the exact,
reproducible set of package versions actually installed on your machine.
Keep it alongside requirements.txt; if you ever rebuild the environment,
`pip install -r requirements.lock.txt` reproduces it exactly.

## Run sequence (every time after setup)

1. **Activate the environment**
   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```

2. **Start Ollama** (skip if it's already running as a Windows service —
   the installer usually sets this up automatically; check the system tray)
   ```powershell
   ollama serve
   ```

3. **Sanity-check the model is present**
   ```powershell
   ollama list
   # should show qwen2.5:7b-instruct-q4_K_M
   ```

4. **Start the FastAPI backend** (new terminal, venv activated)
   ```powershell
   uvicorn app.main:app --host 0.0.0.0 --port 8000
   ```
   Visit `http://localhost:8000/docs` to confirm it's up and test endpoints directly.

5. **Start the Gradio UI** (new terminal, venv activated)
   ```powershell
   python app/ui.py
   ```
   Visit `http://localhost:7860`.

6. **Use it**: upload a PDF/DOCX/PPTX/image, pick source/target language from
   your 7, optionally check "summarize", hit Translate.

From this point, no network access is needed at all — everything runs
locally against the models already on disk.

## Notes / things worth knowing

- **The 7 languages** are set in `app/translate.py` as a `LANGUAGES` dict
  mapping friendly names to FLORES-200 codes. Edit that dict to match your
  actual 7 — NLLB-200 covers 200 languages so almost anything you pick will
  already be supported; you're just choosing which 7 show up in the UI.
- **Ollama's role is intentionally limited to non-translation work**
  (summarizing, cleaning up OCR'd structure, answering questions about a
  document). Don't be tempted to also ask it to translate — NLLB will
  outperform it, especially on your less resource-rich language pairs.
- **If OCR quality on scanned documents is poor**, Docling will fall back
  to worse results on very noisy scans. Tesseract (installed by setup.ps1)
  is there as a manual fallback — you can swap `RapidOcrOptions()` for
  `TesseractOcrOptions()` in `app/document.py` and compare.
- **16GB shared GPU memory on Windows** usually means Ollama runs on
  CPU unless you have a discrete NVIDIA/AMD GPU Ollama recognizes — check
  `ollama ps` while a model is loaded; it'll show whether it's using GPU
  layers. 7B q4 models are still quite usable on CPU with 32GB RAM.
- **First run of `translate.py`** downloads the NLLB tokenizer config
  (small, a few MB) from Hugging Face and caches it — this happens once,
  triggered from `convert_nllb_model.py` during setup, not at request time.
