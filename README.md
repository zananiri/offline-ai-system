# Offline Translator + Document OCR System

Windows / 32GB RAM / 16GB shared-memory GPU. Ollama for orchestration,
MADLAD-400 (CTranslate2) for translation, Docling for document conversion + OCR.

All components are commercially licensed (Apache 2.0 / MIT) — safe to use in
a product you sell. This project originally used NLLB-200 for translation,
but NLLB-200 is CC-BY-NC 4.0 (non-commercial only, per Meta's model card) and
was replaced with MADLAD-400 (Google Research, Apache 2.0) for that reason.

## OCR: two engines, deliberately

- **RapidOCR (default)** — used for every document unless told otherwise.
  Better real-world accuracy than Tesseract: higher precision, built-in
  table/layout detection (useful for invoices), better handling of skewed
  or photographed documents.
- **Tesseract (Hebrew only)** — RapidOCR (and EasyOCR) have no Hebrew model
  at all; this is a known, unaddressed gap in all major open-source OCR
  toolkits. Tesseract is the only engine here that can read Hebrew script,
  so it's used *only* when a document is explicitly marked Hebrew via the
  checkbox in each tab — this avoids trading away RapidOCR's better
  accuracy on the other 7 languages for documents that don't need it.

## Folder layout
```
offline-ai-system/
├── requirements.txt             # pinned Python deps
├── setup.ps1                    # one-time installer (run once, needs internet)
├── run.ps1 / start.bat          # start all services after setup is done
├── scripts/
│   └── convert_translation_model.py  # downloads + quantizes MADLAD-400 (run by setup.ps1)
├── models/
│   └── madlad400-ct2/           # created by setup.ps1, fully offline after
└── app/
    ├── document.py               # Docling: convert + OCR (RapidOCR default, Tesseract for Hebrew) -> markdown, docx export
    ├── translate.py               # MADLAD-400 via CTranslate2
    ├── main.py                    # FastAPI orchestrator
    └── ui.py                      # Gradio front end (Translate / Convert / Chat / Accountant)
```

## One-time setup
```powershell
cd offline-ai-system
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\setup.ps1
```
This installs Python 3.11, Ollama, pandoc, Tesseract, creates a venv,
installs pinned packages, pulls the Ollama model, and downloads + quantizes
MADLAD-400 to CTranslate2 format. Needs internet.

**If you'll process any Hebrew documents**, re-run the Tesseract installer
after setup finishes and tick "Hebrew" on the language selection page — this
is required, not optional, since Tesseract is the only OCR engine here that
supports Hebrew script. See the warning setup.ps1 prints for details.

MADLAD-400-3B is a larger model than the NLLB-1.3B this project used before,
so the download and conversion step will take longer — there's no smaller
official MADLAD-400 checkpoint, and that size is the tradeoff for a
commercially licensed model.

After this finishes, check `requirements.lock.txt` for the exact,
reproducible set of package versions actually installed on your machine.

## Run sequence (every time after setup)

Easiest: double-click `start.bat`, or run `.\run.ps1` — both start Ollama,
the FastAPI backend, and the Gradio UI together.

Manual sequence if you prefer separate terminals:

1. **Activate the environment**
   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```
2. **Start Ollama** (skip if already running as a background service)
   ```powershell
   ollama serve
   ```
3. **Start the FastAPI backend** (new terminal, venv activated)
   ```powershell
   uvicorn app.main:app --host 0.0.0.0 --port 8000
   ```
   Visit `http://localhost:8000/docs` to confirm it's up.
4. **Start the Gradio UI** (new terminal, venv activated)
   ```powershell
   python -m app.ui
   ```
   Visit `http://localhost:7860`.

From this point, no network access is needed at all.

## Notes / things worth knowing

- **The 8 languages** (7 + Hebrew) are set in `app/translate.py` as a
  `LANGUAGES` dict mapping friendly names to MADLAD-400's language codes
  (mostly plain ISO 639-1, e.g. `en`, `fr`, `ar`, `zh`, `he`). MADLAD-400
  covers 400+ languages, so almost anything you pick will already be
  supported — edit the dict to match your actual needs.
- **MADLAD-400 only needs a target-language tag** — it infers the source
  language automatically. The source-language dropdown in the UI is kept
  for clarity but isn't strictly required by the model itself.
- **Every tab that touches OCR has a "Document is in Hebrew" checkbox** —
  Translate, Convert to Word, Chat (as an extra option below the chat box),
  and Accountant. Leave it unchecked for everything else; it routes that
  specific request through Tesseract instead of the default RapidOCR engine.
- **The Accountant tab's Hebrew checkbox applies to the whole batch** — if a
  single ZIP mixes Hebrew and non-Hebrew invoices, process them in two
  separate batches for best accuracy on each.
- **Ollama's role is intentionally limited to non-translation work**
  (summarizing, cleaning up OCR'd structure, answering questions about a
  document, classifying invoices). Don't ask it to also translate — a
  dedicated MT model outperforms a general-purpose LLM at translation.
- **First run of `translate.py`** downloads MADLAD-400's tokenizer config
  (small) from Hugging Face and caches it — this happens once, triggered
  from `convert_translation_model.py` during setup, not at request time.
