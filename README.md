# Offline Translator + Document OCR System

Windows / 32GB RAM / 16GB shared-memory GPU. Ollama for orchestration and
LLM inference (chat, document rewrite, and document translation), Docling
for document conversion + OCR.

Document translation is handled by the Chat tab via two sequential
`qwen2.5:32b` calls per chunk (clean up OCR/PDF extraction artifacts, then
translate) rather than a dedicated machine-translation model — just ask the
Chat tab to translate an attached document into whatever language you want,
in plain language ("translate this to Spanish"). This project originally
used a dedicated offline MT engine (NLLB-200, then MADLAD-400 after
NLLB-200 turned out to be CC-BY-NC 4.0 and unusable commercially) for
translation; that whole pipeline (`app/translate.py`, the MADLAD-400
CTranslate2 model, the dedicated Translate tab) has since been removed in
favor of this simpler LLM-based approach.

All components are commercially licensed (Apache 2.0 / MIT) — safe to use in
a product you sell.

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
└── app/
    ├── document.py               # Docling: convert + OCR (RapidOCR default, Tesseract for Hebrew) -> markdown, docx export
    ├── legal_pipeline_v2.py       # DictaLM-only RAG pipeline backing the Attorney tabs
    ├── main.py                    # FastAPI orchestrator
    └── ui.py                      # Gradio front end (Chat, Convert to Word, Accountant,
                                    # Attorney 32B / Attorney 32B Instruct (Slower) / Attorney 1.7B (Fast),
                                    # Canon AI, GDPR AI, HIPAA AI)
```

## One-time setup
```powershell
cd offline-ai-system
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\setup.ps1
```
This installs Python 3.11, Ollama, pandoc, Tesseract, creates a venv,
installs pinned packages, and pulls the Ollama models (`gpt-oss:20b` for
invoice classification/PowerPoint outlines, `qwen2.5:32b` for the Chat tab's
chat/rewrite/translation, `bge-m3` for the Attorney tabs' retrieval, and
DictaLM for the Legal/Attorney tabs). Needs internet.

**If you'll process any Hebrew documents**, re-run the Tesseract installer
after setup finishes and tick "Hebrew" on the language selection page — this
is required, not optional, since Tesseract is the only OCR engine here that
supports Hebrew script. See the warning setup.ps1 prints for details.

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

- **Document translation is a Chat tab request, not a separate tab.** Attach
  a document and ask the Chat tab to "translate this to [language]" — no
  fixed language list, just say what you want in plain language and the
  model (`qwen2.5:32b`) figures out the target language from your request.
  Internally this runs two sequential LLM calls per chunk: Job 1 cleans up
  OCR/PDF-extraction line breaks and typos *without* translating, Job 2
  translates the cleaned text. See `translate_document_via_llm` and
  `CHAT_MODEL` / `TRANSLATE_MODEL` in `app/ui.py`.
- **Every tab that touches OCR has a "Document is in Hebrew" checkbox** —
  Convert to Word, Chat (as an extra option below the chat box), Accountant,
  and the Attorney tabs. Leave it unchecked for everything else; it routes
  that specific request through Tesseract instead of the default RapidOCR
  engine.
- **The Accountant tab's Hebrew checkbox applies to the whole batch** — if a
  single ZIP mixes Hebrew and non-Hebrew invoices, process them in two
  separate batches for best accuracy on each.
- **`qwen2.5:32b` backs the Chat tab specifically** (chat, document rewrite,
  and document translation) — chosen over the general `gpt-oss:20b` default
  (still used elsewhere: invoice classification, PowerPoint outlines) for
  its stronger multilingual output; `gpt-oss:20b` is primarily
  English-optimized and more prone to drifting back into English on
  non-English text.
- **Attorney tabs** (`Attorney 32B`, `Attorney 32B Instruct (Slower)`,
  `Attorney 1.7B (Fast)`) chat with DictaLM
  (`hf.co/dicta-il/DictaLM-3.0-24B-Thinking-GGUF:Q4_K_M` for the first two,
  a smaller 1.7B variant for the fast one) instead of the Chat tab's
  `qwen2.5:32b`. Every stage of the two DictaLM-backed pipelines — query
  understanding, answering, and verification — runs on DictaLM; there's no
  separate reasoning model involved. See `app/legal_pipeline_v2.py`'s module
  docstring for the full architecture of each tab. Being a "thinking"
  model, DictaLM wraps its reasoning in `<think>...</think>` before the
  actual answer, which is stripped before it reaches the UI either way.
- **Document-chunking reliability fix**: chunks of Hebrew text used to be
  sent to downstream LLM calls (translation, document rewrite, RAG
  ingestion) far larger than the intended per-chunk limit, because the
  sentence-boundary regex used for chunking only recognized a new sentence
  when followed by a Latin capital letter or digit — never a Hebrew (or
  Arabic/Cyrillic/CJK) letter. That silently defeated chunking for any
  non-Latin source script, and long, multi-sentence blocks are less
  reliable for most local models regardless of task. `app/document.py`'s
  `SENTENCE_SPLIT_RE` now recognizes these scripts too, and `chunk_text` has
  a hard character-count fallback (`_hard_split`) so no chunk can silently
  exceed the limit regardless of script. Hebrew OCR text also had invisible
  Unicode bidi-direction marks (inserted by Tesseract to keep mixed
  Hebrew/English text in correct reading order) landing mid-word, which
  fragmented tokenization further — `strip_bidi_controls` removes these at
  Hebrew-OCR extraction time.
