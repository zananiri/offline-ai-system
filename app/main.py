"""
FastAPI orchestrator.

Pipeline: upload -> Docling (convert+OCR) -> language detect -> NLLB translate
          -> optional Ollama cleanup/summarization -> return / export docx
"""
import shutil
import tempfile
from pathlib import Path

import ollama
import py3langid as langid
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse

from app.document import convert_to_markdown, chunk_text
from app.translate import get_translator, LANGUAGES

app = FastAPI(title="Offline Translator + Document OCR")

OLLAMA_MODEL = "qwen2.5:7b-instruct-q4_K_M"


@app.get("/health")
def health():
    return {"status": "ok", "languages": list(LANGUAGES.keys())}


@app.post("/detect-language")
def detect_language(text: str = Form(...)):
    code, _ = langid.classify(text)
    return {"iso_639_1": code}


@app.post("/translate-document")
async def translate_document(
    file: UploadFile = File(...),
    source_lang: str = Form(...),
    target_lang: str = Form(...),
    summarize: bool = Form(False),
):
    # 1. Save upload
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    # 2. Convert + OCR
    markdown_text = convert_to_markdown(tmp_path)
    chunks = chunk_text(markdown_text)

    # 3. Translate each chunk with NLLB
    translator = get_translator()
    translated_chunks = translator.translate_chunks(chunks, source_lang, target_lang)
    translated_text = "\n\n".join(translated_chunks)

    # 4. Optional: Ollama pass to clean up structure / summarize
    summary = None
    if summarize:
        response = ollama.chat(
            model=OLLAMA_MODEL,
            messages=[
                {"role": "system", "content": "You are a precise document summarizer."},
                {"role": "user", "content": f"Summarize this in 3-5 bullet points:\n\n{translated_text}"},
            ],
        )
        summary = response["message"]["content"]

    return JSONResponse({
        "original_markdown": markdown_text,
        "translated_text": translated_text,
        "summary": summary,
    })
