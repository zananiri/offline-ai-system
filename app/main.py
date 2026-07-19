"""
FastAPI orchestrator.

Pipeline: upload -> Docling (convert+OCR) -> language detect -> NLLB translate
          -> optional Ollama cleanup/summarization -> return / export docx
"""
import json
import shutil
import tempfile
from pathlib import Path

import ollama
import py3langid as langid
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse, FileResponse

from app.document import convert_to_markdown, chunk_text, convert_file_to_docx
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


@app.post("/classify-invoice")
async def classify_invoice(payload: dict):
    """
    Classifies OCR-extracted invoice/receipt text as sales or expense and
    extracts key fields. Returns document_type="unrecognized" if the model
    can't confidently classify it or the text isn't an invoice/receipt at all.
    """
    markdown_text = payload.get("markdown", "")
    filename = payload.get("filename", "")
    company_name = payload.get("company_name", "").strip()

    system_prompt = (
        "You are an accounting assistant. You are given OCR-extracted text from a "
        "scanned invoice or receipt. Classify it and extract key fields. "
        "Respond with ONLY valid JSON, no other text, in exactly this shape:\n"
        '{"document_type": "sales" | "expense" | "unrecognized", '
        '"party_name": string, "invoice_number": string, "date": string, '
        '"amount": number, "vat": number, "currency": string}\n\n'
        "\"amount\" is the total amount on the invoice (including tax/VAT if shown). "
        "\"vat\" is the VAT/tax amount shown on the invoice as a number; use 0 if no "
        "VAT/tax line is present or the invoice is not VAT-registered. "
        f"The business being accounted for is: {company_name or '(not specified - infer from context)'}. "
        "Classify as \"sales\" if this business is the SELLER/issuer of the invoice (money coming in). "
        "Classify as \"expense\" if this business is the BUYER/recipient (money going out). "
        "If the text is unreadable, doesn't look like an invoice or receipt, or you cannot "
        "determine the type or amount with reasonable confidence, use \"unrecognized\" and "
        "leave the other fields empty or 0."
    )

    try:
        response = ollama.chat(
            model=OLLAMA_MODEL,
            format="json",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": markdown_text[:4000]},
            ],
        )
        result = json.loads(response["message"]["content"])
        if result.get("document_type") not in ("sales", "expense", "unrecognized"):
            result["document_type"] = "unrecognized"
    except Exception:
        result = {"document_type": "unrecognized", "party_name": "", "invoice_number": "",
                   "date": "", "amount": 0, "vat": 0, "currency": ""}

    result["filename"] = filename
    return result


@app.post("/extract-text")
async def extract_text(file: UploadFile = File(...), hebrew: bool = Form(False)):
    """
    Extracts Markdown text from any supported document (PDF, DOCX, PPTX, image via OCR)
    without exporting to Word. Used by the chat tab to give the model file context.
    Set hebrew=True to route OCR through Tesseract instead of the default RapidOCR engine.
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    markdown_text = convert_to_markdown(tmp_path, hebrew=hebrew)
    return {"filename": file.filename, "markdown": markdown_text}


@app.post("/chat")
async def chat(payload: dict):
    """
    General chat endpoint backed by Ollama.
    Expects: {"messages": [{"role": "user"|"assistant"|"system", "content": "..."}]}
    """
    messages = payload.get("messages", [])
    response = ollama.chat(model=OLLAMA_MODEL, messages=messages)
    return {"content": response["message"]["content"]}


@app.post("/convert-to-word")
async def convert_to_word(file: UploadFile = File(...), hebrew: bool = Form(False)):
    """
    Converts PDF -> Word, or an image (via OCR) -> Word.
    Docling auto-detects the input type; scanned PDFs and images both
    go through its OCR path, native PDFs go through direct text extraction.
    Set hebrew=True to route OCR through Tesseract instead of the default RapidOCR engine.
    """
    suffix = Path(file.filename).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_in:
        shutil.copyfileobj(file.file, tmp_in)
        input_path = tmp_in.name

    output_filename = Path(file.filename).stem + ".docx"
    output_path = str(Path(tempfile.gettempdir()) / output_filename)

    convert_file_to_docx(input_path, output_path, hebrew=hebrew)

    return FileResponse(
        output_path,
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        filename=output_filename,
    )


@app.post("/translate-chunk")
async def translate_chunk(payload: dict):
    """Translates a single chunk of text. Used by the UI to show per-chunk progress."""
    text = payload.get("text", "")
    source_lang = payload.get("source_lang")
    target_lang = payload.get("target_lang")
    translator = get_translator()
    translated = translator.translate(text, source_lang, target_lang)
    return {"translated": translated}


@app.post("/translate-document")
async def translate_document(
    file: UploadFile = File(...),
    source_lang: str = Form(...),
    target_lang: str = Form(...),
    summarize: bool = Form(False),
    hebrew: bool = Form(False),
):
    # 1. Save upload
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    # 2. Convert + OCR
    markdown_text = convert_to_markdown(tmp_path, hebrew=hebrew)
    chunks = chunk_text(markdown_text)

    # 3. Translate each chunk
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
