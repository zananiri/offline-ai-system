"""
FastAPI orchestrator.

Pipeline: upload -> Docling (convert+OCR) -> language detect -> NLLB translate
          -> optional Ollama cleanup/summarization -> return / export docx
"""
import json
import re
import shutil
import tempfile
import uuid
from pathlib import Path

import ollama
import httpx
import py3langid as langid
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse, FileResponse

from app.document import (
    convert_to_markdown,
    chunk_text,
    convert_file_to_docx,
    resolve_hebrew_flag,
    TesseractNotAvailableError,
)
from app.translate import get_translator, LANGUAGES
from app.pptx_generator import generate_pptx

app = FastAPI(title="Offline Translator + Document OCR")

OLLAMA_MODEL = "qwen2.5:7b-instruct-q4_K_M"

# Backs the Legal tab. Must be pulled once via:
#   ollama pull hf.co/dicta-il/DictaLM-3.0-1.7B-Thinking-GGUF:Q4_K_M
# (setup.ps1 does this for you.) NOTE: this string is duplicated in
# app/ui.py (which only talks to this backend over HTTP and can't share a
# Python constant with it) — keep the two in sync if you change the model.
#
# Swapped from the 24B variant to this 1.7B one for speed on CPU/iGPU-bound
# machines -- it's the smallest size Dicta publishes for this family (24B,
# 12B, 1.7B), so this is as fast as a DictaLM "thinking" model gets. Real
# tradeoff: noticeably weaker legal reasoning/citation accuracy than the 24B.
# To go back, just restore the line above (after re-pulling the 24B model).
LEGAL_MODEL = "hf.co/dicta-il/DictaLM-3.0-1.7B-Thinking-GGUF:Q4_K_M"

# ollama-python's module-level ollama.chat(...) convenience function uses a
# default client built with `timeout=None` -- i.e. NO request timeout at
# all. Combined with nothing anywhere capping how many tokens a model is
# allowed to generate, a "thinking" model on a broad/open question can run
# for as long as it wants: DictaLM-3.0-1.7B-Thinking has a native 65k-token
# context, so there's no natural ceiling either (confirmed: a real question
# about the Clean Air Law ran 1500+s and 13k+ tokens with no end in sight).
#
# Two layers of defense, both applied below:
#   1. num_predict (a hard cap on tokens generated) -- the graceful bound.
#      llama.cpp just stops there and returns whatever it has, which is why
#      this pairs with _strip_thinking's already-existing handling for an
#      opened-but-never-closed <think> tag (see chat() below): if the model
#      is still mid-thought when the cap hits, the user gets a clear "ran
#      out of space" message instead of nothing.
#   2. This client's `timeout=` -- a much larger outer safety net, in case
#      something is actually stuck (not just slow) and never returns at all
#      even within its token budget.
_OLLAMA_REQUEST_TIMEOUT_SECONDS = 1800  # 30 min -- safety net, not the normal path
_DEFAULT_NUM_PREDICT = 4096  # ~7-14 min of generation at the ~5-9 tok/s seen on this hardware
_ollama_client = ollama.Client(timeout=_OLLAMA_REQUEST_TIMEOUT_SECONDS)

# num_predict on its own is NOT a real ceiling unless the context window it
# runs inside is at least that big. Ollama defaults to a 4096-token context
# per slot regardless of a model's native window (confirmed via the
# llama.cpp server log: "n_ctx_slot = 4096" even for DictaLM-3.0-1.7B-
# Thinking, which natively supports 65k) -- so a caller asking for
# num_predict=6144 (see app/ui.py's _LEGAL_NUM_PREDICT) could never actually
# reach that cap; the 4096-token window fills first. When that happens,
# llama.cpp falls back to context-shifting (visible in the same log as
# "n_keep = 4") instead of stopping -- it keeps a handful of anchor tokens,
# discards the rest, and keeps decoding. That's what produced the original
# "ran 1500+s and 13k+ tokens with no end in sight" failure on a Clean Air
# Law question: the num_predict cap was never actually reachable, so
# generation looked hung when it was really just shifting its window
# forever. Always request a context window comfortably larger than
# num_predict + the prompt (system prompt + history + any attached-document
# text) so num_predict is the thing that actually stops generation.
_DEFAULT_NUM_CTX = 8192


def _num_ctx_for(num_predict: int, requested: int | None) -> int:
    """Picks a context window big enough that num_predict is reachable.
    Honors an explicit request (e.g. from the Legal tab, which needs more
    room for a bigger num_predict plus attached-document context), but
    never returns something smaller than num_predict itself would need."""
    floor = num_predict + 2048  # headroom for system prompt + history + doc context
    if requested:
        return max(requested, floor)
    return max(_DEFAULT_NUM_CTX, floor)

# DictaLM-3.0-24B-Thinking (like other "thinking" models, e.g. QwQ/R1-style)
# emits its chain-of-thought wrapped in <think>...</think> before the real
# answer. Strip it so the chat UI only ever shows the final response — this
# is a no-op for models (like the default qwen2.5) that never emit the tag.
# If DictaLM turns out to use a different wrapper tag, update this regex.
_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL)

# Fallback for a <think> tag that was opened but never closed -- happens
# when the model runs out of its context/token budget mid-reasoning (a real
# risk for "thinking" models on long prompts, e.g. a legal question with a
# big attached document pushing the request near the model's context
# limit). _THINK_BLOCK_RE requires a matching close tag, so on its own it
# silently leaves the entire raw, unfinished chain-of-thought in place --
# which then renders in the chat UI as if it were the actual answer.
_UNCLOSED_THINK_RE = re.compile(r"<think>.*", re.IGNORECASE | re.DOTALL)


def _strip_thinking(content: str) -> str:
    """Removes a thinking model's <think>...</think> block. Handles both a
    normal, fully-closed block and one truncated by a token/context limit
    (see _UNCLOSED_THINK_RE above) -- in the latter case there's no real
    answer to recover, so that's surfaced as an explicit note instead of
    dumping the raw, unfinished reasoning on the user."""
    stripped = _THINK_BLOCK_RE.sub("", content).strip()
    if "<think>" in stripped.lower():
        stripped = _UNCLOSED_THINK_RE.sub("", stripped).strip()
        if not stripped:
            stripped = (
                "_(The model ran out of space while reasoning and never "
                "reached an answer. Try a shorter question or a smaller "
                "attached document.)_"
            )
    return stripped


def _convert_to_markdown_or_503(path: str, hebrew: bool) -> str:
    try:
        return convert_to_markdown(path, hebrew=hebrew)
    except TesseractNotAvailableError as e:
        raise HTTPException(status_code=503, detail=str(e))


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
        response = _ollama_client.chat(
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

    # Resolved once up front (rather than just passing `hebrew` through)
    # so the response can honestly report which OCR path actually ran --
    # resolve_hebrew_flag can auto-detect and override a False here.
    hebrew_used = resolve_hebrew_flag(tmp_path, hebrew)
    markdown_text = _convert_to_markdown_or_503(tmp_path, hebrew_used)
    return {"filename": file.filename, "markdown": markdown_text, "hebrew_used": hebrew_used}


@app.post("/chat")
async def chat(payload: dict):
    """
    General chat endpoint backed by Ollama.
    Expects: {"messages": [...], "model": "...", "num_predict": int, "num_ctx": int}
    (model optional, defaults to OLLAMA_MODEL -- the Legal tab passes
    LEGAL_MODEL explicitly, everything else uses the default. num_predict
    optional, defaults to _DEFAULT_NUM_PREDICT -- see that constant's
    comment above for why this is capped at all. num_ctx optional --
    see _num_ctx_for's comment above: without this, num_predict is not a
    real ceiling since Ollama's default 4096-token context window fills up
    first and triggers context-shifting instead of a clean stop.)
    """
    messages = payload.get("messages", [])
    model = payload.get("model") or OLLAMA_MODEL
    num_predict = payload.get("num_predict") or _DEFAULT_NUM_PREDICT
    num_ctx = _num_ctx_for(num_predict, payload.get("num_ctx"))
    try:
        response = _ollama_client.chat(
            model=model, messages=messages,
            options={"num_predict": num_predict, "num_ctx": num_ctx},
        )
    except ConnectionError as e:
        # ollama-python already turns "daemon not running/reachable" into a
        # plain ConnectionError with a friendly message -- pass it straight
        # through rather than letting FastAPI turn it into a raw 500.
        raise HTTPException(status_code=503, detail=str(e))
    except ollama.ResponseError as e:
        if e.status_code == 404:
            raise HTTPException(
                status_code=503,
                detail=(
                    f"Model '{model}' isn't pulled in Ollama yet. Run: "
                    f"ollama pull {model}"
                ),
            )
        raise HTTPException(status_code=502, detail=str(e))
    except httpx.TimeoutException as e:
        raise HTTPException(
            status_code=504,
            detail=f"Ollama didn't respond within {_OLLAMA_REQUEST_TIMEOUT_SECONDS}s: {e}",
        )
    content = response["message"]["content"]
    content = _strip_thinking(content)
    return {"content": content}


@app.post("/generate-pptx")
async def generate_pptx_endpoint(payload: dict):
    """
    Generates a PowerPoint (.pptx) file from a plain-language prompt.

    Expects: {"prompt": "...", "model": "..."} (model optional, defaults to
    OLLAMA_MODEL -- same model the Chat tab already uses for everything
    else). The prompt can just be a topic ("the history of coffee") or can
    include attached-document text to summarize into slides (the Chat tab's
    UI builds that combined prompt before calling this).

    Uses qwen2.5 (via Ollama) to draft a JSON slide outline, then
    python-pptx (MIT licensed) to build the actual file -- both steps run
    fully locally, no external/paid presentation-generation service involved.
    """
    prompt = (payload.get("prompt") or "").strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="Missing 'prompt'.")
    model = payload.get("model") or OLLAMA_MODEL

    output_path = str(Path(tempfile.gettempdir()) / f"presentation_{uuid.uuid4().hex}.pptx")
    try:
        generate_pptx(prompt, output_path, model=model)
    except ValueError as e:
        # The model's outline couldn't be parsed as usable JSON even after a
        # retry -- surface this as a clean error rather than a raw 500.
        raise HTTPException(status_code=502, detail=str(e))

    return FileResponse(
        output_path,
        media_type="application/vnd.openxmlformats-officedocument.presentationml.presentation",
        filename="presentation.pptx",
    )


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

    hebrew_used = resolve_hebrew_flag(input_path, hebrew)
    try:
        convert_file_to_docx(input_path, output_path, hebrew=hebrew_used)
    except TesseractNotAvailableError as e:
        raise HTTPException(status_code=503, detail=str(e))

    return FileResponse(
        output_path,
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        filename=output_filename,
        headers={"X-Hebrew-OCR-Used": str(hebrew_used)},
    )


@app.post("/translate-chunk")
async def translate_chunk(payload: dict):
    """Translates a single chunk of text. Used by the UI to show per-chunk progress."""
    text = payload.get("text", "")
    source_lang = payload.get("source_lang")
    target_lang = payload.get("target_lang")
    translator = get_translator()
    translated, ok = translator.translate(text, source_lang, target_lang)
    return {"translated": translated, "ok": ok}


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
    hebrew_used = resolve_hebrew_flag(tmp_path, hebrew)
    markdown_text = _convert_to_markdown_or_503(tmp_path, hebrew_used)
    chunks = chunk_text(markdown_text)

    # 3. Translate each chunk
    translator = get_translator()
    results = translator.translate_chunks(chunks, source_lang, target_lang)
    translated_chunks = [text for text, _ok in results]
    failed_chunks = sum(1 for _text, ok in results if not ok)
    translated_text = "\n\n".join(translated_chunks)

    # 4. Optional: Ollama pass to clean up structure / summarize
    summary = None
    if summarize:
        response = _ollama_client.chat(
            model=OLLAMA_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        f"You are a precise document summarizer. Respond ONLY in "
                        f"{target_lang}, matching the language of the text you are "
                        f"given -- never switch to a different language."
                    ),
                },
                {"role": "user", "content": f"Summarize this in 3-5 bullet points:\n\n{translated_text}"},
            ],
        )
        summary = response["message"]["content"]

    return JSONResponse({
        "original_markdown": markdown_text,
        "translated_text": translated_text,
        "summary": summary,
        "total_chunks": len(chunks),
        "failed_chunks": failed_chunks,
        "hebrew_used": hebrew_used,
    })