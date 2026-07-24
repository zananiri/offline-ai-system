"""
Gradio UI — talks to the FastAPI backend at localhost:8000.
Run after main.py is already running: python -m app.ui
"""
import base64
import re
import tempfile
import zipfile
from pathlib import Path

import requests
import gradio as gr
import openpyxl
from openpyxl.styles import Font
from pypdf import PdfReader, PdfWriter

try:
    import chromadb
except ImportError:
    chromadb = None  # Canon AI tab degrades to a clear error message if this isn't installed

from app.translate import LANGUAGES
from app.document import chunk_text

BACKEND_URL = "http://localhost:8000"

# Matches (and pads slightly past) main.py's own _OLLAMA_REQUEST_TIMEOUT_SECONDS,
# so this client never gives up before the backend's own safety-net timeout
# would already have returned a clean error. Without this, requests has no
# default timeout at all -- a stuck/slow "thinking" model call just hangs
# the whole Gradio UI indefinitely with zero feedback (confirmed in
# practice: a Clean Air Law question to DictaLM ran 1500+s and 13k+ tokens
# with no sign of stopping before main.py's num_predict cap was added).
_CHAT_TIMEOUT_SECONDS = 1830


def _chat_backend(messages, model=None, num_predict=None, num_ctx=None, timeout=None, timeout_hint=None):
    """
    POSTs to the backend's /chat endpoint and returns the response content
    as a plain string. Raises RuntimeError with a clean, user-facing message
    on any failure (backend unreachable, timed out, or a clean error detail
    the backend itself already generated) -- callers decide what to do with
    that: show it as the reply (chat_fn/legal_chat_fn), or fall back
    gracefully without losing other already-successful work (process()'s
    summarizer step, where a failed summary shouldn't also take down the
    translated text that already succeeded).

    timeout/timeout_hint let a caller override the default for a model that
    needs more room (see legal_chat_fn, which passes a larger timeout and a
    more specific hint tailored to a "thinking" model on a broad question).

    num_ctx lets a caller request a bigger context window than the backend's
    own default -- see main.py's _num_ctx_for comment for why this matters:
    without it, num_predict isn't a real ceiling for a big-num_predict
    caller like legal_chat_fn, since Ollama's smaller default context window
    fills up first and the model keeps going via context-shifting instead of
    stopping cleanly.
    """
    payload = {"messages": messages}
    if model:
        payload["model"] = model
    if num_predict:
        payload["num_predict"] = num_predict
    if num_ctx:
        payload["num_ctx"] = num_ctx
    timeout = timeout or _CHAT_TIMEOUT_SECONDS
    try:
        resp = requests.post(f"{BACKEND_URL}/chat", json=payload, timeout=timeout)
    except requests.exceptions.Timeout:
        hint = timeout_hint or "try a shorter question or a smaller attached document."
        raise RuntimeError(
            f"The model didn't respond within {timeout // 60} minutes. "
            f"It may be stuck, or just very slow on this hardware -- {hint}"
        )
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Couldn't reach the backend: {e}")

    if not resp.ok:
        try:
            detail = resp.json().get("detail", resp.text)
        except Exception:
            detail = resp.text
        raise RuntimeError(detail)

    return resp.json()["content"]


# Model backing the Legal tab. NOTE: this string is duplicated in
# app/main.py (ui.py only talks to that backend over HTTP, so it can't
# import a shared Python constant from it) — keep the two in sync if you
# change the model. Must be pulled once via:
#   ollama pull hf.co/dicta-il/DictaLM-3.0-24B-Thinking-GGUF:Q4_K_M
#
# Upgraded from the 1.7B model to Dicta's 24B flagship for noticeably
# stronger legal reasoning and more reliable citations. Q4_K_M quantization
# is a ~14.3GB download/on-disk file. On a 32GB-RAM, offline/CPU (or
# modest-iGPU) machine this fits with room to spare for the OS and the rest
# of this app's own memory use (docling, the MADLAD-400 translation model,
# etc.) -- but it's the biggest single thing this app loads, so:
#   - Real tradeoff vs. the 1.7B: much slower generation (CPU tok/s roughly
#     tracks parameter count, so expect noticeably fewer tokens/sec than the
#     1.7B's ~20 tok/s -- see the timeout constants below, which were raised
#     accordingly).
#   - If you're running the Legal tab at the same time as a large translation
#     or batch-invoice job (i.e. multiple big models resident at once) and
#     see swapping/OOM, drop to the smaller IQ4_XS quant instead (~12.8GB:
#     hf.co/dicta-il/DictaLM-3.0-24B-Thinking-GGUF:IQ4_XS), or fall back to
#     the 1.7B line below.
# Previous (faster, weaker) setting, kept here for an easy revert:
#   LEGAL_MODEL = "hf.co/dicta-il/DictaLM-3.0-1.7B-Thinking-GGUF:Q4_K_M"
LEGAL_MODEL = "hf.co/dicta-il/DictaLM-3.0-24B-Thinking-GGUF:Q4_K_M"

# Backs the separate "Attorney 1.7B" tab (see legal_chat_fn_1_7b below) --
# Dicta's smallest model in this family, kept alongside the 24B as a fast
# option rather than a replacement for it. NOT pulled by setup.ps1/
# install.txt -- pull it manually once, same venv active:
#   ollama pull hf.co/dicta-il/DictaLM-3.0-1.7B-Thinking-GGUF:Q4_K_M
# (~1.1GB download; confirm it landed with `ollama list`)
LEGAL_MODEL_1_7B = "hf.co/dicta-il/DictaLM-3.0-1.7B-Thinking-GGUF:Q4_K_M"

# See main.py's _DEFAULT_NUM_PREDICT comment for the full story: with no cap
# at all, a "thinking" model on a broad question can run for as long as it
# wants (DictaLM-3.0-24B-Thinking has a native 65k-token context, so there's
# no natural ceiling either -- confirmed in practice with the 1.7B model: a
# real question about the Clean Air Law ran 1500+s and 13k+ tokens with no
# end in sight, and the 24B is slower still). This is more generous than the
# backend's own 4096-token default since thinking + a citation for every
# claim genuinely needs more room, but it's still a hard cap -- if it's hit
# mid-thought, _strip_thinking (main.py) turns that into a clear "ran out of
# space" message instead of hanging forever.
_LEGAL_NUM_PREDICT = 6144
# _LEGAL_NUM_PREDICT only works as an actual ceiling if the model's context
# window is at least that big -- see main.py's _num_ctx_for comment. Without
# this, Ollama's default 4096-token window fills up before num_predict does
# and the model keeps decoding via context-shifting instead of stopping
# (this is what "ran 1500+s and 13k+ tokens with no end in sight" on a Clean
# Air Law question actually was). Sized for num_predict (6144) + the system
# prompt + chat history + a full MAX_CONTEXT_CHARS-sized attached document,
# with headroom -- comfortably under DictaLM-3.0-24B-Thinking's native 65k.
# (Note: KV cache at this context size adds real memory on top of the 14.3GB
# weights -- a few more GB. Lower this if you're tight on RAM alongside
# other models this app loads.)
_LEGAL_NUM_CTX = 16384
# Comfortably above the backend's own safety-net timeout (main.py's
# _OLLAMA_REQUEST_TIMEOUT_SECONDS), so that backend's clearer error message
# surfaces first instead of a generic timeout here. Raised well past the
# 1.7B-era value: on CPU, a 24B dense model generating up to 6144 tokens can
# genuinely take tens of minutes rather than single-digit minutes.
_LEGAL_REQUEST_TIMEOUT_SECONDS = 3700

LEGAL_SYSTEM_PROMPT = (
    "You are an Israeli lawyer. Think through and answer every question strictly "
    "according to the laws of the State of Israel -- its statutes, regulations, "
    "and case law -- not the law of any other jurisdiction, unless the user "
    "explicitly asks about a different country's law.\n\n"
    "For every substantive legal claim, name the specific Israeli statute, "
    "regulation, or section you are relying on immediately after the claim -- "
    "for example: 'A contract requires offer and acceptance (Section 1, "
    "Contracts Law (General Part), 5733-1973).' Also cite the relevant "
    "section/clause of any attached document when you rely on it. If you are "
    "not confident of the exact statute, section number, or case citation, say "
    "so explicitly instead of inventing one -- a wrong or fabricated citation "
    "is worse than admitting uncertainty.\n\n"
    "Keep your answer proportionate to the question: for a broad or general "
    "topic, cover the most important, directly relevant points rather than "
    "exhaustively enumerating every provision of a law -- you can offer to go "
    "deeper on a specific part if the user wants that.\n\n"
    "You are not a substitute for advice from a licensed attorney, and you "
    "should say so when a question calls for one. Always reply in the same "
    "language the user's question is written in -- e.g. if they write in "
    "English, your entire answer must be in English, even though the "
    "underlying law and any attached document may be in Hebrew. Never switch "
    "languages on the user unasked."
)

# Best-effort heuristic for "did the answer cite anything at all" -- matches
# common Israeli-law citation shapes in both English and Hebrew (section/
# regulation numbers, named statutes, and Hebrew court-ruling abbreviations
# like בג"ץ/ע"א). This can only detect the ABSENCE of a citation-shaped
# string; it cannot verify that a citation that IS present is real. Small
# models are prone to fabricating plausible-looking statute/section numbers,
# and no regex can catch that -- this is a floor (something was cited),
# not a correctness guarantee. See legal_chat_fn for how it's used.
_LEGAL_CITATION_RE = re.compile(
    r"(סעיף\s*\d+|תקנה\s*\d+|חוק\s+\S+|פסק\s*דין|בג\"?ץ|ע\"?א\s*\d+|"
    r"\bsection\s+\d+|\bregulation\s+\d+|\barticle\s+\d+|\blaw\s*(\(|,)?\s*(19|20|5[6-9])\d{2}\b|"
    r"\bhcj\b|\bbasic\s+law\b)",
    re.IGNORECASE,
)

_NO_CITATION_NOTE = (
    "\n\n---\n⚠️ *This answer doesn't appear to cite a specific law, regulation, "
    "or case. Treat it as general information only and verify against the "
    "actual legislation or with a licensed attorney before relying on it.*"
)

# Splits an answer into sentence-ish chunks so _extract_citations can pull
# out the whole sentence around a citation match (not just the bare
# "Section 1" fragment _LEGAL_CITATION_RE matches on its own) -- that's what
# actually gets shown in the sidebar panel. Handles Hebrew sentence-enders
# too, plus plain newlines (the model often puts one citation per line).
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")

_CITATIONS_PLACEHOLDER = (
    "_No citations detected yet. Once you ask a question, any statute, "
    "regulation, or case citation the model includes in its answer will be "
    "listed here for easy review._"
)
_CITATIONS_NONE_FOUND = (
    "_This answer didn't include anything citation-shaped. See the "
    "warning in the chat -- treat it as general information only._"
)


def _extract_citations(answer: str) -> list[str]:
    """Best-effort pull of citation-bearing sentences out of an answer, for
    display in the Legal tab's sidebar. Same caveats as _LEGAL_CITATION_RE:
    this can only detect something citation-shaped, not verify it's real."""
    seen = set()
    citations = []
    for chunk in _SENTENCE_SPLIT_RE.split(answer):
        sentence = chunk.strip(" -•\t\n")
        if not sentence or sentence in seen:
            continue
        if _LEGAL_CITATION_RE.search(sentence):
            seen.add(sentence)
            citations.append(sentence)
    return citations


def _format_citations_panel(citations: list[str]) -> str:
    if not citations:
        return _CITATIONS_NONE_FOUND
    return "\n\n".join(f"- {c}" for c in citations)

# Languages whose script reads right-to-left. Used to flip the translated-text
# output box's text direction so Arabic/Hebrew results display correctly
# instead of being left-aligned like Latin-script languages.
RTL_LANGUAGES = {"arabic", "hebrew"}


def _is_rtl(lang: str) -> bool:
    return (lang or "").strip().lower() in RTL_LANGUAGES


_RTL_MARK = "\u200f"  # RIGHT-TO-LEFT MARK (invisible, sets bidi direction only)


def _anchor_rtl_lines(text: str) -> str:
    """
    Textbox(rtl=True) sets the box's overall base direction, but that alone
    doesn't stop individual lines from getting visually reordered by the
    browser's bidi algorithm -- a line that happens to start with a digit,
    punctuation, or an embedded Latin word/number (invoice numbers, dates,
    stray English terms) can pull that whole line's layout toward
    left-to-right, scattering words out of their intended order.

    Prefixing every non-empty line with an invisible RIGHT-TO-LEFT MARK
    (U+200F) fixes each line's base direction as RTL regardless of its
    first character, without adding any visible content.
    """
    if not text:
        return text
    return "\n".join(
        f"{_RTL_MARK}{line}" if line.strip() else line
        for line in text.split("\n")
    )


# --- Document preview (Translate tab's right-hand panel) -------------------
#
# There's no single Gradio component that previews every format this app
# accepts, so three different strategies are used depending on file type:
#
#   Images  -> gr.Image with show_fullscreen_button=True. Gives a native
#              click-to-zoom/pan lightbox for free (same mechanism this
#              file already uses for the header logo, just switched on).
#   PDFs    -> embedded via a base64 data: URI inside an <iframe>, so the
#              browser's own PDF viewer renders it -- built-in zoom, scroll,
#              page navigation, and print, no extra dependency. A data URI
#              is used instead of a path-based Gradio static-file URL (e.g.
#              "/file=...") on purpose: that route's exact prefix differs
#              across Gradio major versions and only works for paths inside
#              Gradio's allowed-paths, which is fragile to hardcode. A data
#              URI works identically regardless of Gradio version.
#   Other   -> (DOCX, PPTX, etc.) no in-browser renderer exists for these
#              without extra dependencies, so the existing /extract-text
#              endpoint is reused to show the extracted text instead. Not a
#              visual preview, but it directly shows what's about to be
#              translated, and costs nothing extra: DOCX/PPTX never go
#              through OCR (see document.py), so this second extraction
#              call is fast.
_PREVIEW_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tif", ".tiff", ".webp"}

# Above this, base64-inlining the whole file into the page's HTML would
# bloat the page enough to feel sluggish -- fall back to a plain notice
# instead of a broken/slow preview. Translation itself is unaffected either
# way; this only gates the preview panel.
_MAX_INLINE_PREVIEW_BYTES = 20 * 1024 * 1024  # 20 MB


def _pdf_preview_html(path: Path) -> str:
    b64 = base64.b64encode(path.read_bytes()).decode("ascii")
    data_uri = f"data:application/pdf;base64,{b64}"
    return f"""
<div style="border:1px solid var(--border-color-primary, #ddd); border-radius:8px; overflow:hidden;">
  <iframe src="{data_uri}" style="width:100%; height:520px; border:none;"></iframe>
</div>
<p style="text-align:center; margin-top:6px;">
  <a href="{data_uri}" target="_blank" rel="noopener">🔍 Open full preview in a new tab (zoom, scroll, print)</a>
</p>
"""


def build_preview(file):
    """
    Builds a live preview of the uploaded document for the panel on the
    right of the Translate tab. Wired to file_in.change, so it fires the
    moment a file is picked (or cleared) -- independent of clicking
    Translate -- letting the user confirm it's the right document before
    running anything.

    The whole preview column is hidden by default and only made visible
    once a file is actually present, and hidden again when the file is
    cleared -- there's no "upload a document" placeholder state, the panel
    simply isn't there until there's something to show.

    Returns a 4-tuple of gr.update(...) for (preview_column, image_preview,
    pdf_preview_html, text_preview). Within the column, exactly one of the
    three preview components is made visible at a time (text_preview also
    doubles as the spot for status/error messages, e.g. an oversized PDF
    or a failed extraction).
    """
    hidden = gr.update(visible=False)

    if file is None:
        return gr.update(visible=False), hidden, hidden, hidden

    path = Path(file.name)
    suffix = path.suffix.lower()

    if suffix in _PREVIEW_IMAGE_EXTS:
        return gr.update(visible=True), gr.update(value=str(path), visible=True), hidden, hidden

    if suffix == ".pdf":
        size = path.stat().st_size
        if size > _MAX_INLINE_PREVIEW_BYTES:
            msg = (
                f"⚠️ This PDF is {size / 1_048_576:.1f} MB, too large to preview inline. "
                "It will still be translated normally -- the preview panel just skips it."
            )
            return gr.update(visible=True), hidden, hidden, gr.update(value=msg, visible=True)
        return gr.update(visible=True), hidden, gr.update(value=_pdf_preview_html(path), visible=True), hidden

    # DOCX/PPTX/etc. -- no visual renderer available, fall back to extracted text.
    try:
        with open(path, "rb") as f:
            resp = requests.post(f"{BACKEND_URL}/extract-text", files={"file": f}, data={"hebrew": False})
        resp.raise_for_status()
        text = resp.json().get("markdown", "")
    except Exception as e:
        return gr.update(visible=True), hidden, hidden, gr.update(value=f"⚠️ Couldn't build a preview: {e}", visible=True)

    if not text.strip():
        return gr.update(visible=True), hidden, hidden, gr.update(value="(no previewable text found in this document)", visible=True)
    return gr.update(visible=True), hidden, hidden, gr.update(value=text, visible=True)


def process(file, source_lang, target_lang, summarize, hebrew_doc, progress=gr.Progress()):
    progress(0, desc="Extracting text from document (OCR if needed)...")
    with open(file.name, "rb") as f:
        extract_resp = requests.post(
            f"{BACKEND_URL}/extract-text", files={"file": f}, data={"hebrew": hebrew_doc}
        )
    extract_resp.raise_for_status()
    extract_data = extract_resp.json()
    markdown_text = extract_data["markdown"]
    # The backend can auto-detect Hebrew and route through Tesseract even
    # if hebrew_doc was False (see document.py's resolve_hebrew_flag) --
    # report what actually ran, not just what was requested, so this
    # doesn't silently lie when that override kicks in.
    hebrew_used = extract_data.get("hebrew_used", hebrew_doc)
    ocr_engine = "Tesseract (Hebrew)" if hebrew_used else "RapidOCR (default)"

    chunks = chunk_text(markdown_text)
    if not chunks:
        return (
            gr.update(value="(no text found in document)", rtl=_is_rtl(target_lang)),
            ocr_engine,
            "(not requested)",
        )

    total_steps = len(chunks) + (1 if summarize else 0)
    translated_chunks = []
    failed_chunks = 0
    for i, chunk in enumerate(chunks):
        progress((i + 1) / total_steps, desc=f"Translating chunk {i + 1}/{len(chunks)}...")
        resp = requests.post(
            f"{BACKEND_URL}/translate-chunk",
            json={"text": chunk, "source_lang": source_lang, "target_lang": target_lang},
        )
        resp.raise_for_status()
        data = resp.json()
        translated_value = data.get("translated", "")
        ok = data.get("ok", True)
        if not isinstance(translated_value, str):
            # Defensive: a backend/frontend version mismatch (e.g. main.py
            # not unpacking the (text, ok) tuple translate() now returns)
            # would otherwise land a list/tuple here and crash the whole
            # request at the join() below. Degrade to a visible failure
            # instead of a hard crash.
            translated_value = str(translated_value)
            ok = False
        if ok:
            translated_chunks.append(translated_value)
        else:
            failed_chunks += 1
            translated_chunks.append(
                f">>> COULD NOT TRANSLATE THIS SECTION (shown untranslated below) >>>\n"
                f"{translated_value}\n"
                f"<<< END UNTRANSLATED SECTION <<<"
            )

    translated_text = "\n\n".join(translated_chunks)
    if failed_chunks:
        translated_text = (
            f"⚠️ {failed_chunks} of {len(chunks)} section(s) could not be translated and are shown "
            f"in their ORIGINAL, untranslated form below (marked with >>> / <<<). This almost always "
            f"means the extracted text for those sections was garbled — usually poor OCR quality on "
            f"a scanned page, rather than a translation problem. Check the OCR engine used below, and "
            f"whether the source document/scan quality is high enough.\n\n"
        ) + translated_text

    summary = "(not requested)"
    if summarize:
        progress(1.0, desc="Summarizing with AI...")
        messages = [
            {
                "role": "system",
                "content": (
                    f"You are a precise document summarizer. Respond ONLY in "
                    f"{target_lang}, matching the language of the text you are "
                    f"given -- never switch to a different language."
                ),
            },
            {"role": "user", "content": f"Summarize this in 3-5 bullet points:\n\n{translated_text}"},
        ]
        try:
            summary = _chat_backend(messages)
        except RuntimeError as e:
            summary = f"⚠️ Summary failed: {e}"

    is_rtl = _is_rtl(target_lang)
    display_text = _anchor_rtl_lines(translated_text) if is_rtl else translated_text
    display_summary = (
        gr.update(value=_anchor_rtl_lines(summary), rtl=True) if (is_rtl and summarize)
        else gr.update(value=summary, rtl=False)
    )
    return gr.update(value=display_text, rtl=is_rtl), ocr_engine, display_summary


def convert_to_word(file, hebrew_doc, progress=gr.Progress()):
    progress(0.15, desc="Extracting text (running OCR if needed)...")
    with open(file.name, "rb") as f:
        resp = requests.post(
            f"{BACKEND_URL}/convert-to-word", files={"file": f}, data={"hebrew": hebrew_doc}
        )
    resp.raise_for_status()
    progress(0.9, desc="Saving Word document...")

    out_name = Path(file.name).stem + ".docx"
    out_path = str(Path(tempfile.gettempdir()) / out_name)
    with open(out_path, "wb") as out_f:
        out_f.write(resp.content)
    progress(1.0, desc="Done")

    # Auto-detection (document.py's resolve_hebrew_flag) can route this
    # through Tesseract even if hebrew_doc was left unchecked -- surfaced
    # via a response header since FileResponse can't carry a JSON body.
    hebrew_used = resp.headers.get("X-Hebrew-OCR-Used", str(hebrew_doc)) == "True"
    ocr_engine = "Tesseract (Hebrew)" if hebrew_used else "RapidOCR (default)"
    return out_path, ocr_engine


MAX_CONTEXT_CHARS = 6000  # keep injected document text within the model's comfortable context window


def extract_context_from_files(filepaths, hebrew=False):
    contexts = []
    for path in filepaths:
        with open(path, "rb") as f:
            resp = requests.post(
                f"{BACKEND_URL}/extract-text", files={"file": f}, data={"hebrew": hebrew}
            )
        resp.raise_for_status()
        data = resp.json()
        contexts.append(f"--- Content of {Path(path).name} ---\n{data['markdown']}")
    return "\n\n".join(contexts)


# Matches a request to CREATE a presentation, e.g. "make me a powerpoint
# about X", "generate a slide deck on Y", "can you build a pptx for Z".
# Requires both a presentation-ish noun AND a creation verb, so it doesn't
# fire on unrelated mentions of the word "presentation" or "slides" (e.g.
# "what should I say in my presentation tomorrow?").
_PPTX_NOUN_RE = re.compile(r"\b(power ?point|pptx|slide ?deck|slides?|presentation)\b", re.IGNORECASE)
_PPTX_VERB_RE = re.compile(
    r"\b(make|create|generate|build|write|prepare|put together|draft|produce)\b", re.IGNORECASE
)


def _is_pptx_request(text: str) -> bool:
    text = text or ""
    return bool(_PPTX_NOUN_RE.search(text)) and bool(_PPTX_VERB_RE.search(text))


def generate_presentation(prompt: str) -> str:
    """Calls the backend's /generate-pptx endpoint and saves the returned
    file to a fresh temp directory (per-call, so concurrent chats can't
    clobber each other's presentation.pptx)."""
    resp = requests.post(f"{BACKEND_URL}/generate-pptx", json={"prompt": prompt}, timeout=300)
    resp.raise_for_status()
    out_path = str(Path(tempfile.mkdtemp()) / "presentation.pptx")
    with open(out_path, "wb") as f:
        f.write(resp.content)
    return out_path


def chat_fn(message, history, hebrew_doc=False):
    """
    message is a dict {"text": str, "files": [filepaths]} because the
    ChatInterface below is configured with multimodal=True. hebrew_doc
    comes from the additional_inputs checkbox added to the ChatInterface.
    """
    if isinstance(message, dict):
        user_text = message.get("text", "")
        files = message.get("files", []) or []
    else:
        user_text = message
        files = []

    file_context = ""
    if files:
        file_context = extract_context_from_files(files, hebrew=hebrew_doc)
        if len(file_context) > MAX_CONTEXT_CHARS:
            file_context = file_context[:MAX_CONTEXT_CHARS] + "\n[...truncated, file is longer...]"

    # PowerPoint generation is handled as its own branch rather than folded
    # into the normal /chat call: it needs a structured JSON outline from
    # the model (see app/pptx_generator.py), then a real .pptx file built
    # from that outline and returned as a download, not a chat reply string.
    if _is_pptx_request(user_text):
        prompt = user_text
        if file_context:
            prompt = f"{user_text}\n\nBase the slides on this source material:\n{file_context}"
        try:
            pptx_path = generate_presentation(prompt)
        except requests.HTTPError as e:
            return (
                "Sorry, I couldn't generate that presentation "
                f"({e}). Try rephrasing the topic, or try again."
            )
        return [
            "Here's the presentation you asked for — click below to download it:",
            gr.File(pptx_path),
        ]

    # Keep only plain-text turns from history — earlier attached files aren't
    # re-sent each turn (they already informed the answer they were attached to).
    clean_history = [
        {"role": turn["role"], "content": turn["content"]}
        for turn in history
        if isinstance(turn.get("content"), str)
    ]

    if file_context:
        combined_message = (
            f"The user attached the following document(s):\n\n{file_context}\n\n"
            f"User question: {user_text}"
        )
    else:
        combined_message = user_text

    messages = clean_history + [{"role": "user", "content": combined_message}]
    try:
        return _chat_backend(messages)
    except RuntimeError as e:
        return f"⚠️ {e}"


def legal_chat_fn(message, history, hebrew_doc=False):
    """
    Same shape as chat_fn, but routed to LEGAL_MODEL (DictaLM-3.0-24B-Thinking)
    with a light legal-assistant system prompt. Kept as a separate function
    (rather than parameterizing chat_fn) so the two tabs can diverge later
    without threading a model choice through the general Chat tab's UI.

    Returns (answer, citations_panel_markdown) -- the second value feeds the
    Legal tab's sidebar via gr.ChatInterface's additional_outputs, so any
    citation-shaped text in the answer shows up as its own list on the
    right instead of only being visible inline in the chat transcript.
    """
    if isinstance(message, dict):
        user_text = message.get("text", "")
        files = message.get("files", []) or []
    else:
        user_text = message
        files = []

    file_context = ""
    if files:
        file_context = extract_context_from_files(files, hebrew=hebrew_doc)
        if len(file_context) > MAX_CONTEXT_CHARS:
            file_context = file_context[:MAX_CONTEXT_CHARS] + "\n[...truncated, file is longer...]"

    clean_history = [
        {"role": turn["role"], "content": turn["content"]}
        for turn in history
        if isinstance(turn.get("content"), str)
    ]

    if file_context:
        combined_message = (
            f"The user attached the following document(s):\n\n{file_context}\n\n"
            f"User question: {user_text}"
        )
    else:
        combined_message = user_text

    messages = (
        [{"role": "system", "content": LEGAL_SYSTEM_PROMPT}]
        + clean_history
        + [{"role": "user", "content": combined_message}]
    )
    try:
        answer = _chat_backend(
            messages, model=LEGAL_MODEL, num_predict=_LEGAL_NUM_PREDICT,
            num_ctx=_LEGAL_NUM_CTX, timeout=_LEGAL_REQUEST_TIMEOUT_SECONDS,
            timeout_hint="try a narrower question -- e.g. ask about a specific section rather than a whole law.",
        )
    except RuntimeError as e:
        return f"⚠️ {e}", gr.skip()

    citations = _extract_citations(answer)
    if not citations:
        answer += _NO_CITATION_NOTE
    return answer, _format_citations_panel(citations)


def legal_chat_fn_1_7b(message, history, hebrew_doc=False):
    """
    Identical to legal_chat_fn, routed to LEGAL_MODEL_1_7B (DictaLM-3.0-1.7B-
    Thinking) instead of the 24B flagship. Kept as a fully separate function
    (rather than a model parameter on legal_chat_fn) so the "Attorney 24B"
    and "Attorney 1.7B" tabs stay independently editable, same reasoning as
    legal_chat_fn's own docstring for why it isn't folded into chat_fn.

    Reuses _LEGAL_NUM_PREDICT/_LEGAL_NUM_CTX/_LEGAL_REQUEST_TIMEOUT_SECONDS
    as-is -- these were sized for the slower 24B model, so they're safe
    (generous) upper bounds here too; the 1.7B will simply finish well
    inside them.

    Returns (answer, citations_panel_markdown) -- see legal_chat_fn.
    """
    if isinstance(message, dict):
        user_text = message.get("text", "")
        files = message.get("files", []) or []
    else:
        user_text = message
        files = []

    file_context = ""
    if files:
        file_context = extract_context_from_files(files, hebrew=hebrew_doc)
        if len(file_context) > MAX_CONTEXT_CHARS:
            file_context = file_context[:MAX_CONTEXT_CHARS] + "\n[...truncated, file is longer...]"

    clean_history = [
        {"role": turn["role"], "content": turn["content"]}
        for turn in history
        if isinstance(turn.get("content"), str)
    ]

    if file_context:
        combined_message = (
            f"The user attached the following document(s):\n\n{file_context}\n\n"
            f"User question: {user_text}"
        )
    else:
        combined_message = user_text

    messages = (
        [{"role": "system", "content": LEGAL_SYSTEM_PROMPT}]
        + clean_history
        + [{"role": "user", "content": combined_message}]
    )
    try:
        answer = _chat_backend(
            messages, model=LEGAL_MODEL_1_7B, num_predict=_LEGAL_NUM_PREDICT,
            num_ctx=_LEGAL_NUM_CTX, timeout=_LEGAL_REQUEST_TIMEOUT_SECONDS,
            timeout_hint="try a narrower question -- e.g. ask about a specific section rather than a whole law.",
        )
    except RuntimeError as e:
        return f"⚠️ {e}", gr.skip()

    citations = _extract_citations(answer)
    if not citations:
        answer += _NO_CITATION_NOTE
    return answer, _format_citations_panel(citations)


# --- Canon AI (RAG over the Codice di Diritto Canonico, Italian) -----------
#
# Unlike the Attorney tabs (which stuff an attached document straight into
# the prompt), this tab retrieves relevant canons from a local ChromaDB
# vector store -- built ahead of time by scrape_cic_it.py + embed_to_chroma.py
# -- and grounds the model's answer in whatever it actually retrieves.
#
# Path/collection name here MUST match what embed_to_chroma.py was run
# with (--chroma-dir / --collection). Adjust if you used different values.
CANON_CHROMA_DIR = str(Path(__file__).resolve().parent / "chroma_db")
CANON_COLLECTION_NAME = "cic_it"

# nomic-embed-text via Ollama -- same model/endpoint used to build the
# collection. Query-time text needs the "search_query: " prefix (NOT
# "search_document: ", which is what was used when indexing) -- nomic's
# model was trained with different prefixes for each side of retrieval,
# and mixing them up silently degrades results rather than erroring.
_CANON_OLLAMA_EMBED_URL = "http://localhost:11434/api/embeddings"
_CANON_EMBED_MODEL = "nomic-embed-text"
_CANON_QUERY_PREFIX = "search_query: "

# How many canons to retrieve per question. Higher = more coverage but
# more context spent on possibly-irrelevant canons.
_CANON_TOP_K = 5

# Model that answers using the retrieved canons. Left as None to use
# whatever the backend's own default chat model is (same as the plain
# Chat tab) -- set this to a specific Ollama model string (pulled and
# available to the backend) if you want Canon AI on its own dedicated
# model instead, e.g. one of the LEGAL_MODEL constants above.
CANON_MODEL = None
_CANON_NUM_PREDICT = 2048
_CANON_REQUEST_TIMEOUT_SECONDS = 900

CANON_SYSTEM_PROMPT = (
    "You are a canon lawyer -- an attorney specialized in the Roman "
    "Catholic Church's Codice di Diritto Canonico (Code of Canon Law). "
    "Think through and answer every question strictly according to the "
    "canons provided to you below, retrieved from the official Italian "
    "text on vatican.va -- not from general recall of canon law, and "
    "not from the law of any civil jurisdiction, unless the user "
    "explicitly asks about civil law.\n\n"
    "For every substantive claim, cite the specific canon(s) you are "
    "relying on immediately after the claim -- for example: 'A parish "
    "priest is removed only for a grave cause (Can. 1740).' Base your "
    "answer ONLY on the canons retrieved below plus general background "
    "knowledge of canon law's structure -- never invent a canon number "
    "or a rule that isn't in the excerpts you were given. If the "
    "retrieved canons don't actually answer the question, say so "
    "explicitly instead of guessing -- a wrong or fabricated citation is "
    "worse than admitting the retrieval didn't cover it.\n\n"
    "Keep your answer proportionate to the question: for a broad or "
    "general topic, cover the most important, directly relevant canons "
    "rather than exhaustively enumerating every retrieved excerpt -- you "
    "can offer to go deeper on a specific canon if the user wants that.\n\n"
    "Note where relevant that Book VI (Cann. 1311-1399) was fully "
    "reformed in 2021 ('Pascite Gregem Dei') -- flag if a retrieved Book "
    "VI canon might reflect the pre-2021 text.\n\n"
    "You are not a substitute for advice from a canon lawyer engaged by "
    "the person's diocese or tribunal, or for the guidance of competent "
    "Church authority, and you should say so when a question calls for "
    "one. Always reply in the same language the user's question is "
    "written in -- e.g. if they write in English, your entire answer "
    "must be in English, even though the retrieved canon text is in "
    "Italian. Never switch languages on the user unasked."
)

_CANON_SOURCES_PLACEHOLDER = (
    "_No canons retrieved yet. Once you ask a question, the specific "
    "canons retrieved from the vector database will be listed here._"
)
_CANON_SOURCES_NONE_FOUND = (
    "_Nothing was retrieved for this question -- the answer above (if any) "
    "isn't grounded in a specific canon. Try rephrasing._"
)
_CANON_DB_MISSING_MSG = (
    "⚠️ Canon AI's vector database isn't available. Make sure you've run "
    "scrape_cic_it.py then embed_to_chroma.py (pointed at "
    f"'{CANON_CHROMA_DIR}', collection '{CANON_COLLECTION_NAME}'), and "
    "that chromadb is installed (`pip install chromadb`)."
)

_canon_collection_cache = None


def _get_canon_collection():
    """Lazily connect to the persistent Chroma collection, caching the
    handle across calls. Returns None (rather than raising) on any
    failure -- callers surface a clean in-chat error instead of crashing
    the whole UI process if the vector DB hasn't been built yet."""
    global _canon_collection_cache
    if _canon_collection_cache is not None:
        return _canon_collection_cache
    if chromadb is None:
        return None
    try:
        client = chromadb.PersistentClient(path=CANON_CHROMA_DIR)
        _canon_collection_cache = client.get_collection(CANON_COLLECTION_NAME)
        return _canon_collection_cache
    except Exception as e:
        import traceback
        print(f"[Canon AI] failed to open Chroma collection at "
              f"'{CANON_CHROMA_DIR}' / '{CANON_COLLECTION_NAME}': {e}")
        traceback.print_exc()
        return None


def _embed_canon_query(text: str) -> list[float] | None:
    try:
        resp = requests.post(
            _CANON_OLLAMA_EMBED_URL,
            json={"model": _CANON_EMBED_MODEL, "prompt": _CANON_QUERY_PREFIX + text},
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()["embedding"]
    except Exception:
        return None


def retrieve_canons(query: str, n_results: int = _CANON_TOP_K):
    """
    Embeds the query with nomic-embed-text and retrieves the closest
    canons from ChromaDB. Returns (context_text, sources) where sources
    is a list of dicts used to build the sidebar panel -- or (None, None)
    on any failure (missing DB, unreachable Ollama, etc.), which callers
    turn into a clear in-chat message rather than a silent empty answer.
    """
    collection = _get_canon_collection()
    if collection is None:
        return None, None

    query_vector = _embed_canon_query(query)
    if query_vector is None:
        return None, None

    try:
        results = collection.query(query_embeddings=[query_vector], n_results=n_results)
    except Exception:
        return None, None

    documents = (results.get("documents") or [[]])[0]
    metadatas = (results.get("metadatas") or [[]])[0]

    if not documents:
        return "", []

    context_parts = []
    sources = []
    for doc, meta in zip(documents, metadatas):
        canon_num = meta.get("canon_number", "?")
        path = meta.get("hierarchy_path", "")
        url = meta.get("source_url", "")
        note = meta.get("in_force_note", "")
        context_parts.append(f"[Can. {canon_num}] ({path})\n{doc}")
        sources.append({"canon_number": canon_num, "hierarchy_path": path, "source_url": url, "note": note})

    context_text = "\n\n---\n\n".join(context_parts)
    return context_text, sources


def _format_canon_sources_panel(sources: list[dict] | None) -> str:
    if not sources:
        return _CANON_SOURCES_NONE_FOUND
    lines = []
    for s in sources:
        line = f"- **Can. {s['canon_number']}** — {s['hierarchy_path']}"
        if s.get("source_url"):
            line += f"  \n  [{s['source_url']}]({s['source_url']})"
        if s.get("note"):
            line += f"  \n  ⚠️ {s['note']}"
        lines.append(line)
    return "\n\n".join(lines)


def canon_chat_fn(message, history):
    """
    RAG chat over the scraped/embedded Codice di Diritto Canonico. Not
    multimodal (no file attachment) -- the whole point of this tab is
    grounding answers in the pre-built vector database rather than
    whatever the user happens to attach.

    Returns (answer, sources_panel_markdown) -- see canon_sources_panel's
    wiring below, same additional_outputs pattern as the Attorney tabs'
    citations panel.
    """
    user_text = message.get("text", message) if isinstance(message, dict) else message

    if chromadb is None or _get_canon_collection() is None:
        return _CANON_DB_MISSING_MSG, gr.skip()

    context_text, sources = retrieve_canons(user_text)
    if context_text is None:
        return _CANON_DB_MISSING_MSG, gr.skip()

    clean_history = [
        {"role": turn["role"], "content": turn["content"]}
        for turn in history
        if isinstance(turn.get("content"), str)
    ]

    if context_text:
        combined_message = (
            f"Retrieved canons relevant to the question:\n\n{context_text}\n\n"
            f"User question: {user_text}"
        )
    else:
        combined_message = (
            "No canons were retrieved for this question -- tell the user "
            f"that plainly instead of guessing.\n\nUser question: {user_text}"
        )

    messages = (
        [{"role": "system", "content": CANON_SYSTEM_PROMPT}]
        + clean_history
        + [{"role": "user", "content": combined_message}]
    )
    try:
        answer = _chat_backend(
            messages, model=CANON_MODEL, num_predict=_CANON_NUM_PREDICT,
            timeout=_CANON_REQUEST_TIMEOUT_SECONDS,
            timeout_hint="try a narrower question about a specific canon or topic.",
        )
    except RuntimeError as e:
        return f"⚠️ {e}", gr.skip()

    return answer, _format_canon_sources_panel(sources)


def set_hebrew_from_source_lang(source_lang):
    """
    Fires on the Translate tab's Source language dropdown. Hebrew/Tesseract
    routing is now fully automatic — there's no checkbox for the user to
    manage. Returns (is_hebrew, status_markdown_update).

    NOTE: assumes the LANGUAGES dict in app/translate.py uses the key
    "hebrew" (matching the lowercase convention "english"/"spanish" already
    used as defaults below) — double check that key against translate.py if
    this doesn't trigger correctly.
    """
    is_hebrew = (source_lang or "").strip().lower() == "hebrew"
    if is_hebrew:
        return True, gr.update(
            value="🔤 **Hebrew selected — Tesseract OCR will be used automatically** "
                  "(the only engine here with Hebrew support).",
            visible=True,
        )
    return False, gr.update(value="", visible=False)


LOGO_PATH = "app/assets/logo.png"  # put your logo file here, any size — it's auto-resized below

SUPPORTED_INVOICE_EXTS = {".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def split_pdf_to_pages(pdf_path, output_dir):
    """Splits a multi-page PDF into one single-page PDF per page, so each
    page is later OCR'd/classified as its own separate document."""
    reader = PdfReader(str(pdf_path))
    base_name = Path(pdf_path).stem
    page_paths = []
    for i, page in enumerate(reader.pages):
        writer = PdfWriter()
        writer.add_page(page)
        out_path = Path(output_dir) / f"{base_name}_page{i + 1}.pdf"
        with open(out_path, "wb") as f:
            writer.write(f)
        page_paths.append(out_path)
    return page_paths


def expand_pdfs_to_pages(files, work_dir):
    """Given a mixed list of files, splits every PDF into per-page PDFs
    and leaves image files untouched (they're already single-page)."""
    expanded = []
    for f in files:
        if f.suffix.lower() == ".pdf":
            expanded.extend(split_pdf_to_pages(f, work_dir))
        else:
            expanded.append(f)
    return expanded


def process_invoices(uploaded_file, company_name, hebrew_batch, progress=gr.Progress()):
    if uploaded_file is None:
        return None, "Please upload a ZIP or PDF file first."

    work_dir = tempfile.mkdtemp()
    suffix = Path(uploaded_file.name).suffix.lower()

    if suffix == ".zip":
        try:
            with zipfile.ZipFile(uploaded_file.name, "r") as zf:
                zf.extractall(work_dir)
        except zipfile.BadZipFile:
            return None, "That file doesn't look like a valid ZIP archive."
        raw_files = sorted(
            p for p in Path(work_dir).rglob("*")
            if p.is_file() and p.suffix.lower() in SUPPORTED_INVOICE_EXTS
        )
        files = expand_pdfs_to_pages(raw_files, work_dir)
    elif suffix == ".pdf":
        files = split_pdf_to_pages(Path(uploaded_file.name), work_dir)
    else:
        return None, f"Unsupported file type '{suffix}'. Upload a .zip or a .pdf."

    if not files:
        return None, "No supported files/pages found."

    sales_rows, expense_rows, unrecognized = [], [], []

    for i, fpath in enumerate(files):
        progress((i + 1) / len(files), desc=f"Processing {fpath.name} ({i + 1}/{len(files)})")
        try:
            with open(fpath, "rb") as f:
                extract_resp = requests.post(
                    f"{BACKEND_URL}/extract-text", files={"file": f}, data={"hebrew": hebrew_batch}
                )
            extract_resp.raise_for_status()
            markdown_text = extract_resp.json()["markdown"]

            classify_resp = requests.post(
                f"{BACKEND_URL}/classify-invoice",
                json={"markdown": markdown_text, "filename": fpath.name, "company_name": company_name},
            )
            classify_resp.raise_for_status()
            result = classify_resp.json()
        except Exception:
            unrecognized.append(fpath.name)
            continue

        doc_type = result.get("document_type", "unrecognized")
        row = [
            result.get("filename", fpath.name),
            result.get("date", ""),
            result.get("party_name", ""),
            result.get("invoice_number", ""),
            result.get("amount", 0),
            result.get("vat", 0),
            result.get("currency", ""),
        ]
        if doc_type == "sales":
            sales_rows.append(row)
        elif doc_type == "expense":
            expense_rows.append(row)
        else:
            unrecognized.append(fpath.name)

    # --- Build the Excel report ---
    progress(1.0, desc="Generating Excel report...")
    headers = ["File", "Date", "Party", "Invoice #", "Total", "VAT", "Currency"]

    wb = openpyxl.Workbook()
    ws_sales = wb.active
    ws_sales.title = "Sales"
    ws_sales.append(headers)
    for row in sales_rows:
        ws_sales.append(row)

    ws_expenses = wb.create_sheet("Expenses")
    ws_expenses.append(headers)
    for row in expense_rows:
        ws_expenses.append(row)

    for ws in (ws_sales, ws_expenses):
        for cell in ws[1]:
            cell.font = Font(bold=True)

    if unrecognized:
        ws_unrec = wb.create_sheet("Unrecognized")
        ws_unrec.append(["Filename"])
        for cell in ws_unrec[1]:
            cell.font = Font(bold=True)
        for name in unrecognized:
            ws_unrec.append([name])

    out_path = str(Path(tempfile.gettempdir()) / "accounting_report.xlsx")
    wb.save(out_path)

    total_sales = sum(r[4] for r in sales_rows if isinstance(r[4], (int, float)))
    total_expenses = sum(r[4] for r in expense_rows if isinstance(r[4], (int, float)))
    total_sales_vat = sum(r[5] for r in sales_rows if isinstance(r[5], (int, float)))
    total_expenses_vat = sum(r[5] for r in expense_rows if isinstance(r[5], (int, float)))

    summary = (
        f"Processed {len(files)} page(s)/file(s): {len(sales_rows)} sales, "
        f"{len(expense_rows)} expenses, {len(unrecognized)} unrecognized.\n"
        f"Total sales: {total_sales} (VAT: {total_sales_vat}) | "
        f"Total expenses: {total_expenses} (VAT: {total_expenses_vat})"
    )
    if unrecognized:
        summary += "\n\nNot recognized (check these manually):\n" + "\n".join(f"- {n}" for n in unrecognized)

    return out_path, summary


with gr.Blocks(title=" Ibrahim Zananiri- AI Employee") as demo:
    with gr.Row():
        gr.Image(
            value=LOGO_PATH,
            show_label=False,
            container=False,
            height=60,
            width=60,
            scale=0,
            show_fullscreen_button=False,
            show_download_button=False,
            interactive=False,
        )
        gr.Markdown("# Clara - LPJ AI Agent \n### By Ibrahim Zananiri")

    with gr.Tab("Translate"):
        with gr.Row():
            with gr.Column(scale=3):
                file_in = gr.File(label="Upload document (PDF, DOCX, PPTX, image)")
                with gr.Row():
                    src = gr.Dropdown(choices=list(LANGUAGES.keys()), label="Source language", value="english")
                    tgt = gr.Dropdown(choices=list(LANGUAGES.keys()), label="Target language", value="spanish")
                    summarize = gr.Checkbox(label="Also summarize with AI")
                hebrew_doc_translate = gr.State(value=False)
                hebrew_status_translate = gr.Markdown(value="", visible=False)
                src.change(
                    set_hebrew_from_source_lang,
                    inputs=[src],
                    outputs=[hebrew_doc_translate, hebrew_status_translate],
                )
                run_btn = gr.Button("Translate")
                output_text = gr.Textbox(
                    label="Translated text", lines=20, rtl=_is_rtl(tgt.value), show_copy_button=True,
                )
                tgt.change(
                    lambda target_lang: gr.update(rtl=_is_rtl(target_lang)),
                    inputs=[tgt],
                    outputs=[output_text],
                )
                ocr_engine_out = gr.Textbox(label="OCR engine (used if the document needed OCR)", interactive=False)
                output_summary = gr.Textbox(label="Summary (if requested)", lines=5)
                run_btn.click(
                    process,
                    inputs=[file_in, src, tgt, summarize, hebrew_doc_translate],
                    outputs=[output_text, ocr_engine_out, output_summary],
                )

            with gr.Column(scale=2, visible=False) as preview_column:
                gr.Markdown("### Document preview")
                preview_image = gr.Image(
                    label="Preview", visible=False, interactive=False,
                    show_fullscreen_button=True, height=520,
                )
                preview_pdf_html = gr.HTML(visible=False)
                preview_text = gr.Textbox(
                    label="Preview", visible=False, lines=20, interactive=False,
                )
                file_in.change(
                    build_preview,
                    inputs=[file_in],
                    outputs=[preview_column, preview_image, preview_pdf_html, preview_text],
                )

    with gr.Tab("Convert to Word"):
        gr.Markdown("Upload a PDF (native or scanned) or an image — text is extracted "
                    "(with OCR if needed) and exported as a .docx file.")
        convert_file_in = gr.File(label="Upload PDF or image")
        hebrew_doc_convert = gr.Checkbox(
            label="Document is in Hebrew (uses Tesseract OCR instead of the default engine)"
        )
        convert_btn = gr.Button("Convert to Word")
        convert_output = gr.File(label="Download .docx")
        convert_ocr_engine_out = gr.Textbox(label="OCR engine actually used", interactive=False)
        convert_btn.click(
            convert_to_word,
            inputs=[convert_file_in, hebrew_doc_convert],
            outputs=[convert_output, convert_ocr_engine_out],
        )

    with gr.Tab("Chat"):
        gr.Markdown("Chat with the local AI model. Attach a PDF, DOCX, PPTX, or image "
                    "and ask questions about it — text (with OCR if needed) is extracted "
                    "and given to the model as context.\n\n"
                    "💡 Ask it to **\"make/create/generate a PowerPoint (presentation/slide "
                    "deck) about ...\"** and it will build a downloadable .pptx file — attach "
                    "a document first if you want the slides based on that document.")
        chat_hebrew_checkbox = gr.Checkbox(
            label="Attached document is in Hebrew (uses Tesseract OCR instead of the default engine)"
        )
        gr.ChatInterface(
            fn=chat_fn, type="messages", multimodal=True,
            additional_inputs=[chat_hebrew_checkbox],
        )

    with gr.Tab("Accountant"):
        gr.Markdown(
            "Upload a **ZIP** or a **PDF** containing scans of sales invoices and expense "
            "receipts (PDF pages, PNG, JPG, TIFF, BMP). Each page/file is OCR'd and "
            "classified independently as a sale or an expense, and an Excel report is "
            "generated with both broken out on separate sheets, including totals and VAT.\n\n"
            "⚠️ **Each page must contain exactly one invoice or receipt.** Multi-page "
            "invoices (a single invoice spanning 2+ pages) are not supported — every "
            "page is treated as its own separate document, so a multi-page invoice will "
            "be split and counted incorrectly.\n\n"
            "⚠️ **The Hebrew checkbox applies to the whole batch.** If a single ZIP mixes "
            "Hebrew and non-Hebrew documents, process them in two separate batches for "
            "best accuracy."
        )
        company_name_in = gr.Textbox(
            label="Ibrahim - AI Employee",
            placeholder="e.g. Acme Corp — helps tell sales invoices from expense invoices",
        )
        zip_in = gr.File(label="Upload ZIP or PDF of invoice/receipt scans", file_types=[".zip", ".pdf"])
        hebrew_batch_in = gr.Checkbox(
            label="These documents are in Hebrew (uses Tesseract OCR instead of the default engine)"
        )
        process_btn = gr.Button("Process Invoices", variant="primary")
        report_out = gr.File(label="Download Excel Report")
        summary_out = gr.Textbox(label="Summary / Documents not recognized", lines=10)

        process_btn.click(
            process_invoices,
            inputs=[zip_in, company_name_in, hebrew_batch_in],
            outputs=[report_out, summary_out],
        )

    with gr.Tab("Attorney 24B"):
        gr.Markdown(
            "Chat with the local Hebrew-legal model "
            "(**DictaLM-3.0-24B-Thinking**, served via Ollama). Attach a PDF, DOCX, "
            "PPTX, or image and ask questions about it — text (with OCR if needed) "
            "is extracted and given to the model as context.\n\n"
            "⚠️ This is not a substitute for advice from a licensed attorney. "
            "Citations to specific laws, sections, or cases should be independently "
            "verified — small local models can occasionally cite a law or section "
            "that doesn't actually exist."
        )
        legal_hebrew_checkbox = gr.Checkbox(
            label="Attached document is in Hebrew (uses Tesseract OCR instead of the default engine)"
        )
        # Defined here (render=False) so it can be passed to ChatInterface's
        # additional_outputs below, then actually placed in the right-hand
        # column further down -- gr.ChatInterface requires additional_outputs
        # to already exist in the same Blocks scope, but we want it to render
        # in the sidebar, not wherever ChatInterface would put it by default.
        legal_citations_panel = gr.Markdown(_CITATIONS_PLACEHOLDER, render=False)
        with gr.Row():
            with gr.Column(scale=2):
                gr.ChatInterface(
                    fn=legal_chat_fn, type="messages", multimodal=True,
                    additional_inputs=[legal_hebrew_checkbox],
                    additional_outputs=[legal_citations_panel],
                )
            with gr.Column(scale=1):
                gr.Markdown("### 📚 Citations found")
                legal_citations_panel.render()

    with gr.Tab("Attorney 1.7B"):
        gr.Markdown(
            "Chat with the local Hebrew-legal model "
            "(**DictaLM-3.0-1.7B-Thinking**, served via Ollama). Attach a PDF, DOCX, "
            "PPTX, or image and ask questions about it — text (with OCR if needed) "
            "is extracted and given to the model as context.\n\n"
            "⚠️ This is not a substitute for advice from a licensed attorney. "
            "Citations to specific laws, sections, or cases should be independently "
            "verified — small local models can occasionally cite a law or section "
            "that doesn't actually exist. This is the smaller/faster 1.7B model, so "
            "it's weaker on legal reasoning and citation accuracy than Attorney 24B."
        )
        legal_hebrew_checkbox_1_7b = gr.Checkbox(
            label="Attached document is in Hebrew (uses Tesseract OCR instead of the default engine)"
        )
        legal_citations_panel_1_7b = gr.Markdown(_CITATIONS_PLACEHOLDER, render=False)
        with gr.Row():
            with gr.Column(scale=2):
                gr.ChatInterface(
                    fn=legal_chat_fn_1_7b, type="messages", multimodal=True,
                    additional_inputs=[legal_hebrew_checkbox_1_7b],
                    additional_outputs=[legal_citations_panel_1_7b],
                )
            with gr.Column(scale=1):
                gr.Markdown("### 📚 Citations found")
                legal_citations_panel_1_7b.render()

    with gr.Tab("Canon AI"):
        gr.Markdown(
            "Chat about the **Codice di Diritto Canonico** (Code of Canon "
            "Law), grounded in a local vector database built from the "
            "official Italian text on vatican.va (RAG — Retrieval-"
            "Augmented Generation via ChromaDB + nomic-embed-text).\n\n"
            "⚠️ Answers are only as good as what's retrieved. This is not "
            "a substitute for a canon lawyer or the guidance of competent "
            "Church authority. Always verify against the actual canon "
            "text (linked in Sources) for anything consequential."
        )
        canon_sources_panel = gr.Markdown(_CANON_SOURCES_PLACEHOLDER, render=False)
        with gr.Row():
            with gr.Column(scale=2):
                gr.ChatInterface(
                    fn=canon_chat_fn, type="messages",
                    additional_outputs=[canon_sources_panel],
                )
            with gr.Column(scale=1):
                gr.Markdown("### 📖 Canons retrieved")
                canon_sources_panel.render()

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)