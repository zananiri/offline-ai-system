"""
Document conversion + OCR via Docling. Handles PDF, DOCX, PPTX, images, etc.
and returns clean Markdown, chunked for translation, or exports to Word.

Two separate OCR paths are used, deliberately:
- RapidOCR via Docling (default) — better accuracy on real-world scans,
  built-in table detection, handles skewed/photographed documents well.
  This is what every document uses unless told otherwise.
- Direct Tesseract, bypassing Docling (Hebrew only) — RapidOCR (and EasyOCR)
  have no Hebrew model at all, so Tesseract is the only engine here that
  can read Hebrew script. Docling's own Tesseract integration
  (TesseractCliOcrOptions) has a known, currently unresolved upstream bug:
  it writes the rendered page to a temp image file without embedding DPI
  metadata, so Tesseract can't determine the resolution, guesses a useless
  70 DPI, and aborts the page ("Too few characters. Skipping this page").
  See docling-project/docling-serve#282 and open-webui#15952 — this is not
  fixable via images_scale or any other documented Docling option. Instead,
  Hebrew documents are rendered to images ourselves (via pypdfium2) and OCR'd
  by calling Tesseract directly (via pytesseract) with an explicit DPI, which
  sidesteps the auto-detection failure entirely.

  Trade-off: the Hebrew path returns plain paragraph text per page, not
  Docling's structured Markdown (no table detection). Fine for translation
  and chat use; a Hebrew document with heavy tables will lose that structure.
"""
import html
import os
import re
import shutil
from pathlib import Path

import pypandoc
import pypdfium2 as pdfium
import pytesseract
from PIL import Image
from docx import Document as DocxDocument
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions, RapidOcrOptions
from docling.datamodel.base_models import InputFormat

# Tesseract (the OCR *engine*, not the pytesseract Python wrapper) is a
# separate program that must be installed on the machine and discoverable
# on PATH. pytesseract just shells out to it. If it's not on PATH -- a very
# common setup gap on Windows, especially right after installing it without
# restarting the terminal -- pytesseract raises TesseractNotFoundError deep
# inside a subprocess call, which otherwise surfaces as a raw, unhelpful
# 500 traceback to the user. TESSERACT_CMD lets you point at an explicit
# install location if PATH still doesn't pick it up for some reason (e.g.
# installed for a different Windows user, or a portable install).
_TESSERACT_CMD_OVERRIDE = os.environ.get("TESSERACT_CMD")
if _TESSERACT_CMD_OVERRIDE:
    pytesseract.pytesseract.tesseract_cmd = _TESSERACT_CMD_OVERRIDE


def _tesseract_available() -> bool:
    cmd = pytesseract.pytesseract.tesseract_cmd
    # tesseract_cmd defaults to just "tesseract" (relies on PATH) unless
    # overridden above with an explicit full path.
    return shutil.which(cmd) is not None or Path(cmd).is_file()


class TesseractNotAvailableError(RuntimeError):
    """Raised when Hebrew OCR is requested but Tesseract isn't installed/found."""

# --- Default converter: RapidOCR via Docling (higher accuracy, no Hebrew support) ---
_default_pipeline_options = PdfPipelineOptions()
_default_pipeline_options.do_ocr = True
# Docling's default images_scale=1.0 renders pages at only 72 DPI, which is
# too low for reliable OCR. 3.0 -> ~216 DPI.
_default_pipeline_options.images_scale = 3.0
_default_pipeline_options.ocr_options = RapidOcrOptions()  # ONNX-based, light on CPU
_default_pipeline_options.do_table_structure = True

_default_converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(pipeline_options=_default_pipeline_options),
    }
)

# --- Hebrew OCR: direct Tesseract call, bypassing Docling's OCR pipeline entirely ---
_HEBREW_OCR_LANG = "heb+eng"
_HEBREW_OCR_DPI = 300
_OCR_INPUT_EXTS = {".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}

# Unicode bidi/formatting control characters: ZERO WIDTH SPACE / NON-JOINER /
# JOINER, LEFT-TO-RIGHT MARK, RIGHT-TO-LEFT MARK, the LTR/RTL/POP embedding
# & override marks, the newer directional isolates, and BOM.
_BIDI_CONTROL_CHARS_RE = re.compile("[\u200b-\u200f\u202a-\u202e\u2066-\u2069\ufeff]")


def strip_bidi_controls(text: str) -> str:
    """
    Strips invisible Unicode bidi/formatting control characters from OCR'd
    text.

    Tesseract's Hebrew OCR (lang="heb+eng") inserts these routinely to
    preserve correct reading order around embedded LTR runs (English words,
    dates, numbers) inside RTL text -- completely correct and invisible when
    the text is just *displayed*. But MADLAD-400's SentencePiece tokenizer
    treats them as ordinary characters, and when one lands mid-word it
    fragments tokenization badly enough that the model gives up translating
    and echoes the (still-Hebrew) input back unchanged instead. Observed in
    practice as a 100% "wrong language" rejection rate on Hebrew-sourced
    chunks. They carry no translatable meaning, so it's safe to drop them
    before the text is ever tokenized. See translate.py's _translate_once
    for the other half of this fix (it's applied there too, defensively).
    """
    if not text:
        return text
    return _BIDI_CONTROL_CHARS_RE.sub("", text)


def _render_pdf_pages(file_path: str, dpi: int = _HEBREW_OCR_DPI) -> list[Image.Image]:
    pdf = pdfium.PdfDocument(file_path)
    scale = dpi / 72  # pypdfium2's render scale is relative to a 72-DPI baseline
    images = []
    for page in pdf:
        bitmap = page.render(scale=scale)
        pil_image = bitmap.to_pil()
        pil_image.info["dpi"] = (dpi, dpi)  # embed DPI so Tesseract doesn't need to guess
        images.append(pil_image)
    return images


def _ocr_hebrew(file_path: str) -> str:
    if not _tesseract_available():
        raise TesseractNotAvailableError(
            "Hebrew OCR requires Tesseract, which isn't installed or isn't on PATH. "
            "Install it from https://github.com/UB-Mannheim/tesseract/wiki, make sure "
            "to check the Hebrew language pack during setup, then restart the backend "
            "(PATH changes don't apply to already-running processes/terminals). "
            "If it's installed somewhere PATH doesn't cover, set the TESSERACT_CMD "
            "environment variable to the full path of tesseract.exe instead."
        )

    suffix = Path(file_path).suffix.lower()
    if suffix == ".pdf":
        images = _render_pdf_pages(file_path)
    else:
        img = Image.open(file_path)
        img.info["dpi"] = (_HEBREW_OCR_DPI, _HEBREW_OCR_DPI)
        images = [img]

    page_texts = []
    for img in images:
        # --psm 3: fully automatic page segmentation, explicitly WITHOUT an
        # orientation/script-detection (OSD) sub-pass — OSD is exactly the
        # sub-step that fails with bad DPI in Docling's wrapper, so we skip
        # it here by design. --dpi is passed explicitly as a second safeguard
        # on top of the DPI already embedded in the image itself.
        text = pytesseract.image_to_string(
            img, lang=_HEBREW_OCR_LANG, config=f"--psm 3 --dpi {_HEBREW_OCR_DPI}"
        ).strip()
        if text:
            page_texts.append(text)

    joined = html.unescape("\n\n".join(page_texts))
    # Strip bidi/formatting marks right at the source, so every downstream
    # consumer (translation, chat context, summarization, the raw markdown
    # returned to the UI) sees clean text, not just the translation path.
    return strip_bidi_controls(joined)


def convert_to_markdown(file_path: str, hebrew: bool = False) -> str:
    """
    Convert any supported document (PDF/DOCX/PPTX/HTML/image) to Markdown.
    Set hebrew=True to OCR through Tesseract directly instead of Docling's
    default RapidOCR pipeline — only do this for documents actually in
    Hebrew. Only affects PDFs and raw images; native formats (DOCX/PPTX)
    never go through OCR at all, so the flag has no effect on them.
    """
    if hebrew and Path(file_path).suffix.lower() in _OCR_INPUT_EXTS:
        return _ocr_hebrew(file_path)

    result = _default_converter.convert(Path(file_path))
    markdown_text = html.unescape(result.document.export_to_markdown())
    # Defensive: same bidi-control-character cleanup as the Hebrew/Tesseract
    # path, applied here too in case RapidOCR ever emits stray marks on
    # Arabic (or any other RTL-script) content.
    return strip_bidi_controls(markdown_text)


def _paragraph_is_rtl(text: str, threshold: float = 0.3) -> bool:
    """
    A paragraph is treated as RTL if a meaningful share of its *letters*
    are Hebrew/Arabic -- not "contains any RTL character at all". That
    keeps a mostly-English paragraph that happens to mention one Hebrew
    name from getting force-flipped to right-aligned.
    """
    letters = [ch for ch in text if ch.isalpha()]
    if not letters:
        return False
    rtl_letters = sum(1 for ch in letters if 0x0590 <= ord(ch) <= 0x06FF)
    return (rtl_letters / len(letters)) >= threshold


def _set_paragraph_rtl(paragraph) -> None:
    """
    Marks a python-docx paragraph as right-to-left and right-aligned.

    pandoc's markdown->docx conversion has no concept of paragraph
    direction -- every paragraph comes out as an ordinary left-to-right
    Word paragraph regardless of script. Word's renderer still reorders
    individual Hebrew/Arabic glyphs correctly (that's the Unicode bidi
    algorithm, and it happens automatically), but without the paragraph's
    w:bidi flag set, Word (a) left-aligns the whole paragraph, which reads
    backwards to an RTL reader, and (b) can misplace weakly-directional
    runs -- numbers, dates, embedded Latin words -- relative to the
    surrounding Hebrew/Arabic text. That second failure mode is the same
    root problem ui.py's _anchor_rtl_lines works around for the Gradio
    textbox; here we set the real OOXML property instead, since a Word
    document actually has one.
    """
    paragraph.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    pPr = paragraph._p.get_or_add_pPr()
    bidi = OxmlElement("w:bidi")
    bidi.set(qn("w:val"), "1")
    pPr.append(bidi)
    for run in paragraph.runs:
        rPr = run._r.get_or_add_rPr()
        rtl = OxmlElement("w:rtl")
        rtl.set(qn("w:val"), "1")
        rPr.append(rtl)


def _apply_rtl_formatting(docx_path: str) -> None:
    """
    Walks every paragraph in the generated docx -- including inside tables,
    since invoices/receipts routinely have RTL table content -- and flips
    direction on any paragraph that's predominantly Hebrew/Arabic.
    """
    doc = DocxDocument(docx_path)
    changed = False

    def _process(paragraphs):
        nonlocal changed
        for p in paragraphs:
            if _paragraph_is_rtl(p.text):
                _set_paragraph_rtl(p)
                changed = True

    _process(doc.paragraphs)
    for table in doc.tables:
        for row in table.rows:
            for cell in row.cells:
                _process(cell.paragraphs)

    if changed:
        doc.save(docx_path)


def export_docx(markdown_text: str, output_path: str) -> str:
    """
    Convert Markdown text to a .docx file via pandoc.
    Used for: PDF -> Word, and OCR'd image -> Word.
    Returns the output_path for convenience.
    """
    pypandoc.convert_text(
        markdown_text,
        to="docx",
        format="md",
        outputfile=output_path,
    )
    # pandoc's output is direction-agnostic (see _apply_rtl_formatting's
    # docstring) -- fix up any Hebrew/Arabic paragraphs after the fact.
    _apply_rtl_formatting(output_path)
    return output_path


def convert_file_to_docx(input_path: str, output_path: str, hebrew: bool = False) -> str:
    """
    Full pipeline: any supported input (PDF, image, PPTX, HTML...) -> Word.
    This is what backs the PDF->Word and Image(OCR)->Word conversion endpoints.
    """
    markdown_text = convert_to_markdown(input_path, hebrew=hebrew)
    return export_docx(markdown_text, output_path)


# Matches a sentence boundary: one of .!? followed by whitespace, followed
# by a character that plausibly starts a new sentence.
#
# IMPORTANT: the lookahead character class used to be Latin-only
# (A-Z, 0-9, and accented Latin). That meant it NEVER matched inside
# Hebrew, Arabic, Cyrillic, or CJK text -- there's no such thing as an
# "uppercase" Hebrew letter, so a period-plus-Hebrew-letter never looked
# like a sentence boundary. The practical effect: every long Hebrew
# paragraph was chunked as a single oversized, unsplit block (observed:
# 100-180 word chunks against an intended ~400-char/60-90-word target),
# and translate.py's own sentence-retry fallback degraded to a no-op for
# the same reason (a "sentence list" of length 1 can't be retried
# piecewise). MADLAD-400-3B is documented (see translate.py) to be much
# less reliable on long multi-sentence blocks than short ones, so this
# was the primary cause of Hebrew-source translations failing on every
# single chunk. Fixed by recognizing common non-Latin scripts as valid
# sentence-starting characters too.
SENTENCE_SPLIT_RE = re.compile(
    r"(?<=[.!?])\s+(?=[A-Z0-9\u00c0-\u024f"  # Latin (original)
    r"\u0400-\u04ff"  # Cyrillic
    r"\u0590-\u05ff"  # Hebrew
    r"\u0600-\u06ff"  # Arabic
    r"\u4e00-\u9fff"  # CJK unified ideographs
    r"])"
)


def _split_into_sentences(paragraph: str) -> list[str]:
    """Best-effort sentence splitter. Not perfect (abbreviations, decimals),
    but good enough to keep chunks short -- which matters far more than
    perfect boundaries for translation reliability."""
    sentences = SENTENCE_SPLIT_RE.split(paragraph.strip())
    return [s.strip() for s in sentences if s.strip()]


def _hard_split(unit: str, max_chars: int) -> list[str]:
    """
    Force-splits an oversized unit on whitespace/word boundaries, greedily
    packing words up to max_chars.

    This is a last-resort safety net for chunk_text below: sentence
    splitting is now script-aware (see SENTENCE_SPLIT_RE) but can still
    hand back a single "sentence" that's still too long -- a genuine
    run-on sentence, text with no recognized sentence punctuation at all,
    or some future script/punctuation style this regex doesn't cover.
    Rather than silently letting an oversized chunk through again (which is
    exactly how the Hebrew bug above went unnoticed), every unit is now
    guaranteed to fit within max_chars by the time it reaches the model.
    """
    if len(unit) <= max_chars:
        return [unit]
    words = unit.split()
    if not words:
        return [unit]
    pieces, current = [], ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if len(candidate) > max_chars and current:
            pieces.append(current)
            current = word
        else:
            current = candidate
    if current:
        pieces.append(current)
    return pieces


def chunk_text(markdown_text: str, max_chars: int = 400) -> list[str]:
    """
    Paragraph- and sentence-aware chunking so translation chunks stay small.

    max_chars was lowered from 800 to 400: MADLAD-400-3B (int8) is noticeably
    less reliable translating long, multi-sentence blocks in one shot than
    short ones -- observed failure mode is the model translating correctly
    for a while, then degenerating into a repetition loop and drifting back
    into the source language partway through. Paragraphs longer than
    max_chars are now split into individual sentences (not just left as one
    oversized chunk), and sentences are packed back together up to the limit
    so short sentences still get batched efficiently.

    Sentence splitting is script-aware (see SENTENCE_SPLIT_RE) and, as a
    final safety net, _hard_split guarantees no single piece handed to the
    packer ever exceeds max_chars, regardless of script or punctuation.
    """
    paragraphs = [p for p in markdown_text.split("\n\n") if p.strip()]
    chunks, current = [], ""

    def flush():
        nonlocal current
        if current.strip():
            chunks.append(current.strip())
        current = ""

    for p in paragraphs:
        units = [p] if len(p) <= max_chars else _split_into_sentences(p)
        for unit in units:
            for piece in _hard_split(unit, max_chars):
                if len(current) + len(piece) > max_chars and current:
                    flush()
                current += piece + " "

    flush()
    return chunks