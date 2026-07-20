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

    return html.unescape("\n\n".join(page_texts))


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
    return html.unescape(result.document.export_to_markdown())


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
    return output_path


def convert_file_to_docx(input_path: str, output_path: str, hebrew: bool = False) -> str:
    """
    Full pipeline: any supported input (PDF, image, PPTX, HTML...) -> Word.
    This is what backs the PDF->Word and Image(OCR)->Word conversion endpoints.
    """
    markdown_text = convert_to_markdown(input_path, hebrew=hebrew)
    return export_docx(markdown_text, output_path)


_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\u00c0-\u024f])")


def _split_into_sentences(paragraph: str) -> list[str]:
    """Best-effort sentence splitter. Not perfect (abbreviations, decimals),
    but good enough to keep chunks short -- which matters far more than
    perfect boundaries for translation reliability."""
    sentences = _SENTENCE_SPLIT_RE.split(paragraph.strip())
    return [s.strip() for s in sentences if s.strip()]


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
            if len(current) + len(unit) > max_chars and current:
                flush()
            current += unit + " "

    flush()
    return chunks