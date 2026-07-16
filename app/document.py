"""
Document conversion + OCR via Docling. Handles PDF, DOCX, PPTX, images, etc.
and returns clean Markdown, chunked for translation.
"""
from pathlib import Path
from docling.document_converter import DocumentConverter
from docling.datamodel.pipeline_options import PdfPipelineOptions, RapidOcrOptions
from docling.datamodel.base_models import InputFormat
from docling.document_converter import PdfFormatOption

_pipeline_options = PdfPipelineOptions()
_pipeline_options.do_ocr = True
_pipeline_options.ocr_options = RapidOcrOptions()  # ONNX-based, light on CPU
_pipeline_options.do_table_structure = True

_converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(pipeline_options=_pipeline_options),
    }
)


def convert_to_markdown(file_path: str) -> str:
    """Convert any supported document (PDF/DOCX/PPTX/HTML/image) to Markdown."""
    result = _converter.convert(Path(file_path))
    return result.document.export_to_markdown()


def chunk_text(markdown_text: str, max_chars: int = 800) -> list[str]:
    """Simple paragraph-aware chunking so translation stays within NLLB's comfortable range."""
    paragraphs = [p for p in markdown_text.split("\n\n") if p.strip()]
    chunks, current = [], ""
    for p in paragraphs:
        if len(current) + len(p) > max_chars and current:
            chunks.append(current.strip())
            current = ""
        current += p + "\n\n"
    if current.strip():
        chunks.append(current.strip())
    return chunks
