"""
Document conversion + OCR via Docling. Handles PDF, DOCX, PPTX, images, etc.
and returns clean Markdown, chunked for translation, or exports to Word.
"""
from pathlib import Path
import pypandoc
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


def convert_to_docx(input_path: str, output_path: str) -> str:
    """
    Convert any supported input (PDF, scanned PDF, or a standalone image via
    OCR) to a Word document. Docling already runs OCR automatically for
    scanned pages and standalone images (RapidOCR, configured above); this
    just takes that extracted Markdown and hands it to pandoc for the
    Markdown -> docx conversion, preserving headings/tables/lists.
    """
    import pypandoc

    markdown_text = convert_to_markdown(input_path)
    pypandoc.convert_text(
        markdown_text,
        to="docx",
        format="md",
        outputfile=output_path,
    )
    return output_path


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


def convert_file_to_docx(input_path: str, output_path: str) -> str:
    """
    Full pipeline: any supported input (PDF, image, PPTX, HTML...) -> Word.
    This is what backs the PDF->Word and Image(OCR)->Word conversion endpoints.
    """
    markdown_text = convert_to_markdown(input_path)
    return export_docx(markdown_text, output_path)


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
