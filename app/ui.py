"""
Gradio UI — talks to the FastAPI backend at localhost:8000.
Run after main.py is already running: python -m app.ui
"""
import tempfile
import zipfile
from pathlib import Path

import requests
import gradio as gr
import openpyxl
from openpyxl.styles import Font
from pypdf import PdfReader, PdfWriter

from app.translate import LANGUAGES
from app.document import chunk_text

BACKEND_URL = "http://localhost:8000"

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


def process(file, source_lang, target_lang, summarize, hebrew_doc, progress=gr.Progress()):
    # Deterministic mapping already enforced in document.py: hebrew=True
    # always routes through Tesseract, hebrew=False through RapidOCR — so
    # this can be reported directly without a round trip to the backend.
    ocr_engine = "Tesseract (Hebrew)" if hebrew_doc else "RapidOCR (default)"

    progress(0, desc="Extracting text from document (OCR if needed)...")
    with open(file.name, "rb") as f:
        extract_resp = requests.post(
            f"{BACKEND_URL}/extract-text", files={"file": f}, data={"hebrew": hebrew_doc}
        )
    extract_resp.raise_for_status()
    markdown_text = extract_resp.json()["markdown"]

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
        resp = requests.post(f"{BACKEND_URL}/chat", json={"messages": messages})
        resp.raise_for_status()
        summary = resp.json()["content"]

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
    return out_path


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
    resp = requests.post(f"{BACKEND_URL}/chat", json={"messages": messages})
    resp.raise_for_status()
    return resp.json()["content"]


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


with gr.Blocks(title=" Ibrahim - AI Employee") as demo:
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
        gr.Markdown("# Ibrahim - AI Employee \n### Multi Purpose AI Agent")

    with gr.Tab("Translate"):
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
        output_text = gr.Textbox(label="Translated text", lines=20, rtl=_is_rtl(tgt.value))
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

    with gr.Tab("Convert to Word"):
        gr.Markdown("Upload a PDF (native or scanned) or an image — text is extracted "
                    "(with OCR if needed) and exported as a .docx file.")
        convert_file_in = gr.File(label="Upload PDF or image")
        hebrew_doc_convert = gr.Checkbox(
            label="Document is in Hebrew (uses Tesseract OCR instead of the default engine)"
        )
        convert_btn = gr.Button("Convert to Word")
        convert_output = gr.File(label="Download .docx")
        convert_btn.click(
            convert_to_word, inputs=[convert_file_in, hebrew_doc_convert], outputs=[convert_output]
        )

    with gr.Tab("Chat"):
        gr.Markdown("Chat with the local AI model. Attach a PDF, DOCX, PPTX, or image "
                    "and ask questions about it — text (with OCR if needed) is extracted "
                    "and given to the model as context.")
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

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)