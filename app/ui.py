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


def process(file, source_lang, target_lang, summarize, progress=gr.Progress()):
    progress(0, desc="Extracting text from document (OCR if needed)...")
    with open(file.name, "rb") as f:
        extract_resp = requests.post(f"{BACKEND_URL}/extract-text", files={"file": f})
    extract_resp.raise_for_status()
    markdown_text = extract_resp.json()["markdown"]

    chunks = chunk_text(markdown_text)
    if not chunks:
        return "(no text found in document)", "(not requested)"

    total_steps = len(chunks) + (1 if summarize else 0)
    translated_chunks = []
    for i, chunk in enumerate(chunks):
        progress((i + 1) / total_steps, desc=f"Translating chunk {i + 1}/{len(chunks)}...")
        resp = requests.post(
            f"{BACKEND_URL}/translate-chunk",
            json={"text": chunk, "source_lang": source_lang, "target_lang": target_lang},
        )
        resp.raise_for_status()
        translated_chunks.append(resp.json()["translated"])

    translated_text = "\n\n".join(translated_chunks)

    summary = "(not requested)"
    if summarize:
        progress(1.0, desc="Summarizing with Ollama...")
        messages = [
            {"role": "system", "content": "You are a precise document summarizer."},
            {"role": "user", "content": f"Summarize this in 3-5 bullet points:\n\n{translated_text}"},
        ]
        resp = requests.post(f"{BACKEND_URL}/chat", json={"messages": messages})
        resp.raise_for_status()
        summary = resp.json()["content"]

    return translated_text, summary


def convert_to_word(file, progress=gr.Progress()):
    progress(0.15, desc="Extracting text (running OCR if needed)...")
    with open(file.name, "rb") as f:
        resp = requests.post(f"{BACKEND_URL}/convert-to-word", files={"file": f})
    resp.raise_for_status()
    progress(0.9, desc="Saving Word document...")

    out_name = Path(file.name).stem + ".docx"
    out_path = str(Path(tempfile.gettempdir()) / out_name)
    with open(out_path, "wb") as out_f:
        out_f.write(resp.content)
    progress(1.0, desc="Done")
    return out_path


MAX_CONTEXT_CHARS = 6000  # keep injected document text within the model's comfortable context window


def extract_context_from_files(filepaths):
    contexts = []
    for path in filepaths:
        with open(path, "rb") as f:
            resp = requests.post(f"{BACKEND_URL}/extract-text", files={"file": f})
        resp.raise_for_status()
        data = resp.json()
        contexts.append(f"--- Content of {Path(path).name} ---\n{data['markdown']}")
    return "\n\n".join(contexts)


def chat_fn(message, history):
    """
    message is a dict {"text": str, "files": [filepaths]} because the
    ChatInterface below is configured with multimodal=True.
    """
    if isinstance(message, dict):
        user_text = message.get("text", "")
        files = message.get("files", []) or []
    else:
        user_text = message
        files = []

    file_context = ""
    if files:
        file_context = extract_context_from_files(files)
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


def process_invoices(uploaded_file, company_name, progress=gr.Progress()):
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
                extract_resp = requests.post(f"{BACKEND_URL}/extract-text", files={"file": f})
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


with gr.Blocks(title="Latin Patriarchate of Jerusalem — Offline Translator") as demo:
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
        gr.Markdown("# Latin Patriarchate of Jerusalem \n### LPJ AI Employee")

    with gr.Tab("Translate"):
        file_in = gr.File(label="Upload document (PDF, DOCX, PPTX, image)")
        with gr.Row():
            src = gr.Dropdown(choices=list(LANGUAGES.keys()), label="Source language", value="english")
            tgt = gr.Dropdown(choices=list(LANGUAGES.keys()), label="Target language", value="spanish")
            summarize = gr.Checkbox(label="Summarize with AI")
        run_btn = gr.Button("Translate")
        output_text = gr.Textbox(label="Translated text", lines=20)
        output_summary = gr.Textbox(label="Summary (if requested)", lines=5)
        run_btn.click(process, inputs=[file_in, src, tgt, summarize], outputs=[output_text, output_summary])

    with gr.Tab("Convert to Word"):
        gr.Markdown("Upload a PDF (native or scanned) or an image — text is extracted "
                    "(with OCR if needed) and exported as a .docx file.")
        convert_file_in = gr.File(label="Upload PDF or image")
        convert_btn = gr.Button("Convert to Word")
        convert_output = gr.File(label="Download .docx")
        convert_btn.click(convert_to_word, inputs=[convert_file_in], outputs=[convert_output])

    with gr.Tab("Chat"):
        gr.Markdown("Chat with the local Ollama model. Attach a PDF, DOCX, PPTX, or image "
                    "and ask questions about it — text (with OCR if needed) is extracted "
                    "and given to the model as context.")
        gr.ChatInterface(fn=chat_fn, type="messages", multimodal=True)

    with gr.Tab("Accountant"):
        gr.Markdown(
            "Upload a **ZIP** or a **PDF** containing scans of sales invoices and expense "
            "receipts (PDF pages, PNG, JPG, TIFF, BMP). Each page/file is OCR'd and "
            "classified independently as a sale or an expense, and an Excel report is "
            "generated with both broken out on separate sheets, including totals and VAT.\n\n"
            "⚠️ **Each page must contain exactly one invoice or receipt.** Multi-page "
            "invoices (a single invoice spanning 2+ pages) are not supported — every "
            "page is treated as its own separate document, so a multi-page invoice will "
            "be split and counted incorrectly."
        )
        company_name_in = gr.Textbox(
            label="Your company/business name",
            placeholder="e.g. Acme Corp — helps tell sales invoices from expense invoices",
        )
        zip_in = gr.File(label="Upload ZIP or PDF of invoice/receipt scans", file_types=[".zip", ".pdf"])
        process_btn = gr.Button("Process Invoices", variant="primary")
        report_out = gr.File(label="Download Excel Report")
        summary_out = gr.Textbox(label="Summary / Documents not recognized", lines=10)

        process_btn.click(
            process_invoices,
            inputs=[zip_in, company_name_in],
            outputs=[report_out, summary_out],
        )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
