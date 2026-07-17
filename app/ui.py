"""
Gradio UI — talks to the FastAPI backend at localhost:8000.
Run after main.py is already running: python -m app.ui
"""
import tempfile
from pathlib import Path

import requests
import gradio as gr

from app.translate import LANGUAGES

BACKEND_URL = "http://localhost:8000"


def process(file, source_lang, target_lang, summarize):
    with open(file.name, "rb") as f:
        resp = requests.post(
            f"{BACKEND_URL}/translate-document",
            files={"file": f},
            data={"source_lang": source_lang, "target_lang": target_lang, "summarize": summarize},
        )
    resp.raise_for_status()
    data = resp.json()
    return data["translated_text"], data.get("summary") or "(not requested)"


def convert_to_word(file):
    with open(file.name, "rb") as f:
        resp = requests.post(f"{BACKEND_URL}/convert-to-word", files={"file": f})
    resp.raise_for_status()

    out_name = Path(file.name).stem + ".docx"
    out_path = str(Path(tempfile.gettempdir()) / out_name)
    with open(out_path, "wb") as out_f:
        out_f.write(resp.content)
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

with gr.Blocks(title="LPJ — Offline Translator") as demo:
    with gr.Row():
        gr.Image(
            value=LOGO_PATH,
            show_label=False,
            container=False,
            height=60,
            width=60,
            scale=0,
        )
        gr.Markdown("# Latin Patriarchate of Jerusalem\n### Offline Document Translator + Converter")

    with gr.Tab("Translate"):
        file_in = gr.File(label="Upload document (PDF, DOCX, PPTX, image)")
        with gr.Row():
            src = gr.Dropdown(choices=list(LANGUAGES.keys()), label="Source language", value="english")
            tgt = gr.Dropdown(choices=list(LANGUAGES.keys()), label="Target language", value="spanish")
            summarize = gr.Checkbox(label="Also summarize with Ollama")
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

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
