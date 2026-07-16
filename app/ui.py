"""
Gradio UI — talks to the FastAPI backend at localhost:8000.
Run after main.py is already running: python app/ui.py
"""
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


with gr.Blocks(title="Offline Translator + Document OCR") as demo:
    gr.Markdown("# Offline Document Translator (NLLB-200 + Ollama)")
    with gr.Row():
        file_in = gr.File(label="Upload document (PDF, DOCX, PPTX, image)")
    with gr.Row():
        src = gr.Dropdown(choices=list(LANGUAGES.keys()), label="Source language", value="english")
        tgt = gr.Dropdown(choices=list(LANGUAGES.keys()), label="Target language", value="spanish")
        summarize = gr.Checkbox(label="Also summarize with Ollama")
    run_btn = gr.Button("Translate")
    output_text = gr.Textbox(label="Translated text", lines=20)
    output_summary = gr.Textbox(label="Summary (if requested)", lines=5)

    run_btn.click(process, inputs=[file_in, src, tgt, summarize], outputs=[output_text, output_summary])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
