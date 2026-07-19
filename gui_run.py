r"""
gui_run.py — GUI launcher for the offline AI system.

Starts Ollama, the FastAPI backend, and the Gradio UI, and shows each one's
live console output in its own tab, with a status indicator per service.
Closing the window stops all three processes.

Run with:
    .venv\Scripts\Activate.ps1
    python gui_run.py

Or double-click via a .bat wrapper (see gui_start.bat) to avoid opening a
console window alongside the GUI.
"""
import os
import queue
import subprocess
import sys
import threading
import time
import tkinter as tk
import urllib.request
import webbrowser
from pathlib import Path
from tkinter import scrolledtext, ttk

PROJECT_ROOT = Path(__file__).resolve().parent
BACKEND_URL = "http://localhost:8000"
UI_URL = "http://localhost:7860"

# ---------------------------------------------------------------------------
# Theme — orange & black, matching the Translate tab.
# NOTE: I don't have app/ui.py, so these hex values are my best match using
# Gradio's standard "orange" palette (primary #F97316) rather than an exact
# pull from your theme file. Share app/ui.py (or its theme/CSS block) if you
# want this pixel-matched, and I'll swap in the exact values/font.
# ---------------------------------------------------------------------------
FONT_FAMILY = "Segoe UI"

BG_MAIN = "#101010"      # window background (near-black)
BG_PANEL = "#1A1A1A"     # log pane / tab background
FG_TEXT = "#F2F2F2"      # primary text
ORANGE = "#F97316"       # primary accent (Gradio orange-500)
ORANGE_DARK = "#C2410C"  # pressed/hover accent
ORANGE_LIGHT = "#FDBA74" # subtle highlight

STATUS_COLORS = {
    "waiting": "#6B6B6B",
    "starting": ORANGE_DARK,
    "running": ORANGE,
    "error": "#E5484D",   # kept red for error — universally read as "bad",
                           # even inside an orange/black palette
}


class ServicePanel:
    """One tab: a log pane + status label for a single service."""

    def __init__(self, notebook, name):
        self.name = name
        self.frame = ttk.Frame(notebook, style="Panel.TFrame")
        notebook.add(self.frame, text=name)

        self.status_var = tk.StringVar(value="Waiting...")
        status_row = ttk.Frame(self.frame, style="Panel.TFrame")
        status_row.pack(fill="x", padx=8, pady=(8, 4))
        tk.Label(
            status_row, text=f"{name}:", font=(FONT_FAMILY, 10, "bold"),
            bg=BG_PANEL, fg=FG_TEXT,
        ).pack(side="left")
        self.status_label = tk.Label(
            status_row, textvariable=self.status_var,
            font=(FONT_FAMILY, 10, "bold"),
            bg=BG_PANEL, fg=STATUS_COLORS["waiting"],
        )
        self.status_label.pack(side="left", padx=(6, 0))

        self.text = scrolledtext.ScrolledText(
            self.frame, wrap="word", height=22, bg="#0C0C0C", fg=FG_TEXT,
            insertbackground=FG_TEXT, font=("Consolas", 9),
            borderwidth=0, highlightthickness=1, highlightbackground=ORANGE_DARK,
        )
        self.text.pack(fill="both", expand=True, padx=8, pady=(0, 8))
        self.text.configure(state="disabled")

        self.queue = queue.Queue()
        self.process = None

    def set_status(self, text, kind):
        self.status_var.set(text)
        self.status_label.configure(fg=STATUS_COLORS.get(kind, STATUS_COLORS["waiting"]))

    def append_line(self, line):
        self.text.configure(state="normal")
        self.text.insert("end", line)
        self.text.see("end")
        self.text.configure(state="disabled")


def stream_output(process, out_queue):
    """Runs in a background thread; pushes subprocess output lines onto a queue."""
    try:
        for line in iter(process.stdout.readline, ""):
            if line:
                out_queue.put(line)
            else:
                break
    except Exception as e:
        out_queue.put(f"[reader thread error: {e}]\n")


def is_url_up(url, timeout=1.5):
    try:
        urllib.request.urlopen(url, timeout=timeout)
        return True
    except Exception:
        return False


class App:
    def __init__(self, root):
        self.root = root
        root.title("Ibrahim Zananiri - Starting AI Server")
        root.geometry("880x580")
        root.configure(bg=BG_MAIN)
        # Blank out the default Tk feather icon in the title bar — a 1x1
        # image with no data is fully transparent, so nothing renders there.
        try:
            root.iconphoto(True, tk.PhotoImage(width=1, height=1))
        except tk.TclError:
            pass

        self._configure_style()

        header = tk.Frame(root, bg=BG_MAIN)
        header.pack(fill="x", padx=10, pady=(12, 4))
        tk.Label(
            header, text="AI Server",
            font=(FONT_FAMILY, 20, "bold"), bg=BG_MAIN, fg=ORANGE,
        ).pack(side="left")

        self.stop_btn = tk.Button(
            header, text="Stop All", command=self.stop_all,
            bg=BG_PANEL, fg=FG_TEXT, activebackground=ORANGE_DARK,
            activeforeground="#101010", font=(FONT_FAMILY, 10, "bold"),
            relief="flat", padx=14, pady=6, cursor="hand2",
        )
        self.stop_btn.pack(side="right")

        self.open_btn = tk.Button(
            header, text="Open UI in Browser", command=self.open_ui,
            state="disabled", bg="#3A3A3A", fg="#8A8A8A",
            font=(FONT_FAMILY, 10, "bold"), relief="flat",
            padx=14, pady=6, cursor="hand2", disabledforeground="#8A8A8A",
        )
        self.open_btn.pack(side="right", padx=(0, 10))

        notebook = ttk.Notebook(root, style="Orange.TNotebook")
        notebook.pack(fill="both", expand=True, padx=10, pady=10)

        self.ollama = ServicePanel(notebook, "Ollama")
        self.backend = ServicePanel(notebook, "Backend (FastAPI)")
        self.ui = ServicePanel(notebook, "UI (Gradio)")
        self.panels = [self.ollama, self.backend, self.ui]

        self.closing = False
        root.protocol("WM_DELETE_WINDOW", self.on_close)

        self.start_all()
        self.root.after(150, self.poll_queues)
        self.root.after(1000, self.poll_ollama_health)
        self.root.after(1000, self.poll_backend_health)
        self.root.after(1000, self.poll_ui_health)

    # -----------------------------------------------------------------
    def _configure_style(self):
        """clam is used (instead of vista) because it's the only built-in
        ttk theme on Windows that actually honors custom background/foreground
        colors for frames, notebooks and tabs — vista mostly ignores them."""
        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass

        style.configure("TFrame", background=BG_MAIN)
        style.configure("Panel.TFrame", background=BG_PANEL)
        style.configure(
            "Orange.TNotebook", background=BG_MAIN, borderwidth=0,
        )
        style.configure(
            "Orange.TNotebook.Tab", background=BG_PANEL, foreground=FG_TEXT,
            font=(FONT_FAMILY, 10, "bold"), padding=(14, 7), borderwidth=0,
            focuscolor=BG_PANEL,
        )
        style.map(
            "Orange.TNotebook.Tab",
            background=[("selected", ORANGE)],
            foreground=[("selected", "#101010")],
            # clam draws a dotted focus ring around the selected tab, which
            # made it look visibly bulkier than the others. Setting the
            # ring's color to match each tab's own background hides it.
            focuscolor=[("selected", ORANGE), ("!selected", BG_PANEL)],
            padding=[("selected", (14, 7)), ("!selected", (14, 7))],
        )

    # -----------------------------------------------------------------
    def start_all(self):
        # Run child Python processes unbuffered so their stdout reaches us
        # line-by-line in real time. Without this, Python block-buffers
        # stdout whenever it isn't attached to a real terminal (i.e. whenever
        # it's piped into a subprocess, as it is here) — so "Running on
        # local URL" from Gradio can sit in the child's internal buffer and
        # never show up in our queue, even though the server is genuinely up.
        # This alone was the cause of the UI tab getting stuck on "Starting".
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"

        # Ollama: skip launching a new process if it's already running
        if is_url_up("http://localhost:11434"):
            self.ollama.set_status("Already running", "running")
            self.ollama.append_line("Ollama is already running as a background service — not starting a new copy.\n")
        else:
            self.ollama.set_status("Starting...", "starting")
            self.ollama.process = self._spawn(["ollama", "serve"], self.ollama)

        self.backend.set_status("Starting...", "starting")
        self.backend.process = self._spawn(
            [sys.executable, "-u", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"],
            self.backend, env=env,
        )

        self.ui.set_status("Starting...", "starting")
        self.ui.process = self._spawn([sys.executable, "-u", "-m", "app.ui"], self.ui, env=env)

    def _spawn(self, cmd, panel, env=None):
        try:
            process = subprocess.Popen(
                cmd, cwd=str(PROJECT_ROOT),
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1,
                env=env,
                creationflags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess, "CREATE_NO_WINDOW") else 0,
            )
        except FileNotFoundError as e:
            panel.set_status("Error", "error")
            panel.append_line(f"Failed to start: {e}\n")
            return None
        threading.Thread(target=stream_output, args=(process, panel.queue), daemon=True).start()
        return process

    # -----------------------------------------------------------------
    def poll_queues(self):
        for panel in self.panels:
            drained = False
            while True:
                try:
                    line = panel.queue.get_nowait()
                except queue.Empty:
                    break
                panel.append_line(line)
                drained = True

                if panel is self.backend and "Application startup complete" in line:
                    panel.set_status("Running", "running")
                if panel is self.ui and "Running on local URL" in line:
                    panel.set_status("Running", "running")
                    self.open_btn.configure(state="normal", bg=ORANGE, fg="#101010")

            if drained and panel.process and panel.process.poll() is not None:
                # process exited
                panel.set_status(f"Stopped (exit code {panel.process.returncode})", "error")
                if panel is self.ui:
                    self.open_btn.configure(state="disabled", bg="#3A3A3A", fg="#8A8A8A")

        if not self.closing:
            self.root.after(150, self.poll_queues)

    def poll_ollama_health(self):
        if not self.closing and self.ollama.status_var.get() not in ("Running", "Already running"):
            if is_url_up("http://localhost:11434"):
                self.ollama.set_status("Running", "running")
        if not self.closing:
            self.root.after(1000, self.poll_ollama_health)

    def poll_backend_health(self):
        # Belt-and-suspenders alongside the "Application startup complete"
        # log match — a real HTTP check can't be fooled by log wording
        # changing between versions.
        if not self.closing and self.backend.status_var.get() != "Running":
            if is_url_up(f"{BACKEND_URL}/docs"):
                self.backend.set_status("Running", "running")
        if not self.closing:
            self.root.after(1000, self.poll_backend_health)

    def poll_ui_health(self):
        # Same idea as poll_ollama_health: don't rely solely on scraping the
        # Gradio process's console text for "Running on local URL" — poll the
        # actual port. This is what makes the UI tab reflect reality even if
        # buffering, wording, or Gradio's startup banner formatting changes.
        if not self.closing and self.ui.status_var.get() != "Running":
            if is_url_up(UI_URL):
                self.ui.set_status("Running", "running")
                self.open_btn.configure(state="normal", bg=ORANGE, fg="#101010")
        if not self.closing:
            self.root.after(1000, self.poll_ui_health)

    # -----------------------------------------------------------------
    def open_ui(self):
        webbrowser.open(UI_URL)

    def stop_all(self):
        for panel in self.panels:
            if panel.process and panel.process.poll() is None:
                panel.append_line("\n--- Stopping ---\n")
                panel.process.terminate()
        # give processes a moment, then force-kill any stragglers
        self.root.after(2000, self._force_kill_remaining)

    def _force_kill_remaining(self):
        for panel in self.panels:
            if panel.process and panel.process.poll() is None:
                panel.process.kill()
                panel.set_status("Stopped", "error")
        self.open_btn.configure(state="disabled", bg="#3A3A3A", fg="#8A8A8A")

    def on_close(self):
        self.closing = True
        self.stop_all()
        self.root.after(500, self.root.destroy)


if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()