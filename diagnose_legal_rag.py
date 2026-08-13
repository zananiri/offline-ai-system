#!/usr/bin/env python3
r"""
diagnose_legal_rag.py
-----------------------------------------------------------------------------
Run this from the project root (same venv as the app: `.\.venv\Scripts\
Activate.ps1` first) WHILE the Gradio app is NOT running, so nothing else
is holding a lock on app/chroma_db or a model slot in Ollama.

    python diagnose_legal_rag.py

It walks through every suspect from the "hangs with no error" investigation
as its own isolated step, each with a HARD timeout enforced via a subprocess
(not a plain function call) -- so if one step is genuinely the thing that
hangs forever, THIS script still finishes and tells you which one, instead
of also hanging.

Copy the full printed report and share it back -- that's the fastest path
to a real diagnosis instead of more guessing.
"""
import json
import subprocess
import sys
import time
import socket
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
CHROMA_DIR = PROJECT_ROOT / "app" / "chroma_db"
COLLECTION_NAME = "knesset_laws"  # change if you embedded under a different collection name
OLLAMA_URL = "http://localhost:11434"
EMBED_MODEL = "nomic-embed-text"
STEP_TIMEOUT_SECONDS = 20  # generous for a healthy step; a healthy step should finish in <2s


def run_isolated(label: str, code: str, timeout: int = STEP_TIMEOUT_SECONDS) -> dict:
    """
    Runs `code` in a brand-new Python subprocess with a hard timeout.
    subprocess.run's timeout is enforced by the OS killing the child
    process -- unlike a plain in-process function call, this is guaranteed
    to return control to us even if `code` hangs forever on a socket or a
    file lock with no exception at all.
    """
    print(f"\n--- {label} ---")
    started = time.monotonic()
    try:
        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=str(PROJECT_ROOT),
            capture_output=True, text=True, timeout=timeout,
        )
        elapsed = time.monotonic() - started
        status = "OK" if result.returncode == 0 else f"FAILED (exit {result.returncode})"
        print(f"[{status}] in {elapsed:.2f}s")
        if result.stdout.strip():
            print("stdout:", result.stdout.strip())
        if result.stderr.strip():
            print("stderr:", result.stderr.strip()[-2000:])  # tail only, keep the report readable
        return {"label": label, "ok": result.returncode == 0, "hung": False,
                "elapsed_s": round(elapsed, 2), "stdout": result.stdout, "stderr": result.stderr}
    except subprocess.TimeoutExpired:
        elapsed = time.monotonic() - started
        print(f"[HUNG] did not finish within {timeout}s -- THIS IS LIKELY YOUR CULPRIT")
        return {"label": label, "ok": False, "hung": True, "elapsed_s": round(elapsed, 2)}


results = []


# =============================================================================
# 1. Is Ollama even reachable, fast, with no RAG involved?
# =============================================================================
results.append(run_isolated(
    "1. Ollama reachable",
    "import requests; r = requests.get('http://localhost:11434', timeout=5); print('status', r.status_code)"
))


# =============================================================================
# 2. Are the models you think are pulled actually pulled?
# =============================================================================
results.append(run_isolated(
    "2. Ollama model list",
    "import requests, json; r = requests.get('http://localhost:11434/api/tags', timeout=5); "
    "names = [m['name'] for m in r.json().get('models', [])]; print(json.dumps(names, indent=2))"
))


# =============================================================================
# 3. Time a single embeddings call in isolation -- this is the FIRST network
#    call legal_chat_fn makes before it ever touches Chroma or DictaLM.
# =============================================================================
results.append(run_isolated(
    "3. nomic-embed-text embeddings call (isolated)",
    f"""
import requests, time
t0 = time.monotonic()
r = requests.post('{OLLAMA_URL}/api/embeddings',
                   json={{'model': '{EMBED_MODEL}', 'prompt': 'search_query: test question'}},
                   timeout=15)
r.raise_for_status()
vec = r.json()['embedding']
print(f'got {{len(vec)}}-dim embedding in {{time.monotonic()-t0:.2f}}s')
"""
))


# =============================================================================
# 4. Is anything else already holding app/chroma_db open? (best-effort check)
# =============================================================================
results.append(run_isolated(
    "4. Chroma directory lock check",
    f"""
import sqlite3
from pathlib import Path
db_file = Path(r'{CHROMA_DIR}') / 'chroma.sqlite3'
if not db_file.exists():
    print('no chroma.sqlite3 found at', db_file, '-- check CHROMA_DIR path in this script')
else:
    # A short-timeout exclusive-lock attempt: if another process has this
    # database open with a write lock, this will raise 'database is locked'
    # quickly instead of hanging -- unlike ChromaDB's own client, which has
    # no such timeout.
    conn = sqlite3.connect(str(db_file), timeout=3)
    try:
        conn.execute('BEGIN IMMEDIATE')
        conn.execute('ROLLBACK')
        print('no competing lock detected -- file is free')
    except sqlite3.OperationalError as e:
        print('LOCK DETECTED:', e, '-- something else has this chroma_db open right now')
    finally:
        conn.close()
"""
))


# =============================================================================
# 5. Open the collection and check it actually has your ~7 chunks.
# =============================================================================
results.append(run_isolated(
    "5. Chroma collection contents",
    f"""
import chromadb
client = chromadb.PersistentClient(path=r'{CHROMA_DIR}')
print('collections:', [c.name for c in client.list_collections()])
collection = client.get_collection('{COLLECTION_NAME}')
print('count in {COLLECTION_NAME}:', collection.count())
sample = collection.get(limit=1, include=['metadatas'])
print('sample metadata:', sample.get('metadatas'))
"""
))


# =============================================================================
# 6. THE MAIN SUSPECT: query with query_embeddings (the correct pattern
#    every existing tab in this project uses) -- should be fast.
# =============================================================================
results.append(run_isolated(
    "6. collection.query(query_embeddings=...) -- correct pattern",
    f"""
import chromadb, requests, time
client = chromadb.PersistentClient(path=r'{CHROMA_DIR}')
collection = client.get_collection('{COLLECTION_NAME}')

t0 = time.monotonic()
r = requests.post('{OLLAMA_URL}/api/embeddings',
                   json={{'model': '{EMBED_MODEL}', 'prompt': 'search_query: test question'}},
                   timeout=15)
vec = r.json()['embedding']
print(f'embed step: {{time.monotonic()-t0:.2f}}s')

t1 = time.monotonic()
res = collection.query(query_embeddings=[vec], n_results=3, include=['documents', 'distances'])
print(f'query step: {{time.monotonic()-t1:.2f}}s')
print('got', len(res.get('documents', [[]])[0]), 'results')
"""
))


# =============================================================================
# 7. THE HANG-THEORY REPRODUCTION: query with query_texts, WITHOUT
#    pre-embedding. If this is what your modified code does, Chroma falls
#    back to its OWN default embedder, which needs to download model
#    weights on first use -- and on an offline machine, that download
#    attempt is exactly what hangs with no error. This step has its own
#    tight 12s timeout separate from the others: if it's going to hang,
#    we don't want to wait the full 20s to find out.
# =============================================================================
results.append(run_isolated(
    "7. collection.query(query_texts=...) -- reproduces the suspected bug",
    f"""
import chromadb
client = chromadb.PersistentClient(path=r'{CHROMA_DIR}')
collection = client.get_collection('{COLLECTION_NAME}')
res = collection.query(query_texts=['test question'], n_results=3)
print('got', len(res.get('documents', [[]])[0]), 'results (if you see this, it did NOT hang)')
""",
    timeout=12,
))


# =============================================================================
# 8. Is this machine actually offline right now? Confirms/denies the
#    "silent model download hang" theory directly, with a fast raw socket
#    connect (no library-level hang risk).
# =============================================================================
def check_internet(host: str, port: int = 443, timeout: float = 3.0) -> bool:
    try:
        socket.create_connection((host, port), timeout=timeout).close()
        return True
    except OSError:
        return False

print("\n--- 8. Internet reachability (raw socket, 3s timeout) ---")
for host in ["huggingface.co", "raw.githubusercontent.com"]:
    reachable = check_internet(host)
    print(f"{host}: {'reachable' if reachable else 'UNREACHABLE'}")
    results.append({"label": f"8. reachability: {host}", "ok": reachable, "hung": False})


# =============================================================================
# Summary
# =============================================================================
print("\n" + "=" * 70)
print("SUMMARY -- share everything above and below this line")
print("=" * 70)
for r in results:
    flag = "HUNG (root cause candidate)" if r.get("hung") else ("ok" if r.get("ok") else "FAILED")
    print(f"  [{flag:28s}] {r['label']}")