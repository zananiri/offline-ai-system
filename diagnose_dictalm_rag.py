#!/usr/bin/env python3
r"""
diagnose_dictalm_rag.py
-----------------------------------------------------------------------------
Reproduces exactly what legal_chat_fn does -- embed query -> retrieve from
ChromaDB -> build the grounded prompt -> call DictaLM -- but OUTSIDE Gradio,
with STREAMING output (token-by-token, as it's generated) and Ollama's own
timing breakdown, so you can see directly whether it's actually hung or just
slow, and exactly which stage (retrieval vs prompt-processing vs generation)
the time is going into. The app itself never shows any of this -- it blocks
on one non-streamed request and shows a single static status line the whole
time, which is why "is it stuck?" has been hard to answer from the UI alone.

Run from the project root, venv active:

    python diagnose_dictalm_rag.py
    python diagnose_dictalm_rag.py --query "תקציר תיקון 13 לחוק הגנת הפרטיות"
    python diagnose_dictalm_rag.py --collection knesset_laws --n-results 6

NOTE on retrieval fidelity: this uses a plain top-N-by-vector-similarity
query, NOT ui.py's full hybrid lexical+vector rerank (_rerank_candidates).
That's deliberate -- this script's job is to isolate DICTALM's behavior on a
realistic prompt, not to exactly reproduce retrieval ranking. The retrieved
set may differ slightly from what the app shows (e.g. "Found 6 relevant
statute excerpt(s)"). If retrieval quality itself is what you need to debug
next, that's a separate, smaller diagnostic -- ask and I'll build that one
too rather than bolting it onto this one.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import requests

try:
    import chromadb
except ImportError:
    print("chromadb isn't installed in this venv. pip install chromadb", file=sys.stderr)
    sys.exit(1)

PROJECT_ROOT = Path(__file__).resolve().parent
CHROMA_DIR = str(PROJECT_ROOT / "app" / "chroma_db")
OLLAMA_URL = "http://localhost:11434"
EMBED_MODEL = "nomic-embed-text"
QUERY_PREFIX = "search_query: "  # nomic's query-side prefix (must match indexing-side "search_document: ")

# Duplicated from app/ui.py rather than imported -- importing app.ui as a
# module has real side effects (it builds the entire gr.Blocks() UI tree at
# import time, outside any __main__ guard), which is exactly the kind of
# thing a lightweight diagnostic script shouldn't drag in. Same
# "keep in sync, documented duplication" pattern this project already uses
# for LEGAL_MODEL across main.py/ui.py.
LEGAL_MODEL = "hf.co/dicta-il/DictaLM-3.0-24B-Thinking-GGUF:Q4_K_M"
LEGAL_NUM_PREDICT = 6144
LEGAL_NUM_CTX = 16384
LEGAL_SYSTEM_PROMPT = (
    "You are an Israeli lawyer. Think through and answer every question strictly "
    "according to the laws of the State of Israel -- its statutes, regulations, "
    "and case law -- not the law of any other jurisdiction, unless the user "
    "explicitly asks about a different country's law.\n\n"
    "For every substantive legal claim, name the specific Israeli statute, "
    "regulation, or section you are relying on immediately after the claim. "
    "You may also be given retrieved statute excerpts below, pulled directly "
    "from the Knesset's official legislation database. When they are relevant "
    "to the question, prefer citing and quoting from those exact excerpts over "
    "relying on general recall -- they are authoritative and more current than "
    "your training data. If no excerpts were retrieved, or the retrieved "
    "excerpts don't actually cover the question, say so explicitly. Always "
    "reply in the same language the user's question is written in."
)

DEFAULT_QUERY = "מהם עיקרי תיקון 13 לחוק הגנת הפרטיות?"


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def embed_query(text: str) -> list[float]:
    t0 = time.monotonic()
    resp = requests.post(
        f"{OLLAMA_URL}/api/embeddings",
        json={"model": EMBED_MODEL, "prompt": QUERY_PREFIX + text},
        timeout=30,
    )
    resp.raise_for_status()
    vec = resp.json()["embedding"]
    log(f"embedded query -> {len(vec)}-dim vector in {time.monotonic()-t0:.2f}s")
    return vec


def retrieve(collection, query_vector: list[float], n_results: int) -> tuple[list[str], list[dict], list[float]]:
    t0 = time.monotonic()
    res = collection.query(
        query_embeddings=[query_vector], n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )
    docs = (res.get("documents") or [[]])[0]
    metas = (res.get("metadatas") or [[]])[0]
    dists = (res.get("distances") or [[]])[0]
    log(f"retrieved {len(docs)} chunk(s) from ChromaDB in {time.monotonic()-t0:.2f}s")
    for i, (doc, meta, dist) in enumerate(zip(docs, metas, dists), start=1):
        title = (meta or {}).get("title", "?")
        section = (meta or {}).get("section_number", "")
        preview = doc[:80].replace("\n", " ")
        print(f"    [{i}] distance={dist:.4f}  {title} {section}  -- {preview!r}...")
    return docs, metas, dists


def build_prompt(question: str, docs: list[str], metas: list[dict]) -> str:
    if not docs:
        return (
            "No statutes were retrieved for this question -- tell the user that "
            f"plainly instead of guessing.\n\nUser question: {question}"
        )
    context_parts = []
    for doc, meta in zip(docs, metas):
        title = (meta or {}).get("title", "")
        section = (meta or {}).get("section_number", "")
        label = f"[{title}" + (f", סעיף {section}" if section else "") + "]"
        context_parts.append(f"{label}\n{doc}")
    context_text = "\n\n---\n\n".join(context_parts)
    return (
        f"Retrieved Israeli statute excerpts relevant to the question "
        f"(from the Knesset's official legislation database):\n\n{context_text}\n\n"
        f"User question: {question}"
    )


def stream_dictalm(system_prompt: str, user_prompt: str, model: str,
                    num_predict: int, num_ctx: int, timeout: int) -> None:
    """
    Streams the response token-by-token (Ollama's /api/chat with
    "stream": true returns newline-delimited JSON, one object per token/
    fragment) and prints each fragment AS IT ARRIVES with no buffering --
    this is the main thing the app itself never shows you. A visible first
    token proves the model is alive and actually generating, not stuck.
    """
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "stream": True,
        "options": {"num_predict": num_predict, "num_ctx": num_ctx},
    }

    log(f"sending request to {model} (num_predict={num_predict}, num_ctx={num_ctx})...")
    log("waiting for first token -- this gap is prompt processing "
        "(reading the whole context before generation can start)...")

    t_request_sent = time.monotonic()
    first_token_at = None
    char_count = 0
    final_stats = None

    print("\n----- STREAMED OUTPUT (raw, including any <think> block) -----")
    try:
        with requests.post(f"{OLLAMA_URL}/api/chat", json=payload, stream=True, timeout=timeout) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines(decode_unicode=True):
                if not line:
                    continue
                chunk = json.loads(line)
                content = chunk.get("message", {}).get("content", "")
                if content:
                    if first_token_at is None:
                        first_token_at = time.monotonic()
                        log(f"FIRST TOKEN after {first_token_at - t_request_sent:.2f}s "
                            f"(this was the silent gap -- model is alive)")
                    print(content, end="", flush=True)
                    char_count += len(content)
                if chunk.get("done"):
                    final_stats = chunk
    except requests.exceptions.Timeout:
        log(f"TIMED OUT after {timeout}s with no completion -- genuinely stuck, "
            f"not just slow (a slow-but-alive run would still be streaming tokens).")
        return
    except requests.exceptions.RequestException as e:
        log(f"request failed: {e}")
        return

    print("\n----- END STREAMED OUTPUT -----\n")

    total_elapsed = time.monotonic() - t_request_sent
    log(f"generation finished. wall-clock total: {total_elapsed:.2f}s, "
        f"{char_count} characters streamed.")

    if final_stats:
        # Ollama reports these in nanoseconds; convert to seconds for readability.
        prompt_eval_count = final_stats.get("prompt_eval_count")
        prompt_eval_ns = final_stats.get("prompt_eval_duration", 0)
        eval_count = final_stats.get("eval_count")
        eval_ns = final_stats.get("eval_duration", 0)
        load_ns = final_stats.get("load_duration", 0)

        print("\n----- OLLAMA'S OWN TIMING BREAKDOWN -----")
        print(f"  model load time:        {load_ns / 1e9:.2f}s "
              f"(0 or near-0 means it was already resident, i.e. no model-swap this call)")
        if prompt_eval_count:
            print(f"  prompt processing:      {prompt_eval_ns / 1e9:.2f}s for "
                  f"{prompt_eval_count} input tokens "
                  f"({prompt_eval_count / max(prompt_eval_ns / 1e9, 1e-9):.1f} tok/s)")
        if eval_count:
            print(f"  answer generation:      {eval_ns / 1e9:.2f}s for "
                  f"{eval_count} output tokens "
                  f"({eval_count / max(eval_ns / 1e9, 1e-9):.1f} tok/s)")
        if eval_count and eval_count >= num_predict:
            print(f"  *** hit num_predict cap ({num_predict}) -- the model was CUT OFF, "
                  f"not finished on its own. This is very likely why answers look "
                  f"incomplete/empty: it may have been cut off mid-<think> block. ***")
        print("-------------------------------------------\n")
    else:
        log("no final stats block received -- stream may have been cut short "
            "(connection dropped before Ollama sent its 'done' message).")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--query", default=DEFAULT_QUERY)
    ap.add_argument("--collection", default="knesset_laws")
    ap.add_argument("--n-results", type=int, default=6)
    ap.add_argument("--model", default=LEGAL_MODEL)
    ap.add_argument("--num-predict", type=int, default=LEGAL_NUM_PREDICT)
    ap.add_argument("--num-ctx", type=int, default=LEGAL_NUM_CTX)
    ap.add_argument("--timeout", type=int, default=3700)
    args = ap.parse_args()

    log(f"query: {args.query!r}")

    query_vector = embed_query(args.query)

    client = chromadb.PersistentClient(path=CHROMA_DIR)
    try:
        collection = client.get_collection(args.collection)
    except Exception as e:
        log(f"could not open collection '{args.collection}' at {CHROMA_DIR}: {e}")
        sys.exit(1)
    log(f"collection '{args.collection}' has {collection.count()} chunk(s) total")

    docs, metas, dists = retrieve(collection, query_vector, args.n_results)
    if not docs:
        log("WARNING: zero chunks retrieved -- your uploaded תיקון 13 document may not "
            "actually be matching this query. Check the collection contents and metadata "
            "(title/section_number) printed above against what you expect.")

    prompt = build_prompt(args.query, docs, metas)
    approx_prompt_chars = len(LEGAL_SYSTEM_PROMPT) + len(prompt)
    log(f"built combined prompt: ~{approx_prompt_chars} characters "
        f"(~{approx_prompt_chars // 4} tokens, rough estimate)")

    stream_dictalm(LEGAL_SYSTEM_PROMPT, prompt, args.model,
                   args.num_predict, args.num_ctx, args.timeout)


if __name__ == "__main__":
    main()
