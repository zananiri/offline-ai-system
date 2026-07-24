#!/usr/bin/env python3
"""
Embed & vectorize Codice di Diritto Canonico (Italian) into ChromaDB
---------------------------------------------------------------------
Reads cic_it_canons.jsonl (produced by scrape_cic_it.py), embeds each
canon's `embed_text` field using nomic-embed-text served locally by
Ollama, and upserts everything into a persistent ChromaDB collection.

Requirements:
    pip install chromadb requests
    ollama pull nomic-embed-text
    ollama serve   (usually already running as a background service)

Usage:
    python embed_to_chroma.py \
        --jsonl ./cic_it_output/cic_it_canons.jsonl \
        --chroma-dir ./chroma_db \
        --collection cic_it

Notes on nomic-embed-text:
- Nomic's model expects a task-instruction PREFIX on every input string
  for best retrieval quality:
    - "search_document: "  -> prefix for text you are indexing (used here)
    - "search_query: "     -> prefix to use later on the user's question
                               at query time (see query_example() below)
  Skipping these prefixes still works but measurably hurts retrieval
  quality versus the model's intended usage.
- This script embeds one canon at a time via Ollama's /api/embeddings
  endpoint. That's simple and reliable; if you have thousands of
  canons and want more speed, batch several prompts per request using
  Ollama's newer /api/embed endpoint (accepts a list under "input").
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
    print("Missing dependency: pip install chromadb", file=sys.stderr)
    sys.exit(1)

OLLAMA_URL = "http://localhost:11434/api/embeddings"
MODEL_NAME = "nomic-embed-text"
DOC_PREFIX = "search_document: "
QUERY_PREFIX = "search_query: "


def embed_text(text: str, session: requests.Session, ollama_url: str, retries: int = 3) -> list[float]:
    payload = {"model": MODEL_NAME, "prompt": DOC_PREFIX + text}
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            resp = session.post(ollama_url, json=payload, timeout=60)
            resp.raise_for_status()
            data = resp.json()
            embedding = data.get("embedding")
            if not embedding:
                raise ValueError(f"No 'embedding' field in Ollama response: {data}")
            return embedding
        except (requests.RequestException, ValueError) as e:
            last_err = e
            print(f"  [retry {attempt}/{retries}] embedding failed: {e}", file=sys.stderr)
            time.sleep(1.5 * attempt)
    raise RuntimeError(f"Failed to embed text after {retries} attempts: {last_err}")


def sanitize_metadata(record: dict) -> dict:
    """Chroma metadata values must be str/int/float/bool — no None."""
    fields = (
        "canon_number", "libro", "parte", "sezione", "titolo",
        "capitolo", "articolo", "hierarchy_path", "source_url",
        "in_force_note",
    )
    meta = {}
    for f in fields:
        v = record.get(f)
        meta[f] = v if v is not None else ""
    return meta


def load_records(jsonl_path: Path) -> list[dict]:
    records = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def dedupe_records(records: list[dict]) -> list[dict]:
    """
    Collapse records that share the same chunk_id down to one, keeping
    whichever has the longer text (the more complete version). This is
    needed because the scraper can occasionally produce the same canon
    twice -- e.g. two source pages overlapping at a boundary, or a
    cross-reference like "a norma del can. 230" inside another canon's
    text getting mis-split as if it were canon 230's own heading.

    Chroma's upsert() rejects duplicate ids within a single call outright
    (even though upsert semantics would otherwise mean "last one wins"),
    so this has to happen before anything is embedded, not just before
    each batch is flushed -- two duplicates could land in different
    batches and still silently overwrite one another without this.
    """
    best: dict[str, dict] = {}
    order: list[str] = []
    dup_count = 0
    for r in records:
        cid = r.get("chunk_id") or f"CIC-it-{r.get('canon_number')}"
        candidate_len = len(r.get("embed_text") or r.get("text") or "")
        if cid not in best:
            best[cid] = r
            order.append(cid)
        else:
            dup_count += 1
            existing_len = len(best[cid].get("embed_text") or best[cid].get("text") or "")
            if candidate_len > existing_len:
                best[cid] = r
            print(f"  [dedupe] {cid} appeared more than once; kept the longer version.", file=sys.stderr)

    if dup_count:
        print(f"WARNING: found {dup_count} duplicate chunk_id(s) in the source JSONL "
              f"(kept the longer text for each, discarded the rest). Consider checking "
              f"the scraper output if this number looks unexpectedly high.", file=sys.stderr)

    return [best[cid] for cid in order]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--jsonl", required=True, help="Path to cic_it_canons.jsonl")
    ap.add_argument("--chroma-dir", default="./chroma_db", help="ChromaDB persistent storage dir")
    ap.add_argument("--collection", default="cic_it", help="ChromaDB collection name")
    ap.add_argument("--ollama-url", default=OLLAMA_URL, help="Ollama embeddings endpoint")
    args = ap.parse_args()

    jsonl_path = Path(args.jsonl)
    if not jsonl_path.exists():
        print(f"File not found: {jsonl_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading records from {jsonl_path} ...")
    records = load_records(jsonl_path)
    print(f"Loaded {len(records)} canon chunks.")
    records = dedupe_records(records)
    print(f"{len(records)} unique chunks after deduplication.")

    client = chromadb.PersistentClient(path=args.chroma_dir)
    collection = client.get_or_create_collection(
        name=args.collection,
        metadata={"hnsw:space": "cosine"},
    )

    session = requests.Session()

    ids, embeddings, metadatas, documents = [], [], [], []
    for i, record in enumerate(records, 1):
        chunk_id = record.get("chunk_id") or f"CIC-it-{record.get('canon_number')}"
        text = record.get("embed_text") or record.get("text", "")
        if not text:
            print(f"  [skip] {chunk_id}: empty text", file=sys.stderr)
            continue

        try:
            vector = embed_text(text, session, args.ollama_url)
        except RuntimeError as e:
            print(f"  [error] {chunk_id}: {e} -- skipping", file=sys.stderr)
            continue

        ids.append(chunk_id)
        embeddings.append(vector)
        metadatas.append(sanitize_metadata(record))
        documents.append(record.get("text", ""))

        print(f"  [{i}/{len(records)}] embedded {chunk_id}")

        # Flush in batches of 50 to keep memory bounded on large corpora.
        if len(ids) >= 50:
            collection.upsert(ids=ids, embeddings=embeddings, metadatas=metadatas, documents=documents)
            ids, embeddings, metadatas, documents = [], [], [], []

    if ids:
        collection.upsert(ids=ids, embeddings=embeddings, metadatas=metadatas, documents=documents)

    print(f"\nDone. Collection '{args.collection}' now has {collection.count()} vectors "
          f"stored at {args.chroma_dir}")


def query_example(question: str, chroma_dir: str = "./chroma_db", collection_name: str = "cic_it", n_results: int = 5):
    """
    Example of how to QUERY this collection later. Note the different
    prefix ("search_query: ") required by nomic-embed-text at query time.
    """
    client = chromadb.PersistentClient(path=chroma_dir)
    collection = client.get_collection(collection_name)
    session = requests.Session()

    query_vector = embed_text_for_query(question, session, OLLAMA_URL)
    results = collection.query(query_embeddings=[query_vector], n_results=n_results)
    return results


def embed_text_for_query(text: str, session: requests.Session, ollama_url: str = OLLAMA_URL) -> list[float]:
    payload = {"model": MODEL_NAME, "prompt": QUERY_PREFIX + text}
    resp = session.post(ollama_url, json=payload, timeout=60)
    resp.raise_for_status()
    return resp.json()["embedding"]


if __name__ == "__main__":
    main()