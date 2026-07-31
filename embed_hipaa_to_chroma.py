#!/usr/bin/env python3
"""
Embed & vectorize HIPAA (45 CFR Parts 160 & 164) into ChromaDB
---------------------------------------------------------------------
Reads hipaa_sections.jsonl (produced by scrape_hipaa.py), embeds each
section's `embed_text` field using nomic-embed-text served locally by
Ollama, and upserts everything into a persistent ChromaDB collection.

This is the HIPAA counterpart of embed_gdpr_to_chroma.py / embed_to_chroma.py
(Canon AI) -- same model, same batching/retry/dedupe/skip-unchanged
approach, same Ollama endpoints -- just pointed at a different JSONL and
collection, so it can live in the same Chroma store as "cic_it" and "gdpr"
without conflicting (different --collection name).

Requirements:
    pip install chromadb requests
    ollama pull nomic-embed-text
    ollama serve   (usually already running as a background service)

Usage (run from the project root, alongside app/ -- same convention as
scrape_hipaa.py and this project's other one-off scripts):
    python embed_hipaa_to_chroma.py \
        --jsonl ./hipaa_output/hipaa_sections.jsonl \
        --collection hipaa

Defaults --chroma-dir to ./app/chroma_db (i.e. app/chroma_db relative to
wherever you run this from) -- the SAME directory app/ui.py's Canon AI and
GDPR AI tabs already read their collections from. This script writes into
a separate "hipaa" collection inside that same directory, so it coexists
with "cic_it" and "gdpr" without touching either -- override with
--chroma-dir if your chroma_db actually lives somewhere else.

Notes on nomic-embed-text:
- Nomic's model expects a task-instruction PREFIX on every input string
  for best retrieval quality:
    - "search_document: "  -> prefix for text you are indexing (used here)
    - "search_query: "     -> prefix to use later on the user's question
                               at query time (this exact prefix is also
                               what app/ui.py's HIPAA AI tab uses)
  Skipping these prefixes still works but measurably hurts retrieval
  quality versus the model's intended usage.
- Sections are embedded in batches via Ollama's /api/embed endpoint
  (falls back to one-at-a-time /api/embeddings if that's unavailable --
  e.g. an older Ollama version).
"""

import argparse
import hashlib
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
OLLAMA_BATCH_URL = "http://localhost:11434/api/embed"
MODEL_NAME = "nomic-embed-text"
DOC_PREFIX = "search_document: "
QUERY_PREFIX = "search_query: "
EMBED_BATCH_SIZE = 16  # prompts per /api/embed call


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


def embed_texts_batch(texts: list[str], session: requests.Session, batch_url: str,
                       single_url: str, retries: int = 3) -> list[list[float]]:
    """
    Embeds multiple sections in one HTTP call via Ollama's /api/embed
    endpoint. Falls back to the proven one-at-a-time embed_text() path for
    this batch if the batch endpoint fails outright -- e.g. an older Ollama
    version that only has /api/embeddings -- so a single incompatible/
    unavailable batch endpoint never blocks the whole run.
    """
    payload = {"model": MODEL_NAME, "input": [DOC_PREFIX + t for t in texts]}
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            resp = session.post(batch_url, json=payload, timeout=120)
            resp.raise_for_status()
            data = resp.json()
            embeddings = data.get("embeddings")
            if not embeddings or len(embeddings) != len(texts):
                raise ValueError(f"Unexpected /api/embed response shape: {data}")
            return embeddings
        except (requests.RequestException, ValueError) as e:
            last_err = e
            print(f"  [batch retry {attempt}/{retries}] batch embedding failed: {e}", file=sys.stderr)
            time.sleep(1.5 * attempt)

    print(f"  [batch embed] giving up after {retries} attempts ({last_err}); "
          f"falling back to one-at-a-time embedding for this batch.", file=sys.stderr)
    return [embed_text(t, session, single_url) for t in texts]


def content_hash(text: str) -> str:
    """Short hash of the text actually used for embedding, stored in each
    vector's metadata so a later rerun can tell whether a section's content
    changed since it was last embedded (see --skip-unchanged)."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def sanitize_metadata(record: dict, text_hash: str = "") -> dict:
    """Chroma metadata values must be str/int/float/bool -- no None, and no
    lists (hence `cross_references` -- a list in the JSONL -- gets joined
    into a comma-separated string here). Note section_id is a STRING
    ("164.312"), unlike GDPR's integer article_number -- CFR section
    numbers aren't plain integers."""
    meta = {
        "section_id": record.get("section_id") or "",
        "title": record.get("title") or "",
        "part_number": record.get("part_number") or 0,
        "part_title": record.get("part_title") or "",
        "subpart_letter": record.get("subpart_letter") or "",
        "subpart_title": record.get("subpart_title") or "",
        "hierarchy_path": record.get("hierarchy_path") or "",
        "source_url": record.get("source_url") or "",
        "cross_references": ",".join(record.get("cross_references") or []),
    }
    meta["text_hash"] = text_hash
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
    whichever has the longer text. Not expected to trigger for a clean
    scrape_hipaa.py run (section ids are inherently unique per part), but
    kept for the same reason embed_to_chroma.py has it: Chroma's upsert()
    rejects duplicate ids within a single call outright, so any accidental
    duplicate (e.g. a hand-edited or re-run/merged JSONL) must be resolved
    before anything is embedded, not just before each batch is flushed.
    """
    best: dict[str, dict] = {}
    order: list[str] = []
    dup_count = 0
    for r in records:
        cid = r.get("chunk_id") or f"HIPAA-45-CFR-{r.get('section_id')}"
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
              f"(kept the longer text for each, discarded the rest).", file=sys.stderr)

    return [best[cid] for cid in order]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--jsonl", required=True, help="Path to hipaa_sections.jsonl")
    ap.add_argument(
        "--chroma-dir", default="./app/chroma_db",
        help="ChromaDB persistent storage dir (default: app/chroma_db relative to "
             "the current directory -- the same store app/ui.py's Canon AI / GDPR AI "
             "tabs use, assuming you're running this from the project root)",
    )
    ap.add_argument("--collection", default="hipaa", help="ChromaDB collection name")
    ap.add_argument("--ollama-url", default=OLLAMA_URL, help="Ollama embeddings endpoint")
    ap.add_argument("--ollama-batch-url", default=OLLAMA_BATCH_URL,
                     help="Ollama batch embeddings endpoint (/api/embed)")
    ap.add_argument("--batch-size", type=int, default=EMBED_BATCH_SIZE,
                     help="Sections embedded per /api/embed call")
    ap.add_argument("--skip-unchanged", action="store_true",
                     help="On a rerun, skip re-embedding sections whose text hasn't "
                          "changed since they were last embedded (compares against "
                          "the text_hash stored in each vector's metadata).")
    args = ap.parse_args()

    jsonl_path = Path(args.jsonl)
    if not jsonl_path.exists():
        print(f"File not found: {jsonl_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading records from {jsonl_path} ...")
    records = load_records(jsonl_path)
    print(f"Loaded {len(records)} section chunk(s).")
    records = dedupe_records(records)
    print(f"{len(records)} unique chunk(s) after deduplication.")

    client = chromadb.PersistentClient(path=args.chroma_dir)
    collection = client.get_or_create_collection(
        name=args.collection,
        metadata={"hnsw:space": "cosine"},
    )

    existing_hashes: dict[str, str] = {}
    if args.skip_unchanged and collection.count() > 0:
        existing = collection.get(include=["metadatas"])
        for cid, meta in zip(existing.get("ids", []), existing.get("metadatas", [])):
            existing_hashes[cid] = (meta or {}).get("text_hash", "")
        print(f"--skip-unchanged: found {len(existing_hashes)} existing vectors to compare against.")

    session = requests.Session()

    pending = []
    skipped_unchanged = 0
    for record in records:
        chunk_id = record.get("chunk_id") or f"HIPAA-45-CFR-{record.get('section_id')}"

        embed_source = record.get("embed_text") or record.get("text", "")
        doc_text = record.get("text") or record.get("embed_text", "")

        if not embed_source or not doc_text:
            print(f"  [skip] {chunk_id}: empty text", file=sys.stderr)
            continue

        h = content_hash(embed_source)
        if args.skip_unchanged and existing_hashes.get(chunk_id) == h:
            skipped_unchanged += 1
            continue

        pending.append((chunk_id, embed_source, doc_text, sanitize_metadata(record, h)))

    if skipped_unchanged:
        print(f"--skip-unchanged: {skipped_unchanged} section(s) unchanged since last run, skipping re-embedding.")
    print(f"Embedding {len(pending)} section(s)...")

    total_done = 0
    for batch_start in range(0, len(pending), args.batch_size):
        batch = pending[batch_start:batch_start + args.batch_size]
        chunk_ids = [b[0] for b in batch]
        texts = [b[1] for b in batch]

        try:
            vectors = embed_texts_batch(texts, session, args.ollama_batch_url, args.ollama_url)
        except RuntimeError as e:
            print(f"  [error] batch starting at {chunk_ids[0]}: {e} -- skipping whole batch", file=sys.stderr)
            continue

        collection.upsert(
            ids=chunk_ids,
            embeddings=vectors,
            metadatas=[b[3] for b in batch],
            documents=[b[2] for b in batch],
        )
        total_done += len(batch)
        print(f"  [{total_done}/{len(pending)}] embedded through {chunk_ids[-1]}")

    print(f"\nDone. Collection '{args.collection}' now has {collection.count()} vectors "
          f"stored at {args.chroma_dir}")


def query_example(question: str, chroma_dir: str = "./app/chroma_db", collection_name: str = "hipaa", n_results: int = 5):
    """Example of how to QUERY this collection later. Note the different
    prefix ("search_query: ") required by nomic-embed-text at query time."""
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
