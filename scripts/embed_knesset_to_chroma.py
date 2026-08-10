"""
Embeds Israeli law texts (downloaded by scripts/scrape_knesset_laws.py) into
ChromaDB, for RAG retrieval by the Attorney tab (app/ui.py's legal_chat_fn,
backed by DictaLM).

--- Why PDF -> text reuses app/document.py instead of a new extractor ---

The PDFs downloaded from fs.knesset.gov.il are official, digitally-typeset
government publications (confirmed live: HTTP 200, Content-Type
application/pdf, real selectable text) -- NOT scans. app/document.py's
convert_to_markdown() already has exactly the right pipeline for this:
resolve_hebrew_flag() auto-detects the Hebrew script ratio and routes into
_ocr_hebrew(), which itself tries pypdfium2 native-text extraction FIRST
(_extract_native_pdf_text) before ever falling back to Tesseract OCR -- see
that function's docstring for why pypdfium2 (not pypdf) matters for
preserving correct logical reading order on RTL text. Since these PDFs have
a clean text layer, the OCR fallback should essentially never trigger in
practice; if it does trigger a lot, that's a signal worth investigating
(e.g. a scanned-only historical law slipping through), not something to
silently paper over.

--- Chunking: per-סעיף (section), not fixed-size ---

Unlike this project's translate.py chunker (max_chars, sentence-based --
built for MT reliability, not retrieval), statute text should be chunked at
its natural legal unit: the סעיף (section). This mirrors exactly what this
project already does for Canon AI ([Can. N]) and GDPR AI ([Art. N]) --
same reasoning: retrieval quality and citation precision both improve when
a chunk boundary is a real legal boundary, not an arbitrary character
count. _SECTION_SPLIT_RE below finds "סעיף <number>" headers; text before
the first match (title/preamble) and any section that's still oversized
after splitting falls back to document.py's chunk_text() as a safety net,
same pattern translate.py/document.py already use elsewhere in this
project.

--- Why no exact-number lookup (unlike Canon/GDPR/HIPAA) ---

Canon law canons and GDPR/HIPAA articles/sections are numbered within a
SINGLE corpus, so "canon 1055" or "Art. 6" unambiguously identifies one
chunk. Section numbers in Israeli law are numbered PER LAW ("סעיף 1" exists
in hundreds of different laws), so a bare section-number regex match
against the whole knesset_laws collection would be actively misleading here
-- it can't tell you *which* law's סעיף 1 the user means. This script still
stores section_number in metadata (useful once a law is identified another
way, e.g. by name), but retrieval in ui.py deliberately relies on the
hybrid semantic+lexical rerank only, not an exact-match short-circuit.

Run from the project root, after scripts/scrape_knesset_laws.py:
    python scripts/embed_knesset_to_chroma.py
"""
import json
import re
import sys
from pathlib import Path

import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))  # so `import app.*` works when run as a script

from app.document import convert_to_markdown, chunk_text  # noqa: E402

MANIFEST_PATH = PROJECT_ROOT / "data" / "knesset_laws" / "manifest.jsonl"
EMBEDDED_STATE_PATH = PROJECT_ROOT / "data" / "knesset_laws" / "embedded_state.json"

CHROMA_DIR = str(PROJECT_ROOT / "app" / "chroma_db")   # same shared dir as Canon/GDPR/HIPAA
COLLECTION_NAME = "knesset_laws"

# Same embedding model/endpoint/prefix convention as Canon/GDPR/HIPAA in
# app/ui.py (_CANON_EMBED_MODEL / _CANON_OLLAMA_EMBED_URL) -- kept as a
# literal duplicate here rather than imported from ui.py, since ui.py pulls
# in gradio and this script has no reason to depend on it. Keep these two
# in sync if you ever change the embedding model.
OLLAMA_EMBED_URL = "http://localhost:11434/api/embeddings"
EMBED_MODEL = "nomic-embed-text"
DOCUMENT_PREFIX = "search_document: "  # nomic's INDEXING-side prefix (query-side uses "search_query: ", in ui.py)

# Statute text runs long per section; this is deliberately closer to the
# GDPR/HIPAA article-chunk size than translate.py's 400-char MT chunks --
# retrieval quality benefits from a whole section staying in one chunk
# where possible, unlike translation reliability which wants short chunks.
_MAX_CHUNK_CHARS = 2500

# Matches a סעיף header at the start of a line: "סעיף 5", "סעיף 12א",
# "סעיף 3(א)" -- captures the bare number+optional-Hebrew-letter suffix;
# any parenthetical sub-point is left in the following text rather than
# parsed out, since sub-points aren't separately citable in the way
# top-level sections are.
_SECTION_SPLIT_RE = re.compile(r"^\s*(סעיף\s+(\d+[א-ת]?))", re.MULTILINE)


def _embed_text(text: str) -> list[float] | None:
    try:
        resp = requests.post(
            OLLAMA_EMBED_URL,
            json={"model": EMBED_MODEL, "prompt": DOCUMENT_PREFIX + text},
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json()["embedding"]
    except Exception as e:
        print(f"[embed] embedding call failed: {e}")
        return None


def _split_into_sections(markdown_text: str) -> list[tuple[str | None, str]]:
    """Returns [(section_number_or_None, section_text), ...]. The first
    tuple (preamble/title, before any סעיף header) always has section_number
    None. Any individual piece still over _MAX_CHUNK_CHARS afterward is
    further split by document.py's chunk_text() and re-tagged with the same
    section_number, so a long section still becomes multiple chunks rather
    than one oversized one."""
    matches = list(_SECTION_SPLIT_RE.finditer(markdown_text))
    if not matches:
        return [(None, markdown_text)]

    pieces = []
    preamble = markdown_text[:matches[0].start()].strip()
    if preamble:
        pieces.append((None, preamble))

    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(markdown_text)
        section_num = m.group(2)
        section_text = markdown_text[start:end].strip()
        if section_text:
            pieces.append((section_num, section_text))

    final = []
    for section_num, text in pieces:
        if len(text) <= _MAX_CHUNK_CHARS:
            final.append((section_num, text))
        else:
            for sub_chunk in chunk_text(text, max_chars=_MAX_CHUNK_CHARS):
                final.append((section_num, sub_chunk))
    return final


def _load_embedded_state() -> dict:
    if EMBEDDED_STATE_PATH.exists():
        return json.loads(EMBEDDED_STATE_PATH.read_text(encoding="utf-8"))
    return {}  # law_id (str) -> last_updated_iso already embedded


def _save_embedded_state(state: dict) -> None:
    EMBEDDED_STATE_PATH.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    try:
        import chromadb
    except ImportError:
        print("chromadb isn't installed. Run: pip install chromadb", file=sys.stderr)
        sys.exit(1)

    if not MANIFEST_PATH.exists():
        print(f"No manifest found at {MANIFEST_PATH}. Run scripts/scrape_knesset_laws.py first.",
              file=sys.stderr)
        sys.exit(1)

    client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = client.get_or_create_collection(COLLECTION_NAME)

    embedded_state = _load_embedded_state()

    entries = [json.loads(line) for line in MANIFEST_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
    print(f"[embed] {len(entries)} law entries in manifest.")

    n_embedded_laws = 0
    n_chunks = 0
    for entry in entries:
        law_id = str(entry["law_id"])
        last_updated = entry.get("last_updated_iso")
        # Skip laws already embedded at this same (or a newer-looking)
        # version -- the manifest is append-only across scraper runs, so
        # the same law can appear more than once if it was updated again.
        if embedded_state.get(law_id) == last_updated and last_updated is not None:
            continue

        pdf_paths = entry.get("pdf_paths") or []
        if not pdf_paths:
            continue

        # A law can have more than one associated document (e.g. an
        # amendment plus the consolidated text) -- embed each, all tagged
        # with the same law_id/name so they're still grouped at query time.
        all_text_parts = []
        for pdf_path in pdf_paths:
            if not Path(pdf_path).exists():
                print(f"[embed] LawID {law_id}: missing PDF file {pdf_path}, skipping it.")
                continue
            try:
                markdown_text = convert_to_markdown(pdf_path, hebrew=False)  # auto-detects Hebrew
            except Exception as e:
                print(f"[embed] LawID {law_id}: failed to extract text from {pdf_path}: {e}")
                continue
            all_text_parts.append(markdown_text)

        if not all_text_parts:
            print(f"[embed] LawID {law_id} ({entry.get('name')!r}): no extractable text, skipping.")
            continue

        full_text = "\n\n".join(all_text_parts)
        sections = _split_into_sections(full_text)

        # Remove any previously-embedded chunks for this law before
        # re-adding -- otherwise a re-embedded (updated) law would leave
        # stale old-version chunks sitting alongside the new ones forever.
        try:
            existing = collection.get(where={"law_id": law_id})
            if existing.get("ids"):
                collection.delete(ids=existing["ids"])
        except Exception:
            pass  # nothing to delete, or collection doesn't support the filter yet

        ids, embeddings, documents, metadatas = [], [], [], []
        for i, (section_num, section_text) in enumerate(sections):
            vector = _embed_text(section_text)
            if vector is None:
                print(f"[embed] LawID {law_id}: skipping one chunk, embedding call failed "
                      "(is `ollama serve` running with nomic-embed-text pulled?)")
                continue
            chunk_id = f"{law_id}_{section_num or 'preamble'}_{i}"
            ids.append(chunk_id)
            embeddings.append(vector)
            documents.append(section_text)
            metadatas.append({
                "law_id": law_id,
                "title": entry.get("name", ""),
                "section_number": section_num or "",
                "sub_type": entry.get("sub_type_desc", ""),
                "knesset_num": str(entry.get("knesset_num") or ""),
                "last_updated": last_updated or "",
                "source_url": (entry.get("source_urls") or [""])[0],
                "hierarchy_path": entry.get("name", ""),
            })

        if not ids:
            continue

        collection.add(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)
        n_embedded_laws += 1
        n_chunks += len(ids)
        embedded_state[law_id] = last_updated
        if n_embedded_laws % 5 == 0:
            _save_embedded_state(embedded_state)  # checkpoint periodically, not just at the very end
            print(f"[embed] ...{n_embedded_laws} laws / {n_chunks} chunks embedded so far")

    _save_embedded_state(embedded_state)
    print(f"[embed] Done. {n_embedded_laws} law(s), {n_chunks} chunk(s) embedded into "
          f"'{COLLECTION_NAME}' at {CHROMA_DIR}.")
    if n_embedded_laws == 0:
        print("[embed] Nothing new to embed. If this is your first run, check that "
              f"{MANIFEST_PATH} actually has entries.")


if __name__ == "__main__":
    main()
