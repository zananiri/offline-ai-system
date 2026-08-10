"""
Manual alternative to scripts/scrape_knesset_laws.py -- for law PDFs you
already have (downloaded by hand, received by email, scanned yourself,
whatever), instead of pulling them from the Knesset OData API.

Drop PDF files into a folder at the project root called `uploads/`
(create it if it doesn't exist), then run this from the project root:

    python scripts/embed_local_law_pdfs.py

Every PDF found is embedded into the SAME ChromaDB collection
(app/chroma_db, collection "knesset_laws") that scripts/embed_knesset_to_chroma.py
writes to -- so app/ui.py's Attorney tab retrieves from both sources
together with no code changes needed. See "Why the same collection" below.

--- Folder layout expected ---

    uploads/
        chok_hozeh_1973.pdf
        chok_hozeh_1973.json      <- OPTIONAL sidecar metadata, same
                                       filename stem as its PDF (see below)
        some_subfolder/
            another_law.pdf        <- picked up too if --recursive is passed

--- Optional per-PDF metadata sidecar ---

If a PDF doesn't have a real Knesset LawID/publication date behind it (it
was scanned, emailed to you, downloaded by hand, etc.), this script guesses
reasonable defaults -- see _guess_title / _derive_law_id below. If you
actually know the real details, drop a same-named .json file next to the
PDF to override any of them:

    {
      "title": "חוק החוזים (חלק כללי), תשל\"ג-1973",
      "source_url": "https://fs.knesset.gov.il/...",
      "sub_type": "נוסח משולב",
      "publication_date": "1973-05-17"
    }

Every field is optional; anything not given falls back to the guessed
default. The file is matched purely by filename stem (mylaw.pdf +
mylaw.json), not by content.

--- Why the same collection, and how re-run/idempotency works ---

Re-running this script is safe: each file's law_id is derived from its
path relative to uploads/ (stable across runs regardless of content), and
a SHA-256 hash of the PDF's bytes is cached in
data/local_law_uploads/state.json. A file is only re-extracted/re-embedded
if that hash changed since last time (edited/replaced PDF) or it's new --
unchanged files are skipped, and changed files have their old chunks
deleted before the new ones are added (same pattern
embed_knesset_to_chroma.py already uses for updated laws), so nothing
stale lingers.

law_id values from THIS script are always prefixed "local_" (e.g.
"local_chok_hozeh_1973"), while scripts/scrape_knesset_laws.py's law_id
values are always bare Knesset LawID integers as strings (e.g. "18734") --
this guarantees the two ingestion paths can never collide on the same id
even if you run both against the same chroma_db.

--- Chunking / embedding logic ---

Deliberately duplicated from scripts/embed_knesset_to_chroma.py rather
than imported from it (same reasoning this project already applies
elsewhere -- see e.g. ui.py's LEGAL_MODEL / main.py's LEGAL_MODEL comments
on why some small constants are kept duplicated with a "keep in sync"
note instead of factored out): these are two independent, standalone
scripts under scripts/, and forcing an import dependency between them for
~80 lines of shared logic isn't worth the coupling. If you change the
section-splitting regex, chunk size, or embedding model/prefix in one,
change it in the other too, or retrieval quality will differ depending on
which ingestion path a given chunk came through.
"""
import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))  # so `import app.*` works when run as a script

from app.document import convert_to_markdown, chunk_text, resolve_hebrew_flag  # noqa: E402

DEFAULT_UPLOADS_DIR = PROJECT_ROOT / "uploads"
STATE_PATH = PROJECT_ROOT / "data" / "local_law_uploads" / "state.json"

CHROMA_DIR = str(PROJECT_ROOT / "app" / "chroma_db")   # same shared dir as scrape_knesset_laws.py's target
COLLECTION_NAME = "knesset_laws"                        # same collection -- see module docstring

# Must match scripts/embed_knesset_to_chroma.py -- see module docstring's
# "Chunking / embedding logic" note.
OLLAMA_EMBED_URL = "http://localhost:11434/api/embeddings"
EMBED_MODEL = "nomic-embed-text"
DOCUMENT_PREFIX = "search_document: "
_MAX_CHUNK_CHARS = 2500
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
    """Identical logic to embed_knesset_to_chroma.py's function of the same
    name -- splits on סעיף headers, falls back to document.py's chunk_text
    for any piece still over _MAX_CHUNK_CHARS. See that script's docstring
    for the full reasoning."""
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


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 16), b""):
            h.update(block)
    return h.hexdigest()


def _slugify(text: str) -> str:
    """Turns a relative path like 'contracts/chok_hozeh 1973.pdf' into a
    safe, stable id component. Keeps Hebrew letters (Python's \\w is
    Unicode-aware by default) since ChromaDB ids are plain strings with no
    charset restriction -- only path separators and whitespace need
    normalizing so the id stays a single readable token."""
    text = text.replace("/", "__").replace("\\", "__")
    text = re.sub(r"\s+", "_", text.strip())
    text = re.sub(r"[^\w\-]+", "", text, flags=re.UNICODE)
    return text[:120] if text else "unnamed"


def _derive_law_id(pdf_path: Path, uploads_dir: Path) -> str:
    rel = pdf_path.relative_to(uploads_dir).with_suffix("")
    return "local_" + _slugify(str(rel))


def _load_sidecar_metadata(pdf_path: Path) -> dict:
    sidecar = pdf_path.with_suffix(".json")
    if not sidecar.exists():
        return {}
    try:
        return json.loads(sidecar.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"[embed] WARNING: couldn't parse sidecar metadata {sidecar}: {e} -- ignoring it.")
        return {}


def _guess_title(markdown_text: str, pdf_path: Path) -> str:
    """No API metadata available for a manually-supplied PDF, so the title
    is guessed from the document's own first non-empty line (Israeli law
    PDFs conventionally open with the law's full name), falling back to
    the filename if extraction produced nothing usable."""
    for line in markdown_text.splitlines():
        line = line.strip().lstrip("#").strip()
        if line:
            return line[:200]
    return pdf_path.stem


def _load_state() -> dict:
    if STATE_PATH.exists():
        return json.loads(STATE_PATH.read_text(encoding="utf-8"))
    return {}  # law_id -> {"sha256": ..., "relative_path": ...}


def _save_state(state: dict) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--uploads-dir", type=Path, default=DEFAULT_UPLOADS_DIR,
                         help=f"Folder to read PDFs from (default: {DEFAULT_UPLOADS_DIR})")
    parser.add_argument("--recursive", action="store_true",
                         help="Also look in subfolders of the uploads dir.")
    parser.add_argument("--force", action="store_true",
                         help="Re-extract and re-embed every PDF, ignoring the unchanged-file cache.")
    args = parser.parse_args()

    uploads_dir = args.uploads_dir
    if not uploads_dir.exists():
        uploads_dir.mkdir(parents=True, exist_ok=True)
        print(f"[embed] Created empty {uploads_dir} -- drop your law PDFs in there and re-run this script.")
        return

    try:
        import chromadb
    except ImportError:
        print("chromadb isn't installed. Run: pip install chromadb", file=sys.stderr)
        sys.exit(1)

    pdf_paths = sorted(uploads_dir.rglob("*.pdf") if args.recursive else uploads_dir.glob("*.pdf"))
    if not pdf_paths:
        scope = "(recursively) " if args.recursive else ""
        print(f"[embed] No .pdf files found {scope}in {uploads_dir}. Nothing to do.")
        return
    print(f"[embed] Found {len(pdf_paths)} PDF(s) in {uploads_dir}.")

    client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = client.get_or_create_collection(COLLECTION_NAME)
    state = _load_state()

    n_embedded, n_skipped_unchanged, n_failed, n_chunks_total = 0, 0, 0, 0

    for pdf_path in pdf_paths:
        law_id = _derive_law_id(pdf_path, uploads_dir)
        rel_path = str(pdf_path.relative_to(uploads_dir))
        file_hash = _file_sha256(pdf_path)

        cached = state.get(law_id)
        if not args.force and cached and cached.get("sha256") == file_hash:
            n_skipped_unchanged += 1
            continue

        print(f"[embed] Processing {rel_path}  (law_id={law_id})")
        try:
            hebrew_used = resolve_hebrew_flag(str(pdf_path), False)  # auto-detects; logged for visibility
            markdown_text = convert_to_markdown(str(pdf_path), hebrew=hebrew_used)
        except Exception as e:
            print(f"[embed]   FAILED to extract text: {e}")
            n_failed += 1
            continue

        if len(markdown_text.strip()) < 20:
            print(f"[embed]   WARNING: extraction produced almost no text -- skipping "
                  "(likely a blank/rotated/low-quality scan). See document.py's OCR notes.")
            n_failed += 1
            continue

        meta_overrides = _load_sidecar_metadata(pdf_path)
        title = meta_overrides.get("title") or _guess_title(markdown_text, pdf_path)
        source_url = meta_overrides.get("source_url", "")
        sub_type = meta_overrides.get("sub_type", "manual upload")
        publication_date = meta_overrides.get("publication_date", "")

        sections = _split_into_sections(markdown_text)

        # Purge any chunks from a previous embedding of this same law_id
        # before re-adding, so an edited/replaced PDF doesn't leave stale
        # old-version chunks sitting alongside the new ones.
        try:
            existing = collection.get(where={"law_id": law_id})
            if existing.get("ids"):
                collection.delete(ids=existing["ids"])
        except Exception:
            pass

        ids, embeddings, documents, metadatas = [], [], [], []
        for i, (section_num, section_text) in enumerate(sections):
            vector = _embed_text(section_text)
            if vector is None:
                print(f"[embed]   skipping one chunk, embedding call failed "
                      "(is `ollama serve` running with nomic-embed-text pulled?)")
                continue
            chunk_id = f"{law_id}_{section_num or 'preamble'}_{i}"
            ids.append(chunk_id)
            embeddings.append(vector)
            documents.append(section_text)
            metadatas.append({
                "law_id": law_id,
                "title": title,
                "section_number": section_num or "",
                "sub_type": sub_type,
                "knesset_num": "",
                "last_updated": publication_date,
                "source_url": source_url,
                "hierarchy_path": title,
            })

        if not ids:
            print(f"[embed]   No chunks embedded for {rel_path} (every embedding call failed) -- not marking as done.")
            n_failed += 1
            continue

        collection.add(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)
        n_embedded += 1
        n_chunks_total += len(ids)
        state[law_id] = {"sha256": file_hash, "relative_path": rel_path}
        _save_state(state)  # checkpoint after every file -- a crash partway through a big batch loses nothing
        print(f"[embed]   OK -- {len(ids)} chunk(s) embedded as {title!r}"
              + (f" ({'Hebrew OCR/native' if hebrew_used else 'default'} extraction)" if True else ""))

    print(
        f"\n[embed] Done. {n_embedded} PDF(s) embedded ({n_chunks_total} chunk(s) total), "
        f"{n_skipped_unchanged} unchanged/skipped, {n_failed} failed. "
        f"Collection '{COLLECTION_NAME}' at {CHROMA_DIR}."
    )
    if n_failed:
        print("[embed] Some PDFs failed -- see FAILED/WARNING lines above. They were NOT marked "
              "as done, so re-running this script will retry them.")


if __name__ == "__main__":
    main()
