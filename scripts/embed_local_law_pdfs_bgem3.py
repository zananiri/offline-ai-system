"""
scripts/embed_local_law_pdfs_bgem3.py

Ingests law PDFs from uploads/ into ChromaDB, following the "Simple Israeli
Legal AI" architecture doc: BGE-M3 embeddings, one collection acting as the
system's legal memory, and a metadata schema that carries version/source/
in-force information on every chunk (mandatory per that doc -- a legal
system cannot answer safely without knowing WHICH version of a provision a
chunk represents).

This is a sibling to scripts/embed_local_law_pdfs.py, not a replacement of
it -- that script feeds the OLD nomic-embed-text "knesset_laws" collection
still used by app/ui.py's existing tabs. This one writes to a NEW, separate
collection ("israeli_legal_db") using a different embedding model (BGE-M3),
consumed by app/legal_pipeline_v2.py's hybrid BM25+vector retrieval instead.
Keeping them separate avoids mixing two different embedding spaces inside
one Chroma collection, which would silently corrupt vector search (cosine
similarity between a nomic-embed-text vector and a BGE-M3 vector is
meaningless -- they aren't the same space at all).

--- Usage ---

Drop PDFs into uploads/ at the project root (same folder/layout as
embed_local_law_pdfs.py -- see that script's docstring for the sidecar
.json convention, which this script extends with a few extra fields):

    python scripts/embed_local_law_pdfs_bgem3.py
    python scripts/embed_local_law_pdfs_bgem3.py --recursive
    python scripts/embed_local_law_pdfs_bgem3.py --force   # re-embed everything

--- Requires ---

    ollama pull bge-m3
    pip install chromadb

--- Sidecar metadata (uploads/<name>.json, same filename stem as the PDF) ---

Every field is optional; anything omitted falls back to a guessed default
(same title-guessing heuristic as embed_local_law_pdfs.py) or a safe
default value:

    {
      "law_name": "חוק החוזים (חלק כללי), תשל\"ג-1973",
      "document_type": "law",              # law | regulation | amendment | ruling | other
      "source": "official",                # official | manual_upload | scan
      "source_url": "https://fs.knesset.gov.il/...",
      "effective_from": "1973-05-17",       # ISO date, or "" if unknown
      "effective_until": null,              # ISO date, or null if still in force
      "is_current": true
    }

--- Metadata schema written on every chunk (per the architecture doc) ---

    {
      "document_type": "law",
      "law_name": "...",
      "chapter": "פרק שני" | "",     # nearest chapter heading above this section, if any
      "section": "12" | "",          # סעיף number
      "subsection": "א,ב" | "",      # subsection letters found INSIDE this section, if any
      "language": "he",
      "effective_from": "1973-05-17" | "",
      "effective_until": "" | "2020-01-01",
      "is_current": true,
      "source": "manual_upload",
      "source_url": "...",
      "title": "...",                 # duplicate of law_name -- kept so this collection
      "hierarchy_path": "...",        # stays readable by any code expecting the same
                                       # metadata keys as the older knesset_laws collection
    }

id shape: LAW_<law_id>_SECTION_<section-or-'preamble'>_<i>, matching the
architecture doc's "LAW_001234_SECTION_12" convention. law_id is derived
the same way embed_local_law_pdfs.py does it (from the file's path
relative to uploads/, not from content), so ids are stable across reruns.

--- Idempotency ---

Same pattern as embed_local_law_pdfs.py: a SHA-256 of each PDF's bytes is
cached in data/local_law_uploads/state_bgem3.json (a SEPARATE state file --
this script's cache must not be shared with the nomic-embed-text script's,
since "already embedded" means something different in each collection).
An unchanged file is skipped; a changed file has its old chunks deleted
before new ones are added.
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
STATE_PATH = PROJECT_ROOT / "data" / "local_law_uploads" / "state_bgem3.json"

CHROMA_DIR = str(PROJECT_ROOT / "app" / "chroma_db")
COLLECTION_NAME = "israeli_legal_db"   # the "ISRAELI LEGAL DATABASE" of the architecture doc

OLLAMA_EMBED_URL = "http://localhost:11434/api/embeddings"
EMBED_MODEL = "bge-m3"
# Unlike nomic-embed-text (which needs "search_document: " / "search_query: "
# task prefixes for good retrieval quality -- see embed_knesset_to_chroma.py),
# BGE-M3's dense-embedding mode was NOT trained with a required prefix
# convention, so none is added here. The SAME model/endpoint must be used
# at query time in app/legal_pipeline_v2.py -- see that module's docstring.
_MAX_CHUNK_CHARS = 2500

_SECTION_SPLIT_RE = re.compile(r"^\s*(סעיף\s+(\d+[א-ת]?))", re.MULTILINE)
# Chapter headings ("פרק ראשון", "פרק 3", "פרק שלישי: הגדרות") -- used only
# to tag each section with the chapter it falls under, per the doc's
# Law -> Chapter -> Section -> Subsection hierarchy. Deliberately loose (no
# attempt to parse/normalize the Hebrew ordinal word itself) -- the raw
# heading text is stored as-is; good enough for citation/display purposes.
_CHAPTER_RE = re.compile(r"^\s*(פרק\s+[^\n:]{1,30})", re.MULTILINE)
# Subsection markers INSIDE a section's own text, e.g. "(א)", "(ב1)" at the
# start of a line -- collected as metadata, not used as a further chunk
# boundary (splitting per-subsection would frequently orphan a subsection
# from the chapeau/opening clause of its own section, which is exactly the
# kind of context-loss the architecture doc warns against ("arbitrary
# 500-token pieces")). A subsection citation like "12(א)" still resolves
# fine against a whole-section chunk.
_SUBSECTION_RE = re.compile(r"^\s*\(([א-ת]\d?)\)", re.MULTILINE)


def _embed_text(text: str) -> list[float] | None:
    try:
        resp = requests.post(
            OLLAMA_EMBED_URL,
            json={"model": EMBED_MODEL, "prompt": text},
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json()["embedding"]
    except Exception as e:
        print(f"[embed-bgem3] embedding call failed: {e}")
        return None


def _nearest_chapter_for(pos: int, chapters: list[tuple[int, str]]) -> str:
    """chapters is a list of (char_offset, chapter_label) sorted ascending.
    Returns the label of the last chapter heading at or before `pos`, or ''
    if the section appears before any chapter heading (a law with no
    chapters at all, or a preamble before Chapter 1)."""
    label = ""
    for offset, chapter_label in chapters:
        if offset > pos:
            break
        label = chapter_label
    return label


def _split_into_sections(markdown_text: str) -> list[tuple[str | None, str, str]]:
    """Returns [(section_number_or_None, chapter_label, section_text), ...].
    Same סעיף-header splitting as embed_local_law_pdfs.py, plus chapter
    tracking (see _CHAPTER_RE above)."""
    chapters = [(m.start(), m.group(1).strip()) for m in _CHAPTER_RE.finditer(markdown_text)]
    matches = list(_SECTION_SPLIT_RE.finditer(markdown_text))

    if not matches:
        return [(None, "", markdown_text)]

    pieces = []
    preamble = markdown_text[:matches[0].start()].strip()
    if preamble:
        pieces.append((None, _nearest_chapter_for(0, chapters), preamble))

    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(markdown_text)
        section_num = m.group(2)
        section_text = markdown_text[start:end].strip()
        if section_text:
            pieces.append((section_num, _nearest_chapter_for(start, chapters), section_text))

    final = []
    for section_num, chapter_label, text in pieces:
        if len(text) <= _MAX_CHUNK_CHARS:
            final.append((section_num, chapter_label, text))
        else:
            for sub_chunk in chunk_text(text, max_chars=_MAX_CHUNK_CHARS):
                final.append((section_num, chapter_label, sub_chunk))
    return final


def _subsections_in(text: str) -> str:
    letters = list(dict.fromkeys(_SUBSECTION_RE.findall(text)))  # dedupe, keep order
    return ",".join(letters)


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 16), b""):
            h.update(block)
    return h.hexdigest()


def _slugify(text: str) -> str:
    text = text.replace("/", "__").replace("\\", "__")
    text = re.sub(r"\s+", "_", text.strip())
    text = re.sub(r"[^\w\-]+", "", text, flags=re.UNICODE)
    return text[:120] if text else "unnamed"


def _derive_law_id(pdf_path: Path, uploads_dir: Path) -> str:
    rel = pdf_path.relative_to(uploads_dir).with_suffix("")
    return _slugify(str(rel))


def _load_sidecar_metadata(pdf_path: Path) -> dict:
    sidecar = pdf_path.with_suffix(".json")
    if not sidecar.exists():
        return {}
    try:
        return json.loads(sidecar.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"[embed-bgem3] WARNING: couldn't parse sidecar metadata {sidecar}: {e} -- ignoring it.")
        return {}


# Same masthead/title-guessing heuristic as embed_local_law_pdfs.py -- see
# that script's _guess_title docstring for the full reasoning (in short:
# naive "first non-empty line" picks up government-gazette boilerplate
# like "רשומות" instead of the real law title).
_MASTHEAD_LINES = {
    "רשומות", "ילקוט הפרסומים", "ספר החוקים", "קובץ התקנות",
    "המדינה", "מדינת ישראל", "State of Israel", "Reshumot",
}
_LAW_TITLE_START_RE = re.compile(r"^(חוק|תקנות|פקודת|פקודה|צו|חוזר)\b")


def _guess_title(markdown_text: str, pdf_path: Path) -> str:
    lines = [ln.strip().lstrip("#").strip() for ln in markdown_text.splitlines()]
    lines = [ln for ln in lines if ln][:40]
    for line in lines:
        if _LAW_TITLE_START_RE.match(line):
            return line[:200]
    for line in lines:
        if line in _MASTHEAD_LINES or len(line.split()) < 2:
            continue
        return line[:200]
    return pdf_path.stem


def _load_state() -> dict:
    if STATE_PATH.exists():
        return json.loads(STATE_PATH.read_text(encoding="utf-8"))
    return {}


def _save_state(state: dict) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


_VALID_DOCUMENT_TYPES = {"law", "regulation", "amendment", "ruling", "other"}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--uploads-dir", type=Path, default=DEFAULT_UPLOADS_DIR)
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--force", action="store_true",
                         help="Re-extract and re-embed every PDF, ignoring the unchanged-file cache.")
    args = parser.parse_args()

    uploads_dir = args.uploads_dir
    if not uploads_dir.exists():
        uploads_dir.mkdir(parents=True, exist_ok=True)
        print(f"[embed-bgem3] Created empty {uploads_dir} -- drop law PDFs in there and re-run.")
        return

    try:
        import chromadb
    except ImportError:
        print("chromadb isn't installed. Run: pip install chromadb", file=sys.stderr)
        sys.exit(1)

    pdf_paths = sorted(uploads_dir.rglob("*.pdf") if args.recursive else uploads_dir.glob("*.pdf"))
    if not pdf_paths:
        print(f"[embed-bgem3] No .pdf files found in {uploads_dir}. Nothing to do.")
        return
    print(f"[embed-bgem3] Found {len(pdf_paths)} PDF(s) in {uploads_dir}.")

    client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = client.get_or_create_collection(COLLECTION_NAME, metadata={"hnsw:space": "cosine"})
    state = _load_state()

    n_embedded, n_skipped, n_failed, n_chunks_total = 0, 0, 0, 0

    for pdf_path in pdf_paths:
        law_id = _derive_law_id(pdf_path, uploads_dir)
        rel_path = str(pdf_path.relative_to(uploads_dir))
        file_hash = _file_sha256(pdf_path)

        cached = state.get(law_id)
        if not args.force and cached and cached.get("sha256") == file_hash:
            n_skipped += 1
            continue

        print(f"[embed-bgem3] Processing {rel_path}  (law_id={law_id})")
        try:
            hebrew_used = resolve_hebrew_flag(str(pdf_path), False)
            markdown_text = convert_to_markdown(str(pdf_path), hebrew=hebrew_used)
        except Exception as e:
            print(f"[embed-bgem3]   FAILED to extract text: {e}")
            n_failed += 1
            continue

        if len(markdown_text.strip()) < 20:
            print(f"[embed-bgem3]   WARNING: extraction produced almost no text -- skipping.")
            n_failed += 1
            continue

        meta_overrides = _load_sidecar_metadata(pdf_path)
        title = meta_overrides.get("law_name") or _guess_title(markdown_text, pdf_path)
        document_type = meta_overrides.get("document_type", "law")
        if document_type not in _VALID_DOCUMENT_TYPES:
            print(f"[embed-bgem3]   WARNING: document_type {document_type!r} not one of "
                  f"{sorted(_VALID_DOCUMENT_TYPES)} -- storing as 'other'.")
            document_type = "other"
        source = meta_overrides.get("source", "manual_upload")
        source_url = meta_overrides.get("source_url", "")
        effective_from = meta_overrides.get("effective_from", "")
        effective_until = meta_overrides.get("effective_until") or ""
        is_current = bool(meta_overrides.get("is_current", True))

        sections = _split_into_sections(markdown_text)

        try:
            existing = collection.get(where={"law_id": law_id})
            if existing.get("ids"):
                collection.delete(ids=existing["ids"])
        except Exception:
            pass

        ids, embeddings, documents, metadatas = [], [], [], []
        for i, (section_num, chapter_label, section_text) in enumerate(sections):
            vector = _embed_text(section_text)
            if vector is None:
                print(f"[embed-bgem3]   skipping one chunk, embedding call failed "
                      "(is `ollama serve` running with `bge-m3` pulled?)")
                continue
            chunk_id = f"LAW_{law_id}_SECTION_{section_num or 'preamble'}_{i}"
            ids.append(chunk_id)
            embeddings.append(vector)
            documents.append(section_text)
            metadatas.append({
                "document_type": document_type,
                "law_name": title,
                "law_id": law_id,               # not in the doc's example schema, but needed so
                                                  # this collection still supports "fetch the whole
                                                  # law" the way app/ui.py's existing tabs do
                "chapter": chapter_label,
                "section": section_num or "",
                "subsection": _subsections_in(section_text),
                "language": "he",
                "effective_from": effective_from,
                "effective_until": effective_until,
                "is_current": is_current,
                "source": source,
                "source_url": source_url,
                # Duplicated under the older knesset_laws collection's key
                # names too, so any code written against that schema (e.g.
                # app/ui.py's _section_sort_key) keeps working unmodified
                # against this collection as well.
                "title": title,
                "section_number": section_num or "",
                "hierarchy_path": title,
            })

        if not ids:
            print(f"[embed-bgem3]   No chunks embedded for {rel_path} -- not marking as done.")
            n_failed += 1
            continue

        collection.add(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)
        n_embedded += 1
        n_chunks_total += len(ids)
        state[law_id] = {"sha256": file_hash, "relative_path": rel_path}
        _save_state(state)
        print(f"[embed-bgem3]   OK -- {len(ids)} chunk(s) embedded as {title!r} "
              f"(document_type={document_type}, is_current={is_current})")

    print(
        f"\n[embed-bgem3] Done. {n_embedded} PDF(s) embedded ({n_chunks_total} chunk(s) total), "
        f"{n_skipped} unchanged/skipped, {n_failed} failed. "
        f"Collection '{COLLECTION_NAME}' at {CHROMA_DIR}."
    )
    if n_failed:
        print("[embed-bgem3] Some PDFs failed -- see WARNING/FAILED lines above. Re-running will retry them.")


if __name__ == "__main__":
    main()
