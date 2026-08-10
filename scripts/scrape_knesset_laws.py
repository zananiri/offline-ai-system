"""
Pulls new/updated Israeli law texts from the Knesset's official OData API
and downloads their official PDF documents, ready for embed_knesset_to_chroma.py.

--- Why this is shaped the way it is (read before changing anything) ---

The Knesset OData service (https://knesset.gov.il/Odata/ParliamentInfo.svc)
exposes TWO different, DISJOINT id spaces for "a law", confirmed by hand
against the live API (not assumed from docs, which don't document this):

    KNS_IsraelLaw   -- a legislation *registry*: name, in-force/repealed
                       status, Basic Law flag. IsraelLawID keys. NO document
                       / full-text link at all.
    KNS_Law         -- published law *versions* (including consolidated
                       "נוסח משולב" texts and individual amendments). LawID
                       keys. This is what actually has text behind it.
    KNS_DocumentLaw -- official PDF documents for a KNS_Law row, filtered
                       by LawID. Returns real fs.knesset.gov.il PDF links.

There is no documented/working join between IsraelLawID and LawID (the
obvious-looking KNS_IsraelLawBinding table only covers law-*replacement*
events and is sparse). So this script deliberately queries KNS_Law directly
-- NOT KNS_IsraelLaw -- and walks straight to KNS_DocumentLaw from there.
If you came here wanting "is law X still in force" metadata, that's a
separate KNS_IsraelLaw query and a separate, smaller script; this one is
about getting real statute TEXT into the RAG corpus.

The API is classic OData v2 (ASP.NET Data Services): $format=json,
substringof(...), $orderby, $top/$skip paging, datetime'...' literals.
It is NOT geo-blocked (re-verified live from outside Israel as of the
writing of the code this script's approach was modeled on).

--- What "pull new laws" means here ---

Incremental by LastUpdatedDate (client-side sorted/filtered -- see
_iter_recent_laws' comment for why the sort is done client-side rather
than trusted to $orderby). A checkpoint file records the newest
LastUpdatedDate seen on the last successful run; each run only downloads
KNS_Law rows updated after that checkpoint. Delete the checkpoint file to
force a full re-pull.

--- Output ---

    data/knesset_laws/raw/{law_id}_{doc_id}.pdf   -- downloaded PDFs
    data/knesset_laws/manifest.jsonl               -- one line per LAW
                                                        (not per PDF) with
                                                        metadata + local
                                                        PDF paths, for
                                                        embed_knesset_to_chroma.py
    data/knesset_laws/state.json                   -- checkpoint

Run from the project root (same convention as the other scripts/ files):
    python scripts/scrape_knesset_laws.py               # incremental
    python scripts/scrape_knesset_laws.py --full         # ignore checkpoint
    python scripts/scrape_knesset_laws.py --limit 200    # cap this run
"""
import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "data" / "knesset_laws"
RAW_DIR = OUTPUT_DIR / "raw"
MANIFEST_PATH = OUTPUT_DIR / "manifest.jsonl"
STATE_PATH = OUTPUT_DIR / "state.json"

BASE_URL = "https://knesset.gov.il/Odata/ParliamentInfo.svc"
LAW_ENTITY = "KNS_Law"
DOCUMENT_ENTITY = "KNS_DocumentLaw"

PAGE_SIZE = 100          # OData server-side default page size for this service
REQUEST_TIMEOUT = 60
RETRY_ATTEMPTS = 3
RETRY_BACKOFF_SECONDS = 5

# The exact field name for "when was this law version last touched" isn't
# nailed down with 100% certainty from documentation alone (the Knesset
# doesn't publish a schema doc for this service) -- so rather than hardcode
# one guess and silently sort/filter on nothing if it's wrong, every
# candidate is tried against a live sample record and the first one that's
# actually present on real rows wins. Logged clearly either way.
_DATE_FIELD_CANDIDATES = ["LastUpdatedDate", "PublicationDate", "LawDate", "CreatedDate"]


def _http_get_json(url: str, params: dict | None = None) -> dict:
    """GET with retries. Raises on final failure -- callers decide how to
    handle a fully-unreachable API rather than this silently returning {}."""
    last_exc = None
    for attempt in range(1, RETRY_ATTEMPTS + 1):
        try:
            resp = requests.get(url, params=params, timeout=REQUEST_TIMEOUT,
                                 headers={"Accept": "application/json"})
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            last_exc = e
            print(f"[knesset] GET {url} attempt {attempt}/{RETRY_ATTEMPTS} failed: {e}")
            if attempt < RETRY_ATTEMPTS:
                time.sleep(RETRY_BACKOFF_SECONDS)
    raise RuntimeError(f"Could not reach Knesset OData API at {url}: {last_exc}")


def _detect_date_field(sample_row: dict) -> str | None:
    for candidate in _DATE_FIELD_CANDIDATES:
        if candidate in sample_row:
            return candidate
    return None


def _parse_odata_date(value) -> datetime | None:
    """OData v2 JSON dates commonly come back either as an ISO string or
    the classic '/Date(1234567890000)/' epoch-millisecond wrapper. Handles
    both defensively; returns None (never raises) if the shape is
    unrecognized, since a missing sort key shouldn't crash the whole run."""
    if not value:
        return None
    if isinstance(value, str) and value.startswith("/Date(") and value.endswith(")/"):
        try:
            millis = int(value[len("/Date("):-len(")/")].split("+")[0].split("-")[0])
            return datetime.fromtimestamp(millis / 1000, tz=timezone.utc)
        except Exception:
            return None
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except Exception:
            return None
    return None


def _load_state() -> dict:
    if STATE_PATH.exists():
        return json.loads(STATE_PATH.read_text(encoding="utf-8"))
    return {"last_seen_date_iso": None, "date_field": None}


def _save_state(state: dict) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def _iter_all_law_rows(page_limit: int | None):
    """
    Pages through the FULL KNS_Law collection ($top/$skip), yielding raw
    rows as-is.

    Deliberately does NOT rely on $orderby against the live service: OData
    v2 servers built on older ASP.NET Data Services stacks (which this one
    is) sometimes silently ignore an $orderby on a field that isn't
    indexed for it rather than erroring, which would make an
    "incremental by date" script quietly wrong instead of loudly broken.
    Safer to page through everything the API is willing to give us in this
    run and sort/filter client-side once we can see the real field names
    on real rows (see _detect_date_field).
    """
    skip = 0
    fetched = 0
    while True:
        params = {"$format": "json", "$top": PAGE_SIZE, "$skip": skip}
        url = f"{BASE_URL}/{LAW_ENTITY}"
        data = _http_get_json(url, params=params)
        rows = data.get("value") if isinstance(data, dict) else None
        if rows is None:
            # Single-entity responses aren't wrapped in "value" (confirmed
            # behavior of this service) -- a collection query returning an
            # un-wrapped dict would be a real schema surprise, so surface it
            # instead of pretending there's no more data.
            raise RuntimeError(
                f"Unexpected response shape from {url} (no 'value' key): "
                f"{str(data)[:300]}"
            )
        if not rows:
            break
        for row in rows:
            yield row
        fetched += len(rows)
        print(f"[knesset] fetched {fetched} {LAW_ENTITY} rows so far...")
        if page_limit and fetched >= page_limit:
            return
        if len(rows) < PAGE_SIZE:
            break  # short page = last page
        skip += PAGE_SIZE


def fetch_documents_for_law(law_id) -> list[dict]:
    """Returns the KNS_DocumentLaw rows (PDF links) for one LawID."""
    url = f"{BASE_URL}/{DOCUMENT_ENTITY}"
    params = {"$format": "json", "$filter": f"LawID eq {int(law_id)}"}
    data = _http_get_json(url, params=params)
    return data.get("value", []) if isinstance(data, dict) else []


def _download_pdf(url: str, dest_path: Path) -> bool:
    if dest_path.exists() and dest_path.stat().st_size > 0:
        return True  # already have it
    try:
        resp = requests.get(url, timeout=REQUEST_TIMEOUT, stream=True)
        resp.raise_for_status()
        content_type = resp.headers.get("Content-Type", "")
        if "pdf" not in content_type.lower() and not url.lower().endswith(".pdf"):
            print(f"[knesset] WARNING: {url} did not look like a PDF "
                  f"(Content-Type: {content_type!r}) -- saving anyway, "
                  "verify manually if extraction later fails.")
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        with open(dest_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=1 << 16):
                f.write(chunk)
        return True
    except Exception as e:
        print(f"[knesset] Failed to download {url}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true",
                         help="Ignore the checkpoint and re-pull everything.")
    parser.add_argument("--limit", type=int, default=None,
                         help="Cap the number of KNS_Law rows fetched this run (for testing).")
    args = parser.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    state = {} if args.full else _load_state()
    last_seen_date = None
    if state.get("last_seen_date_iso"):
        last_seen_date = datetime.fromisoformat(state["last_seen_date_iso"])
        print(f"[knesset] Incremental run: only laws updated after {last_seen_date.isoformat()}")
    else:
        print("[knesset] Full run: no checkpoint found (or --full passed).")

    date_field = state.get("date_field")
    newest_seen = last_seen_date
    manifest_entries = []
    n_new = 0

    with open(MANIFEST_PATH, "a", encoding="utf-8") as manifest_f:
        for row in _iter_all_law_rows(page_limit=args.limit):
            if date_field is None:
                date_field = _detect_date_field(row)
                if date_field:
                    print(f"[knesset] Detected date field on {LAW_ENTITY}: {date_field!r}")
                else:
                    print(f"[knesset] WARNING: none of {_DATE_FIELD_CANDIDATES} found on a "
                          f"sample row (keys present: {list(row.keys())}). Incremental "
                          "filtering by date will be skipped this run -- every row will "
                          "be treated as new. Re-check this script's _DATE_FIELD_CANDIDATES "
                          "against the live schema.")

            row_date = _parse_odata_date(row.get(date_field)) if date_field else None
            if last_seen_date and row_date and row_date <= last_seen_date:
                continue  # already pulled this one in a previous run

            law_id = row.get("LawID")
            if law_id is None:
                continue
            name = row.get("Name", "") or row.get("LawName", "")

            docs = fetch_documents_for_law(law_id)
            pdf_paths = []
            source_urls = []
            for doc in docs:
                file_url = doc.get("FilePath")
                if not file_url:
                    continue
                doc_id = doc.get("Id") or doc.get("DocumentId") or law_id
                dest = RAW_DIR / f"{law_id}_{doc_id}.pdf"
                if _download_pdf(file_url, dest):
                    pdf_paths.append(str(dest))
                    source_urls.append(file_url)

            if not pdf_paths:
                print(f"[knesset] LawID {law_id} ({name!r}): no downloadable PDF found, skipping.")
                continue

            entry = {
                "law_id": law_id,
                "name": name,
                "sub_type_desc": row.get("SubTypeDesc", ""),
                "knesset_num": row.get("KnessetNum"),
                "last_updated_raw": row.get(date_field) if date_field else None,
                "last_updated_iso": row_date.isoformat() if row_date else None,
                "pdf_paths": pdf_paths,
                "source_urls": source_urls,
                "fetched_at": datetime.now(timezone.utc).isoformat(),
            }
            manifest_f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            manifest_entries.append(entry)
            n_new += 1

            if row_date and (newest_seen is None or row_date > newest_seen):
                newest_seen = row_date

            if n_new % 10 == 0:
                print(f"[knesset] ...{n_new} new/updated laws downloaded so far")

    if newest_seen:
        _save_state({
            "last_seen_date_iso": newest_seen.isoformat(),
            "date_field": date_field,
        })

    print(f"[knesset] Done. {n_new} new/updated law(s) with PDFs written to {MANIFEST_PATH}")
    print("[knesset] Next: python scripts/embed_knesset_to_chroma.py")

    if n_new == 0 and not last_seen_date:
        print("[knesset] NOTE: zero laws found on a full run usually means the field names "
              "this script assumes (Name, LawID, SubTypeDesc, FilePath) don't match the live "
              "schema anymore. Try a manual check:\n"
              f"  {BASE_URL}/{LAW_ENTITY}?$format=json&$top=1", file=sys.stderr)


if __name__ == "__main__":
    main()
