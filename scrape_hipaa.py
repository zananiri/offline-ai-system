#!/usr/bin/env python3
"""
Scrape HIPAA's regulatory text -- 45 CFR Part 160 (General Administrative
Requirements) and Part 164 (Security and Privacy) -- into a JSONL file, one
record per section. Same output shape/spirit as scrape_gdpr.py, so it can be
fed straight into embed_hipaa_to_chroma.py below.

Source: https://www.law.cornell.edu/cfr/text/45 (Cornell Law School's LII --
Legal Information Institute), a well-established, widely-cited, official
electronic CFR mirror. Unlike GDPR's 99 sequentially-numbered articles,
HIPAA's sections aren't a clean 1..N range (160.101, 160.102, ... 164.534,
164.535, with gaps and [Reserved] slots) -- so this is a two-phase scrape:

  1. DISCOVERY: fetch each Part/Subpart's table-of-contents page (e.g.
     .../cfr/text/45/part-164/subpart-E) and pull the list of section links
     it publishes -- this is the authoritative, always-current list of
     which sections actually exist in that subpart.
  2. SCRAPE: fetch each discovered section's own page and extract its body.

Scope: Part 160 (Subparts A-E: General Provisions, Preemption of State Law,
Compliance and Investigations, Civil Money Penalties, Procedures for
Hearings) and Part 164 (Subparts A, C, D, E: General Provisions, Security
Standards, Breach Notification, Privacy -- Subpart B is officially
"Reserved" and skipped). Part 162 (Transactions and Code Sets -- mostly
administrative/EDI standards, not the "HIPAA Rules" people usually mean)
is deliberately out of scope; pass --parts 160,162,164 to include it if
you want it, once you've confirmed its subpart slugs match the pattern
this script assumes.

This is a courtesy scrape of a small (~120-page), public, non-paywalled
legal reference site -- polite by design: a single sequential pass, a real
User-Agent identifying the tool, a default 1s delay between requests, and
built-in retry/backoff instead of hammering on failure.

Usage:
    pip install requests beautifulsoup4
    python scrape_hipaa.py --output-dir ./hipaa_output

Output:
    ./hipaa_output/hipaa_sections.jsonl -- one JSON object per section:
        {
          "chunk_id": "HIPAA-45-CFR-164.312",
          "section_id": "164.312",
          "title": "Technical safeguards.",
          "part_number": 164,
          "part_title": "Security and Privacy",
          "subpart_letter": "C",
          "subpart_title": "Security Standards for the Protection of Electronic Protected Health Information",
          "hierarchy_path": "Part 164 (Security and Privacy) > Subpart C (...) > \u00a7 164.312",
          "text": "A covered entity or business associate must...",
          "embed_text": "HIPAA 45 CFR \u00a7 164.312 -- Technical safeguards. ...",
          "cross_references": ["164.306", "164.308"],
          "source_url": "https://www.law.cornell.edu/cfr/text/45/164.312"
        }
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup, NavigableString, Tag

BASE_URL = "https://www.law.cornell.edu/cfr/text/45"

USER_AGENT = (
    "hipaa-scraper/1.0 (offline-ai-system; local research/RAG corpus build; "
    "polite sequential scrape)"
)

# Which Part/Subpart table-of-contents pages to crawl. This is fixed,
# official CFR structure (like scrape_gdpr.py's hardcoded chapter map) --
# stable even as the underlying section TEXT gets amended over time. Titles
# are still read live from the h1 of the TOC pages themselves, not
# hardcoded, but a "part-{N}" whose 5-subpart structure changes entirely
# would be a genuine, rare restructuring of the CFR -- worth re-checking by
# hand if this ever needs updating (see README note this script prints on
# a total mismatch).
PART_TITLES = {
    160: "General Administrative Requirements",
    162: "Administrative Requirements",  # Transactions and Code Sets -- out of scope by default
    164: "Security and Privacy",
}
DEFAULT_SUBPART_SLUGS = {
    160: ["subpart-A", "subpart-B", "subpart-C", "subpart-D", "subpart-E"],
    162: [],  # not scraped by default -- see module docstring
    164: ["subpart-A", "subpart-C", "subpart-D", "subpart-E"],  # Subpart B is [Reserved]
}
DEFAULT_PARTS = [160, 164]


def _dump_debug_html(debug_dump_dir: Path, name: str, html: str, reason: str) -> None:
    debug_dump_dir.mkdir(parents=True, exist_ok=True)
    path = debug_dump_dir / f"{name}-{reason}.html"
    path.write_text(html, encoding="utf-8")
    print(f"  [debug] saved raw HTML to {path} (reason: {reason})", file=sys.stderr)


def _build_session(retries: int = 3, backoff: float = 1.5) -> requests.Session:
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})
    adapter = requests.adapters.HTTPAdapter(
        max_retries=requests.adapters.Retry(
            total=retries, backoff_factor=backoff,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=("GET",),
        )
    )
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


# --- Phase 1: discovery -----------------------------------------------------

_SUBPART_H1_RE = re.compile(
    r"^\d+\s*CFR\s*Part\s*(\d+)\s*-\s*Subpart\s*([A-Z])\s*-\s*(.+?)\s*$", re.IGNORECASE
)
_SECTION_LINK_RE = re.compile(r"/cfr/text/45/(\d{3}\.\d+[a-zA-Z]?)$")


def discover_sections_in_subpart(session: requests.Session, part: int, subpart_slug: str,
                                  debug_dump_dir: Path | None = None):
    """
    Fetches one Part/Subpart TOC page (e.g. .../part-164/subpart-E) and
    returns (subpart_letter, subpart_title, [(section_id, title, url), ...])
    in the order the page lists them -- or None on failure.

    Section links on these TOC pages are identified purely by URL shape
    (.../cfr/text/45/{NNN.NNN}) rather than by guessing a container class,
    since these index pages have no other body content that would produce
    a false-positive match (confirmed against live examples: prev/next
    links point at other subpart/part pages, which don't match this
    pattern at all).
    """
    url = f"{BASE_URL}/part-{part}/{subpart_slug}"
    try:
        resp = session.get(url, timeout=30)
        resp.raise_for_status()
    except requests.RequestException as e:
        print(f"  [error] failed to fetch subpart TOC {url}: {e}", file=sys.stderr)
        return None

    soup = BeautifulSoup(resp.text, "html.parser")
    h1 = soup.find("h1")
    subpart_letter, subpart_title = None, None
    if h1:
        m = _SUBPART_H1_RE.match(h1.get_text(" ", strip=True))
        if m:
            subpart_letter, subpart_title = m.group(2), m.group(3).rstrip(".")
    if subpart_letter is None:
        # Fall back to the slug itself ("subpart-E" -> "E") rather than
        # failing outright -- title just won't be as nicely worded.
        subpart_letter = subpart_slug.replace("subpart-", "")
        subpart_title = ""
        print(f"  [warn] couldn't parse subpart title from h1 on {url}; "
              f"using slug-derived letter {subpart_letter!r}", file=sys.stderr)
        if debug_dump_dir is not None:
            _dump_debug_html(debug_dump_dir, f"part-{part}-{subpart_slug}", resp.text, "no-h1-title-match")

    sections = []
    seen_ids = set()
    for a in soup.find_all("a", href=True):
        path = urlparse(a["href"]).path
        m = _SECTION_LINK_RE.search(path)
        if not m:
            continue
        section_id = m.group(1)
        if section_id in seen_ids:
            continue
        link_text = a.get_text(" ", strip=True)
        if "[reserved]" in link_text.lower():
            print(f"  [skip] {section_id}: marked [Reserved] in TOC, no content to scrape")
            seen_ids.add(section_id)
            continue
        # Link text looks like "§ 164.312 Technical safeguards." -- strip
        # the leading "§ {id}" so `title` holds just the human title.
        title = re.sub(rf"^§\s*{re.escape(section_id)}\s*", "", link_text).strip()
        seen_ids.add(section_id)
        sections.append((section_id, title, f"{BASE_URL}/{section_id}"))

    return subpart_letter, subpart_title, sections


# --- Phase 2: section body extraction ---------------------------------------
#
# Cornell LII's CFR section pages render regulatory numbering ("(a)", "(1)",
# "(i)") as literal, already-present TEXT inside nested <div>/<p> wrappers --
# unlike gdpr-info.eu, where equivalent markers are CSS-generated and have to
# be reconstructed (see scrape_gdpr.py's render_list). So here the job is
# simpler: walk the DOM in document order collecting each leaf block's own
# text, without needing to invent any numbering ourselves. "Leaf block" means
# an element with no non-inline child elements (a <div> that's just text +
# hyperlinks); an element that itself has such element AND further nested
# containers (e.g. a "(2) Implementation specifications:" div that also
# wraps "(i)"/"(ii)" sub-divs) contributes its own direct text as one line,
# then recurses into the nested containers for more lines -- so nothing is
# silently dropped or duplicated.

_INLINE_TAGS = {"a", "em", "strong", "b", "i", "span", "sup", "sub", "br", "small", "code", "u"}


def _is_inline(tag_name: str) -> bool:
    return tag_name in _INLINE_TAGS


def _own_text(node: Tag) -> str:
    parts = []
    for child in node.children:
        if isinstance(child, NavigableString):
            parts.append(str(child))
        elif isinstance(child, Tag) and _is_inline(child.name):
            parts.append(child.get_text(" "))
        # else: a non-inline container child -- handled by recursion elsewhere.
    text = " ".join(p.strip() for p in parts if p.strip())
    return " ".join(text.split())


def collect_leaf_blocks(root: Tag) -> list[str]:
    """Returns the page's block-level text content, in document order, one
    entry per leaf block (see module comment above)."""
    lines: list[str] = []

    def _walk(node: Tag):
        container_children = [
            c for c in node.children if isinstance(c, Tag) and not _is_inline(c.name)
        ]
        if not container_children:
            text = node.get_text(" ", strip=True)
            if text:
                lines.append(text)
            return
        own = _own_text(node)
        if own:
            lines.append(own)
        for child in container_children:
            if child.name in ("ol", "ul"):
                for li in child.find_all("li", recursive=False):
                    _walk(li)
            else:
                _walk(child)

    _walk(root)
    return lines


_H1_TITLE_RE = re.compile(r"^\d+\s*CFR\s*§\s*([\d.]+[A-Za-z]?)\s*-\s*(.+?)\.?\s*$", re.IGNORECASE)
_CROSS_REF_RE = re.compile(r"§+\s*(\d{3}\.\d+[a-zA-Z]?)")
_STOP_MARKERS = {"cfr toolbox"}


def parse_section_page(html: str, expected_section_id: str, source_url: str,
                        part: int, part_title: str, subpart_letter: str, subpart_title: str,
                        debug_dump_dir: Path | None = None) -> dict | None:
    soup = BeautifulSoup(html, "html.parser")
    for junk in soup(["script", "style", "noscript"]):
        junk.decompose()

    h1 = soup.find("h1")
    section_id, title = expected_section_id, None
    if h1:
        m = _H1_TITLE_RE.match(h1.get_text(" ", strip=True))
        if m:
            section_id, title = m.group(1), m.group(2).strip()
    if title is None:
        title = h1.get_text(" ", strip=True) if h1 else ""

    body_root = soup.body or soup
    lines = collect_leaf_blocks(body_root)

    marker_prefix = f"§ {section_id}"
    start_idx = next((i for i, l in enumerate(lines) if l.startswith(marker_prefix)), None)
    if start_idx is None:
        if debug_dump_dir is not None:
            _dump_debug_html(debug_dump_dir, f"section-{section_id}", html, "no-start-marker")
        return None

    end_idx = next(
        (i for i in range(start_idx, len(lines)) if lines[i].strip().lower() in _STOP_MARKERS),
        len(lines),
    )
    if end_idx <= start_idx:
        if debug_dump_dir is not None:
            _dump_debug_html(debug_dump_dir, f"section-{section_id}", html, "empty-slice")
        return None

    body_lines = list(lines[start_idx:end_idx])

    # The marker line doubles as the first sentence of substantive text
    # (e.g. "§ 164.312 Technical safeguards. A covered entity ... must,
    # in accordance with § 164.306:") -- strip just the "§ {id} {title}"
    # label prefix off its front, keeping whatever real text follows it.
    first = body_lines[0]
    stripped = re.sub(rf"^§\s*{re.escape(section_id)}\s*", "", first).strip()
    if title:
        # Title text in the body may have slightly different trailing
        # punctuation than the h1's version -- strip it case-insensitively
        # up to the title's own length as a best-effort, falling back to
        # leaving it untouched if it doesn't actually match (better to
        # keep a harmless duplicate label than accidentally eat real text).
        title_norm = title.rstrip(".")
        if stripped.lower().startswith(title_norm.lower()):
            stripped = stripped[len(title_norm):].lstrip(". ").strip()
    body_lines[0] = stripped

    body_text = "\n\n".join(l for l in body_lines if l).strip()
    if not body_text:
        if debug_dump_dir is not None:
            _dump_debug_html(debug_dump_dir, f"section-{section_id}", html, "empty-body-after-strip")
        return None

    cross_refs = sorted({
        ref for ref in _CROSS_REF_RE.findall(body_text) if ref != section_id
    })

    hierarchy_path = (
        f"Part {part} ({part_title}) > Subpart {subpart_letter}"
        + (f" ({subpart_title})" if subpart_title else "")
        + f" > § {section_id}"
    )
    embed_text = (
        f"HIPAA 45 CFR § {section_id} -- {title}\n"
        f"({hierarchy_path})\n\n"
        f"{body_text}"
    )

    return {
        "chunk_id": f"HIPAA-45-CFR-{section_id}",
        "section_id": section_id,
        "title": title,
        "part_number": part,
        "part_title": part_title,
        "subpart_letter": subpart_letter,
        "subpart_title": subpart_title,
        "hierarchy_path": hierarchy_path,
        "text": body_text,
        "embed_text": embed_text,
        "cross_references": cross_refs,
        "source_url": source_url,
    }


# --- Orchestration -----------------------------------------------------------

def scrape(output_dir: Path, parts: list[int], delay: float,
           debug_dump_dir: Path | None = None, include_reserved: bool = False) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "hipaa_sections.jsonl"

    session = _build_session()
    records = []
    failures = []

    # --- Phase 1: discover every section across the requested parts ---
    to_scrape = []  # (section_id, title, url, part, part_title, subpart_letter, subpart_title)
    for part in parts:
        part_title = PART_TITLES.get(part, "")
        slugs = DEFAULT_SUBPART_SLUGS.get(part)
        if not slugs:
            print(f"[discover] no known subpart list for Part {part} -- skipping "
                  f"(add it to DEFAULT_SUBPART_SLUGS if you need this part).", file=sys.stderr)
            continue
        for slug in slugs:
            print(f"[discover] Part {part} / {slug} ...")
            result = discover_sections_in_subpart(session, part, slug, debug_dump_dir)
            time.sleep(delay)
            if result is None:
                failures.append(f"part-{part}/{slug} (TOC fetch failed)")
                continue
            subpart_letter, subpart_title, sections = result
            print(f"  -> Subpart {subpart_letter} ({subpart_title or '?'}): {len(sections)} section(s)")
            for section_id, title, url in sections:
                to_scrape.append((section_id, title, url, part, part_title, subpart_letter, subpart_title))

    print(f"\n[discover] {len(to_scrape)} section(s) found across {len(parts)} part(s). Scraping...\n")

    # --- Phase 2: scrape each discovered section ---
    for i, (section_id, toc_title, url, part, part_title, subpart_letter, subpart_title) in enumerate(to_scrape, 1):
        print(f"[{i}/{len(to_scrape)}] fetching § {section_id} -- {url}")
        try:
            resp = session.get(url, timeout=30)
            resp.raise_for_status()
        except requests.RequestException as e:
            print(f"  [error] failed to fetch § {section_id}: {e}", file=sys.stderr)
            failures.append(section_id)
            time.sleep(delay)
            continue

        record = parse_section_page(
            resp.text, section_id, url, part, part_title, subpart_letter, subpart_title,
            debug_dump_dir=debug_dump_dir,
        )
        if record is None:
            print(f"  [warn] § {section_id}: could not extract section body, skipping", file=sys.stderr)
            failures.append(section_id)
        else:
            records.append(record)
            print(f"  -> {record['title']!r} ({len(record['text'])} chars, "
                  f"{len(record['cross_references'])} cross-ref(s))")

        time.sleep(delay)

    with open(out_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"\nDone. Wrote {len(records)} section(s) to {out_path}")
    if failures:
        print(f"WARNING: {len(failures)} item(s) failed/skipped: {failures}", file=sys.stderr)
        if debug_dump_dir is not None:
            print(f"Raw HTML of failed pages saved under {debug_dump_dir} -- inspect those "
                  f"to see why parsing failed.", file=sys.stderr)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output-dir", default="./hipaa_output", help="Directory to write hipaa_sections.jsonl into")
    ap.add_argument(
        "--parts", default=",".join(str(p) for p in DEFAULT_PARTS),
        help=f"Comma-separated CFR Part numbers to scrape (default: {','.join(str(p) for p in DEFAULT_PARTS)}). "
             "Part 162 is recognized but has no subpart slugs configured by default -- see module docstring.",
    )
    ap.add_argument("--delay", type=float, default=1.0, help="Seconds to wait between requests (default 1.0 -- be polite)")
    ap.add_argument(
        "--debug-dir", default=None,
        help="If set, saves the raw HTML of any page that fails to parse into this directory "
             "(e.g. ./debug_html), so you can inspect why. Off by default.",
    )
    args = ap.parse_args()

    try:
        parts = [int(p.strip()) for p in args.parts.split(",") if p.strip()]
    except ValueError:
        ap.error("--parts must be a comma-separated list of integers, e.g. 160,164")

    debug_dump_dir = Path(args.debug_dir) if args.debug_dir else None
    scrape(Path(args.output_dir), parts, args.delay, debug_dump_dir=debug_dump_dir)


if __name__ == "__main__":
    main()
