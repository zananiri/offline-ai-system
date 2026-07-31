#!/usr/bin/env python3
"""
Scrape the GDPR (Regulation (EU) 2016/679) into a JSONL file, one record per
article -- same output shape/spirit as scrape_cic_it.py (which does this for
the Codice di Diritto Canonico), so it can be fed straight into
embed_gdpr_to_chroma.py below.

Source: https://gdpr-info.eu -- a well-established, widely-cited, clean
article-by-article mirror of the official text (OJ L 119, 04.05.2016; corr.
OJ L 127, 23.5.2018), maintained by intersoft consulting. Each article lives
at its own stable URL: https://gdpr-info.eu/art-{N}-gdpr/ for N in 1..99
(the GDPR has exactly 99 articles across 11 chapters).

This is a courtesy scrape of a small (99-page), public, non-paywalled legal
reference site -- polite by design: a single sequential pass, a real
User-Agent identifying the tool, a default 1s delay between requests, and
built-in retry/backoff instead of hammering on failure. Re-run only when you
actually need to refresh the corpus.

Usage:
    pip install requests beautifulsoup4
    python scrape_gdpr.py --output-dir ./gdpr_output

Output:
    ./gdpr_output/gdpr_articles.jsonl -- one JSON object per article:
        {
          "chunk_id": "GDPR-art-5",
          "article_number": 5,
          "title": "Principles relating to processing of personal data",
          "chapter_number": 2,
          "chapter_title": "Principles",
          "hierarchy_path": "Chapter 2 (Principles) > Art. 5",
          "text": "1. Personal data shall be:\\n  (a) processed lawfully...",
          "embed_text": "GDPR Article 5 -- Principles relating to ... \\n\\n1. Personal data shall be...",
          "recitals": [39, 74],
          "source_url": "https://gdpr-info.eu/art-5-gdpr/"
        }
"""

import argparse
import json
import re
import string
import sys
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup, Tag

BASE_URL = "https://gdpr-info.eu"
TOTAL_ARTICLES = 99  # GDPR has exactly 99 articles, Art. 1 through Art. 99

USER_AGENT = (
    "gdpr-scraper/1.0 (offline-ai-system; local research/RAG corpus build; "
    "polite sequential scrape, contact: set via --contact if needed)"
)

# The GDPR's chapter structure is fixed, official, and effectively never
# changes (this is a ratified regulation, not a living wiki page) -- hardcoded
# here rather than re-derived from the page's sidebar TOC on every run, which
# would mean parsing a second, more fragile piece of markup for no benefit.
# (chapter_number, chapter_title, first_article, last_article)
CHAPTERS = [
    (1, "General provisions", 1, 4),
    (2, "Principles", 5, 11),
    (3, "Rights of the data subject", 12, 23),
    (4, "Controller and processor", 24, 43),
    (5, "Transfers of personal data to third countries or international organisations", 44, 50),
    (6, "Independent supervisory authorities", 51, 59),
    (7, "Cooperation and consistency", 60, 76),
    (8, "Remedies, liability and penalties", 77, 84),
    (9, "Provisions relating to specific processing situations", 85, 91),
    (10, "Delegated acts and implementing acts", 92, 93),
    (11, "Final provisions", 94, 99),
]


def _chapter_for_article(article_number: int):
    for chapter_number, chapter_title, first, last in CHAPTERS:
        if first <= article_number <= last:
            return chapter_number, chapter_title
    return None, None


# --- Legal-style ordered-list rendering -------------------------------------
#
# gdpr-info.eu's nested <ol>/<ul> markers ((a), (b), (i), (ii), ...) are
# applied purely via CSS (list-style-type), so plain .get_text() on the raw
# HTML loses them entirely -- you're left with an unmarked blob of clauses,
# which is exactly the kind of information a lawyer/compliance reader (and a
# retrieval system asked to answer "what does Art. 6(1)(f) say") needs intact.
# This regenerates markers ourselves using the standard legal-drafting
# convention GDPR itself uses: depth 0 = "1.", "2." (arabic); depth 1 =
# "(a)", "(b)" (lower-alpha); depth 2+ = "(i)", "(ii)" (lower-roman).

def _roman(n: int) -> str:
    vals = [
        (1000, "m"), (900, "cm"), (500, "d"), (400, "cd"), (100, "c"),
        (90, "xc"), (50, "l"), (40, "xl"), (10, "x"), (9, "ix"),
        (5, "v"), (4, "iv"), (1, "i"),
    ]
    result = ""
    for v, sym in vals:
        while n >= v:
            result += sym
            n -= v
    return result


def _list_marker(depth: int, index: int) -> str:
    if depth == 0:
        return f"{index}."
    if depth == 1:
        return f"({string.ascii_lowercase[(index - 1) % 26]})"
    return f"({_roman(index)})"


def _li_own_text(li: Tag) -> str:
    """Text belonging directly to this <li>, excluding any nested <ol>/<ul>
    (those are rendered separately, indented, by render_list's recursion)."""
    clone = BeautifulSoup(str(li), "html.parser").find("li")
    for nested in clone.find_all(["ol", "ul"]):
        nested.decompose()
    return clone.get_text(" ", strip=True)


def render_list(ol_or_ul: Tag, depth: int = 0) -> list[str]:
    lines = []
    for i, li in enumerate(ol_or_ul.find_all("li", recursive=False), start=1):
        marker = _list_marker(depth, i)
        own_text = _li_own_text(li)
        indent = "  " * depth
        line = f"{indent}{marker} {own_text}".rstrip()
        if line.strip():
            lines.append(line)
        for nested in li.find_all(["ol", "ul"], recursive=False):
            lines.extend(render_list(nested, depth + 1))
    return lines


def _render_body_node(node: Tag) -> str:
    """Renders a single top-level content node (paragraph or list) to text."""
    if node.name in ("ol", "ul"):
        return "\n".join(render_list(node, depth=0))
    return node.get_text(" ", strip=True)


# --- Page parsing ------------------------------------------------------------

def _dump_debug_html(debug_dump_dir: Path, article_number: int, html: str, reason: str) -> None:
    debug_dump_dir.mkdir(parents=True, exist_ok=True)
    path = debug_dump_dir / f"art-{article_number}-{reason}.html"
    path.write_text(html, encoding="utf-8")
    print(f"  [debug] saved raw HTML to {path} (reason: {reason})", file=sys.stderr)


_TITLE_PREFIX_RE = re.compile(r"^Art\.?\s*\d+\s*GDPR\s*", re.IGNORECASE)
_RECITAL_NUM_RE = re.compile(r"\((\d+)\)")

# Candidate selectors for the element that actually holds the article body,
# tried in order. gdpr-info.eu runs a WordPress "Twenty Fifteen" child theme
# (visible in its asset paths, e.g. wp-content/themes/twentyfifteen-child/),
# whose standard markup is:
#   <article>
#     <header class="entry-header"><h1 class="entry-title">...</h1></header>
#     <div class="entry-content"> ... the actual body ... </div>
#   </article>
# i.e. the title and body live in SEPARATE sibling elements, not as flat
# siblings of the <h1> itself -- walking h1.find_next_sibling() (an earlier
# version of this scraper did that) finds nothing, because the body isn't a
# sibling of the h1 at all, it's a sibling of the h1's *parent* header.
# ".entry-content" is tried first as the specific, correct target; the
# generic fallbacks after it are only a safety net in case the theme changes.
_CONTENT_SELECTORS = [
    "div.entry-content",
    "article .entry-content",
    ".post .entry-content",
    "article",
    "main",
    "#content",
]


def _find_title_and_container(soup: BeautifulSoup):
    """Returns (title_text, container_tag) or (None, None) if neither the
    title nor a plausible content container could be found at all."""
    h1 = soup.select_one("h1.entry-title") or soup.find("h1")
    title = _TITLE_PREFIX_RE.sub("", h1.get_text(" ", strip=True)).strip() if h1 else None

    for selector in _CONTENT_SELECTORS:
        container = soup.select_one(selector)
        if container is not None:
            return title, container
    return title, None


def parse_article_page(html: str, article_number: int, source_url: str,
                        debug_dump_dir: Path | None = None) -> dict | None:
    soup = BeautifulSoup(html, "html.parser")

    title, container = _find_title_and_container(soup)
    if title is None or container is None:
        if debug_dump_dir is not None:
            _dump_debug_html(debug_dump_dir, article_number, html, "no-title-or-container")
        return None

    # Walk the container's DIRECT children (not find_all, which would also
    # pull in <li> text a second time from inside nested <ol>s, and would
    # wander into unrelated nested widgets) collecting body content --
    # paragraphs and top-level ordered/unordered lists -- stopping at the
    # "Suitable Recitals" heading or a <nav> (the prev/next article links).
    body_parts: list[str] = []
    recitals: list[int] = []
    seen_recitals_heading = False

    for node in container.find_all(recursive=False):
        if not isinstance(node, Tag):
            continue

        text_preview = node.get_text(" ", strip=True)
        if not text_preview:
            continue

        if node.name in ("h2", "h3", "h4") and "recital" in text_preview.lower():
            seen_recitals_heading = True
            continue

        if node.name == "nav":
            break

        if seen_recitals_heading:
            # Recital links in this section look like "(39) Principles of
            # Data Processing" -- pull just the leading recital number(s).
            for m in _RECITAL_NUM_RE.finditer(text_preview):
                recitals.append(int(m.group(1)))
            # Stop once we hit the prev/next article links or the "Table of
            # contents" link, which sit right after the recitals block.
            if "table of contents" in text_preview.lower():
                break
            continue

        if node.name in ("h2", "h3", "h4"):
            # A genuine new section heading before recitals shouldn't occur
            # in practice for this site's article pages -- bail out rather
            # than risk slurping unrelated trailing content if it does.
            break

        if node.name in ("p", "ol", "ul"):
            rendered = _render_body_node(node)
            if rendered:
                body_parts.append(rendered)
        elif node.name == "div":
            # Some article pages wrap the actual <ol>/<p> body one level
            # deeper inside a plain <div> (no distinguishing class) rather
            # than putting them directly under .entry-content -- recurse one
            # level into any such div and pull its own direct p/ol/ul
            # children the same way, without going further (avoids
            # accidentally descending into unrelated nested widgets).
            for inner in node.find_all(["p", "ol", "ul"], recursive=False):
                rendered = _render_body_node(inner)
                if rendered:
                    body_parts.append(rendered)

    body_text = "\n\n".join(body_parts).strip()
    if not body_text:
        if debug_dump_dir is not None:
            _dump_debug_html(debug_dump_dir, article_number, html, "empty-body")
        return None

    chapter_number, chapter_title = _chapter_for_article(article_number)
    hierarchy_path = (
        f"Chapter {chapter_number} ({chapter_title}) > Art. {article_number}"
        if chapter_number else f"Art. {article_number}"
    )

    embed_text = (
        f"GDPR Article {article_number} -- {title}\n"
        f"({hierarchy_path})\n\n"
        f"{body_text}"
    )

    return {
        "chunk_id": f"GDPR-art-{article_number}",
        "article_number": article_number,
        "title": title,
        "chapter_number": chapter_number,
        "chapter_title": chapter_title,
        "hierarchy_path": hierarchy_path,
        "text": body_text,
        "embed_text": embed_text,
        "recitals": sorted(set(recitals)),
        "source_url": source_url,
    }


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


def scrape(output_dir: Path, start: int, end: int, delay: float, base_url: str,
           debug_dump_dir: Path | None = None) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "gdpr_articles.jsonl"

    session = _build_session()
    records = []
    failures = []

    for n in range(start, end + 1):
        url = f"{base_url}/art-{n}-gdpr/"
        print(f"[{n - start + 1}/{end - start + 1}] fetching Art. {n} -- {url}")
        try:
            resp = session.get(url, timeout=30)
            resp.raise_for_status()
        except requests.RequestException as e:
            print(f"  [error] failed to fetch Art. {n}: {e}", file=sys.stderr)
            failures.append(n)
            time.sleep(delay)
            continue

        record = parse_article_page(resp.text, n, url, debug_dump_dir=debug_dump_dir)
        if record is None:
            print(f"  [warn] Art. {n}: could not extract article body, skipping", file=sys.stderr)
            failures.append(n)
        else:
            records.append(record)
            print(f"  -> {record['title']!r} ({len(record['text'])} chars, "
                  f"{len(record['recitals'])} recital(s))")

        time.sleep(delay)  # politeness delay between requests

    with open(out_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"\nDone. Wrote {len(records)} article(s) to {out_path}")
    if failures:
        print(f"WARNING: {len(failures)} article(s) failed/skipped: {failures}", file=sys.stderr)
        print("Re-run with --start/--end targeting just those numbers to retry.", file=sys.stderr)
        if debug_dump_dir is not None:
            print(f"Raw HTML of failed pages saved under {debug_dump_dir} -- inspect those "
                  f"to see why parsing failed.", file=sys.stderr)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output-dir", default="./gdpr_output", help="Directory to write gdpr_articles.jsonl into")
    ap.add_argument("--start", type=int, default=1, help="First article number to scrape (default 1)")
    ap.add_argument("--end", type=int, default=TOTAL_ARTICLES, help=f"Last article number to scrape (default {TOTAL_ARTICLES})")
    ap.add_argument("--delay", type=float, default=1.0, help="Seconds to wait between requests (default 1.0 -- be polite)")
    ap.add_argument("--base-url", default=BASE_URL, help="Base URL of the GDPR mirror to scrape")
    ap.add_argument(
        "--debug-dir", default=None,
        help="If set, saves the raw HTML of any article page that fails to parse into this "
             "directory, so you can inspect why (e.g. ./debug_html). Off by default.",
    )
    args = ap.parse_args()

    if args.start < 1 or args.end > TOTAL_ARTICLES or args.start > args.end:
        ap.error(f"--start/--end must be within 1..{TOTAL_ARTICLES} and start <= end")

    debug_dump_dir = Path(args.debug_dir) if args.debug_dir else None
    scrape(Path(args.output_dir), args.start, args.end, args.delay, args.base_url.rstrip("/"),
           debug_dump_dir=debug_dump_dir)


if __name__ == "__main__":
    main()