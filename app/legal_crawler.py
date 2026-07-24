"""
app/legal_crawler.py

Crawls kolzchut.org.il ("כל-זכות") for PDF copies of Israeli laws and
regulations, and keeps a local mirror of them in this project's upload/
folder -- re-downloading only files that are new or have actually changed
since the last run.

Why kolzchut.org.il: it's a Hebrew-language nonprofit legal-information wiki
whose "law" (חוק) and "regulations" (תקנות) pages attach a source-text PDF
(a Reshumot/official-gazette scan, or the wiki's own uploaded copy). Site
content is published under CC BY-NC-SA 2.5 IL -- non-commercial, share-alike
(see https://www.kolzchut.org.il/he/כל-זכות:מדיניות#זכויות_יוצרים) -- so
anything this crawler downloads carries that license; it doesn't change that,
and this mirror should only be used non-commercially, with attribution back
to kolzchut/the original legal source when the material is used or shown.

Crawler etiquette this module follows (all of these matter more than
squeezing out a faster crawl -- this is someone else's server):
  - robots.txt is fetched fresh and checked before *every* URL request (via
    urllib.robotparser), rather than a hardcoded copy of today's rules. As of
    this writing the site disallows /w/*/index.php, /w/*/api.php,
    /ChangeRequest/ and /forms/, and allows everything else -- including one
    specific index.php recent-changes feed URL, which is why that one exact
    URL is special-cased in _recent_changes_feed_url().
  - Fully sequential, single connection, with a polite delay between every
    request (default 2s) -- no concurrency, no bursts.
  - A descriptive User-Agent so kolzchut's admins have a way to reach out if
    this ever needs to be tuned down further.
  - Conditional GETs (If-None-Match / If-Modified-Since) plus a sha256 body
    hash, so a daily run only re-downloads a PDF that actually changed --
    everything else is a cheap "still the same" check.

Two discovery modes:
  - "full"        -- walks the site's XML sitemap to enumerate every Hebrew
                     page once. Slow (one request per page), meant for the
                     first-ever run (or an occasional re-index to catch pages
                     the daily feed might miss), not the daily job.
  - "incremental" -- reads kolzchut's Special:RecentChanges Atom feed (the
                     one index.php URL robots.txt explicitly allows) for
                     pages that changed in roughly the last day, and only
                     re-checks those. This is the one meant to run daily.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import logging
import re
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser

import httpx

logger = logging.getLogger("legal_crawler")

BASE_URL = "https://www.kolzchut.org.il"
LANGUAGE_PATH_PREFIX = "/he/"  # Hebrew namespace only -- matches this app's Hebrew-legal focus
SITEMAP_URL = f"{BASE_URL}/sitemap.xml"

# The one recent-changes feed URL robots.txt explicitly carves out an
# "Allow:" for, even though index.php is disallowed in general. days=1 and a
# generous limit keep this matched to "check daily for changes".
_RECENT_CHANGES_TITLE = "מיוחד:שינויים_אחרונים"  # Special:RecentChanges


def _recent_changes_feed_url(days: int = 1, limit: int = 500) -> str:
    from urllib.parse import quote
    title = quote(_RECENT_CHANGES_TITLE, safe="")
    return f"{BASE_URL}/w/he/index.php?title={title}&feed=atom&days={days}&limit={limit}"


# kolzchut renders each page's HTML <title> as "<Page name> (<Type>) – כל-זכות"
# whenever the page name needs a type qualifier -- "(חוק)" for a law, "(תקנות)"
# for regulations, "(חקיקה)" is used for some legislation-adjacent pages. This
# is the signal used to decide "this page is a law/regulations page" (as
# opposed to a general rights-explainer page, which is what most of the site
# is) before bothering to look for an attached PDF on it.
_LAW_TITLE_RE = re.compile(r"\((?:חוק|תקנות|חקיקה)\)\s*[-–]\s*כל.?זכות")

_PDF_HREF_RE = re.compile(r"\.pdf(?:[?#]|$)", re.IGNORECASE)

USER_AGENT = (
    "OfflineLegalDocsBot/1.0 "
    "(+local non-commercial legal-reference mirror; "
    "contact: set CRAWLER_CONTACT_EMAIL before running in production)"
)

_REQUEST_DELAY_SECONDS = 2.0  # polite minimum gap between any two requests
_REQUEST_TIMEOUT_SECONDS = 30.0


@dataclass
class Discovery:
    pdf_url: str
    source_page_url: str
    source_title: str


@dataclass
class CrawlStats:
    pages_checked: int = 0
    pdfs_seen: int = 0
    pdfs_new: int = 0
    pdfs_updated: int = 0
    pdfs_unchanged: int = 0
    errors: list[str] = field(default_factory=list)


class _TitleAndPdfLinkParser(HTMLParser):
    """Pulls out <title> text and every <a href="...pdf"> from one page,
    without pulling in a BeautifulSoup dependency for something this small."""

    def __init__(self):
        super().__init__()
        self.title_parts: list[str] = []
        self._in_title = False
        self.pdf_hrefs: list[str] = []

    def handle_starttag(self, tag, attrs):
        if tag == "title":
            self._in_title = True
        elif tag == "a":
            href = dict(attrs).get("href")
            if href and _PDF_HREF_RE.search(href):
                self.pdf_hrefs.append(href)

    def handle_endtag(self, tag):
        if tag == "title":
            self._in_title = False

    def handle_data(self, data):
        if self._in_title:
            self.title_parts.append(data)

    @property
    def title(self) -> str:
        return "".join(self.title_parts).strip()


class RateLimiter:
    """Simple sequential rate limiter -- one shared clock for every request
    this crawler makes, regardless of which function makes it."""

    def __init__(self, delay_seconds: float = _REQUEST_DELAY_SECONDS):
        self.delay_seconds = delay_seconds
        self._last_request_at = 0.0

    def wait(self):
        elapsed = time.monotonic() - self._last_request_at
        remaining = self.delay_seconds - elapsed
        if remaining > 0:
            time.sleep(remaining)
        self._last_request_at = time.monotonic()


class RobotsGate:
    """Fetches robots.txt once per run and answers can_fetch() for every URL
    this crawler considers requesting. Fails closed: if robots.txt can't be
    retrieved, nothing is allowed."""

    def __init__(self, client: httpx.Client):
        self._parser = RobotFileParser()
        self._ok = False
        try:
            resp = client.get(f"{BASE_URL}/robots.txt", timeout=_REQUEST_TIMEOUT_SECONDS)
            resp.raise_for_status()
            self._parser.parse(resp.text.splitlines())
            self._ok = True
        except httpx.HTTPError as e:
            logger.error("Could not fetch robots.txt (%s) -- refusing to crawl.", e)

    def can_fetch(self, url: str) -> bool:
        if not self._ok:
            return False
        return self._parser.can_fetch(USER_AGENT, url)


def _sanitize_filename(name: str, max_len: int = 150) -> str:
    """Strips characters invalid on Windows filesystems while keeping Hebrew
    text intact, so downloaded files stay identifiable by law name."""
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', " ", name).strip()
    name = re.sub(r"\s+", " ", name)
    return name[:max_len] or "document"


def _maybe_gunzip(url: str, content: bytes) -> bytes:
    """kolzchut serves some sitemap files as literal .xml.gz files -- the
    gzip is the file's actual content, not an HTTP Content-Encoding the
    server declares (httpx already handles that transport-level case on its
    own). Detect it either by the .gz extension or the gzip magic bytes
    (in case a .xml URL is gzipped without saying so) and decompress before
    handing it to the XML parser."""
    if url.endswith(".gz") or content[:2] == b"\x1f\x8b":
        return gzip.decompress(content)
    return content


def _resolve_page_urls_from_sitemap(client: httpx.Client, robots: RobotsGate,
                                     rate_limiter: RateLimiter,
                                     max_pages: int) -> list[str]:
    """Recursively walks sitemap index files (if any) down to per-page
    <loc> entries, filtered to the Hebrew namespace."""
    to_visit = [SITEMAP_URL]
    visited_sitemaps: set[str] = set()
    page_urls: list[str] = []

    while to_visit and len(page_urls) < max_pages:
        sitemap_url = to_visit.pop()
        if sitemap_url in visited_sitemaps:
            continue
        visited_sitemaps.add(sitemap_url)

        if not robots.can_fetch(sitemap_url):
            logger.info("robots.txt disallows %s -- skipping.", sitemap_url)
            continue

        rate_limiter.wait()
        try:
            resp = client.get(sitemap_url, timeout=_REQUEST_TIMEOUT_SECONDS)
            resp.raise_for_status()
            root = ET.fromstring(_maybe_gunzip(sitemap_url, resp.content))
        except (httpx.HTTPError, ET.ParseError, gzip.BadGzipFile) as e:
            logger.warning("Failed to fetch/parse sitemap %s: %s", sitemap_url, e)
            continue

        tag = root.tag.rsplit("}", 1)[-1]  # strip XML namespace
        locs = [
            el.text.strip()
            for el in root.iter()
            if el.tag.rsplit("}", 1)[-1] == "loc" and el.text
        ]
        if tag == "sitemapindex":
            to_visit.extend(locs)
        else:  # urlset -- actual page URLs
            for loc in locs:
                path = urlparse(loc).path
                if path.startswith(LANGUAGE_PATH_PREFIX):
                    page_urls.append(loc)
                    if len(page_urls) >= max_pages:
                        break

    return page_urls


def _resolve_page_urls_from_recent_changes(client: httpx.Client, robots: RobotsGate,
                                            rate_limiter: RateLimiter,
                                            days: int) -> list[str]:
    feed_url = _recent_changes_feed_url(days=days)
    if not robots.can_fetch(feed_url):
        # Shouldn't happen -- robots.txt explicitly allows this exact URL --
        # but fail closed if the site's rules ever change.
        logger.error("robots.txt no longer allows the recent-changes feed -- "
                      "falling back to no incremental results this run.")
        return []

    rate_limiter.wait()
    try:
        resp = client.get(feed_url, timeout=_REQUEST_TIMEOUT_SECONDS)
        resp.raise_for_status()
        root = ET.fromstring(_maybe_gunzip(feed_url, resp.content))
    except (httpx.HTTPError, ET.ParseError, gzip.BadGzipFile) as e:
        logger.warning("Failed to fetch/parse recent-changes feed: %s", e)
        return []

    ns = {"atom": "http://www.w3.org/2005/Atom"}
    urls = []
    for entry in root.findall("atom:entry", ns):
        link_el = entry.find("atom:link", ns)
        if link_el is not None and link_el.get("href"):
            urls.append(link_el.get("href"))
    return urls


def _check_page_for_pdfs(client: httpx.Client, robots: RobotsGate,
                          rate_limiter: RateLimiter, page_url: str) -> list[Discovery]:
    if not robots.can_fetch(page_url):
        logger.debug("robots.txt disallows %s -- skipping.", page_url)
        return []

    rate_limiter.wait()
    try:
        resp = client.get(page_url, timeout=_REQUEST_TIMEOUT_SECONDS)
        resp.raise_for_status()
    except httpx.HTTPError as e:
        logger.warning("Failed to fetch %s: %s", page_url, e)
        return []

    parser = _TitleAndPdfLinkParser()
    parser.feed(resp.text)

    if not _LAW_TITLE_RE.search(parser.title):
        return []  # not a law/regulations-type page -- not in scope

    discoveries = []
    for href in parser.pdf_hrefs:
        pdf_url = urljoin(page_url, href)
        if urlparse(pdf_url).netloc.endswith("kolzchut.org.il"):
            discoveries.append(Discovery(pdf_url=pdf_url, source_page_url=page_url,
                                          source_title=parser.title))
    return discoveries


def discover(client: httpx.Client, robots: RobotsGate, rate_limiter: RateLimiter,
             mode: str, max_pages: int, stats: CrawlStats) -> list[Discovery]:
    """Returns the list of law/regulation PDFs found across the pages
    relevant to `mode` ("full" or "incremental")."""
    if mode == "full":
        page_urls = _resolve_page_urls_from_sitemap(client, robots, rate_limiter, max_pages)
    elif mode == "incremental":
        page_urls = _resolve_page_urls_from_recent_changes(client, robots, rate_limiter, days=1)
    else:
        raise ValueError(f"Unknown mode: {mode!r} (expected 'full' or 'incremental')")

    logger.info("Discovery mode=%s found %d candidate page(s) to check.", mode, len(page_urls))

    all_discoveries: list[Discovery] = []
    for page_url in page_urls:
        stats.pages_checked += 1
        try:
            found = _check_page_for_pdfs(client, robots, rate_limiter, page_url)
        except Exception as e:  # keep one bad page from aborting the whole run
            logger.exception("Unexpected error checking %s", page_url)
            stats.errors.append(f"{page_url}: {e}")
            continue
        all_discoveries.extend(found)

    return all_discoveries


# ---------------------------------------------------------------------------
# Manifest + conditional download
# ---------------------------------------------------------------------------

def load_manifest(manifest_path: Path) -> dict:
    if manifest_path.exists():
        try:
            return json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            logger.warning("Manifest at %s was corrupt -- starting a fresh one.", manifest_path)
    return {}


def save_manifest(manifest_path: Path, manifest: dict):
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def download_if_changed(client: httpx.Client, robots: RobotsGate, rate_limiter: RateLimiter,
                         discovery: Discovery, dest_dir: Path, manifest: dict,
                         stats: CrawlStats) -> str:
    """Downloads discovery.pdf_url into dest_dir only if it's new or its
    content has changed since the last run. Returns one of
    "new" / "updated" / "unchanged" / "error"."""
    stats.pdfs_seen += 1
    pdf_url = discovery.pdf_url

    if not robots.can_fetch(pdf_url):
        logger.debug("robots.txt disallows %s -- skipping.", pdf_url)
        return "error"

    entry = manifest.get(pdf_url, {})
    headers = {}
    if entry.get("etag"):
        headers["If-None-Match"] = entry["etag"]
    if entry.get("last_modified"):
        headers["If-Modified-Since"] = entry["last_modified"]

    rate_limiter.wait()
    try:
        resp = client.get(pdf_url, headers=headers, timeout=_REQUEST_TIMEOUT_SECONDS)
    except httpx.HTTPError as e:
        logger.warning("Failed to fetch %s: %s", pdf_url, e)
        stats.errors.append(f"{pdf_url}: {e}")
        return "error"

    if resp.status_code == 304:
        stats.pdfs_unchanged += 1
        entry["last_checked"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        manifest[pdf_url] = entry
        return "unchanged"

    if resp.status_code != 200:
        logger.warning("Unexpected status %s for %s", resp.status_code, pdf_url)
        stats.errors.append(f"{pdf_url}: HTTP {resp.status_code}")
        return "error"

    content = resp.content
    content_hash = hashlib.sha256(content).hexdigest()

    if entry.get("sha256") == content_hash:
        stats.pdfs_unchanged += 1
        entry["last_checked"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        entry["etag"] = resp.headers.get("ETag", entry.get("etag"))
        entry["last_modified"] = resp.headers.get("Last-Modified", entry.get("last_modified"))
        manifest[pdf_url] = entry
        return "unchanged"

    is_new = pdf_url not in manifest
    original_name = Path(urlparse(pdf_url).path).name or "document.pdf"
    safe_title = _sanitize_filename(re.sub(r"\s*\((?:חוק|תקנות|חקיקה)\)\s*[-–]\s*כל.?זכות\s*$",
                                            "", discovery.source_title))
    local_filename = entry.get("local_filename") or f"{safe_title} -- {original_name}"
    local_path = dest_dir / local_filename

    dest_dir.mkdir(parents=True, exist_ok=True)
    local_path.write_bytes(content)

    manifest[pdf_url] = {
        "etag": resp.headers.get("ETag"),
        "last_modified": resp.headers.get("Last-Modified"),
        "sha256": content_hash,
        "local_filename": local_filename,
        "source_page_url": discovery.source_page_url,
        "source_title": discovery.source_title,
        "first_seen": entry.get("first_seen") or time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "last_checked": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    if is_new:
        stats.pdfs_new += 1
        logger.info("New: %s", local_filename)
        return "new"
    else:
        stats.pdfs_updated += 1
        logger.info("Updated: %s", local_filename)
        return "updated"


def run_crawl(dest_dir: Path, mode: str = "incremental", max_pages: int = 20000,
              request_delay_seconds: float = _REQUEST_DELAY_SECONDS,
              contact_email: str | None = None) -> CrawlStats:
    """Runs one full crawl cycle and returns summary stats. Safe to call
    daily -- pass mode="incremental" for the normal daily job, and
    mode="full" occasionally (e.g. weekly/monthly) to catch anything the
    recent-changes feed missed."""
    stats = CrawlStats()
    rate_limiter = RateLimiter(request_delay_seconds)
    manifest_path = dest_dir / "_crawler_manifest.json"
    manifest = load_manifest(manifest_path)

    user_agent = USER_AGENT
    if contact_email:
        user_agent = (
            f"OfflineLegalDocsBot/1.0 (+local non-commercial legal-reference "
            f"mirror; contact: {contact_email})"
        )

    with httpx.Client(headers={"User-Agent": user_agent}, follow_redirects=True) as client:
        robots = RobotsGate(client)
        if not robots._ok:
            stats.errors.append("robots.txt unavailable -- aborting run.")
            return stats

        discoveries = discover(client, robots, rate_limiter, mode, max_pages, stats)
        logger.info("Found %d law/regulation PDF link(s) across checked pages.", len(discoveries))

        # De-duplicate: the same PDF can be linked from more than one page.
        seen_urls = set()
        unique_discoveries = []
        for d in discoveries:
            if d.pdf_url not in seen_urls:
                seen_urls.add(d.pdf_url)
                unique_discoveries.append(d)

        for discovery in unique_discoveries:
            download_if_changed(client, robots, rate_limiter, discovery, dest_dir, manifest, stats)
            save_manifest(manifest_path, manifest)  # persist progress incrementally

    return stats