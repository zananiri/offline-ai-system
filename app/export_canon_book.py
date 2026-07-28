#!/usr/bin/env python3
"""
Export the Canon AI ChromaDB collection to a readable PDF "book"
====================================================================
Pulls every canon out of your ChromaDB collection and lays it out as a
proper book: title page, auto-generated table of contents with real page
numbers, and each canon under its correct Libro / Parte / Sezione / Titolo
/ Capitolo / Articolo heading -- using the same hierarchy metadata fields
that embed_to_chroma.py already stores on every vector.

Ordering: canons are sorted purely by canon_number (not by trying to
parse/sort the Italian hierarchy labels alphabetically, which wouldn't
order Roman numerals correctly -- "IX" would sort before "V" as plain
text). The 1983 Code numbers canons sequentially from Book I through
Book VII, so sorting by canon_number alone reconstructs the correct
reading order for free, and headings are then emitted automatically
whenever the hierarchy metadata changes between one canon and the next.

Usage:
    python export_canon_book.py
    python export_canon_book.py --chroma-dir ./chroma_db --collection cic_it
    python export_canon_book.py --output "Codice di Diritto Canonico.pdf"
    python export_canon_book.py --limit 50          # quick preview build
"""

import argparse
import sys
from pathlib import Path

try:
    import chromadb
except ImportError:
    print("Missing dependency: pip install chromadb", file=sys.stderr)
    sys.exit(1)

try:
    from reportlab.lib.pagesizes import LETTER
    from reportlab.lib.units import inch
    from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
    from reportlab.platypus.tableofcontents import TableOfContents
except ImportError:
    print("Missing dependency: pip install reportlab", file=sys.stderr)
    sys.exit(1)

HIERARCHY_FIELDS = ["libro", "parte", "sezione", "titolo", "capitolo", "articolo"]
HEADING_STYLE_NAMES = ["Heading1", "Heading2", "Heading2", "Heading3", "Heading4", "Heading5"]
# Only the first 4 hierarchy levels get a Table of Contents entry (Capitolo/
# Articolo still render as visible headings in the body, just not listed in
# the TOC -- most CIC editions don't index down to that level either, and a
# ~1700-canon TOC going 6 levels deep would be enormous).
TOC_LEVELS = 4


def load_canons(chroma_dir: str, collection_name: str, limit: int | None = None) -> list[dict]:
    client = chromadb.PersistentClient(path=chroma_dir)
    try:
        collection = client.get_collection(collection_name)
    except Exception as e:
        print(f"ERROR: couldn't open collection '{collection_name}' at '{chroma_dir}': {e}", file=sys.stderr)
        sys.exit(1)

    data = collection.get(include=["metadatas", "documents"])
    canons = []
    for meta, doc in zip(data.get("metadatas", []), data.get("documents", [])):
        number = meta.get("canon_number")
        if not number or not doc or not doc.strip():
            continue
        entry = {"number": str(number), "text": doc.strip(), "note": meta.get("in_force_note", "")}
        for field in HIERARCHY_FIELDS:
            entry[field] = (meta.get(field) or "").strip()
        canons.append(entry)

    def sort_key(c):
        try:
            return (0, int(c["number"]))
        except (TypeError, ValueError):
            return (1, c["number"])  # non-numeric canon numbers sort after, alphabetically

    canons.sort(key=sort_key)
    if limit:
        canons = canons[:limit]
    return canons


class BookDocTemplate(SimpleDocTemplate):
    """Subclassed purely to hook afterFlowable(), which is how ReportLab's
    TableOfContents flowable learns what page each heading landed on --
    this requires a two-pass build (see multiBuild() below) since page
    numbers aren't known until the whole document has been laid out once."""

    def afterFlowable(self, flowable):
        if not isinstance(flowable, Paragraph):
            return
        style_name = flowable.style.name
        level_map = {name: i for i, name in enumerate(HEADING_STYLE_NAMES[:TOC_LEVELS])}
        # dict comprehension above collapses duplicate names (Heading2 used
        # twice, for Parte and Sezione) to the LAST matching index -- fine
        # here since we only need "is this a headingish style", not which
        # exact field produced it, and TOC entries are visually grouped by
        # indentation (levelStyles below) rather than needing a unique id.
        if style_name in HEADING_STYLE_NAMES[:TOC_LEVELS]:
            level = HEADING_STYLE_NAMES.index(style_name)
            level = min(level, TOC_LEVELS - 1)
            self.notify("TOCEntry", (level, flowable.getPlainText(), self.page))


def build_pdf(canons: list[dict], out_path: Path, title: str, subtitle: str, source_note: str):
    doc = BookDocTemplate(
        str(out_path), pagesize=LETTER,
        topMargin=0.9 * inch, bottomMargin=0.9 * inch,
        leftMargin=1 * inch, rightMargin=1 * inch,
        title=title,
    )
    styles = getSampleStyleSheet()

    canon_number_style = ParagraphStyle(
        "CanonNumber", parent=styles["Heading5"], fontSize=11,
        spaceBefore=12, spaceAfter=3, textColor="#1e3a5f",
    )
    body_style = ParagraphStyle(
        "CanonBody", parent=styles["Normal"], fontSize=10.5,
        leading=15, alignment=TA_JUSTIFY, spaceAfter=6,
    )
    note_style = ParagraphStyle(
        "CanonNote", parent=styles["Italic"], fontSize=8.5,
        textColor="#666666", spaceAfter=10,
    )

    toc = TableOfContents()
    toc.levelStyles = [
        ParagraphStyle(name="TOCHeading1", fontSize=13, leftIndent=0, firstLineIndent=0, spaceBefore=10, leading=16),
        ParagraphStyle(name="TOCHeading2", fontSize=11, leftIndent=16, firstLineIndent=0, spaceBefore=5, leading=13),
        ParagraphStyle(name="TOCHeading3", fontSize=10, leftIndent=32, firstLineIndent=0, spaceBefore=2, leading=12),
        ParagraphStyle(name="TOCHeading4", fontSize=9, leftIndent=48, firstLineIndent=0, spaceBefore=1, leading=11),
    ][:TOC_LEVELS]

    story = []

    # --- Title page ----------------------------------------------------
    story.append(Spacer(1, 2 * inch))
    story.append(Paragraph(title, ParagraphStyle(
        "BookTitle", parent=styles["Title"], fontSize=26, alignment=TA_CENTER)))
    story.append(Spacer(1, 0.25 * inch))
    story.append(Paragraph(subtitle, ParagraphStyle(
        "BookSubtitle", parent=styles["Normal"], fontSize=12, alignment=TA_CENTER, textColor="#555555")))
    story.append(Spacer(1, 0.6 * inch))
    story.append(Paragraph(f"{len(canons)} canons" if canons else "", ParagraphStyle(
        "BookMeta", parent=styles["Normal"], fontSize=10, alignment=TA_CENTER, textColor="#888888")))
    if source_note:
        story.append(Spacer(1, 0.15 * inch))
        story.append(Paragraph(source_note, ParagraphStyle(
            "BookSource", parent=styles["Normal"], fontSize=8.5, alignment=TA_CENTER, textColor="#999999")))
    story.append(PageBreak())

    # --- Table of contents ----------------------------------------------
    story.append(Paragraph("Table of Contents", styles["Heading1"]))
    story.append(toc)
    story.append(PageBreak())

    # --- Body: walk canons in order, emitting headings on hierarchy change
    prev = tuple(None for _ in HIERARCHY_FIELDS)
    is_first_heading_batch = True
    for c in canons:
        current = tuple(c[field] for field in HIERARCHY_FIELDS)
        first_diff = next((i for i in range(len(HIERARCHY_FIELDS)) if current[i] != prev[i]), None)

        if first_diff is not None:
            for i in range(first_diff, len(HIERARCHY_FIELDS)):
                if current[i]:
                    if i == 0 and not is_first_heading_batch:
                        # skip the page break before the very first Libro --
                        # we already just landed on a fresh page right after
                        # the TOC, so breaking again here would leave a
                        # pointless blank page between TOC and Libro I
                        story.append(PageBreak())
                    story.append(Paragraph(current[i], styles[HEADING_STYLE_NAMES[i]]))
            prev = current
            is_first_heading_batch = False

        story.append(Paragraph(f"Can. {c['number']}", canon_number_style))
        story.append(Paragraph(c["text"], body_style))
        if c["note"]:
            story.append(Paragraph(c["note"], note_style))

    if not canons:
        story.append(Paragraph("No canons found in this collection.", styles["Normal"]))

    doc.multiBuild(story)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--chroma-dir", default="./chroma_db", help="ChromaDB persistent storage dir")
    ap.add_argument("--collection", default="cic_it", help="ChromaDB collection name")
    ap.add_argument("--output", default="Codice di Diritto Canonico.pdf", help="Output PDF filename")
    ap.add_argument("--title", default="Codice di Diritto Canonico")
    ap.add_argument("--subtitle", default="1983 Code of Canon Law -- Italian text")
    ap.add_argument("--limit", type=int, default=None,
                     help="Only include the first N canons (quick preview build)")
    args = ap.parse_args()

    print(f"Loading canons from {args.chroma_dir} (collection '{args.collection}') ...")
    canons = load_canons(args.chroma_dir, args.collection, limit=args.limit)
    print(f"Loaded {len(canons)} canons.")
    if not canons:
        print("WARNING: no canons found -- check --chroma-dir/--collection, or that "
              "embed_to_chroma.py has actually been run.", file=sys.stderr)

    out_path = Path(args.output)
    source_note = f"Generated from local ChromaDB collection '{args.collection}'."
    print(f"Building PDF -> {out_path} ...")
    build_pdf(canons, out_path, args.title, args.subtitle, source_note)
    print(f"Done. Wrote {out_path.resolve()}")


if __name__ == "__main__":
    main()