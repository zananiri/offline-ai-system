#!/usr/bin/env python3
"""
Canon AI RAG evaluation harness
================================
Tests two things separately, since they can fail independently:

1. ChromaDB retrieval quality -- does the right canon actually get pulled
   out of the vector database for a given question?
2. The offline LLM's use of what was retrieved -- does the generated
   answer actually cite the canon it was given?

WHERE THE 100 QUESTIONS COME FROM (important -- read this before trusting
the numbers)
-------------------------------------------------------------------------
This script does NOT hard-code 100 canon-law trivia questions with
asserted "correct" canon numbers written from memory. Doing that would be
dangerous for a benchmark: any mistake in that ground truth would silently
corrupt every score this script reports, and you'd have no way to tell a
real retrieval failure apart from the benchmark itself being wrong.

Instead:

  * AUTO-SAMPLED questions (default 35 canons x 2 phrasings = 70) are
    generated live from YOUR OWN ChromaDB collection at runtime. For each
    sampled canon, the ground truth is simply "the retriever should find
    THIS canon, which is the one this question was built from" -- correct
    by construction, not by my recollection of canon law. Two phrasings
    are generated per sampled canon:
      - "numeric": names the canon number outright (e.g. "What does
        canon 1055 establish?") -- exercises the exact-match code path.
      - "semantic": built from the canon's own first sentence with the
        number stripped out -- exercises real vector-similarity /
        hybrid-rerank retrieval, no shortcut available. This is the one
        that actually tells you how good the vector DB is.
    These give hard, trustworthy numbers: Recall@1/3/5 and Mean
    Reciprocal Rank (MRR) for retrieval, and citation accuracy for
    generation.

  * CURATED conceptual questions (30, fixed list below) cover broad canon
    law topics (marriage, ordination, penal law, tribunals, rights of the
    faithful, religious life, etc.) written in general terms with NO
    specific canon number asserted as "correct" -- because I can't
    guarantee 100% accuracy on every specific citation from memory, and a
    wrong assertion here would be worse than not scoring it at all. These
    are reported for you to spot-check qualitatively (did it retrieve
    something plausible, did it cite a canon) -- not pass/fail scored.

Usage
-----
    python test_canon_rag.py                      # retrieval-only, fast
    python test_canon_rag.py --full                # + LLM generation (slow)
    python test_canon_rag.py --n-sampled 50 --seed 7
    python test_canon_rag.py --full --limit 10     # quick smoke test
    python test_canon_rag.py --ui-path ../app/ui.py

Output
------
Prints a summary scorecard to the console and writes a detailed
per-question CSV (retrieved canons, rank, hit@k, latency, generated
answer if --full) to canon_rag_eval_<timestamp>.csv for manual review.
"""

import argparse
import csv
import importlib.util
import random
import re
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Curated conceptual questions -- broad canon-law topics, deliberately
# phrased WITHOUT asserting a specific canon number as ground truth (see
# module docstring for why). Mix of English and Italian, since Canon AI is
# used in both. Qualitative review only, not scored pass/fail.
# ---------------------------------------------------------------------------
CURATED_QUESTIONS = [
    "What is required for a marriage to be valid?",
    "Can a Catholic marry someone who isn't baptized?",
    "What are the grounds for a declaration of nullity of marriage?",
    "What is required for someone to be validly ordained a priest?",
    "What are the impediments to receiving holy orders?",
    "Can a priest be removed from his parish, and under what conditions?",
    "What is the process for a canonical penal trial?",
    "What is excommunication and how is it incurred?",
    "What are the rights of the Christian faithful within the Church?",
    "What are the obligations of clerics regarding celibacy?",
    "Can a diocesan bishop dispense from Church law?",
    "What is required to establish a religious institute?",
    "What is the role of a diocesan tribunal?",
    "What is required for a valid baptism?",
    "Who can validly witness a marriage?",
    "What is the seal of confession and can it ever be broken?",
    "What are the requirements for someone to be a godparent?",
    "Can a Catholic be denied Holy Communion, and under what circumstances?",
    "What is required for a valid canonical appeal?",
    "What is the difference between a latae sententiae and ferendae sententiae penalty?",
    "Cosa è richiesto per la validità di un matrimonio canonico?",
    "Quali sono gli impedimenti al matrimonio nel diritto canonico?",
    "Come viene nominato un parroco?",
    "Quali sono i diritti dei fedeli laici nella Chiesa?",
    "Cosa comporta la scomunica?",
    "Quali sono le cause di nullità del matrimonio?",
    "Cosa dice il diritto canonico sul segreto della confessione?",
    "Come si svolge un processo penale canonico?",
    "Quali sono gli obblighi di un chierico riguardo al celibato?",
    "Chi può celebrare validamente il sacramento del battesimo?",
]

CITATION_RE = re.compile(r"\bcan\.?\s*\d{1,4}\b", re.IGNORECASE)


def load_reference_qa(path: Path):
    """Loads the curated reference Q&A set (question + Claude-authored
    reference answer + key canon numbers). See that file's _disclaimer
    field: these are a strong comparison baseline, not a re-verified
    authoritative answer key -- treat 'confidence: medium' items with
    appropriate caution, and spot-check anything consequential against
    the actual code text."""
    if not path.exists():
        print(f"NOTE: reference Q&A file {path} not found -- running "
              f"without the curated comparison set.", file=sys.stderr)
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get("items", [])


_STOPWORDS_EN = frozenset("""
a an the of to in on for and or is are was were be been being this that
these those what which who whom does do did can may must shall not with
as by from about into over under between it its i we you he she they
""".split())


def _content_tokens(text: str) -> set[str]:
    """Very rough content-word extractor for the reference-vs-generated
    overlap score below -- deliberately simple (no stemming, no Italian
    stopword list here) since this score is explicitly a coarse proxy for
    manual review, not a claimed semantic judge."""
    return {w for w in re.findall(r"[a-z]+", text.lower()) if w not in _STOPWORDS_EN and len(w) > 3}


def content_overlap_score(reference_answer: str, generated_answer: str) -> float:
    """Fraction of the reference answer's content words that also appear
    in the generated answer. A coarse proxy for 'did it touch on the same
    substance', NOT a legal-correctness judgment -- always read the
    side-by-side text in the CSV for anything that matters."""
    ref_tokens = _content_tokens(reference_answer)
    if not ref_tokens:
        return 0.0
    gen_tokens = _content_tokens(generated_answer or "")
    return len(ref_tokens & gen_tokens) / len(ref_tokens)


def load_ui_module(ui_path: Path):
    """Imports ui.py as a module by file path (rather than requiring this
    script to sit in the same package) so retrieve_canons(), _chat_backend(),
    CANON_SYSTEM_PROMPT etc. are reused directly from production code --
    the test then exercises exactly what real users hit, instead of a
    reimplementation that could quietly drift out of sync over time."""
    spec = importlib.util.spec_from_file_location("ui", ui_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def first_sentence_stripped_of_number(text: str, canon_number: str) -> str:
    """Builds a 'semantic' test query from a canon's own text: takes
    (roughly) its first sentence and removes the canon number itself, so
    the retriever can't just pattern-match the digits back out -- it has
    to actually find the passage by meaning/vector similarity."""
    # Canon text stored in Chroma sometimes starts with a "Can. N." style
    # self-reference (mirroring how it appears on vatican.va); strip that
    # off before taking the working sentence.
    cleaned = re.sub(r"^\s*(?:can(?:one)?\.?\s*\d+\.?\s*)", "", text, flags=re.IGNORECASE)
    sentence = re.split(r"(?<=[.!?])\s", cleaned.strip(), maxsplit=1)[0]
    sentence = re.sub(rf"\b{re.escape(canon_number)}\b", "", sentence)
    sentence = sentence.strip()
    # Guard against a degenerate/empty result (very short canon text, or a
    # canon whose whole first sentence was just its own number).
    if len(sentence) < 15:
        sentence = cleaned.strip()[:200]
    return sentence


def build_auto_sampled_questions(ui, n_sampled: int, seed: int):
    """Pulls real canons directly from the live ChromaDB collection and
    builds two question phrasings per sampled canon (numeric + semantic),
    with ground truth = the canon they were built from. See module
    docstring for why this is safer than hand-written ground truth."""
    collection = ui._get_canon_collection()
    if collection is None:
        print("ERROR: couldn't open the ChromaDB collection -- check "
              "CANON_CHROMA_DIR/CANON_COLLECTION_NAME in ui.py and that "
              "chromadb is installed.", file=sys.stderr)
        sys.exit(1)

    all_data = collection.get(include=["metadatas", "documents"])
    ids = all_data.get("ids", [])
    metas = all_data.get("metadatas", [])
    docs = all_data.get("documents", [])

    pool = [
        (meta.get("canon_number"), doc)
        for meta, doc in zip(metas, docs)
        if meta.get("canon_number") and doc and len(doc.strip()) > 20
    ]
    if not pool:
        print("ERROR: collection has no usable canon_number/document pairs "
              "to sample from.", file=sys.stderr)
        sys.exit(1)

    # Sort by canon number (numeric where possible) and take evenly spaced
    # samples across the whole range, so the test set spans the whole Code
    # (Books I-VII) instead of clustering wherever insertion order happened
    # to put things.
    def sort_key(item):
        try:
            return (0, int(item[0]))
        except (TypeError, ValueError):
            return (1, str(item[0]))

    pool.sort(key=sort_key)
    n_sampled = min(n_sampled, len(pool))
    step = len(pool) / n_sampled
    rng = random.Random(seed)
    sampled = []
    for i in range(n_sampled):
        # small jitter within each stratum so reruns with a different seed
        # don't always land on the exact same canon at each position
        lo = int(i * step)
        hi = max(lo + 1, int((i + 1) * step))
        idx = rng.randrange(lo, min(hi, len(pool)))
        sampled.append(pool[idx])

    questions = []
    for canon_number, doc_text in sampled:
        questions.append({
            "type": "auto_numeric",
            "expected_canon": str(canon_number),
            "question": f"What does canon {canon_number} establish?",
        })
        semantic_q = first_sentence_stripped_of_number(doc_text, str(canon_number))
        questions.append({
            "type": "auto_semantic",
            "expected_canon": str(canon_number),
            "question": f"Which canon addresses this: {semantic_q}",
        })
    return questions


def evaluate_retrieval(ui, question: str, expected_canon: str, fetch_k: int = 10):
    """Runs the actual production retrieve_canons() and checks whether the
    expected canon shows up, and at what rank, among the sources returned."""
    t0 = time.perf_counter()
    context_text, sources = ui.retrieve_canons(question, n_results=fetch_k)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    if sources is None:
        return {"retrieved": [], "rank": None, "latency_ms": elapsed_ms, "error": "retrieval_failed"}

    retrieved = [str(s.get("canon_number")) for s in sources]
    rank = None
    if expected_canon in retrieved:
        rank = retrieved.index(expected_canon) + 1  # 1-indexed for MRR
    return {"retrieved": retrieved, "rank": rank, "latency_ms": elapsed_ms, "error": None}


def evaluate_generation(ui, question: str, expected_canon: str | None):
    """Runs the actual production canon_chat_fn's underlying pieces
    (retrieve_canons + _chat_backend with CANON_SYSTEM_PROMPT) to get a
    real generated answer, and checks whether it cites the expected canon
    (when we have one) and/or any canon at all (for curated questions)."""
    context_text, sources = ui.retrieve_canons(question)
    if context_text is None:
        return {"answer": None, "cited_expected": False, "cited_any": False,
                "latency_s": None, "error": "retrieval_failed"}

    messages = [
        {"role": "system", "content": ui.CANON_SYSTEM_PROMPT},
        {"role": "user", "content": f"Canons retrieved:\n\n{context_text}\n\nQuestion: {question}"},
    ]
    t0 = time.perf_counter()
    try:
        answer = ui._chat_backend(
            messages, model=ui.CANON_MODEL, num_predict=ui._CANON_NUM_PREDICT,
            timeout=ui._CANON_REQUEST_TIMEOUT_SECONDS,
        )
        error = None
    except RuntimeError as e:
        answer = None
        error = str(e)
    elapsed_s = time.perf_counter() - t0

    cited_any = bool(answer and CITATION_RE.search(answer))
    cited_expected = bool(
        answer and expected_canon and
        re.search(rf"\bcan\.?\s*{re.escape(expected_canon)}\b", answer, re.IGNORECASE)
    )
    return {"answer": answer, "cited_expected": cited_expected, "cited_any": cited_any,
             "latency_s": elapsed_s, "error": error}


def summarize(label: str, rows: list[dict]):
    n = len(rows)
    if n == 0:
        return
    hit1 = sum(1 for r in rows if r["rank"] == 1) / n
    hit3 = sum(1 for r in rows if r["rank"] is not None and r["rank"] <= 3) / n
    hit5 = sum(1 for r in rows if r["rank"] is not None and r["rank"] <= 5) / n
    mrr = sum((1 / r["rank"]) if r["rank"] else 0 for r in rows) / n
    lat = [r["latency_ms"] for r in rows if r["latency_ms"] is not None]
    print(f"\n  {label}  (n={n})")
    print(f"    Recall@1: {hit1:6.1%}   Recall@3: {hit3:6.1%}   Recall@5: {hit5:6.1%}   MRR: {mrr:.3f}")
    if lat:
        print(f"    Retrieval latency (ms) -- mean: {statistics.mean(lat):.0f}  "
              f"median: {statistics.median(lat):.0f}  p95: {sorted(lat)[int(len(lat)*0.95)-1]:.0f}")


def results_as_csv_rows(results: list[dict]) -> list[dict]:
    """Mirrors exactly how csv.DictWriter below serializes these same
    results (None -> '', bool -> 'True'/'False', list -> joined string),
    so compute_metrics() sees identical string values whether it's reading
    a freshly-written CSV back from disk or these in-memory results
    right after a run -- no separate/divergent code path for the two."""
    rows = []
    for r in results:
        row = dict(r)
        if isinstance(row.get("retrieved"), list):
            row["retrieved"] = ", ".join(row["retrieved"])
        for k, v in list(row.items()):
            if v is None:
                row[k] = ""
            elif isinstance(v, bool):
                row[k] = str(v)
        rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ui-path", default="ui.py", help="Path to ui.py (default: ./ui.py)")
    ap.add_argument("--n-sampled", type=int, default=35,
                     help="Number of canons to auto-sample from ChromaDB (x2 questions each)")
    ap.add_argument("--seed", type=int, default=42, help="Random seed, for reproducible sampling")
    ap.add_argument("--full", action="store_true",
                     help="Also run LLM generation and score citation accuracy (slow -- one "
                          "real local LLM call per question)")
    ap.add_argument("--limit", type=int, default=None,
                     help="Only run the first N questions total (quick smoke test)")
    ap.add_argument("--fetch-k", type=int, default=10,
                     help="How many candidates to check for Recall@k / MRR (k up to this value)")
    args = ap.parse_args()

    ui_path = Path(args.ui_path).resolve()
    if not ui_path.exists():
        print(f"ERROR: {ui_path} not found. Pass --ui-path pointing at your ui.py.", file=sys.stderr)
        sys.exit(1)

    print(f"Loading production RAG code from {ui_path} ...")
    ui = load_ui_module(ui_path)

    print(f"Sampling {args.n_sampled} canons directly from ChromaDB "
          f"({ui.CANON_CHROMA_DIR}, collection '{ui.CANON_COLLECTION_NAME}') "
          f"to build ground-truth questions...")
    auto_questions = build_auto_sampled_questions(ui, args.n_sampled, args.seed)
    all_questions = auto_questions + [
        {"type": "curated", "expected_canon": None, "question": q} for q in CURATED_QUESTIONS
    ]
    if args.limit:
        all_questions = all_questions[:args.limit]

    print(f"Running {len(all_questions)} questions "
          f"({len(auto_questions)} auto-sampled + {len(CURATED_QUESTIONS)} curated)"
          + (" -- retrieval only" if not args.full else " -- retrieval + generation") + " ...\n")

    results = []
    for i, q in enumerate(all_questions, 1):
        retrieval = evaluate_retrieval(ui, q["question"], q["expected_canon"], fetch_k=args.fetch_k)
        row = {**q, **retrieval}
        if args.full:
            gen = evaluate_generation(ui, q["question"], q["expected_canon"])
            row.update(gen)
        results.append(row)
        status = "hit" if row.get("rank") else ("n/a" if q["type"] == "curated" else "MISS")
        print(f"  [{i}/{len(all_questions)}] ({q['type']:>13}) {status:>4}  {q['question'][:70]}")

    # --- Scorecard ----------------------------------------------------------
    print("\n" + "=" * 72)
    print("RETRIEVAL SCORECARD (ground truth = the canon each question was built from)")
    print("=" * 72)
    numeric_rows = [r for r in results if r["type"] == "auto_numeric"]
    semantic_rows = [r for r in results if r["type"] == "auto_semantic"]
    summarize("Numeric lookups   (tests the exact-match code path)", numeric_rows)
    summarize("Semantic queries  (tests real vector-DB / hybrid-rerank quality)", semantic_rows)
    summarize("Combined auto-sampled", numeric_rows + semantic_rows)

    curated_rows = [r for r in results if r["type"] == "curated"]
    if curated_rows:
        n_retrieved_something = sum(1 for r in curated_rows if r["retrieved"])
        print(f"\n  Curated conceptual questions (n={len(curated_rows)}) -- QUALITATIVE ONLY, "
              f"no ground truth asserted:")
        print(f"    Retrieved >=1 canon for {n_retrieved_something}/{len(curated_rows)} questions.")
        print(f"    Review the CSV to judge whether what was retrieved actually looks relevant.")

    if args.full:
        gen_rows = [r for r in results if r.get("answer") is not None]
        if gen_rows:
            auto_gen = [r for r in gen_rows if r["type"].startswith("auto")]
            if auto_gen:
                cite_acc = sum(1 for r in auto_gen if r["cited_expected"]) / len(auto_gen)
                print(f"\nGENERATION SCORECARD")
                print(f"  Citation accuracy on auto-sampled questions (cited the expected "
                      f"canon by number): {cite_acc:.1%}")
            gen_lat = [r["latency_s"] for r in gen_rows if r["latency_s"] is not None]
            if gen_lat:
                print(f"  Generation latency (s) -- mean: {statistics.mean(gen_lat):.1f}  "
                      f"median: {statistics.median(gen_lat):.1f}")
            curated_gen = [r for r in gen_rows if r["type"] == "curated"]
            if curated_gen:
                cited_any_pct = sum(1 for r in curated_gen if r["cited_any"]) / len(curated_gen)
                print(f"  Curated questions where the answer cited at least one canon: {cited_any_pct:.1%}")

    # --- CSV ------------------------------------------------------------
    out_path = Path(f"canon_rag_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    fieldnames = ["type", "question", "expected_canon", "retrieved", "rank",
                  "latency_ms", "error"]
    if args.full:
        fieldnames += ["answer", "cited_expected", "cited_any", "latency_s"]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in results:
            row = dict(r)
            row["retrieved"] = ", ".join(row.get("retrieved", []))
            writer.writerow(row)
    print(f"\nDetailed per-question results written to: {out_path}")

    try:
        import plot_canon_accuracy as plot_mod
        metrics = plot_mod.compute_metrics(results_as_csv_rows(results))
        chart_path = out_path.with_name(out_path.stem + "_chart.png")
        plot_mod.render_chart(metrics, chart_path, title_suffix=f"  ({out_path.name})")
        print(f"Accuracy chart saved to: {chart_path}")
    except ImportError:
        print("\n(plot_canon_accuracy.py not found alongside this script -- skipping chart "
              "generation. You can still chart the CSV later with:\n"
              f"  python plot_canon_accuracy.py --csv {out_path}")


if __name__ == "__main__":
    main()
