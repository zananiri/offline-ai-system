#!/usr/bin/env python3
"""
Chart the accuracy results from test_canon_rag.py
===================================================
Reads the per-question CSV produced by test_canon_rag.py (either passed
directly, or auto-detects the most recent canon_rag_eval_*.csv in the
current directory) and renders an accuracy chart covering:

  1. Retrieval Recall@1 / @3 / @5 and MRR, broken out by question type
     (numeric lookups vs. semantic queries vs. combined) -- this is the
     one that actually reflects vector-DB quality, since numeric lookups
     get an assist from the exact-match code path.
  2. Generation citation accuracy and latency, if the CSV was produced
     with --full (skipped gracefully if not present).

Can be run two ways:
  - Standalone, against an existing CSV:
        python plot_canon_accuracy.py
        python plot_canon_accuracy.py --csv canon_rag_eval_20260726_140512.csv
  - Imported and called directly by test_canon_rag.py right after a run,
    so a chart is produced automatically without a separate step.
"""

import argparse
import csv
import glob
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # no display available in most environments this runs in
import matplotlib.pyplot as plt


def _to_float(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def load_results(csv_path: Path) -> list[dict]:
    with open(csv_path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def compute_metrics(rows: list[dict]) -> dict:
    def group_metrics(group_rows):
        n = len(group_rows)
        if n == 0:
            return None
        ranks = [_to_float(r.get("rank")) for r in group_rows]
        hit1 = sum(1 for r in ranks if r == 1) / n
        hit3 = sum(1 for r in ranks if r is not None and r <= 3) / n
        hit5 = sum(1 for r in ranks if r is not None and r <= 5) / n
        mrr = sum((1 / r) if r else 0 for r in ranks) / n
        return {"n": n, "recall@1": hit1, "recall@3": hit3, "recall@5": hit5, "mrr": mrr}

    numeric = [r for r in rows if r.get("type") == "auto_numeric"]
    semantic = [r for r in rows if r.get("type") == "auto_semantic"]
    combined = numeric + semantic
    curated = [r for r in rows if r.get("type") == "curated"]

    metrics = {
        "numeric": group_metrics(numeric),
        "semantic": group_metrics(semantic),
        "combined": group_metrics(combined),
    }

    # Generation metrics, only present if the CSV was produced with --full
    has_generation = any("cited_expected" in r for r in rows)
    if has_generation:
        auto_gen = [r for r in combined if r.get("cited_expected") not in (None, "")]
        if auto_gen:
            cite_acc = sum(1 for r in auto_gen if r.get("cited_expected") == "True") / len(auto_gen)
            metrics["citation_accuracy"] = cite_acc
        if curated:
            cited_any = [r for r in curated if r.get("cited_any") not in (None, "")]
            if cited_any:
                metrics["curated_cited_any"] = sum(1 for r in cited_any if r.get("cited_any") == "True") / len(cited_any)
        gen_latencies = [_to_float(r.get("latency_s")) for r in rows if _to_float(r.get("latency_s")) is not None]
        if gen_latencies:
            metrics["mean_gen_latency_s"] = sum(gen_latencies) / len(gen_latencies)

    retrieval_latencies = [_to_float(r.get("latency_ms")) for r in rows if _to_float(r.get("latency_ms")) is not None]
    if retrieval_latencies:
        metrics["mean_retrieval_latency_ms"] = sum(retrieval_latencies) / len(retrieval_latencies)

    return metrics


def render_chart(metrics: dict, out_path: Path, title_suffix: str = ""):
    has_generation = "citation_accuracy" in metrics
    fig, axes = plt.subplots(1, 2 if has_generation else 1, figsize=(13 if has_generation else 7, 5.5))
    if not has_generation:
        axes = [axes]

    # --- Panel 1: Recall@k + MRR by question type ---------------------------
    ax = axes[0]
    groups = [g for g in ("numeric", "semantic", "combined") if metrics.get(g)]
    labels = {"numeric": "Numeric\n(exact-match)", "semantic": "Semantic\n(vector DB)", "combined": "Combined"}
    metric_keys = ["recall@1", "recall@3", "recall@5", "mrr"]
    metric_colors = ["#8fb4e3", "#6d8fd9", "#5c7dc7", "#2c3e63"]
    x = range(len(groups))
    bar_width = 0.2
    for i, mk in enumerate(metric_keys):
        vals = [metrics[g][mk] for g in groups]
        offsets = [xi + (i - 1.5) * bar_width for xi in x]
        bars = ax.bar(offsets, vals, width=bar_width, label=mk.replace("recall@", "Recall@").upper()
                       if mk == "mrr" else mk.replace("recall@", "Recall@"), color=metric_colors[i])
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.2f}", ha="center", fontsize=8)
    ax.set_xticks(list(x))
    ax.set_xticklabels([labels[g] for g in groups])
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score")
    ax.set_title("Retrieval accuracy" + title_suffix)
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    # --- Panel 2: Generation citation accuracy + latency (if --full) --------
    if has_generation:
        ax2 = axes[1]
        bars_data = [("Cited expected\ncanon (auto-sampled)", metrics.get("citation_accuracy", 0))]
        if "curated_cited_any" in metrics:
            bars_data.append(("Cited any canon\n(curated, qualitative)", metrics["curated_cited_any"]))
        names = [b[0] for b in bars_data]
        vals = [b[1] for b in bars_data]
        bars = ax2.bar(names, vals, color=["#2c3e63", "#8fb4e3"][:len(names)])
        for b, v in zip(bars, vals):
            ax2.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.0%}", ha="center", fontsize=9)
        ax2.set_ylim(0, 1.15)
        ax2.set_ylabel("Accuracy")
        ax2.set_title("LLM generation accuracy" + title_suffix)
        ax2.grid(axis="y", alpha=0.3)
        if "mean_gen_latency_s" in metrics:
            ax2.text(0.5, -0.22, f"Mean generation latency: {metrics['mean_gen_latency_s']:.1f}s",
                      transform=ax2.transAxes, ha="center", fontsize=8, color="#555")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def find_latest_csv() -> Path | None:
    candidates = sorted(glob.glob("canon_rag_eval_*.csv"))
    return Path(candidates[-1]) if candidates else None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default=None, help="Path to a canon_rag_eval_*.csv (default: most recent in cwd)")
    ap.add_argument("--out", default=None, help="Output PNG path (default: <csv-name>_chart.png)")
    args = ap.parse_args()

    csv_path = Path(args.csv) if args.csv else find_latest_csv()
    if csv_path is None or not csv_path.exists():
        print("ERROR: no CSV found. Run test_canon_rag.py first, or pass --csv explicitly.", file=sys.stderr)
        sys.exit(1)

    rows = load_results(csv_path)
    if not rows:
        print(f"ERROR: {csv_path} has no rows.", file=sys.stderr)
        sys.exit(1)

    metrics = compute_metrics(rows)
    out_path = Path(args.out) if args.out else csv_path.with_name(csv_path.stem + "_chart.png")
    render_chart(metrics, out_path, title_suffix=f"  ({csv_path.name})")
    print(f"Chart saved to: {out_path}")


if __name__ == "__main__":
    main()
