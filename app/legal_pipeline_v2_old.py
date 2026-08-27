"""
app/legal_pipeline_v2.py

Implements the multi-model pipeline from the "Simple Israeli Legal AI —
Programmer Workflow" architecture doc, grounding the Attorney 24B tab:

    Dicta (query understanding)
      -> BGE-M3 (embed)      -\
      -> BM25 (keyword)       |-> Reciprocal Rank Fusion -> hybrid rerank
      -> ChromaDB (vector)   -/
      -> DeepSeek (legal reasoning over retrieved sources)
      -> Dicta (Hebrew draft, no new sources)
      -> DeepSeek (verification: are the claims/citations actually supported?)
      -> retry retrieval+reasoning up to MAX_VERIFICATION_CYCLES on FAIL
      -> Dicta final answer

This module is deliberately self-contained (talks to Ollama and ChromaDB
directly, no dependency on app/ui.py) so app/ui.py can import it without a
circular import -- same "duplicate small pieces rather than couple two
independent modules" reasoning already used throughout this project (see
e.g. scripts/embed_local_law_pdfs.py's module docstring).

--- What this expects to already exist ---

  - ChromaDB collection "israeli_legal_db" at app/chroma_db, built by
    scripts/embed_local_law_pdfs_bgem3.py -- BGE-M3 embeddings, the rich
    metadata schema described in that script's docstring.
  - `ollama pull bge-m3`               (embeddings -- MUST match the model
                                         used to build the collection, or
                                         vector search is comparing vectors
                                         from two different embedding spaces)
  - `ollama pull deepseek-r1:32b`      (DeepSeek-R1-Distill-Qwen-32B -- see
                                         DEEPSEEK_MODEL below)
  - The existing DictaLM pull already used by app/ui.py's LEGAL_MODEL
  - pip install chromadb rank_bm25

--- Degraded operation ---

Like the rest of this app's RAG tabs, every external dependency (Chroma,
Ollama, rank_bm25) is checked defensively -- PIPELINE_AVAILABLE is False if
chromadb or rank_bm25 aren't importable, and answer_legal_question() raises
a plain RuntimeError with a clear message on any live failure (Ollama
unreachable, collection missing, etc.) rather than crashing. The caller
(app/ui.py) is expected to catch that and show it in-chat, same pattern as
_chat_backend's RuntimeError elsewhere in this project.
"""
import json
import re
import unicodedata
from pathlib import Path

import requests

try:
    import chromadb
except ImportError:
    chromadb = None

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    BM25Okapi = None

PIPELINE_AVAILABLE = chromadb is not None and BM25Okapi is not None

CHROMA_DIR = str(Path(__file__).resolve().parent / "chroma_db")
COLLECTION_NAME = "israeli_legal_db"

OLLAMA_CHAT_URL = "http://localhost:11434/api/chat"
OLLAMA_EMBED_URL = "http://localhost:11434/api/embeddings"
EMBED_MODEL = "bge-m3"  # must match scripts/embed_local_law_pdfs_bgem3.py

# NOTE: duplicated from app/ui.py's LEGAL_MODEL rather than imported (see
# this module's docstring) -- keep the two in sync if you change the model.
DICTA_MODEL = "hf.co/dicta-il/DictaLM-3.0-24B-Thinking-GGUF:Q4_K_M"
# DeepSeek-R1-Distill-Qwen-32B via Ollama's "deepseek-r1:32b" tag (Ollama's
# deepseek-r1 tags are the distilled variants -- 32b is specifically the
# Qwen-32B distillation, not the full 671B model). Meaningfully stronger
# reasoning than the smaller distills, at a real cost: ~19-20GB on disk at
# Q4_K_M, and its KV cache on top of that. Alongside DictaLM-3.0-24B-
# Thinking's ~14.3GB (DICTA_MODEL above), this pair does NOT both fit
# resident in memory at once on the 32GB-RAM/no-GPU hardware this project
# targets (see app/ui.py's LEGAL_MODEL comment for that machine's numbers).
# With OLLAMA_MAX_LOADED_MODELS=1 (see run.ps1/setup.ps1) this still works
# correctly -- Ollama unloads one model before loading the other -- but
# every switch between a Dicta call and a DeepSeek call now pays a real
# model-load cost, on top of the several sequential calls each verification
# cycle already makes, which can push a single answer to 10+ minutes.
#
# DEEPSEEK_MODEL_DEFAULT / DEEPSEEK_MODEL_FAST exist as two named presets
# (rather than one hardcoded DEEPSEEK_MODEL constant) specifically so
# app/ui.py can offer two Attorney tabs on the same pipeline logic, one per
# DeepSeek size -- see legal_chat_fn / legal_chat_fn_deepseek14b there.
# answer_legal_question()/deepseek_analyze()/deepseek_verify() below all
# take an explicit deepseek_model parameter (defaulting to
# DEEPSEEK_MODEL_DEFAULT) rather than reading a single module-level
# constant, so a caller genuinely picks per-call, not just at import time.
DEEPSEEK_MODEL_DEFAULT = "deepseek-r1:32b"
# ~9GB at Q4_K_M -- comfortably fits alongside DictaLM-3.0-24B-Thinking on
# this project's target 32GB-RAM/no-GPU hardware without the swap-heavy
# latency the 32B variant above incurs. Meaningfully weaker legal reasoning
# than the 32B, same tradeoff shape as LEGAL_MODEL_FAST vs LEGAL_MODEL in
# app/ui.py's older Knesset-RAG pipeline.
DEEPSEEK_MODEL_FAST = "deepseek-r1:14b"
# Backward-compat alias -- earlier versions of this module only had one
# DEEPSEEK_MODEL constant. Kept pointing at the default/strongest variant.
DEEPSEEK_MODEL = DEEPSEEK_MODEL_DEFAULT

_CHAT_TIMEOUT_SECONDS = 1800
_ANALYZE_NUM_PREDICT = 2048
_VERIFY_NUM_PREDICT = 1024
_DRAFT_NUM_PREDICT = 2048
_UNDERSTAND_NUM_PREDICT = 512

MAX_VERIFICATION_CYCLES = 3

_FETCH_K_VECTOR = 30
_FETCH_K_BM25 = 30
_RRF_K = 60           # standard RRF damping constant
_RERANK_TOP_N = 10    # "TOP 5-15 SOURCES" per the doc -- 10 is the midpoint
_MIN_RELEVANCE_SIMILARITY = 0.20  # same floor used elsewhere in this project
                                   # (app/ui.py's _MIN_RELEVANCE_SIMILARITY) --
                                   # below this, a candidate is not "relevant",
                                   # it's just the least-bad thing retrieved.
_LEXICAL_WEIGHT = 0.35

# --- Context-window budgeting for the Dicta/DeepSeek calls below -----------
#
# hybrid_retrieve can hand back up to _RERANK_TOP_N (10) source chunks, each
# up to 2500 characters of statute text (see scripts/embed_local_law_pdfs_
# bgem3.py's _MAX_CHUNK_CHARS). Worst case that's ~25000 characters of
# Hebrew legal text -- comfortably enough on its own to blow past any of the
# fixed num_ctx values below, even before the system prompt, the question,
# or (for deepseek_verify) the analysis + draft are added on top.
# deepseek_analyze/deepseek_verify used to pass sources straight through
# with a FIXED, unchecked num_ctx -- which is exactly the failure this
# closes: "request (10045 tokens) exceeds the available context size
# (8192 tokens)".
#
# 11264 matches app/ui.py's _LEGAL_NUM_CTX -- already confirmed safe on this
# project's target 32GB-RAM/no-GPU hardware for DictaLM-3.0-24B-Thinking's
# KV cache. DEEPSEEK_MODEL_FAST (14B) has a meaningfully smaller footprint
# than that 24B model (see DEEPSEEK_MODEL_FAST's own comment above), so the
# same context size is comfortable headroom here, not a stretch -- and
# DEEPSEEK_MODEL_DEFAULT (32B) never needs to share RAM with anything else
# at the same instant anyway, since OLLAMA_MAX_LOADED_MODELS=1 (see run.ps1)
# means only one model is ever resident at once regardless of which tab is
# active.
_DICTA_NUM_CTX = 11264

# DeepSeek-R1-Distill-Qwen-14B's weights (~9GB) leave meaningfully more RAM
# headroom on this project's 32GB-RAM/no-GPU hardware than the 32B variant
# does (see DEEPSEEK_MODEL_FAST's own comment above: "comfortably fits
# alongside DictaLM-3.0-24B-Thinking... without the swap-heavy latency the
# 32B variant incurs"). A bigger context window here directly reduces how
# often _trim_sources_to_budget has to cut a genuinely relevant retrieval
# down to one narrow fragment -- which was a real contributor to the
# hallucination this is fixing: a small model reasoning from a single
# partial excerpt is under real pressure to "fill in" the rest from its
# own parametric memory, dressed up as grounded fact.
#
# DEEPSEEK_MODEL_DEFAULT (32B) stays at the DictaLM-matched 11264 -- that
# model does NOT comfortably fit alongside DictaLM per this module's own
# earlier comment, so it gets no extra headroom assumed here. Re-check via
# `ollama ps`/the server log if you push _DEEPSEEK_NUM_CTX_BY_MODEL[14B]
# further -- this is a reasoned starting point, not independently
# benchmarked on this exact hardware.
_DEEPSEEK_NUM_CTX_BY_MODEL = {
    DEEPSEEK_MODEL_FAST: 16384,
}
_DEEPSEEK_NUM_CTX_DEFAULT = 11264


def _deepseek_num_ctx(model: str) -> int:
    return _DEEPSEEK_NUM_CTX_BY_MODEL.get(model, _DEEPSEEK_NUM_CTX_DEFAULT)


# Budget for the formatted RETRIEVED-SOURCES text specifically, in the same
# weighted units as app/ui.py's _estimate_token_weight (see that function's
# docstring for why Hebrew needs a heavier per-character weight than a flat
# chars-per-token guess -- duplicated here rather than imported, same
# "small helpers stay duplicated across independent modules" reasoning
# already used throughout this project, e.g. scripts/embed_local_law_pdfs.py).
#
# Sized against deepseek_verify -- the TIGHTEST consumer, since its prompt
# carries the sources AND the analysis AND the draft all at once:
#   num_ctx
#   - _VERIFY_NUM_PREDICT (1024, verify's own JSON output reserve)
#   - _ANALYZE_NUM_PREDICT (2048, analysis's real token cap -- already an
#     exact token count, not a weight estimate, since num_predict directly
#     bounds how many tokens that stage could have generated)
#   - _DRAFT_NUM_PREDICT (2048, draft's real token cap, same reasoning)
#   - ~800 headroom for the system prompt + "Question: ...\n\nRetrieved
#     sources:\n\n" wrapper text
# A function of num_ctx (not a flat constant) so a model with a larger
# budget (see _DEEPSEEK_NUM_CTX_BY_MODEL) automatically gets a
# correspondingly larger source allowance instead of two numbers needing
# to be kept in sync by hand.
def _sources_max_weight_for(num_ctx: int) -> float:
    reserved = _VERIFY_NUM_PREDICT + _ANALYZE_NUM_PREDICT + _DRAFT_NUM_PREDICT + 800
    return max(0.0, (num_ctx - reserved) * 0.9)  # 10% cushion against the weight estimate being imperfect


_SOURCES_CONTEXT_MAX_WEIGHT = _sources_max_weight_for(_DEEPSEEK_NUM_CTX_DEFAULT)  # ~4900; default/fallback


def _estimate_token_weight(text: str) -> float:
    """Rough, script-aware proxy for prompt-token cost -- see app/ui.py's
    function of the same name for the full reasoning (Hebrew routinely
    needs more tokens per character than Latin script for a general-
    purpose BPE tokenizer, so Hebrew characters are weighted ~2x more
    heavily). Intentionally pessimistic: better to trim a source that
    would actually have fit than to let one through that overflows the
    context window the way the un-budgeted code before this did."""
    hebrew_chars = sum(1 for ch in text if 0x0590 <= ord(ch) <= 0x05FF)
    other_chars = len(text) - hebrew_chars
    return hebrew_chars * 2.0 + other_chars * 1.0


def _trim_sources_to_budget(sources: list[dict], max_weight: float = _SOURCES_CONTEXT_MAX_WEIGHT) -> list[dict]:
    """
    Drops the lowest-ranked sources (from the end -- hybrid_retrieve's
    rerank already sorts `sources` best-first) until the formatted source
    text fits under max_weight weighted units.

    Drops whole sources rather than truncating one mid-text, so what's left
    is always a complete, citable excerpt -- same "never truncate
    mid-section" principle app/ui.py's _fetch_whole_law already applies.
    Always keeps at least the single best source, even if it alone exceeds
    max_weight -- better to attempt the call with the best available
    excerpt (and risk a slower/larger single-source prompt) than to
    silently answer with zero grounding.

    Applied ONCE, right after retrieval (see answer_legal_question), rather
    than separately inside deepseek_analyze/deepseek_verify -- so every
    downstream consumer (analysis, verification, and the sources panel
    shown to the user) is working from the exact same set of sources,
    instead of a set that quietly differs stage to stage.
    """
    if not sources:
        return sources
    kept, total = [], 0.0
    for s in sources:
        weight = _estimate_token_weight(_format_sources_for_prompt([s]))
        if kept and total + weight > max_weight:
            break
        kept.append(s)
        total += weight
    if len(kept) < len(sources):
        print(f"[legal_pipeline_v2] trimmed {len(sources)} retrieved source(s) down to "
              f"{len(kept)} to fit the {max_weight:.0f}-weighted-unit source budget "
              f"(~{total:.0f} units kept).")
    return kept


# --- Ollama helpers ---------------------------------------------------------

def _ollama_chat(model: str, messages: list[dict], *, format_json: bool = False,
                  num_predict: int = 1024, num_ctx: int = 8192,
                  timeout: int = _CHAT_TIMEOUT_SECONDS) -> str:
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {"num_predict": num_predict, "num_ctx": num_ctx},
    }
    if format_json:
        payload["format"] = "json"
    try:
        resp = requests.post(OLLAMA_CHAT_URL, json=payload, timeout=timeout)
        resp.raise_for_status()
    except requests.exceptions.Timeout:
        raise RuntimeError(f"{model} didn't respond within {timeout // 60} minutes.")
    except requests.exceptions.HTTPError:
        # requests' raise_for_status() only gives the status code/reason
        # phrase, not Ollama's own error body -- which is where the actually
        # useful message lives. Ollama's error body shape is
        # {"error": {"code": ..., "message": ..., "type": ..., ...}} (a
        # NESTED object, not a plain string) for structured errors like
        # context-size overflow -- treating resp.json().get("error", ...)
        # as if it were already the message string was the earlier bug
        # here: it silently produced the wrong text instead of the actual
        # "message" field, and (worse) blanket-classified EVERY 400 as
        # "model not pulled," which is wrong for a context-size error and
        # actively misleading (it tells you to `ollama pull` a model that
        # was never the problem).
        try:
            body = resp.json()
        except Exception:
            body = {}
        err_obj = body.get("error") if isinstance(body, dict) else None
        if isinstance(err_obj, dict):
            message = err_obj.get("message", "")
            err_type = err_obj.get("type", "")
        else:
            # Some Ollama versions/errors return "error" as a plain string
            # instead of a nested object -- handle both shapes.
            message = err_obj or ""
            err_type = ""
        message = message or resp.text or "(no detail returned)"

        if resp.status_code == 404 or "not found" in message.lower():
            raise RuntimeError(
                f"Model '{model}' doesn't seem to be pulled in Ollama "
                f"(HTTP {resp.status_code}: {message}). Run: ollama pull {model}"
            )
        if err_type == "exceed_context_size_error" or "exceeds the available context" in message.lower():
            raise RuntimeError(
                f"The request to {model!r} was too large for its context window "
                f"({message}). This call's num_ctx (currently {num_ctx}) needs to be "
                f"raised, or the retrieved sources trimmed further -- see "
                f"_SOURCES_CONTEXT_MAX_WEIGHT / _trim_sources_to_budget in "
                f"app/legal_pipeline_v2.py."
            )
        raise RuntimeError(f"Ollama rejected the request for model {model!r} "
                            f"(HTTP {resp.status_code}, type={err_type or 'unknown'}): {message}")
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Couldn't reach Ollama for model {model!r}: {e}")
    data = resp.json()
    content = data.get("message", {}).get("content", "")
    return _strip_think(content)


# DeepSeek-R1 and DictaLM-3.0-Thinking both emit <think>...</think> chain-of-
# thought before the real answer -- same issue app/main.py's _strip_thinking
# handles for the backend's /chat endpoint. Duplicated here (not imported --
# main.py pulls in FastAPI/ollama-python machinery this module doesn't need)
# since this module talks to Ollama's raw HTTP API directly, not through the
# FastAPI backend.
_THINK_RE = re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL)
_UNCLOSED_THINK_RE = re.compile(r"<think>.*", re.IGNORECASE | re.DOTALL)


def _strip_think(content: str) -> str:
    stripped = _THINK_RE.sub("", content).strip()
    if "<think>" in stripped.lower():
        stripped = _UNCLOSED_THINK_RE.sub("", stripped).strip()
    return stripped


def _embed(text: str) -> list[float] | None:
    try:
        resp = requests.post(OLLAMA_EMBED_URL, json={"model": EMBED_MODEL, "prompt": text}, timeout=30)
        resp.raise_for_status()
        return resp.json()["embedding"]
    except Exception as e:
        print(f"[legal_pipeline_v2] embedding call failed: {e}")
        return None


def _parse_json_response(raw: str) -> dict | None:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```[a-zA-Z]*\n?", "", raw)
        raw = re.sub(r"```\s*$", "", raw)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


# --- Collection / BM25 index -----------------------------------------------

_collection_cache = None


def _get_collection():
    global _collection_cache
    if _collection_cache is not None:
        return _collection_cache
    if chromadb is None:
        return None
    try:
        client = chromadb.PersistentClient(path=CHROMA_DIR)
        _collection_cache = client.get_collection(COLLECTION_NAME)
        return _collection_cache
    except Exception as e:
        print(f"[legal_pipeline_v2] couldn't open Chroma collection "
              f"'{COLLECTION_NAME}' at '{CHROMA_DIR}': {e}")
        return None


_STOPWORDS_HE = frozenset("""
של על עם לא כן זה זו אלה אלו הוא היא הם הן אני אתה את אנחנו אתם
מה מי איך למה כאשר כי אם או גם רק כל כמה יותר פחות
""".split())


def _tokenize_he(text: str) -> list[str]:
    """Hebrew/Latin/digit word tokenizer for BM25, deliberately more
    permissive than app/ui.py's Canon-AI-oriented _tokenize (that one is
    Latin-only). Used both to build the BM25 corpus index and to tokenize
    each query."""
    ascii_norm = unicodedata.normalize("NFKD", text)
    tokens = re.findall(r"[א-ת]+|[a-zA-Z]+|\d+", ascii_norm)
    return [t for t in tokens if t not in _STOPWORDS_HE and len(t) > 1]


# Common short greetings/chit-chat in Hebrew and English -- checked in
# addition to (not instead of) the length/stopword-based _has_meaningful_content
# test below, since "hi", "hey", "thanks" etc. are exactly the short,
# real-looking tokens that test alone would NOT catch (they're not
# stopwords, and they clear the length>1 filter).
_GREETING_ONLY_RE = re.compile(
    r"^\s*(hi|hey|hello|yo|sup|thanks?|thank you|thx|ok|okay|bye|goodbye|"
    r"שלום|היי|הי|תודה|ביי|נשיקה|בוקר טוב|ערב טוב|לילה טוב)[\s!.?]*$",
    re.IGNORECASE,
)


def _has_meaningful_content(query: str) -> bool:
    """
    True if the query has an actual legal topic to search on. Mirrors
    app/ui.py's _has_meaningful_content (used by the Canon/GDPR/HIPAA
    tabs) for the same reason: without this, a plain "hi" still triggers
    the FULL pipeline -- query understanding, hybrid retrieval, DeepSeek
    analysis, a Dicta draft, DeepSeek verification, and up to
    MAX_VERIFICATION_CYCLES retries once nothing relevant is ever found --
    which is both a multi-minute round trip and a confusing answer for
    what should just be a normal greeting response. Checked in
    answer_legal_question() before any model call is made at all.
    """
    if _GREETING_ONLY_RE.match(query.strip()):
        return False
    return bool(_tokenize_he(query))


_GREETING_REPLY = (
    "שלום! אני עוזר משפטי לחוק הישראלי, מבוסס על אחזור מקורות מהמאגר "
    "המשפטי (BGE-M3 + BM25) וניתוח/אימות באמצעות DeepSeek ו-Dicta. "
    "אשמח לעזור -- מה השאלה המשפטית שלך?\n\n"
    "*(Hi! I'm an Israeli-law assistant grounded in retrieved statute "
    "text, with DeepSeek reasoning/verification and a Dicta draft. Ask "
    "me a legal question and I'll search the database for you.)*"
)


_bm25_cache = {"count": None, "index": None, "ids": [], "docs": [], "metas": []}


def _get_bm25_index(collection):
    """Loads/caches a BM25 index over the WHOLE collection, rebuilt only
    when collection.count() changes (same caching pattern as app/ui.py's
    _get_knesset_law_titles) -- rebuilding on every question would mean
    re-tokenizing the entire legal corpus per request, which doesn't scale
    even at this app's modest personal/small-team size."""
    global _bm25_cache
    if BM25Okapi is None:
        return None
    try:
        current_count = collection.count()
    except Exception:
        current_count = None
    if current_count is not None and _bm25_cache["count"] == current_count:
        return _bm25_cache

    try:
        rows = collection.get(include=["documents", "metadatas"])
    except Exception as e:
        print(f"[legal_pipeline_v2] couldn't load documents for BM25 index: {e}")
        return _bm25_cache if _bm25_cache["index"] is not None else None

    ids = rows.get("ids") or []
    docs = rows.get("documents") or []
    metas = rows.get("metadatas") or []
    tokenized = [_tokenize_he(d) for d in docs]
    index = BM25Okapi(tokenized) if tokenized else None
    _bm25_cache = {"count": current_count, "index": index, "ids": ids, "docs": docs, "metas": metas}
    return _bm25_cache


# --- Retrieval: hybrid vector + BM25 + RRF ----------------------------------

def _cosine_similarity(a, b) -> float:
    if a is None or b is None or len(a) == 0 or len(b) == 0:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(x * x for x in b) ** 0.5
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def _vector_rankings(collection, query_vector: list[float], top_k: int) -> list[str]:
    """Returns a list of chunk ids ranked by vector similarity (best first).
    Actual similarity scores are recomputed later during rerank, from raw
    embeddings -- this step only needs rank order for RRF."""
    try:
        results = collection.query(
            query_embeddings=[query_vector], n_results=top_k,
            include=["distances"],
        )
        return (results.get("ids") or [[]])[0]
    except Exception as e:
        print(f"[legal_pipeline_v2] vector search failed: {e}")
        return []


def _bm25_rankings(bm25_cache: dict, query: str, top_k: int) -> list[str]:
    index = bm25_cache["index"]
    if index is None:
        return []
    scores = index.get_scores(_tokenize_he(query))
    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    return [bm25_cache["ids"][i] for i in ranked if scores[i] > 0]


def _reciprocal_rank_fusion(rankings_lists: list[list[str]], k: int = _RRF_K) -> list[str]:
    """Standard RRF: score(doc) = sum over rankings of 1 / (k + rank). Merges
    the vector-search rankings and BM25 rankings from EVERY generated query
    into one fused ranking, so a candidate that shows up (even mid-pack)
    across several signals outranks one that was merely the single best
    vector hit for one query -- exactly the "30-50 candidates" pool the
    architecture doc describes handing to the reranker."""
    scores: dict[str, float] = {}
    for ranking in rankings_lists:
        for rank, doc_id in enumerate(ranking):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    return [doc_id for doc_id, _ in sorted(scores.items(), key=lambda kv: kv[1], reverse=True)]


def hybrid_retrieve(queries: list[str]) -> list[dict]:
    """Runs every generated query through both BGE-M3 vector search and
    BM25, fuses the rankings with RRF, then reranks the fused candidate
    pool by a hybrid vector-similarity + lexical-overlap score (a cheap,
    dependency-free stand-in for a dedicated cross-encoder reranker -- see
    _rerank's docstring for how to swap in a real one).

    Returns up to _RERANK_TOP_N dicts: {"id", "document", "metadata",
    "score"}, most relevant first. Returns [] if the collection/BM25 index
    isn't available, or nothing clears the relevance floor.
    """
    collection = _get_collection()
    if collection is None:
        raise RuntimeError(
            f"Legal vector database (collection '{COLLECTION_NAME}') isn't available. "
            "Run scripts/embed_local_law_pdfs_bgem3.py first."
        )
    bm25_cache = _get_bm25_index(collection)
    if bm25_cache is None or bm25_cache["index"] is None:
        raise RuntimeError(
            "BM25 index couldn't be built (either rank_bm25 isn't installed, or the "
            "collection is empty). Run `pip install rank_bm25` and/or the ingestion script."
        )

    query_vectors = {}
    all_rankings = []
    for q in queries:
        vec = _embed(q)
        if vec is None:
            continue
        query_vectors[q] = vec
        all_rankings.append(_vector_rankings(collection, vec, _FETCH_K_VECTOR))
        all_rankings.append(_bm25_rankings(bm25_cache, q, _FETCH_K_BM25))

    if not all_rankings:
        raise RuntimeError("Couldn't reach the bge-m3 embedding model via Ollama.")

    fused_ids = _reciprocal_rank_fusion(all_rankings)
    if not fused_ids:
        return []

    return _rerank(queries, query_vectors, fused_ids, collection, bm25_cache, top_n=_RERANK_TOP_N)


def _rerank(queries: list[str], query_vectors: dict, fused_ids: list[str],
            collection, bm25_cache: dict, top_n: int) -> list[dict]:
    """Hybrid vector-similarity + lexical-overlap rerank of the fused
    candidate pool, matching the pattern already used elsewhere in this
    project (app/ui.py's _rerank_candidates for Canon/GDPR/HIPAA/Knesset).

    This is a pragmatic substitute for a real cross-encoder reranker (e.g.
    BAAI/bge-reranker-v2-m3): it needs no extra model download and runs
    entirely on vectors/text already in hand. If you later install
    sentence-transformers and want a proper cross-encoder instead, swap
    this function's body for a CrossEncoder(...).predict(...) call over
    (query, candidate_text) pairs -- the rest of the pipeline (RRF input,
    _RERANK_TOP_N output shape) doesn't need to change.
    """
    id_to_doc = dict(zip(bm25_cache["ids"], bm25_cache["docs"]))
    id_to_meta = dict(zip(bm25_cache["ids"], bm25_cache["metas"]))

    # Best query-token set across all generated queries, for lexical scoring.
    all_query_tokens: set[str] = set()
    for q in queries:
        all_query_tokens |= set(_tokenize_he(q))

    scored = []
    for doc_id in fused_ids:
        doc = id_to_doc.get(doc_id)
        meta = id_to_meta.get(doc_id, {})
        if doc is None:
            continue

        # Best vector similarity across every query variant, not just one.
        best_vec_sim = 0.0
        try:
            got = collection.get(ids=[doc_id], include=["embeddings"])
            embeds = got.get("embeddings")
            doc_vector = (embeds[0].tolist() if hasattr(embeds[0], "tolist") else embeds[0]) if embeds else None
        except Exception:
            doc_vector = None
        if doc_vector is not None:
            for vec in query_vectors.values():
                best_vec_sim = max(best_vec_sim, _cosine_similarity(vec, doc_vector))

        if best_vec_sim < _MIN_RELEVANCE_SIMILARITY and not (all_query_tokens & set(_tokenize_he(doc))):
            continue  # neither signal thinks this is relevant -- drop it, don't just rank it low

        doc_tokens = set(_tokenize_he(doc))
        lexical = len(all_query_tokens & doc_tokens) / len(all_query_tokens) if all_query_tokens else 0.0
        hybrid_score = (1 - _LEXICAL_WEIGHT) * best_vec_sim + _LEXICAL_WEIGHT * lexical
        scored.append({"id": doc_id, "document": doc, "metadata": meta, "score": hybrid_score})

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_n]


# --- Stage 1: Dicta query understanding -------------------------------------

_UNDERSTAND_SYSTEM_PROMPT = (
    "You turn an Israeli legal question (in Hebrew or English) into a "
    "structured search request. Respond with ONLY valid JSON, no other "
    "text, in exactly this shape:\n"
    '{"legal_topic": string, "keywords": [string, ...], "queries": [string, ...]}\n\n'
    "\"legal_topic\" is a short English label for the area of law (e.g. "
    "\"employment termination\"). \"keywords\" are 3-6 precise Hebrew legal "
    "terms someone would expect to find verbatim in the relevant statute "
    "text. \"queries\" are 2-4 different Hebrew phrasings of the underlying "
    "legal question, varied enough to catch a relevant provision that "
    "doesn't share the user's exact wording."
)


def understand_query(user_text: str) -> dict:
    messages = [
        {"role": "system", "content": _UNDERSTAND_SYSTEM_PROMPT},
        {"role": "user", "content": user_text},
    ]
    try:
        raw = _ollama_chat(DICTA_MODEL, messages, format_json=True, num_predict=_UNDERSTAND_NUM_PREDICT)
        parsed = _parse_json_response(raw)
    except RuntimeError:
        parsed = None
    if not parsed or not parsed.get("queries"):
        # Fall back to the raw question as the only query -- retrieval still
        # runs, just without the benefit of query expansion.
        return {"legal_topic": "", "keywords": [], "queries": [user_text]}
    queries = [q for q in parsed.get("queries", []) if isinstance(q, str) and q.strip()]
    if user_text not in queries:
        queries.append(user_text)  # always search the user's own wording too
    return {
        "legal_topic": parsed.get("legal_topic", ""),
        "keywords": parsed.get("keywords", []),
        "queries": queries,
    }


# --- Stage 2: DeepSeek reasoning --------------------------------------------

_ANALYZE_SYSTEM_PROMPT = (
    "You are a legal reasoning engine for Israeli law. You will be given a "
    "user's question and a set of retrieved Israeli legal sources (statute "
    "excerpts, each labeled with its law name, section, and in-force "
    "status). Analyze the question using ONLY these sources.\n\n"
    "Do not invent laws, sections, cases, or citations that are not in the "
    "sources given to you.\n\n"
    "Structure your analysis explicitly:\n"
    "- Applicable law (which sources actually govern this question)\n"
    "- Exceptions (any carve-outs or conditions visible in the sources)\n"
    "- Precedent (only if a source itself references case law)\n"
    "- Facts (what the user's question establishes)\n"
    "- Assumptions (anything you must assume because the question is silent)\n"
    "- Conclusion\n\n"
    "If the sources are insufficient to answer confidently, say so plainly "
    "instead of filling the gap from general knowledge."
)


def _format_sources_for_prompt(sources: list[dict]) -> str:
    parts = []
    for s in sources:
        meta = s["metadata"]
        label = f"[{meta.get('law_name', '')}"
        if meta.get("section"):
            label += f", סעיף {meta['section']}"
        if meta.get("subsection"):
            label += f"({meta['subsection']})"
        label += "]"
        status = "בתוקף" if meta.get("is_current", True) else "לא בתוקף / הוחלף"
        eff = meta.get("effective_from") or "לא ידוע"
        label += f" (סטטוס: {status}; תחילה: {eff})"
        parts.append(f"{label}\n{s['document']}")
    return "\n\n---\n\n".join(parts)


def deepseek_analyze(user_text: str, sources: list[dict], model: str = DEEPSEEK_MODEL_DEFAULT) -> str:
    if not sources:
        return "לא נמצאו מקורות רלוונטיים במאגר -- אין בסיס לניתוח משפטי."
    messages = [
        {"role": "system", "content": _ANALYZE_SYSTEM_PROMPT},
        {"role": "user", "content": f"Question: {user_text}\n\nRetrieved sources:\n\n"
                                     f"{_format_sources_for_prompt(sources)}"},
    ]
    return _ollama_chat(model, messages, num_predict=_ANALYZE_NUM_PREDICT, num_ctx=_deepseek_num_ctx(model))


# --- Stage 3: Dicta draft ----------------------------------------------------

_DRAFT_SYSTEM_PROMPT = (
    "אתה עורך דין ישראלי הכותב תשובה בעברית ברורה למשתמש, על בסיס ניתוח "
    "משפטי שכבר בוצע. שמור במדויק על המשמעות המשפטית מהניתוח -- אל תוסיף "
    "מקור משפטי חדש, אל תמציא עובדה משפטית. הפוך את הניתוח לתשובה קריאה "
    "וברורה, וציין בבירור כל אי-ודאות שהניתוח הצביע עליה.\n\n"
    "חשוב במיוחד: מותר לך לצטט (שם חוק + מספר סעיף) אך ורק מתוך 'רשימת "
    "המקורות המאומתים' שתסופק לך למטה -- זו הרשימה היחידה שאומתה מול "
    "המאגר בפועל. אם הניתוח מזכיר חוק או סעיף שאינו ברשימה הזו, אל תעתיק "
    "את הציטוט הזה כפי שהוא -- נסח את הטענה בזהירות כידע כללי לא-מאומת "
    "בלבד, ואל תציין מספר סעיף שלא ברשימה. סיים תמיד בהערה שזו אינה "
    "תחליף לייעוץ מעורך דין מוסמך."
)


def _citation_whitelist(sources: list[dict]) -> str:
    """A short, cheap (law_name, section) list -- NOT the full source text
    -- of what was genuinely retrieved this cycle, handed to dicta_draft
    alongside the analysis.

    This exists because dicta_draft previously had ZERO ground truth of
    its own: it only ever saw DeepSeek's prose analysis and was told
    (purely by instruction) not to invent a citation, with nothing to
    actually check that against. If the analysis itself contained a
    fabricated law/section -- a real, observed failure mode for a
    distilled reasoning model -- Dicta had no way to notice and would
    faithfully launder it into a confident-sounding Hebrew answer. Handing
    over the full source TEXT a second time (on top of what
    deepseek_analyze already consumed it for) would be needlessly
    expensive against dicta_draft's own num_ctx budget; a short list of
    just the (law, section) labels is nearly free and gives Dicta a
    concrete, checkable reference instead of only a system-prompt
    instruction to police itself."""
    if not sources:
        return "(לא אותרו מקורות מאומתים -- אין רשימה לצטט ממנה)"
    labels = []
    for s in sources:
        meta = s["metadata"]
        label = meta.get("law_name", "")
        if meta.get("section"):
            label += f", סעיף {meta['section']}"
        if label and label not in labels:
            labels.append(label)
    return "\n".join(f"- {l}" for l in labels)


def dicta_draft(user_text: str, analysis: str, sources: list[dict],
                 history: list[dict] | None = None) -> str:
    messages = [{"role": "system", "content": _DRAFT_SYSTEM_PROMPT}]
    messages += (history or [])
    messages.append({
        "role": "user",
        "content": (
            f"שאלת המשתמש: {user_text}\n\n"
            f"הניתוח המשפטי שבוצע:\n\n{analysis}\n\n"
            f"רשימת המקורות המאומתים (ניתן לצטט אך ורק מתוכה):\n\n"
            f"{_citation_whitelist(sources)}"
        ),
    })
    return _ollama_chat(DICTA_MODEL, messages, num_predict=_DRAFT_NUM_PREDICT, num_ctx=_DICTA_NUM_CTX)


# --- Stage 4: DeepSeek verification -----------------------------------------

_VERIFY_SYSTEM_PROMPT = (
    "You are a strict legal QA reviewer. You will be given a user's "
    "question, the legal sources that were retrieved, the reasoning "
    "analysis, and a drafted answer. Check:\n"
    "1. Are the legal claims actually supported by the sources given?\n"
    "2. Are the citations (law name, section number) accurate to those sources?\n"
    "3. Does the cited section actually say what the answer claims it says?\n"
    "4. Were any important exceptions visible in the sources missed?\n"
    "5. Do any sources contradict each other, and if so was that handled?\n"
    "6. Is any cited source marked as not currently in force (is_current=false) "
    "and being relied on anyway without flagging that?\n"
    "7. Did the draft introduce any citation not present in the sources?\n"
    "8. Would additional retrieval likely find something important that's missing?\n\n"
    "Respond with ONLY valid JSON, no other text, in exactly this shape:\n"
    '{"status": "PASS" | "FAIL", "issues": [string, ...], "needs_more_retrieval": true | false}'
)


_SECTION_MENTION_RE = re.compile(r"סעיף\s+(\d+[א-ת]?)")


def _ground_draft_citations(draft: str, sources: list[dict]) -> list[str]:
    """
    Deterministic (non-LLM) hallucination guard, run IN ADDITION TO the LLM
    verifier above, not instead of it.

    deepseek_verify asks a model to judge whether a draft (written by a
    DIFFERENT model, or sometimes the same model family) is grounded in the
    sources -- but a 14B/32B model is not a reliable judge of a citation
    it would itself have plausibly fabricated; it's exactly the kind of
    small, plausible-looking error a local model tends to rubber-stamp
    rather than catch. Cross-checking the specific סעיף numbers the draft
    cites against the section numbers that were ACTUALLY retrieved this
    cycle (ground truth -- it's literally what was fed to the model) is
    deterministic and can't be talked out of flagging a real mismatch.

    This only catches one specific, checkable failure mode -- a cited
    section number that was never retrieved at all -- not every possible
    hallucination (e.g. a real section cited but mischaracterized). It's a
    hard floor underneath the LLM verifier's softer semantic judgment, not
    a replacement for it.
    """
    retrieved_sections = {
        str(s["metadata"].get("section", "")).strip()
        for s in sources
        if s.get("metadata", {}).get("section")
    }
    cited_sections = {m.group(1) for m in _SECTION_MENTION_RE.finditer(draft)}
    ungrounded = sorted(cited_sections - retrieved_sections)
    if not ungrounded:
        return []
    retrieved_str = ", ".join(sorted(retrieved_sections)) or "(none)"
    return [
        f"Draft cites סעיף {sec}, but no retrieved source has that section "
        f"number this cycle (retrieved: {retrieved_str}) -- this looks like "
        f"a fabricated citation, not one grounded in the retrieved sources."
        for sec in ungrounded
    ]


def deepseek_verify(user_text: str, sources: list[dict], analysis: str, draft: str,
                     model: str = DEEPSEEK_MODEL_DEFAULT) -> dict:
    messages = [
        {"role": "system", "content": _VERIFY_SYSTEM_PROMPT},
        {"role": "user", "content": (
            f"Question: {user_text}\n\n"
            f"Retrieved sources:\n\n{_format_sources_for_prompt(sources)}\n\n"
            f"Analysis:\n\n{analysis}\n\n"
            f"Draft answer:\n\n{draft}"
        )},
    ]
    try:
        raw = _ollama_chat(model, messages, format_json=True,
                            num_predict=_VERIFY_NUM_PREDICT, num_ctx=_deepseek_num_ctx(model))
        parsed = _parse_json_response(raw)
    except RuntimeError as e:
        # CHANGED: this used to degrade to PASS-with-a-note on the theory
        # that a verification step which couldn't run shouldn't block the
        # answer. That's backwards for a pipeline whose whole purpose is
        # catching hallucination -- "verification didn't run" is NOT
        # evidence the draft is safe, it's an absence of evidence either
        # way, and mislabeling that as PASS is exactly what let an
        # ungrounded answer through looking falsely verified. The answer
        # still isn't blocked (answer_legal_question returns the last
        # draft either way once retries are exhausted) -- only the label
        # shown to the user changes, from a false "verified" to an honest
        # "couldn't verify."
        result = {"status": "FAIL", "issues": [f"(verification step failed to run: {e})"],
                   "needs_more_retrieval": True}
    else:
        if not parsed or parsed.get("status") not in ("PASS", "FAIL"):
            result = {"status": "FAIL",
                       "issues": ["(verifier returned an unparseable response -- treating as unverified, not PASS)"],
                       "needs_more_retrieval": True}
        else:
            result = {
                "status": parsed["status"],
                "issues": parsed.get("issues", []) or [],
                "needs_more_retrieval": bool(parsed.get("needs_more_retrieval", False)),
            }

    # Deterministic grounding check runs regardless of what the LLM
    # verifier concluded -- a hard floor under its soft judgment. Any
    # citation this flags forces FAIL even if the LLM verifier said PASS.
    grounding_issues = _ground_draft_citations(draft, sources)
    if grounding_issues:
        result["status"] = "FAIL"
        result["issues"] = grounding_issues + result["issues"]
        result["needs_more_retrieval"] = True

    return result


def _followup_queries(user_text: str, issues: list[str]) -> list[str]:
    """Generates additional search queries targeting the verifier's specific
    complaints, for a correction cycle -- reuses understand_query's model
    call but feeds it the verifier's issues as extra context so retry #2
    isn't just re-running the exact same searches that already missed."""
    issues_text = "; ".join(issues) if issues else "the retrieval may be incomplete"
    messages = [
        {"role": "system", "content": _UNDERSTAND_SYSTEM_PROMPT},
        {"role": "user", "content": (
            f"{user_text}\n\n(A legal reviewer flagged this problem with the first attempt: "
            f"{issues_text}. Generate queries that specifically target what's missing.)"
        )},
    ]
    try:
        raw = _ollama_chat(DICTA_MODEL, messages, format_json=True, num_predict=_UNDERSTAND_NUM_PREDICT)
        parsed = _parse_json_response(raw)
    except RuntimeError:
        parsed = None
    if not parsed or not parsed.get("queries"):
        return [user_text]
    return [q for q in parsed["queries"] if isinstance(q, str) and q.strip()] or [user_text]


# --- Orchestration -----------------------------------------------------------

def _sources_panel_data(sources: list[dict]) -> list[dict]:
    out = []
    for s in sources:
        meta = s["metadata"]
        out.append({
            "law_name": meta.get("law_name", ""),
            "section": meta.get("section", ""),
            "subsection": meta.get("subsection", ""),
            "is_current": meta.get("is_current", True),
            "effective_from": meta.get("effective_from", ""),
            "source_url": meta.get("source_url", ""),
        })
    return out


def answer_legal_question(user_text: str, history: list[dict] | None = None,
                           file_context: str = "", on_progress=None,
                           deepseek_model: str = DEEPSEEK_MODEL_DEFAULT) -> dict:
    """
    Runs the full pipeline for one question and returns:
        {"answer": str, "sources": list[dict], "cycles_used": int,
         "verification": {"status": ..., "issues": [...]}}

    on_progress, if given, is called with a short status string before each
    stage -- app/ui.py's ChatInterface generator wraps these into the
    yielded "thinking..." messages the other tabs already show.

    deepseek_model selects which DeepSeek size runs the reasoning +
    verification stages (DEEPSEEK_MODEL_DEFAULT / DEEPSEEK_MODEL_FAST, or
    any other Ollama tag) -- this is how app/ui.py's two Attorney 24B tabs
    (one per DeepSeek size) share this single pipeline implementation
    instead of each needing their own copy. Query understanding and
    drafting always run on DICTA_MODEL regardless of this choice -- only
    the reasoning/verification stages change.

    Raises RuntimeError (with a message safe to show the user) if a hard
    dependency is missing or unreachable -- see this module's docstring.
    """
    if not PIPELINE_AVAILABLE:
        raise RuntimeError(
            "The legal pipeline's dependencies aren't installed. Run: "
            "pip install chromadb rank_bm25"
        )

    if not _has_meaningful_content(user_text) and not file_context:
        # A greeting/chit-chat message with nothing to search on -- return
        # immediately rather than running query understanding, retrieval,
        # DeepSeek analysis, a Dicta draft, DeepSeek verification, and up to
        # MAX_VERIFICATION_CYCLES retries against a question that was never
        # going to retrieve anything relevant in the first place. Skipped
        # entirely (not just short-circuited after one cheap check) if a
        # file is attached -- an attached document can supply real content
        # even when the typed message alone is just "hi, can you look at this?"
        return {
            "answer": _GREETING_REPLY,
            "sources": [],
            "cycles_used": 0,
            "verification": {"status": "PASS", "issues": [], "needs_more_retrieval": False},
        }

    def progress(msg):
        if on_progress:
            on_progress(msg)

    combined_question = user_text
    if file_context:
        combined_question = f"{user_text}\n\n(User also attached this document:\n{file_context})"

    progress("🧭 מבין את השאלה ובונה שאילתות חיפוש…")
    understood = understand_query(combined_question)
    queries = understood["queries"]

    sources: list[dict] = []
    analysis = ""
    draft = ""
    verification = {"status": "FAIL", "issues": [], "needs_more_retrieval": True}

    for cycle in range(1, MAX_VERIFICATION_CYCLES + 1):
        progress(f"📚 מחפש מקורות משפטיים (סבב {cycle}/{MAX_VERIFICATION_CYCLES})…")
        sources = _trim_sources_to_budget(
            hybrid_retrieve(queries),
            max_weight=_sources_max_weight_for(_deepseek_num_ctx(deepseek_model)),
        )

        progress("🧠 DeepSeek מנתח את המקורות שאותרו…")
        analysis = deepseek_analyze(combined_question, sources, model=deepseek_model)

        progress("✍️ Dicta מנסח תשובה בעברית…")
        draft = dicta_draft(combined_question, analysis, sources, history)

        progress("🔎 DeepSeek בודק את התשובה מול המקורות…")
        verification = deepseek_verify(combined_question, sources, analysis, draft, model=deepseek_model)

        if verification["status"] == "PASS":
            break
        if cycle == MAX_VERIFICATION_CYCLES:
            break  # out of retries -- return the last draft with a visible warning
        queries = _followup_queries(combined_question, verification["issues"])

    final_answer = draft
    if verification["status"] == "FAIL":
        issues_md = "\n".join(f"- {i}" for i in verification["issues"]) or "- (no specific issue text returned)"
        final_answer += (
            "\n\n---\n🚨 *A verification pass over this answer did not fully pass, even after "
            f"retrying retrieval. Outstanding concerns:*\n{issues_md}\n\n"
            "*Treat this answer as unverified and check it against the actual legislation.*"
        )
    else:
        final_answer += (
            "\n\n---\n⚖️ *This answer was checked against its retrieved sources before being "
            "shown to you. It is not legal advice and is not a substitute for a licensed attorney.*"
        )

    return {
        "answer": final_answer,
        "sources": _sources_panel_data(sources),
        "cycles_used": cycle,
        "verification": verification,
    }