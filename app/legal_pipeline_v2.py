"""
app/legal_pipeline_v2.py

DictaLM-only pipeline grounding the Attorney tab. DeepSeek has been removed
entirely -- every reasoning/drafting/verification step below runs on
DICTA_MODEL. Flow:

    Dicta (query understanding)
      -> BGE-M3 (embed)      -\
      -> BM25 (keyword)       |-> Reciprocal Rank Fusion -> hybrid rerank
      -> ChromaDB (vector)   -/
      -> [source resolver: if a specific law is named, fetch it whole
          instead of competing for top-K slots against the rest of the
          collection]
      -> Dicta (answers directly from the retrieved sources, or says
          plainly that the sources aren't relevant/sufficient)
      -> Dicta (independent pass: checks the answer's claims/citations
          against the sources)
         + deterministic (non-LLM) citation-existence check -- the actual
           hard guarantee, since a model checking its own answer is not a
           fully independent check (see dicta_verify's docstring)
      -> retry retrieval+answering up to MAX_VERIFICATION_CYCLES on FAIL

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
  - The existing DictaLM pull already used by app/ui.py's LEGAL_MODEL
  - pip install chromadb rank_bm25

--- Why DictaLM does its own verification, and what that does and doesn't buy you ---

Asking DictaLM to check a DictaLM-written answer is NOT a fully independent
check -- a model verifying its own output tends toward self-consistency,
not correctness, especially on an error it would itself have plausibly
made. dicta_verify() below still runs it (it does catch real issues, and
the retry loop still uses it to decide whether to search again), but the
one check in this pipeline that's actually immune to that blind spot is
_ground_draft_citations(): a deterministic, non-LLM diff between the
section numbers the answer cites and the section numbers that were
literally fed to the model. That's what should be trusted as the hard
guarantee that citations exist; the LLM verify pass is a softer,
best-effort semantic check on top of it, not a replacement for it.

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
# This is now the ONLY chat model this module ever calls.
DICTA_MODEL = "hf.co/dicta-il/DictaLM-3.0-24B-Thinking-GGUF:Q4_K_M"

# No timeout on DictaLM calls at all (was 1800s/30min) -- this module talks
# to Ollama directly (see OLLAMA_CHAT_URL), so unlike app/ui.py's Attorney
# 1.7B (Fast) tab (which is routed through main.py's backend and therefore
# still has that backend's own _OLLAMA_REQUEST_TIMEOUT_SECONDS as a
# secondary cap), removing this is a genuinely complete, uncapped wait for
# every DictaLM call this module makes: understand_query, dicta_answer,
# dicta_verify, and _followup_queries all default to this. requests.post
# treats timeout=None as "block forever" -- that's the correct way to
# remove a timeout, not just picking a bigger number.
_CHAT_TIMEOUT_SECONDS = None
_UNDERSTAND_NUM_PREDICT = 512
# Was two separate 2048-token stages (deepseek_analyze + dicta_draft)
# merged into one pass -- given a bit more headroom than either alone had,
# since a single pass now has to both reason over the sources AND produce
# the final Hebrew prose in one shot.
_ANSWER_NUM_PREDICT = 3072
_VERIFY_NUM_PREDICT = 1024

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

# --- Context-window budgeting ------------------------------------------------
#
# Only ONE model/context window is in play now (DICTA_MODEL), so this is
# simpler than the old split DeepSeek/Dicta budgeting: one num_ctx, one
# source-weight budget, applied once right after retrieval and reused for
# both dicta_answer and dicta_verify.
#
# 11264 matches app/ui.py's _LEGAL_NUM_CTX -- already confirmed safe on this
# project's target 32GB-RAM/no-GPU hardware for DictaLM-3.0-24B-Thinking's
# KV cache. Don't raise this without re-checking `ollama ps`/the server log
# the way that constant's own comment describes -- this machine sits close
# to its RAM ceiling with this model's KV cache already.
_DICTA_NUM_CTX = 11264

# Budget for the formatted RETRIEVED-SOURCES text specifically, in weighted
# units (see _estimate_token_weight below). Sized against dicta_verify --
# the tighter of the two consumers, since its prompt carries the sources
# AND the answer at once:
#   _DICTA_NUM_CTX (11264)
#   - _VERIFY_NUM_PREDICT (1024, verify's own JSON output reserve)
#   - _ANSWER_NUM_PREDICT (3072, the answer's real token cap -- already an
#     exact token count, not a weight estimate, since num_predict directly
#     bounds how many tokens that stage could have generated)
#   - ~800 headroom for the system prompt + question wrapper text
#   = ~6368, *0.9 for a safety cushion against the weight estimate being
#   imperfect. dicta_answer's own budget is more generous (doesn't need to
#   reserve room for verify's output), so this same trim safely covers both.
_SOURCES_CONTEXT_MAX_WEIGHT = max(0.0, (_DICTA_NUM_CTX - _VERIFY_NUM_PREDICT - _ANSWER_NUM_PREDICT - 800) * 0.9)


def _estimate_token_weight(text: str) -> float:
    """Rough, script-aware proxy for prompt-token cost -- see app/ui.py's
    function of the same name for the full reasoning (Hebrew routinely
    needs more tokens per character than Latin script for a general-
    purpose BPE tokenizer, so Hebrew characters are weighted ~2x more
    heavily). Intentionally pessimistic: better to trim a source that
    would actually have fit than to let one through that overflows the
    context window."""
    hebrew_chars = sum(1 for ch in text if 0x0590 <= ord(ch) <= 0x05FF)
    other_chars = len(text) - hebrew_chars
    return hebrew_chars * 2.0 + other_chars * 1.0


def _trim_sources_to_budget(sources: list[dict], max_weight: float = _SOURCES_CONTEXT_MAX_WEIGHT) -> list[dict]:
    """
    Drops the lowest-ranked sources (from the end -- hybrid_retrieve's
    rerank, or the whole-law fetch's reading order, already sorts `sources`
    with the most important first) until the formatted source text fits
    under max_weight weighted units.

    Drops whole sources rather than truncating one mid-text, so what's left
    is always a complete, citable excerpt -- same "never truncate
    mid-section" principle app/ui.py's _fetch_whole_law already applies.
    Always keeps at least the single best source, even if it alone exceeds
    max_weight -- better to attempt the call with the best available
    excerpt than to silently answer with zero grounding.

    Applied ONCE, right after retrieval (see answer_legal_question), so
    every downstream consumer (the answer, the verification pass, and the
    sources panel shown to the user) works from the exact same set.
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
                  timeout: int | None = _CHAT_TIMEOUT_SECONDS) -> str:
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
        # context-size overflow.
        try:
            body = resp.json()
        except Exception:
            body = {}
        err_obj = body.get("error") if isinstance(body, dict) else None
        if isinstance(err_obj, dict):
            message = err_obj.get("message", "")
            err_type = err_obj.get("type", "")
        else:
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


# DictaLM-3.0-Thinking emits <think>...</think> chain-of-thought before the
# real answer -- same issue app/main.py's _strip_thinking handles for the
# backend's /chat endpoint. Duplicated here (not imported -- main.py pulls
# in FastAPI/ollama-python machinery this module doesn't need) since this
# module talks to Ollama's raw HTTP API directly, not through the FastAPI
# backend.
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
    the FULL pipeline -- query understanding, hybrid retrieval, a Dicta
    answer, Dicta verification, and up to MAX_VERIFICATION_CYCLES retries
    once nothing relevant is ever found -- which is both a multi-minute
    round trip and a confusing answer for what should just be a normal
    greeting response. Checked in answer_legal_question() before any model
    call is made at all.
    """
    if _GREETING_ONLY_RE.match(query.strip()):
        return False
    return bool(_tokenize_he(query))


_GREETING_REPLY = (
    "שלום! אני עוזר משפטי לחוק הישראלי, מבוסס על אחזור מקורות מהמאגר "
    "המשפטי (BGE-M3 + BM25) וניתוח/אימות באמצעות Dicta. "
    "אשמח לעזור -- מה השאלה המשפטית שלך?\n\n"
    "*(Hi! I'm an Israeli-law assistant grounded in retrieved statute "
    "text, answered and independently checked by Dicta. Ask me a legal "
    "question and I'll search the database for you.)*"
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

def _embeddings_to_list(value):
    """Chroma's get()/query() results return `embeddings` as a numpy array
    in this project's installed chromadb version (confirmed directly:
    type(embeds) is numpy.ndarray), unlike documents/metadatas/distances,
    which come back as plain Python lists/dicts. A bare truthiness check
    crashes on a multi-element numpy array with "The truth value of an
    array... is ambiguous" instead of behaving like an ordinary Python
    falsy/truthy check.

    Recurses so a nested numpy array (dtype=object rows, each itself a
    separate array) is fully converted to plain Python floats, not just
    the outer layer. Same defensive pattern already established in
    app/ui.py's function of the same name for the identical chromadb
    behavior -- duplicated here rather than imported, since these are two
    independent modules (ui.py pulls in gradio; this module has no reason
    to depend on it)."""
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, list):
        return [_embeddings_to_list(v) for v in value]
    return value


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


# --- Source resolver: bypass top-K competition when a law is named ---------
#
# hybrid_retrieve (below) always competes a law's sections against every
# OTHER law in the collection for _RERANK_TOP_N slots -- fine for an
# open-ended question, but actively wrong when the user names one specific
# law: a relevant section can lose its slot to something semantically
# louder from an unrelated law, and even when sections DO win, they come
# back in relevance order, not reading order, which can omit a definition
# or exception a later section depends on. app/ui.py's older Knesset
# pipeline already solved exactly this (_detect_named_law +
# _fetch_whole_law_sections) -- this ports the same approach to this
# module's BGE-M3/israeli_legal_db schema, returning sources in the same
# {"id","document","metadata","score"} shape hybrid_retrieve produces, so
# nothing downstream (trim, format, grounding check) needs to change.

# How much of a law's title's meaningful words must appear in the question
# before it's treated as confidently named rather than just topically
# related. Deliberately high, same reasoning as app/ui.py's
# _KNESSET_WHOLE_LAW_TOKEN_OVERLAP_THRESHOLD: a false positive here means
# potentially dumping an entire UNRELATED law into the model's context,
# which is worse than just falling through to ordinary hybrid search.
_NAMED_LAW_OVERLAP_THRESHOLD = 0.7
_NAMED_LAW_MIN_TITLE_TOKENS = 3

_law_titles_cache = {"count": None, "map": {}}


def _get_law_titles(collection) -> dict[str, str]:
    """Returns {law_id: law_name} for every law currently embedded. Cached,
    invalidated when collection.count() changes -- same pattern as
    _get_bm25_index above and app/ui.py's _get_knesset_law_titles."""
    global _law_titles_cache
    try:
        current_count = collection.count()
    except Exception:
        current_count = None
    if current_count is not None and _law_titles_cache["count"] == current_count:
        return _law_titles_cache["map"]
    try:
        rows = collection.get(include=["metadatas"])
    except Exception as e:
        print(f"[legal_pipeline_v2] couldn't refresh law-title map: {e}")
        return _law_titles_cache["map"]  # stale is better than nothing
    law_map: dict[str, str] = {}
    for meta in rows.get("metadatas") or []:
        law_id = meta.get("law_id")
        law_name = meta.get("law_name")
        if law_id and law_name:
            law_map[law_id] = law_name
    _law_titles_cache = {"count": current_count, "map": law_map}
    return law_map


def _detect_named_law(query: str, law_map: dict[str, str]) -> tuple[str, str, float] | None:
    """Returns (law_id, law_name, overlap_score) for the best-matching law
    if the question clearly names it, else None. See
    _NAMED_LAW_OVERLAP_THRESHOLD's comment above for why this stays
    conservative on purpose."""
    query_tokens = set(_tokenize_he(query))
    if not query_tokens:
        return None
    best = None
    for law_id, law_name in law_map.items():
        title_tokens = set(_tokenize_he(law_name))
        if len(title_tokens) < _NAMED_LAW_MIN_TITLE_TOKENS:
            continue  # too generic/short a title to match confidently at all
        overlap = len(title_tokens & query_tokens) / len(title_tokens)
        if overlap < _NAMED_LAW_OVERLAP_THRESHOLD:
            continue
        if best is None or overlap > best[2]:
            best = (law_id, law_name, overlap)
    return best


def _section_sort_key(meta: dict):
    """Sorts a law's chunks into reading order: preamble/no-section-number
    chunks first, then numeric section order (not string order, which
    would put '10' before '2'). A trailing Hebrew letter on a section
    number (e.g. '12א', an inserted amendment section) sorts right after
    its base number. Same logic as app/ui.py's function of the same name,
    adapted to this schema's 'section' key (vs. that one's
    'section_number')."""
    section = str(meta.get("section") or "").strip()
    if not section:
        return (0, 0, "")
    m = re.match(r"(\d+)([א-ת]?)", section)
    if not m:
        return (1, 0, section)
    return (1, int(m.group(1)), m.group(2))


def _fetch_whole_law_sources(collection, law_id: str) -> list[dict]:
    """Fetches EVERY chunk for one law_id directly via a metadata filter --
    bypassing top-K competition against the rest of the collection
    entirely -- sorted into reading order. Returns the same
    {"id","document","metadata","score"} shape hybrid_retrieve produces
    (score is a constant sentinel here, not a real similarity -- reading
    order matters more than relevance order for a whole-law fetch, and
    nothing downstream actually re-sorts by score). _trim_sources_to_budget
    (already in the pipeline) is what keeps a genuinely long law from
    overflowing context -- this function itself does no length limiting,
    same "never truncate mid-section, drop whole sections instead"
    principle already established there."""
    try:
        result = collection.get(where={"law_id": law_id}, include=["documents", "metadatas"])
    except Exception as e:
        print(f"[legal_pipeline_v2] whole-law fetch failed for law_id={law_id}: {e}")
        return []
    docs = result.get("documents") or []
    metas = result.get("metadatas") or []
    ids = result.get("ids") or []
    triples = sorted(zip(ids, docs, metas), key=lambda t: _section_sort_key(t[2]))
    return [{"id": i, "document": d, "metadata": m, "score": 1.0} for i, d, m in triples]


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

    # Batch-fetch embeddings for the WHOLE candidate pool in one call,
    # instead of one collection.get() per candidate.
    id_to_vector: dict[str, list[float]] = {}
    if fused_ids:
        try:
            got = collection.get(ids=fused_ids, include=["embeddings"])
            fetched_ids = got.get("ids") or []
            fetched_vectors = _embeddings_to_list(got.get("embeddings")) or []
            id_to_vector = dict(zip(fetched_ids, fetched_vectors))
        except Exception as e:
            print(f"[legal_pipeline_v2] batched embedding fetch for rerank failed: {e} "
                  "-- falling back to lexical-only scoring for this rerank pass.")

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
        doc_vector = id_to_vector.get(doc_id)
        best_vec_sim = 0.0
        if doc_vector:
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


def _citation_whitelist(sources: list[dict]) -> str:
    """A short, cheap (law_name, section) list -- alongside, not instead of,
    the full source text -- of what was genuinely retrieved this cycle.
    Gives Dicta a concrete, checkable reference on top of the raw text, and
    is nearly free against the context budget."""
    if not sources:
        return "(לא אותרו מקורות -- אין רשימה לצטט ממנה)"
    labels = []
    for s in sources:
        meta = s["metadata"]
        label = meta.get("law_name", "")
        if meta.get("section"):
            label += f", סעיף {meta['section']}"
        if label and label not in labels:
            labels.append(label)
    return "\n".join(f"- {l}" for l in labels)


# --- Stage 2: Dicta answers directly from the retrieved sources ------------

_ANSWER_SYSTEM_PROMPT = (
    "אתה עורך דין ישראלי. תפקידך לענות על שאלה משפטית בעברית, בהתבסס אך "
    "ורק על טקסט המקורות המשפטיים שאוחזרו מהמאגר ותסופק לך למטה -- אלה "
    "המקור הסמכותי היחיד שלך.\n\n"
    "לפני שאתה עונה לגופו של עניין, בדוק תחילה האם המקורות שסופקו אכן "
    "רלוונטיים לשאלה. אם המקורות אינם רלוונטיים, חלקיים מדי, או אינם "
    "מכסים את השאלה שנשאלה -- אמור זאת במפורש ובבירור, ואל תנסה להשלים "
    "את החסר מהידע הכללי שלך. זו תשובה לגיטימית: 'המקורות שאוחזרו אינם "
    "עונים על שאלה זו' עדיפה על תשובה מומצאת.\n\n"
    "אם המקורות כן רלוונטיים: אל תוסיף מקור משפטי חדש, אל תמציא עובדה "
    "משפטית, ואל תצטט (שם חוק + מספר סעיף) שאינו מופיע במפורש בטקסט "
    "המקורות. בנה את תשובתך כך: החוק הרלוונטי (אילו מהמקורות אכן חלים "
    "על השאלה), חריגים או תנאים הנראים במקורות, ואז מסקנה ברורה. ציין "
    "בבירור כל אי-ודאות. סיים תמיד בהערה שזו אינה תחליף לייעוץ מעורך "
    "דין מוסמך."
)


def dicta_answer(user_text: str, sources: list[dict], history: list[dict] | None = None) -> str:
    """Single-pass reasoning + Hebrew drafting, directly from the retrieved
    source text -- replaces the old deepseek_analyze -> dicta_draft
    two-model handoff. Returns a fixed Hebrew "nothing relevant" message
    without calling the model at all if sources is empty (cheap short-
    circuit -- same as the old deepseek_analyze's empty-sources check).
    The subtler case -- sources WERE retrieved but aren't actually
    relevant/sufficient -- is handled inside the model call itself via
    _ANSWER_SYSTEM_PROMPT's explicit relevance check, since only the model
    (seeing the actual question next to the actual text) can judge that;
    the retrieval floor upstream only guarantees "least-bad candidate that
    cleared a similarity threshold," not "actually answers this question."
    """
    if not sources:
        return "לא נמצאו מקורות רלוונטיים במאגר לשאלה זו -- אין בסיס לתשובה מבוססת."

    messages = [{"role": "system", "content": _ANSWER_SYSTEM_PROMPT}]
    messages += (history or [])
    messages.append({
        "role": "user",
        "content": (
            f"שאלת המשתמש: {user_text}\n\n"
            f"טקסט המקורות המשפטיים שאוחזרו:\n\n{_format_sources_for_prompt(sources)}\n\n"
            f"רשימת (חוק, סעיף) שמותר לצטט ממנה:\n\n{_citation_whitelist(sources)}"
        ),
    })
    return _ollama_chat(DICTA_MODEL, messages, num_predict=_ANSWER_NUM_PREDICT, num_ctx=_DICTA_NUM_CTX)


# --- Stage 3: Dicta verification (independent pass) + deterministic check ---

_VERIFY_SYSTEM_PROMPT = (
    "You are a strict legal QA reviewer checking an answer written by "
    "another legal assistant. You will be given a user's question, the "
    "legal sources that were retrieved, and the drafted answer. This is "
    "an INDEPENDENT check -- be skeptical, actively look for problems "
    "rather than confirming the answer looks reasonable. Check:\n"
    "1. Does every citation (law name, section number) in the answer "
    "literally appear among the retrieved sources? This is the single "
    "most important check -- flag ANY citation you cannot find verbatim "
    "in the sources given below.\n"
    "2. Are the legal claims actually supported by the sources given, not "
    "just plausible-sounding?\n"
    "3. Does the cited section actually say what the answer claims it says?\n"
    "4. Were any important exceptions visible in the sources missed?\n"
    "5. Do any sources contradict each other, and if so was that handled?\n"
    "6. Is any cited source marked as not currently in force (is_current=false) "
    "and being relied on anyway without flagging that?\n"
    "7. If the sources were actually insufficient/irrelevant, did the "
    "answer correctly say so instead of answering anyway?\n\n"
    "Respond with ONLY valid JSON, no other text, in exactly this shape:\n"
    '{"status": "PASS" | "FAIL", "issues": [string, ...], "needs_more_retrieval": true | false}'
)


_SECTION_MENTION_RE = re.compile(r"סעיף\s+(\d+[א-ת]?)")


def _ground_draft_citations(answer: str, sources: list[dict]) -> list[str]:
    """
    Deterministic (non-LLM) citation-existence check, run IN ADDITION TO
    the LLM verify pass above, not instead of it -- and this is the check
    that should actually be trusted as "citations exist," since
    dicta_verify asking DictaLM to judge a DictaLM-written answer is not a
    fully independent check (see this module's docstring). Cross-checking
    the specific סעיף numbers the answer cites against the section numbers
    that were ACTUALLY retrieved this cycle is ground truth -- it's
    literally what was fed to the model -- and deterministic: it can't be
    talked out of flagging a real mismatch the way an LLM verifier can.

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
    cited_sections = {m.group(1) for m in _SECTION_MENTION_RE.finditer(answer)}
    ungrounded = sorted(cited_sections - retrieved_sections)
    if not ungrounded:
        return []
    retrieved_str = ", ".join(sorted(retrieved_sections)) or "(none)"
    return [
        f"Answer cites סעיף {sec}, but no retrieved source has that section "
        f"number this cycle (retrieved: {retrieved_str}) -- this looks like "
        f"a fabricated citation, not one grounded in the retrieved sources."
        for sec in ungrounded
    ]


def dicta_verify(user_text: str, sources: list[dict], answer: str) -> dict:
    """Independent(-ish) DictaLM pass checking the answer against the
    sources, PLUS the deterministic grounding check above -- see this
    module's docstring for why the deterministic check, not this LLM call,
    is what should be trusted as the actual "citations exist" guarantee.
    """
    messages = [
        {"role": "system", "content": _VERIFY_SYSTEM_PROMPT},
        {"role": "user", "content": (
            f"Question: {user_text}\n\n"
            f"Retrieved sources:\n\n{_format_sources_for_prompt(sources)}\n\n"
            f"Answer to check:\n\n{answer}"
        )},
    ]
    try:
        raw = _ollama_chat(DICTA_MODEL, messages, format_json=True,
                            num_predict=_VERIFY_NUM_PREDICT, num_ctx=_DICTA_NUM_CTX)
        parsed = _parse_json_response(raw)
    except RuntimeError as e:
        # A verification step that couldn't run is NOT evidence the answer
        # is safe -- it's absence of evidence either way, so this defaults
        # to FAIL (unverified), not PASS. The answer still isn't blocked
        # (answer_legal_question returns the last answer either way once
        # retries are exhausted) -- only the label shown to the user
        # changes, from a false "verified" to an honest "couldn't verify."
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
    grounding_issues = _ground_draft_citations(answer, sources)
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
                           file_context: str = "", on_progress=None) -> dict:
    """
    Runs the full pipeline for one question and returns:
        {"answer": str, "sources": list[dict], "cycles_used": int,
         "verification": {"status": ..., "issues": [...]}}

    on_progress, if given, is called with a short status string before each
    stage -- app/ui.py's ChatInterface generator wraps these into the
    yielded "thinking..." messages the other tabs already show.

    Every stage -- query understanding, answering, verification, and
    retry-query generation -- runs on DICTA_MODEL. There is no DeepSeek
    dependency anywhere in this module.

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
        # immediately rather than running query understanding, retrieval, a
        # Dicta answer, Dicta verification, and up to MAX_VERIFICATION_CYCLES
        # retries against a question that was never going to retrieve
        # anything relevant in the first place. Skipped entirely (not just
        # short-circuited after one cheap check) if a file is attached -- an
        # attached document can supply real content even when the typed
        # message alone is just "hi, can you look at this?"
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
    answer = ""
    verification = {"status": "FAIL", "issues": [], "needs_more_retrieval": True}

    for cycle in range(1, MAX_VERIFICATION_CYCLES + 1):
        progress(f"📚 מחפש מקורות משפטיים (סבב {cycle}/{MAX_VERIFICATION_CYCLES})…")

        named_sources: list[dict] | None = None
        if cycle == 1:
            # Only attempted on the first cycle -- a retry after a FAIL
            # already has everything this same whole-law fetch would
            # return again (it's deterministic), so a retry falls through
            # to ordinary semantic search instead, which can surface the
            # SPECIFIC section relevant to whatever the verifier flagged
            # even if it got trimmed out of the whole-law pass.
            collection = _get_collection()
            if collection is not None:
                law_map = _get_law_titles(collection)
                named = _detect_named_law(user_text, law_map)
                if named:
                    law_id, law_name, score = named
                    print(f"[legal_pipeline_v2] question names a specific law "
                          f"({law_name!r}, token-overlap={score:.2f}) -- using a direct "
                          f"whole-law fetch instead of top-K competition.")
                    fetched = _fetch_whole_law_sources(collection, law_id)
                    if fetched:
                        named_sources = fetched
                    else:
                        print(f"[legal_pipeline_v2] {law_name!r} matched by name but "
                              "returned no chunks -- falling back to semantic search.")

        sources = _trim_sources_to_budget(named_sources if named_sources is not None else hybrid_retrieve(queries))

        progress("🧠 Dicta עונה על סמך המקורות שאותרו…")
        answer = dicta_answer(combined_question, sources, history)

        progress("🔎 Dicta בודק את התשובה מול המקורות…")
        verification = dicta_verify(combined_question, sources, answer)

        if verification["status"] == "PASS":
            break
        if cycle == MAX_VERIFICATION_CYCLES:
            break  # out of retries -- return the last answer with a visible warning
        queries = _followup_queries(combined_question, verification["issues"])

    final_answer = answer
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


def answer_legal_question_instruct(user_text: str, history: list[dict] | None = None,
                                    file_context: str = "", on_progress=None) -> dict:
    """
    Leaner, single-pass pipeline behind the "Attorney 32B Instruct (Slower)"
    tab -- deliberately simpler than answer_legal_question's retry +
    dual-verification loop:

        Step 1: Dicta, prompted for JSON            -> generates search terms
        Step 2: ChromaDB                             -> hybrid vector/BM25 search
        Step 3: Dicta, prompted for RAG               -> reasons in <think>,
                                                          outputs a cited answer
        Step 4: Python engine (regex, no LLM)         -> validates every cited
                                                          citation against the
                                                          retrieved sources'
                                                          own metadata
        -> final answer (no retry loop, no second LLM verification call)

    "Instruct" describes how step 1 is prompted (a plain JSON-instruction
    call -- the same understand_query() the other pipeline also uses), not
    a different model or checkpoint: both Attorney tabs run the exact same
    DICTA_MODEL (hf.co/dicta-il/DictaLM-3.0-24B-Thinking-GGUF:Q4_K_M).
    Step 3's "reasons in <think>" is just naming what a "Thinking" model
    already does internally before its real answer -- dicta_answer's
    result has that chain-of-thought stripped already (via _strip_think),
    same convention as every other thinking-model call in this project;
    it isn't shown to the user.

    Trades answer_legal_question's higher recall (keeps searching across
    up to MAX_VERIFICATION_CYCLES until something verifies) for
    predictable, bounded latency: exactly one search-term call, one
    retrieval, one answer call, and one free (non-LLM) validation pass,
    every time, with no retry branch back to retrieval.

    Returns the same {"answer", "sources", "cycles_used", "verification"}
    shape as answer_legal_question, so app/ui.py's existing sources-panel
    formatting works unchanged. cycles_used is always 1. "verification"
    here reflects ONLY the deterministic citation check (step 4) -- no LLM
    verifier runs in this pipeline at all, which is a narrower guarantee
    than answer_legal_question's combined LLM+deterministic result, but
    per this module's docstring, the deterministic check is the part
    actually worth trusting as "citations exist" either way.
    """
    if not PIPELINE_AVAILABLE:
        raise RuntimeError(
            "The legal pipeline's dependencies aren't installed. Run: "
            "pip install chromadb rank_bm25"
        )

    if not _has_meaningful_content(user_text) and not file_context:
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

    # --- Step 1: Dicta, prompted for JSON -- generates search terms ---
    progress("🧭 שלב 1/4: Dicta מייצר מונחי חיפוש (JSON)…")
    understood = understand_query(combined_question)
    queries = understood["queries"]

    # --- Step 2: ChromaDB -- hybrid vector/BM25 search ---
    # Reuses the same named-law fast path as answer_legal_question: still
    # "ChromaDB hybrid retrieval" from this pipeline's point of view, just
    # a smarter strategy for HOW step 2 fetches chunks when a specific law
    # is confidently named, rather than a departure from the diagram.
    progress("📚 שלב 2/4: מחפש בבסיס הנתונים המשפטי (ChromaDB, וקטורי + BM25)…")
    named_sources: list[dict] | None = None
    collection = _get_collection()
    if collection is not None:
        law_map = _get_law_titles(collection)
        named = _detect_named_law(user_text, law_map)
        if named:
            law_id, law_name, score = named
            print(f"[legal_pipeline_v2:instruct] question names a specific law "
                  f"({law_name!r}, token-overlap={score:.2f}) -- using a direct "
                  f"whole-law fetch instead of top-K competition.")
            fetched = _fetch_whole_law_sources(collection, law_id)
            if fetched:
                named_sources = fetched

    sources = _trim_sources_to_budget(named_sources if named_sources is not None else hybrid_retrieve(queries))

    # --- Step 3: Dicta, prompted for RAG -- reasons in <think>, outputs a
    # cited answer ---
    progress("🧠 שלב 3/4: Dicta מנמק (<think>) ועונה עם ציטוטים…")
    answer = dicta_answer(combined_question, sources, history)

    # --- Step 4: Python engine -- programmatic regex check, no LLM
    # involved, validating every cited סעיף number against the retrieved
    # sources' own metadata. Same function answer_legal_question uses as a
    # hard floor UNDER an LLM verifier -- here it's the ONLY verification
    # step, not a backstop under a second model call. ---
    progress("🔎 שלב 4/4: בדיקה פרוגרמטית של הציטוטים מול המקורות…")
    grounding_issues = _ground_draft_citations(answer, sources)
    verification = {
        "status": "FAIL" if grounding_issues else "PASS",
        "issues": grounding_issues,
        "needs_more_retrieval": False,  # this pipeline never retries
    }

    final_answer = answer
    if verification["status"] == "FAIL":
        issues_md = "\n".join(f"- {i}" for i in verification["issues"])
        final_answer += (
            "\n\n---\n🚨 *A programmatic check found citation(s) in this answer that "
            f"don't match anything actually retrieved:*\n{issues_md}\n\n"
            "*This pipeline doesn't retry automatically -- treat this answer as "
            "unverified and check it against the actual legislation.*"
        )
    else:
        final_answer += (
            "\n\n---\n⚖️ *Every citation in this answer was programmatically checked against "
            "its retrieved sources. This is not legal advice and is not a substitute for a "
            "licensed attorney.*"
        )

    return {
        "answer": final_answer,
        "sources": _sources_panel_data(sources),
        "cycles_used": 1,
        "verification": verification,
    }
