"""
Offline translation via MADLAD-400 (CTranslate2, int8). No network calls.

Apache 2.0 licensed (Google Research) — safe for commercial use. This
replaces NLLB-200, which is CC-BY-NC 4.0 and cannot legally be sold.
"""
from pathlib import Path
import re
import sys
import ctranslate2
import py3langid as langid
from transformers import AutoTokenizer

from app.document import strip_bidi_controls, SENTENCE_SPLIT_RE as _SENTENCE_SPLIT_RE, _hard_split

# Windows' console defaults to a legacy codepage (cp1252) that can't
# represent Hebrew, Arabic, or many other characters — any print() with
# such text would crash the request that triggered it. Forcing UTF-8 here
# makes the diagnostic logging below (and any future print in this module)
# safe regardless of what codepage the terminal happens to be using.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="backslashreplace")

MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "madlad400-ct2"
TOKENIZER_DIR = Path(__file__).resolve().parent.parent / "models" / "madlad400-tokenizer"
HF_TOKENIZER_ID = "google/madlad400-3b-mt"  # only used the first time, to populate TOKENIZER_DIR locally


_TOKEN_SPLIT_RE = re.compile(r"[\s.]+")


def _looks_degenerate(text: str, min_words: int = 40, max_repeat_ratio: float = 0.3,
                       prefix_len: int = 3, max_consecutive_same_prefix: int = 5) -> bool:
    """
    Detects a decoding-degeneration loop (the model getting stuck repeating
    tokens/n-grams — common on non-prose input like tables, ID numbers, or
    dates). Real translated prose has a much higher unique-word ratio than
    this even accounting for normal repetition of common words.
    """
    words = text.split()
    if len(words) < min_words:
        return False

    unique_ratio = len(set(words)) / len(words)
    if unique_ratio < max_repeat_ratio:
        return True

    # A second, distinct failure mode the exact-word check above misses
    # entirely: the model looping through different word-forms of the same
    # root instead of repeating one word verbatim. Confirmed directly in
    # practice on Hebrew output: "בעטרים.בעטירות.בעטורות.בעטריות.בעטראות.
    # בעטלות..." -- none of these individual forms repeat verbatim (so
    # unique_ratio above stays high, ~0.91 on the real example this was
    # found on), and worse, text.split() alone doesn't even see them as
    # separate words: a chain of periods with no following space (common
    # in this app's Hebrew OCR/extraction output) glues a 17-word run-on
    # into a SINGLE whitespace-delimited "word", hiding it completely from
    # any word-level check. Re-tokenizing on periods too, then checking for
    # a long CONSECUTIVE run sharing the same short prefix, is a specific,
    # false-positive-resistant signal for true degeneration: genuine prose
    # can repeat a place name or term many times across a passage, but
    # essentially never produces 5+ variants of the same root back-to-back
    # with nothing real interspersed -- degeneration loops are bursty and
    # consecutive, legitimate repetition is scattered.
    tokens = [t for t in _TOKEN_SPLIT_RE.split(text) if t]
    prefixes = [t[:prefix_len] for t in tokens if len(t) >= prefix_len]
    longest_run = current_run = 1
    for i in range(1, len(prefixes)):
        if prefixes[i] == prefixes[i - 1]:
            current_run += 1
            longest_run = max(longest_run, current_run)
        else:
            current_run = 1
    if longest_run >= max_consecutive_same_prefix:
        return True

    return False


# py3langid frequently confuses short/noisy text between languages that
# share the same script (Hebrew <-> Yiddish, Arabic <-> Persian/Urdu/Pashto)
# -- and text here is especially prone to that: OCR'd input is often short
# and imperfect. Without this, a same-script misdetection on a genuinely
# correct translation causes _wrong_language to wrongly reject it, and the
# caller then falls back to showing the original untranslated (and
# possibly still-noisy OCR'd) text instead -- i.e. a good translation gets
# thrown away because of a language-ID false positive, not a real model
# failure.
_SCRIPT_EQUIVALENT_LANGS = {
    "he": {"he", "yi"},
    "ar": {"ar", "fa", "ur", "ps"},
}


def _wrong_language(text: str, expected_code: str, min_words: int = 6) -> bool:
    """
    Detects a distinct failure mode from _looks_degenerate: the model
    ignoring the "<2xx>" target-language tag and performing an identity
    copy of the source text instead of translating it. That output is
    completely normal, non-repetitive prose -- just in the wrong language --
    so the repetition check above never catches it. Too short a text makes
    language detection unreliable, so very short chunks are skipped.
    """
    words = text.split()
    if len(words) < min_words:
        return False
    detected, _ = langid.classify(text)
    if detected == expected_code:
        return False
    if detected in _SCRIPT_EQUIVALENT_LANGS.get(expected_code, set()):
        return False
    return True

# Edit this to match your 7 target languages.
# MADLAD-400 covers 400+ languages, mostly using plain ISO 639-1 codes.
# Full list: https://github.com/google-research/google-research/blob/master/madlad_400/languages.md
LANGUAGES = {
    "english":   "en",
    "french":    "fr",
    "spanish":   "es",
    "italian":   "it",
    "german":    "de",
    "arabic":    "ar",
    "chinese":   "zh",
    "russian":   "ru",
    "hebrew":    "he",
}


class Translator:
    def __init__(self, device: str = "auto", compute_type: str = "int8"):
        self.translator = ctranslate2.Translator(str(MODEL_DIR), device=device, compute_type=compute_type)
        self.tokenizer = self._load_tokenizer()

    @staticmethod
    def _load_tokenizer():
        if TOKENIZER_DIR.exists() and any(TOKENIZER_DIR.iterdir()):
            # A genuine local directory makes transformers treat this as a
            # purely local load, which skips a Hub-only network check that
            # newer transformers versions run inside AutoTokenizer.from_pretrained()
            # (internally: `_is_local or (not _is_local and is_base_mistral(...))`
            # -- is_base_mistral() calls the HF Hub API unconditionally
            # unless the path is already local). That Hub call is what was
            # crashing every /translate-chunk request with no internet.
            # local_files_only=True is a second safety net on top of that.
            return AutoTokenizer.from_pretrained(str(TOKENIZER_DIR), local_files_only=True)

        # No local copy yet (first run on this machine, or models/ wasn't
        # copied over in full) -- download once from the Hub, then save it
        # locally so every future run loads purely from disk and never
        # touches the network again.
        try:
            tokenizer = AutoTokenizer.from_pretrained(HF_TOKENIZER_ID)
        except Exception as e:
            raise RuntimeError(
                f"No local tokenizer cache found at {TOKENIZER_DIR}, and "
                "downloading it from Hugging Face failed (are you offline?). "
                "Connect to the internet once so it can be cached locally "
                "(or run scripts/convert_translation_model.py), or copy an "
                "existing models/madlad400-tokenizer/ folder from another "
                "machine that already has it."
            ) from e
        TOKENIZER_DIR.mkdir(parents=True, exist_ok=True)
        tokenizer.save_pretrained(str(TOKENIZER_DIR))
        return tokenizer

    # Floor for retry splitting: below this, a piece is treated as
    # irreducible and any further failure is surfaced rather than retried
    # again, so a pathological chunk can't loop forever.
    _MIN_RETRY_CHARS = 40
    _MAX_RETRY_DEPTH = 4

    def translate(self, text: str, source_lang: str, target_lang: str) -> tuple[str, bool]:
        """Returns (text, ok). ok=False means this section was NOT translated —
        the original text is returned as-is so nothing is silently lost, but
        callers must check ok and surface the failure instead of treating the
        return value as normal translated output.

        source_lang is kept in the signature for API/UI compatibility with the
        rest of the app (which still lets the user pick a source language) —
        MADLAD-400 only needs a target-language tag; it infers the source
        language itself.
        """
        if not text or not text.strip():
            return text, True
        return self._translate_with_retry(text.strip(), target_lang, depth=0)

    def _translate_with_retry(self, text: str, target_lang: str, depth: int) -> tuple[str, bool]:
        """
        Recursively shrinks and retries a piece of text until it translates
        successfully or becomes irreducible.

        A one-shot version of this (split into sentences, retry each once,
        no further fallback) can leave a large "leftover" piece that fails
        for the exact same reason the original chunk did -- e.g. a chunk
        with no internal sentence boundary (common in OCR'd/converted text)
        just gives up immediately with no retry at all, confirmed directly
        in production logs (8 of 13 chunks failing outright on one
        document, each shown as exactly one failed attempt with no retry).

        This version keeps splitting and retrying recursively -- sentence
        boundaries first, falling back to a hard word-count split whenever
        sentence-splitting doesn't actually shrink the piece -- until every
        piece either succeeds or hits _MIN_RETRY_CHARS / _MAX_RETRY_DEPTH,
        at which point it's returned as-is with ok=False rather than
        retried indefinitely.
        """
        out_text, ok = self._translate_once(text, target_lang)
        if ok:
            return out_text, True

        if depth >= self._MAX_RETRY_DEPTH or len(text) <= self._MIN_RETRY_CHARS:
            return text, False  # as small as it's worth going; surface the failure

        # Confirmed in practice: MADLAD-400-3B is markedly more reliable on
        # single short sentences than on long multi-sentence blocks.
        # _SENTENCE_SPLIT_RE is script-aware for Latin/Cyrillic/Hebrew/
        # Arabic/CJK -- see document.py for why this used to silently fail
        # for Hebrew specifically.
        pieces = [s.strip() for s in _SENTENCE_SPLIT_RE.split(text) if s.strip()]

        # If sentence-splitting found no real boundary (<=1 piece), or every
        # "sentence" it found is still nearly the whole original text (a
        # long run-on with no punctuation-based break -- exactly what
        # happens when periods aren't followed by a space, which is common
        # in this app's Hebrew OCR/extraction output), fall back to a hard
        # word-count split so the retry is guaranteed to actually get
        # smaller instead of re-trying the same oversized text forever.
        if len(pieces) <= 1 or all(len(p) >= len(text) * 0.9 for p in pieces):
            pieces = _hard_split(text, max_chars=max(self._MIN_RETRY_CHARS, len(text) // 2))

        if len(pieces) <= 1:
            return text, False  # genuinely irreducible

        translated_pieces = []
        any_failed = False
        for piece in pieces:
            p_out, p_ok = self._translate_with_retry(piece, target_lang, depth + 1)
            translated_pieces.append(p_out)
            any_failed = any_failed or not p_ok

        return " ".join(translated_pieces), not any_failed

    def _translate_once(self, text: str, target_lang: str) -> tuple[str, bool]:
        """Single translation attempt for one piece of text, with no retry logic."""
        # Defense-in-depth: strip bidi/formatting control characters here too
        # (not just at Hebrew-OCR extraction time in document.py), so any
        # chunk that picked up stray marks another way -- RTL content pasted
        # into an otherwise-LTR document, chat input, etc. -- still tokenizes
        # cleanly. Only the text fed to the model is cleaned; `text` itself
        # is left untouched so the untranslated-section fallback below still
        # shows exactly what was extracted, not a modified version of it.
        clean_text = strip_bidi_controls(text)
        tgt_code = LANGUAGES[target_lang.lower()]
        tagged_text = f"<2{tgt_code}> {clean_text}"
        tokens = self.tokenizer.convert_ids_to_tokens(self.tokenizer.encode(tagged_text))

        results = self.translator.translate_batch(
            [tokens],
            beam_size=4,
            no_repeat_ngram_size=3,       # hard-blocks repeating any 3-token sequence
            repetition_penalty=1.3,        # discourages the model from repeating tokens at all
            max_decoding_length=max(256, len(tokens) * 3),  # safety cap: can't run away even if it still loops
        )
        out_tokens = results[0].hypotheses[0]
        out_ids = self.tokenizer.convert_tokens_to_ids(out_tokens)
        out_text = self.tokenizer.decode(out_ids, skip_special_tokens=True)

        # TEMPORARY diagnostic logging -- remove once the "always English"
        # bug is confirmed fixed. Shows the model's ACTUAL raw output, which
        # the checks below can otherwise discard before it ever reaches the UI.
        print(f"[translate] target={target_lang!r} tgt_code={tgt_code!r}")
        print(f"[translate] raw model output ({len(out_text.split())} words): {out_text!r}")

        if _looks_degenerate(out_text):
            print("[translate] REJECTED: flagged as degenerate (repetition loop)")
            # The generation-time guards above didn't fully prevent a
            # repetition loop. Surface that plainly instead of silently
            # returning garbage — this is most common on non-prose input
            # (tables, ID numbers, dates) where there isn't really anything
            # meaningful to "translate" in the first place, but it's also
            # exactly what happens when upstream OCR extraction was poor, so
            # ok=False must be checked and shown to the user, not swallowed.
            return text, False

        if _wrong_language(out_text, tgt_code):
            detected, _ = langid.classify(out_text)
            print(f"[translate] REJECTED: wrong language -- langid detected {detected!r}, expected {tgt_code!r}")
            # The model produced fine, coherent text -- just not in the
            # requested target language (usually an identity copy of the
            # source). This is a real, observed T5/MADLAD failure mode,
            # distinct from degenerate repetition above.
            return text, False

        print("[translate] OK")
        return out_text, True

    def translate_chunks(self, chunks: list[str], source_lang: str, target_lang: str) -> list[tuple[str, bool]]:
        return [self.translate(c, source_lang, target_lang) for c in chunks]


# Lazily instantiated singleton so the model loads once per process
_translator: Translator | None = None

def get_translator() -> Translator:
    global _translator
    if _translator is None:
        _translator = Translator()
    return _translator