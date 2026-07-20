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

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\u00c0-\u024f])")

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


def _looks_degenerate(text: str, min_words: int = 40, max_repeat_ratio: float = 0.3) -> bool:
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
    return unique_ratio < max_repeat_ratio


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
    return detected != expected_code

# Edit this to match your 7 target languages.
# MADLAD-400 covers 400+ languages, mostly using plain ISO 639-1 codes.
# Full list: https://github.com/google-research/google-research/blob/master/madlad_400/languages.md
LANGUAGES = {
    "english":   "en",
    "french":    "fr",
    "spanish":   "es",
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

        out_text, ok = self._translate_once(text, target_lang)
        if ok:
            return out_text, True

        # Whole-chunk translation was rejected (degenerate output, or the
        # model drifting back into the source language). Confirmed in
        # practice: MADLAD-400-3B is markedly more reliable on single short
        # sentences than on long multi-sentence blocks, so before giving up
        # on the whole chunk, retry sentence-by-sentence -- one bad sentence
        # shouldn't sink an otherwise-translatable paragraph.
        sentences = [s.strip() for s in _SENTENCE_SPLIT_RE.split(text.strip()) if s.strip()]
        if len(sentences) <= 1:
            return text, False  # already as small as it gets, nothing left to retry

        pieces = []
        any_failed = False
        for sentence in sentences:
            s_out, s_ok = self._translate_once(sentence, target_lang)
            pieces.append(s_out if s_ok else sentence)
            any_failed = any_failed or not s_ok

        return " ".join(pieces), not any_failed

    def _translate_once(self, text: str, target_lang: str) -> tuple[str, bool]:
        """Single translation attempt for one piece of text, with no retry logic."""
        tgt_code = LANGUAGES[target_lang.lower()]
        tagged_text = f"<2{tgt_code}> {text}"
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