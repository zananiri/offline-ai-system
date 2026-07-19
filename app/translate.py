"""
Offline translation via MADLAD-400 (CTranslate2, int8). No network calls.

Apache 2.0 licensed (Google Research) — safe for commercial use. This
replaces NLLB-200, which is CC-BY-NC 4.0 and cannot legally be sold.
"""
from pathlib import Path
import ctranslate2
from transformers import AutoTokenizer

MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "madlad400-ct2"
HF_TOKENIZER_ID = "google/madlad400-3b-mt"  # tokenizer config only, cached locally after first run


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
        self.tokenizer = AutoTokenizer.from_pretrained(HF_TOKENIZER_ID)

    def translate(self, text: str, source_lang: str, target_lang: str) -> str:
        # MADLAD-400 only needs a target-language tag prepended to the source
        # text (e.g. "<2es> Hello") — it infers the source language itself.
        # source_lang is kept in the signature for API/UI compatibility with
        # the rest of the app, which still lets the user pick a source language.
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

        if _looks_degenerate(out_text):
            # The generation-time guards above didn't fully prevent a
            # repetition loop. Surface that plainly instead of silently
            # returning garbage — this is most common on non-prose input
            # (tables, ID numbers, dates) where there isn't really anything
            # meaningful to "translate" in the first place.
            return f"[Translation failed for this section — likely a table, ID numbers, or non-sentence content. Original text:]\n{text}"

        return out_text

    def translate_chunks(self, chunks: list[str], source_lang: str, target_lang: str) -> list[str]:
        return [self.translate(c, source_lang, target_lang) for c in chunks]


# Lazily instantiated singleton so the model loads once per process
_translator: Translator | None = None

def get_translator() -> Translator:
    global _translator
    if _translator is None:
        _translator = Translator()
    return _translator