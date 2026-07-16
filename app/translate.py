"""
Offline translation via NLLB-200 (CTranslate2, int8). No network calls.
"""
from pathlib import Path
import ctranslate2
from transformers import AutoTokenizer

MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "nllb-ct2"
HF_TOKENIZER_ID = "facebook/nllb-200-distilled-1.3B"  # tokenizer config only, cached locally after first run

# Edit this to match your 7 target languages.
# Full FLORES-200 code list: https://github.com/facebookresearch/flores/blob/main/flores200/README.md
LANGUAGES = {
    "english":   "eng_Latn",
    "french":    "fra_Latn",
    "spanish":   "spa_Latn",
    "german":    "deu_Latn",
    "arabic":    "arb_Arab",
    "chinese":   "zho_Hans",
    "russian":   "rus_Cyrl",
}


class Translator:
    def __init__(self, device: str = "auto", compute_type: str = "int8"):
        self.translator = ctranslate2.Translator(str(MODEL_DIR), device=device, compute_type=compute_type)
        self.tokenizer = AutoTokenizer.from_pretrained(HF_TOKENIZER_ID)

    def translate(self, text: str, source_lang: str, target_lang: str) -> str:
        src_code = LANGUAGES[source_lang.lower()]
        tgt_code = LANGUAGES[target_lang.lower()]

        self.tokenizer.src_lang = src_code
        tokens = self.tokenizer.convert_ids_to_tokens(self.tokenizer.encode(text))

        results = self.translator.translate_batch(
            [tokens],
            target_prefix=[[tgt_code]],
            beam_size=4,
        )
        out_tokens = results[0].hypotheses[0][1:]  # drop the target-lang prefix token
        out_ids = self.tokenizer.convert_tokens_to_ids(out_tokens)
        return self.tokenizer.decode(out_ids, skip_special_tokens=True)

    def translate_chunks(self, chunks: list[str], source_lang: str, target_lang: str) -> list[str]:
        return [self.translate(c, source_lang, target_lang) for c in chunks]


# Lazily instantiated singleton so the model loads once per process
_translator: Translator | None = None

def get_translator() -> Translator:
    global _translator
    if _translator is None:
        _translator = Translator()
    return _translator
