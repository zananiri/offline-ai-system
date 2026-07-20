"""
One-time conversion: HuggingFace MADLAD-400-3B-MT -> CTranslate2 int8 format,
plus caching the tokenizer locally.

MADLAD-400 (Google Research) replaces NLLB-200 in this project. NLLB-200 is
licensed CC-BY-NC 4.0 (non-commercial only, per Meta's model card) and cannot
legally be used in a product you sell. MADLAD-400 is Apache 2.0 — free for
commercial use, and it covers even more languages (400+ vs. 200).

Run once during setup (needs internet). After this, models/madlad400-ct2/
and models/madlad400-tokenizer/ are both fully self-contained and the app
never touches the network again for translation.

Note: MADLAD-400-3B is a larger model than the NLLB-1.3B this project used
before, so both the download and translation speed will be noticeably heavier
on CPU-only hardware. This is the necessary tradeoff for a commercially
licensed model — there is no smaller official MADLAD-400 checkpoint.
"""
import subprocess
import sys
from pathlib import Path

from transformers import AutoTokenizer

MODEL_ID = "google/madlad400-3b-mt"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "models" / "madlad400-ct2"
TOKENIZER_DIR = Path(__file__).resolve().parent.parent / "models" / "madlad400-tokenizer"

def main():
    if OUTPUT_DIR.exists() and any(OUTPUT_DIR.iterdir()):
        print(f"Already converted at {OUTPUT_DIR}, skipping.")
    else:
        OUTPUT_DIR.parent.mkdir(parents=True, exist_ok=True)

        cmd = [
            sys.executable, "-m", "ctranslate2.converters.transformers",
            "--model", MODEL_ID,
            "--output_dir", str(OUTPUT_DIR),
            "--quantization", "int8",
        ]
        print("Running:", " ".join(cmd))
        print("This downloads several GB from Hugging Face — it will take a while.")
        subprocess.run(cmd, check=True)
        print(f"Done. CTranslate2 model at: {OUTPUT_DIR}")

    # app/translate.py loads the tokenizer from this local folder at runtime
    # instead of the "google/madlad400-3b-mt" Hub ID directly. Newer
    # transformers versions run an extra Hugging Face Hub network check
    # inside AutoTokenizer.from_pretrained() even when a cached copy already
    # exists elsewhere on disk -- passing a real local directory instead of
    # a Hub ID string skips that check entirely, which is what makes the
    # app genuinely offline-capable at request time.
    if TOKENIZER_DIR.exists() and any(TOKENIZER_DIR.iterdir()):
        print(f"Tokenizer already cached at {TOKENIZER_DIR}, skipping.")
    else:
        print(f"Caching tokenizer to {TOKENIZER_DIR} ...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        TOKENIZER_DIR.mkdir(parents=True, exist_ok=True)
        tokenizer.save_pretrained(str(TOKENIZER_DIR))
        print("Done.")

if __name__ == "__main__":
    main()