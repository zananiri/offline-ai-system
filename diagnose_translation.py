"""
Diagnostic: inspect exactly what MADLAD-400 receives and produces for
different target languages. Run this directly (not through the app) to see
whether the target-language tag is actually being recognized correctly.

Usage:
    .venv\\Scripts\\Activate.ps1      (Windows)
    source .venv/bin/activate         (Mac)
    python diagnose_translation.py
"""
import sys

print("Starting diagnostic script...", flush=True)
print("Importing app.translate (this itself can take a moment)...", flush=True)

from app.translate import get_translator, LANGUAGES

print("Import done. Loading the MADLAD-400 model (several GB — this is the", flush=True)
print("slow part, can take 30-90+ seconds depending on your disk)...", flush=True)

translator = get_translator()

print("Model loaded. Running translations now.\n", flush=True)

TEST_TEXT = "The quick brown fox jumps over the lazy dog."
TARGETS = ["spanish", "french", "arabic", "hebrew"]

for target in TARGETS:
    tgt_code = LANGUAGES[target]
    tagged_text = f"<2{tgt_code}> {TEST_TEXT}"

    # Show EXACTLY what tokens are being fed to CTranslate2 —
    # this is the critical piece of evidence.
    tokens = translator.tokenizer.convert_ids_to_tokens(
        translator.tokenizer.encode(tagged_text)
    )

    print(f"=== Target: {target} (tag: <2{tgt_code}>) ===", flush=True)
    print(f"First 5 tokens sent to the model: {tokens[:5]}", flush=True)

    out_text, ok = translator.translate(TEST_TEXT, "english", target)
    print(f"Output (ok={ok}): {out_text}\n", flush=True)

print("Diagnostic complete.", flush=True)