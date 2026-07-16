"""
One-time conversion: HuggingFace NLLB-200-distilled-1.3B -> CTranslate2 int8 format.

Run once during setup (needs internet). After this, models/nllb-ct2/ is fully
self-contained and the app never touches the network again for translation.
"""
import subprocess
import sys
from pathlib import Path

MODEL_ID = "facebook/nllb-200-distilled-1.3B"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "models" / "nllb-ct2"

def main():
    if OUTPUT_DIR.exists() and any(OUTPUT_DIR.iterdir()):
        print(f"Already converted at {OUTPUT_DIR}, skipping.")
        return

    OUTPUT_DIR.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, "-m", "ctranslate2.converters.transformers",
        "--model", MODEL_ID,
        "--output_dir", str(OUTPUT_DIR),
        "--quantization", "int8",
    ]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    print(f"Done. CTranslate2 model at: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()