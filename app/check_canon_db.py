"""
Standalone diagnostic for the Canon AI Chroma DB.
Run from the SAME directory (and same Python/venv) you use for `python -m app.ui`:

    python check_canon_db.py
"""
import sys

try:
    import chromadb
except ImportError:
    print("ERROR: chromadb is not installed in THIS Python interpreter:")
    print("  ", sys.executable)
    print("Run: pip install chromadb")
    print("(then make sure this is the same interpreter that runs `python -m app.ui`)")
    sys.exit(1)

print("Using Python:", sys.executable)
print("chromadb version:", chromadb.__version__)

client = chromadb.PersistentClient(path="./chroma_db")

collections = client.list_collections()
names = [c.name for c in collections]
print("Collections found in ./chroma_db:", names)

if "cic_it" not in names:
    print("\n>>> 'cic_it' collection NOT found. This is the mismatch.")
    print(">>> Either re-run embed_to_chroma.py with --collection cic_it,")
    print(">>> or update CANON_COLLECTION_NAME in ui.py to match:", names)
    sys.exit(1)

col = client.get_collection("cic_it")
count = col.count()
print("cic_it document count:", count)

if count == 0:
    print("\n>>> Collection exists but is EMPTY. embed_to_chroma.py likely")
    print(">>> ran but failed partway through, or the scrape produced no data.")
else:
    print("\n>>> Looks healthy. If Canon AI still errors, the issue is likely")
    print(">>> that `python -m app.ui` is running under a DIFFERENT Python")
    print(">>> interpreter/venv than this one:", sys.executable)
