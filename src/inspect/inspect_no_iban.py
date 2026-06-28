# -*- coding: utf-8 -*-
"""Dimostra che il dataset Ai4Privacy (italiano) NON ha il tag IBAN."""
import io, json, sys, collections

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
from pathlib import Path
PATH = str(Path(__file__).resolve().parents[2] / "dataset" / "raw" / "ai4privacy_500k" / "data" / "train" / "train.jsonl")

labels = collections.Counter()
iban_rows = 0
example = None
finance_example = None
KEYS = ("iban", "conto", "bancar", "bonifico", "pagament", "versare", "accredit")

for line in open(PATH, encoding="utf-8"):
    r = json.loads(line)
    if r.get("language") != "it":
        continue
    cls = set(c[2:] for c in r["mbert_token_classes"] if c != "O")
    for c in cls:
        labels[c] += 1
    if any("IBAN" in c for c in cls):
        iban_rows += 1
    if example is None:
        example = r
    low = r["source_text"].lower()
    if finance_example is None and any(k in low for k in KEYS):
        finance_example = r

print("Tutti i tag presenti nell'italiano (entity-level):")
print("  ", ", ".join(sorted(labels)))
print(f"\nRighe italiane con un tag IBAN: {iban_rows}   <-- (zero = IBAN non esiste)\n")

def show(title, r):
    cls = sorted(set(c[2:] for c in r["mbert_token_classes"] if c != "O"))
    print("=" * 80)
    print(title)
    print("source:", r["source_text"][:300])
    print("masked:", r["masked_text"][:300])
    print("tag in questo esempio:", cls)
    print("-> contiene IBAN?", "SI" if any("IBAN" in c for c in cls) else "NO")

show("ESEMPIO GENERICO", example)
if finance_example:
    show("ESEMPIO CON CONTESTO FINANZIARIO (pagamento/banca) ma SENZA tag IBAN",
         finance_example)
