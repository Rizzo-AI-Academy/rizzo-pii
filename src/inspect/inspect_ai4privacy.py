# -*- coding: utf-8 -*-
import json, collections, sys, io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
from pathlib import Path
BASE = str(Path(__file__).resolve().parents[2] / "dataset" / "raw" / "ai4privacy_500k" / "data")

def count(path):
    lang = collections.Counter()
    it_first = None
    n = 0
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        lang[r["language"]] += 1
        n += 1
        if r["language"] == "it" and it_first is None:
            it_first = r
    return n, lang, it_first

n, lang, it_first = count(f"{BASE}/train/train.jsonl")
nv, langv, _ = count(f"{BASE}/validation/test.jsonl")

print(f"TRAIN: {n} esempi | VALIDATION: {nv} esempi")
print("\nEsempi train per lingua:")
for k, v in lang.most_common():
    print(f"  {k}: {v}")
print(f"\nItaliano: train={lang.get('it',0)}  val={langv.get('it',0)}")

if it_first:
    print("\n--- Esempio italiano ---")
    print("source:", it_first["source_text"][:220])
    print("masked:", it_first["masked_text"][:220])
    labels = sorted(set(c[2:] for c in it_first["mbert_token_classes"] if c != "O"))
    print("tag in questo esempio:", labels)

# tassonomia globale dei tag (campione)
tags = collections.Counter()
for line in open(f"{BASE}/train/train.jsonl", encoding="utf-8"):
    r = json.loads(line)
    if r["language"] != "it":
        continue
    for c in r["mbert_token_classes"]:
        if c != "O":
            tags[c[2:]] += 1
print("\nTag piu' frequenti (solo italiano):")
for k, v in tags.most_common(40):
    print(f"  {k}: {v}")
