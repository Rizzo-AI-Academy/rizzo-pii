# -*- coding: utf-8 -*-
"""Misura la lunghezza dei testi del dataset: parole (mbert_tokens) e subword (mmBERT)."""
import io, json, sys, random
import numpy as np
from transformers import AutoTokenizer

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
from pathlib import Path
PATH = str(Path(__file__).resolve().parents[2] / "dataset" / "raw" / "ai4privacy_500k" / "data" / "train" / "train.jsonl")
tok = AutoTokenizer.from_pretrained("jhu-clsp/mmBERT-base")

word_lens, it_rows = [], []
all_word_max = 0
for line in open(PATH, encoding="utf-8"):
    r = json.loads(line)
    t = r.get("mbert_tokens") or []
    all_word_max = max(all_word_max, len(t))
    if r.get("language") == "it":
        word_lens.append(len(t))
        it_rows.append(t)

word_lens = np.array(word_lens)
pct = lambda a, p: int(np.percentile(a, p))
print(f"ITALIANO: {len(word_lens)} righe")
print(f"  parole/riga  -> max {word_lens.max()} | p50 {pct(word_lens,50)} | "
      f"p90 {pct(word_lens,90)} | p95 {pct(word_lens,95)} | p99 {pct(word_lens,99)}")
print(f"  (max parole/riga su TUTTE le lingue: {all_word_max})")

# subword reali: campione + le 200 righe italiane piu' lunghe (per il vero massimo)
random.seed(0)
sample = random.sample(it_rows, min(8000, len(it_rows)))
longest = sorted(it_rows, key=len, reverse=True)[:200]
def sub_lens(rows):
    enc = tok(rows, is_split_into_words=True, add_special_tokens=True)
    return np.array([len(x) for x in enc["input_ids"]])

ss = sub_lens(sample)
sl = sub_lens(longest)
print(f"\nSUBWORD (campione {len(ss)} righe it):")
print(f"  max {ss.max()} | p50 {pct(ss,50)} | p90 {pct(ss,90)} | p95 {pct(ss,95)} | p99 {pct(ss,99)}")
print(f"SUBWORD sulle 200 righe it piu' lunghe: max {sl.max()}")
print(f"\n>> Con MAX_LEN=256 verrebbero troncate le righe sopra i 256 subword.")
for thr in (128, 256, 512):
    over = (ss > thr).mean() * 100
    print(f"   righe (campione) > {thr} subword: {over:.2f}%")
