#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Costruisce UNA validation reale unificata (validation_real.jsonl).

  - Base REALE held-out: Ai4Privacy validation (it) + DeepMount test.
  - Per i tag che hanno 0 esempi nella base (CF, PIVA, CATASTO, DOCID, PROVINCE) -
    che non esistono come dato reale da nessuna parte - inietta l'entita' generata
    dentro FRASI REALI held-out (frasi della validation Ai4, NON nel training).
    Cosi' il CONTESTO resta reale e non c'e' leakage col train (che usa frasi di TRAIN).

Output: {tokens, bio_labels} (grezzi) -> train_pii.py li normalizza al caricamento.
Seed 2024 (diverso dal 42 del training) => entita' indipendenti.

Uso:  python build_validation.py
"""

import json
import random

import augment_real_pii as aug   # snippet_tokens, sentence_boundaries (riconfigura gia' stdout UTF-8)

random.seed(2024)

from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "dataset"
VAL_PATH = str(DATA / "raw" / "ai4privacy_500k" / "data" / "validation" / "test.jsonl")
DM_TEST = str(DATA / "processed" / "deepmount_pii_it_test.jsonl")
OUT = str(DATA / "validation" / "validation_real.jsonl")

AI4_BASE = 2500       # frasi reali Ai4 (it) usate as-is
DM_BASE = 2500        # frasi DeepMount test usate as-is
PER_MISSING = 400     # frasi reali in cui iniettare ciascun tag mancante

# tag senza alcun esempio reale -> iniettati in frasi reali held-out
MISSING = {
    "CF": ["C.F. {CF}", "codice fiscale {CF}"],
    "PIVA": ["P.IVA {PIVA}", "partita IVA {PIVA}"],
    "CATASTO": ["immobile censito al Catasto al {CATASTO}", "identificato catastalmente al {CATASTO}"],
    "DOCID": ["prot. n. {DOCID}", "giusta sentenza n. {DOCID}"],
    "PROVINCE": ["in provincia di {PROVINCE}", "prov. {PROVINCE}"],
}


def load_ai4_it():
    recs = []
    with open(VAL_PATH, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r.get("language") != "it":
                continue
            t, l = r.get("mbert_tokens"), r.get("mbert_token_classes")
            if t and l and len(t) == len(l):
                recs.append((list(t), list(l)))
    return recs


def load_dm():
    recs = []
    with open(DM_TEST, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            t, l = r["tokens"], r["bio_labels"]
            if len(t) == len(l):
                recs.append((list(t), list(l)))
    return recs


def inject(toks, labs, snippet):
    s_tok, s_lab = aug.snippet_tokens(snippet)
    spots = aug.sentence_boundaries(labs, toks)
    pos = random.choice(spots) if spots else len(toks)
    t, l = list(toks), list(labs)
    t[pos:pos] = s_tok
    l[pos:pos] = s_lab
    return t, l


def main():
    ai4 = load_ai4_it()
    random.shuffle(ai4)
    dm = load_dm()
    random.shuffle(dm)
    print(f"Ai4 it val: {len(ai4)} frasi | DeepMount test: {len(dm)} frasi")

    rows = ai4[:AI4_BASE] + dm[:DM_BASE]   # base reale held-out

    # supplementi: tag mancanti iniettati in frasi reali Ai4 NON usate nella base
    pool = ai4[AI4_BASE:] or ai4
    pi = 0
    for tag, snips in MISSING.items():
        for _ in range(PER_MISSING):
            base = pool[pi % len(pool)]
            pi += 1
            rows.append(inject(base[0], base[1], random.choice(snips)))

    random.shuffle(rows)
    with open(OUT, "w", encoding="utf-8") as f:
        for t, l in rows:
            f.write(json.dumps({"tokens": t, "bio_labels": l}, ensure_ascii=False) + "\n")
    print(f"Scritte {len(rows)} righe -> {OUT} "
          f"(base reale {AI4_BASE + DM_BASE} + supplementi {len(MISSING) * PER_MISSING})")


if __name__ == "__main__":
    main()
