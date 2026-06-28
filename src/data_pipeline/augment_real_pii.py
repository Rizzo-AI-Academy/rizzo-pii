#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Augmentation: inietta entita' sintetiche nei testi REALI di Ai4Privacy (italiano).

Scopo: i tag che esistono SOLO nei sintetici (CF, IBAN, PIVA, AMOUNT, TARGA, ORG,
DOCID, CATASTO) il modello li vedrebbe solo dentro i ~pochi template ->
imparerebbe la struttura, non l'entita'. Qui li mostriamo dentro PROSA REALE varia,
in POSIZIONI DIVERSE (inseriti a un confine di frase casuale, non sempre in coda).

Riusa gli iniettori (con checksum validi) di generate_synthetic_pii.py: i valori
sono matematicamente validi e le label BIO restano esatte (sappiamo cosa iniettiamo).

Output: JSONL con schema {tokens, bio_labels} (come i sintetici) -> caricabile da
train_pii.py, che rimappa i tag grezzi via TAG_MAP al caricamento.

Uso:
  python augment_real_pii.py -n 40000 --out synthetic_pii_it_realaug.jsonl
"""

import argparse
import io
import json
import random
import sys

import generate_synthetic_pii as g

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
random.seed(42)

from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
TRAIN_PATH = str(ROOT / "dataset" / "raw" / "ai4privacy_500k" / "data" / "train" / "train.jsonl")

# frammenti a UN solo slot, con connettore in italiano: il connettore resta O,
# il valore prende B-/I-. Priorita' ai 9 tag che vengono solo dai sintetici.
INJECTION_SNIPPETS = [
    "C.F. {CF}",
    "codice fiscale {CF}",
    "P.IVA {PIVA}",
    "partita IVA {PIVA}",
    "IBAN {IBAN}",
    "coordinate bancarie IBAN {IBAN}",
    "per l'importo di {AMOUNT}",
    "pari a {AMOUNT}",
    "veicolo targato {TARGA}",
    "autovettura targa {TARGA}",
    "presso {ORG}",
    "la societa' {ORG}",
    "la ditta {ORG}",
    "il gruppo {ORG}",
    "la cooperativa {ORG}",
    "appaltatore {ORG}",
    "fattura emessa da {ORG}",
    "contratto stipulato con {ORG}",
    "le societa' {ORG}, {ORG} e {ORG}",
    "prot. n. {DOCID}",
    "R.G. n. {DOCID}",
    "giusta sentenza n. {DOCID}",
    "immobile censito al Catasto al {CATASTO}",
    "identificato catastalmente al {CATASTO}",
    # PROVINCE: nei template appare quasi solo come "Citta' (XX)"; qui la mostriamo in testo
    # reale con connettori vari -> mitiga l'overfit strutturale (era l'unico tag IT-only assente).
    "in provincia di {PROVINCE}",
    "prov. {PROVINCE}",
    "Prov. di {PROVINCE}",
    "in provincia ({PROVINCE})",
]


def snippet_tokens(snippet):
    """Costruisce (tokens, bio_labels) grezzi di un frammento con un valore iniettato."""
    text, ents = g.build_example(0, [snippet])
    return g.to_bio(text, ents)


def sentence_boundaries(labels, tokens):
    """Indici DOPO un punto fermo che e' fuori da un'entita' (confine di frase sicuro)."""
    return [i + 1 for i, (t, l) in enumerate(zip(tokens, labels))
            if t in (".", ";") and l == "O"]


def augment(tokens, labels, k):
    """Inserisce k frammenti in posizioni di confine casuali (o in coda se non ce ne sono)."""
    toks, labs = list(tokens), list(labels)
    for _ in range(k):
        s_tok, s_lab = snippet_tokens(random.choice(INJECTION_SNIPPETS))
        spots = sentence_boundaries(labs, toks)
        pos = random.choice(spots) if spots else len(toks)
        toks[pos:pos] = s_tok
        labs[pos:pos] = s_lab
    return toks, labs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", type=int, default=40000, help="righe augmentate da produrre")
    ap.add_argument("--out", default=str(ROOT / "dataset" / "synthetic" / "synthetic_pii_it_realaug.jsonl"))
    ap.add_argument("--max-inject", type=int, default=2, help="frammenti max per frase")
    args = ap.parse_args()

    # carica i testi reali italiani (tokens word-level + BIO grezzo)
    reals = []
    with open(TRAIN_PATH, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r.get("language") != "it":
                continue
            toks, labs = r.get("mbert_tokens"), r.get("mbert_token_classes")
            if toks and labs and len(toks) == len(labs):
                reals.append((toks, labs))
    print(f"Testi reali italiani disponibili: {len(reals)}")
    if not reals:
        sys.exit("Nessun testo reale italiano trovato: controlla TRAIN_PATH.")

    counts = {}
    written = 0
    with open(args.out, "w", encoding="utf-8") as out:
        for i in range(args.n):
            toks, labs = reals[i % len(reals)]
            k = random.randint(1, args.max_inject)
            atoks, alabs = augment(toks, labs, k)
            for l in alabs:
                if l.startswith("B-"):
                    counts[l[2:]] = counts.get(l[2:], 0) + 1
            out.write(json.dumps({"tokens": atoks, "bio_labels": alabs},
                                 ensure_ascii=False) + "\n")
            written += 1
    print(f"Scritte {written} righe augmentate -> {args.out}\n")
    print("Entita' B- per tipo (grezzo, prima di TAG_MAP):")
    for k, v in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"  {k:18s} {v}")


if __name__ == "__main__":
    main()
