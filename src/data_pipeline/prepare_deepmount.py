#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prepara il dataset DeepMount00/pii-masking-ita per il training, unificandolo
sulla nostra tassonomia a 22 tag.

Il dataset usa la tassonomia ai4privacy-200k (56 tipi, Faker, prosa varia NON legale)
tradotta in italiano. Qui:
  - ri-tokenizziamo `unmasked_text` a PAROLA (stesso TOKEN_RE dei sintetici) e
    ricaviamo le BIO dagli span a caratteri (`span_labels`) -> niente mismatch col
    `tokenised_text` subword del dataset;
  - rimappiamo i 56 tipi sui nostri 22 (MAP); i tipi fuori scope -> O.

Output: JSONL {tokens, bio_labels} (come i sintetici) -> caricabile da train_pii.py.
Valore: dà contesto naturale e vario a ORG/IBAN/CREDITCARDNUMBER/AMOUNT/TARGA che
noi avevamo solo dai template (mitiga l'overfit strutturale).

Uso:  python prepare_deepmount.py
"""

import ast
import io
import json
import re
import sys

import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = str(ROOT / "dataset" / "raw" / "deepmount_pii_ita" / "data") + "/"
OUT_DIR = ROOT / "dataset" / "processed"
TRAIN_PARQUET = DATA_DIR + "train-00000-of-00001-ba82306000a2501b.parquet"
TEST_PARQUET = DATA_DIR + "test-00000-of-00001-abf75b392e5f38bd.parquet"

TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)   # identico a generate_synthetic_pii.py

# DeepMount (ai4privacy-200k) -> nostra tassonomia a 22 tag. Assente o None => O.
MAP = {
    # nomi
    "FIRSTNAME": "FULLNAME", "LASTNAME": "FULLNAME", "MIDDLENAME": "FULLNAME",
    # contatti
    "EMAIL": "EMAIL", "PHONENUMBER": "TELEPHONENUM",
    # bancari / finanziari
    "IBAN": "IBAN", "ACCOUNTNUMBER": "IBAN", "CREDITCARDNUMBER": "CREDITCARDNUMBER",
    "AMOUNT": "AMOUNT",
    # tempo / demografia
    "DATE": "DATE", "DOB": "DATE", "TIME": "TIME", "AGE": "AGE",
    "GENDER": "GENDER", "SEX": "GENDER",
    # indirizzo
    "STREET": "STREET", "BUILDINGNUMBER": "BUILDINGNUM", "ZIPCODE": "ZIPCODE", "CITY": "CITY",
    # enti
    "COMPANYNAME": "ORG",
    # documento identita'
    "SSN": "ID_DOC",
    # veicolo
    "VEHICLEVRM": "TARGA",
}

# tutto il resto -> O (digitali, credenziali, valuta, lavoro, fisici, BIC, borderline)


def to_word_bio(text, span_labels_str):
    """Ri-tokenizza il testo a parola e assegna BIO dai char-span, rimappando i tipi."""
    spans = [(s, e, lab) for s, e, lab in ast.literal_eval(span_labels_str) if lab != "O"]
    toks = [(m.group(), m.start(), m.end()) for m in TOKEN_RE.finditer(text)]
    labels = ["O"] * len(toks)
    for s, e, lab in spans:
        base = lab.rsplit("_", 1)[0]          # FIRSTNAME_1 -> FIRSTNAME
        final = MAP.get(base)
        if not final:
            continue
        first = True
        for i, (tok, ts, te) in enumerate(toks):
            if ts >= s and te <= e:
                labels[i] = ("B-" if first else "I-") + final
                first = False
    return [t for t, _, _ in toks], labels


def convert(parquet_path, out_path):
    df = pd.read_parquet(parquet_path)
    counts, dropped_types = {}, {}
    n = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            toks, labs = to_word_bio(row["unmasked_text"], row["span_labels"])
            if len(toks) != len(labs):
                continue
            for l in labs:
                if l.startswith("B-"):
                    counts[l[2:]] = counts.get(l[2:], 0) + 1
            # traccia cosa abbiamo scartato (per trasparenza)
            for s, e, lab in ast.literal_eval(row["span_labels"]):
                if lab != "O":
                    base = lab.rsplit("_", 1)[0]
                    if base not in MAP:
                        dropped_types[base] = dropped_types.get(base, 0) + 1
            f.write(json.dumps({"tokens": toks, "bio_labels": labs}, ensure_ascii=False) + "\n")
            n += 1
    return n, counts, dropped_types


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for split, src, out in [("train", TRAIN_PARQUET, str(OUT_DIR / "deepmount_pii_it_train.jsonl")),
                            ("test", TEST_PARQUET, str(OUT_DIR / "deepmount_pii_it_test.jsonl"))]:
        n, counts, dropped = convert(src, out)
        print(f"\n=== {split.upper()} -> {out} ({n} righe) ===")
        print("Tag finali (B- per tipo):")
        for k, v in sorted(counts.items(), key=lambda x: -x[1]):
            print(f"  {k:18s} {v}")
        if split == "train":
            tot_drop = sum(dropped.values())
            print(f"\nTipi scartati -> O ({len(dropped)} tipi, {tot_drop} entita'):")
            for k, v in sorted(dropped.items(), key=lambda x: -x[1]):
                print(f"  {k:20s} {v}")


if __name__ == "__main__":
    main()
