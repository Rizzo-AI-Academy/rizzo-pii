# -*- coding: utf-8 -*-
"""
Costruisce due subset RAPPRESENTATIVI per smoke test / tuning veloce prima del
training grande (rizzo-pii:0.3B):

  - train_subset_10k.jsonl   (10.000 righe)  stratificato per (fonte x lingua) +
                                             floor sui tag rari (proporzionale + floor)
  - val_subset_5k.jsonl      ( 5.000 righe)  stratificato per tag (val e' tutta IT)

Schema di output: {tokens, bio_labels, language, source}. I campi language/source
sono solo per ispezione; train_pii.py legge solo tokens/bio_labels (load_synth).

Strategia:
  * stratificazione PRIMARIA per (fonte x lingua) in modo proporzionale -> riproduce
    da sola la distribuzione dei tag (i tag sono legati alle fonti), e mantiene il mix
    multilingua di Ai4Privacy (it/en/fr/de/es/hi/te/nl).
  * FLOOR sui tag rari (TARGA, CREDITCARDNUMBER, GENDER, AGE) -> garantisce abbastanza
    righe perche' le metriche per-tag siano calcolabili anche a 10k.

Implementazione a DUE PASSATE con offset dei byte: la pass 1 indicizza solo
(offset, fonte, lingua, set di tag) per riga, senza tenere i token in RAM; la pass 2
rilegge per seek solo le righe selezionate. Cosi' non si caricano gli ~1,3 GB di token.

Le label vengono normalizzate con la STESSA logica di train_pii.py (TAG_MAP/DROP_TYPES)
cosi' i tag finali coincidono con quelli che vede il modello.
"""

import io
import json
import os
import random
import sys
from collections import Counter, defaultdict

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

SEED = 1234
random.seed(SEED)

# --------------------------------------------------------------------------- #
# Tassonomia: identica a train_pii.py (i file grezzi restano intatti)
# --------------------------------------------------------------------------- #
TAG_MAP = {
    "GIVENNAME": "FULLNAME", "SURNAME": "FULLNAME",
    "GIUDICE": "FULLNAME", "AVVOCATO": "FULLNAME", "CONVENUTO": "FULLNAME",
    "ATTORE": "FULLNAME", "TESTIMONE": "FULLNAME",
    "SEX": "GENDER",
    "TAXNUM": "PIVA",
    "PEC": "EMAIL",
    "RG": "DOCID",
    "IDCARDNUM": "ID_DOC", "PASSPORTNUM": "ID_DOC",
    "DRIVERLICENSENUM": "ID_DOC", "SOCIALNUM": "ID_DOC",
    "CONTO": "IBAN",
}
DROP_TYPES = {"TITLE", "TRIBUNAL"}


def normalize_labels(labels):
    out, prev = [], None
    for l in labels:
        typ = TAG_MAP.get(l[2:], l[2:]) if l != "O" else None
        if typ is None or typ in DROP_TYPES:
            out.append("O"); prev = None
            continue
        out.append(("I-" if typ == prev else "B-") + typ)
        prev = typ
    return out


def tags_of(norm_labels):
    """Insieme dei tipi (senza B-/I-) presenti nella riga."""
    return {l[2:] for l in norm_labels if l != "O"}


# --------------------------------------------------------------------------- #
# Sorgenti
# --------------------------------------------------------------------------- #
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "dataset"

TRAIN_FILES = [
    dict(path=str(DATA / "raw/ai4privacy_500k/data/train/train.jsonl"), schema="ai4", source="ai4"),
    dict(path=str(DATA / "synthetic/synthetic_pii_it_200k.jsonl"), schema="synth", source="synth_template"),
    dict(path=str(DATA / "synthetic/synthetic_pii_it_realaug.jsonl"), schema="synth", source="synth_augment"),
    dict(path=str(DATA / "processed/deepmount_pii_it_train.jsonl"), schema="synth", source="deepmount"),
]
VAL_FILES = [
    dict(path=str(DATA / "validation/validation_real.jsonl"), schema="synth", source="val"),
]

# Floor (numero MINIMO di righe nel subset che contengono il tag).
TRAIN_RARE_FLOOR = {"TARGA": 200, "CREDITCARDNUMBER": 200, "GENDER": 250, "AGE": 250}
TRAIN_DEFAULT_FLOOR = 40        # rete di sicurezza: ogni tag almeno N righe (se disponibili)


def parse_line(raw, schema):
    """Ritorna (tokens, raw_labels, language) oppure None se la riga non e' valida."""
    r = json.loads(raw)
    if schema == "ai4":
        toks, labs = r.get("mbert_tokens"), r.get("mbert_token_classes")
        lang = r.get("language") or "?"
    else:
        toks, labs = r.get("tokens"), r.get("bio_labels")
        lang = r.get("language") or "it"
    if toks and labs and len(toks) == len(labs):
        return toks, labs, lang
    return None


# --------------------------------------------------------------------------- #
# PASSATA 1: indicizza (file_i, offset, source, lang, tagset) per ogni riga valida
# --------------------------------------------------------------------------- #
def index_files(files):
    rows = []                      # lista di dict leggeri
    pool_tagrows = Counter()       # righe per tag (sull'intero pool)
    pool_by_stratum = Counter()    # righe per (source, lang)
    for fi, cfg in enumerate(files):
        path, schema, source = cfg["path"], cfg["schema"], cfg["source"]
        if not os.path.exists(path):
            print(f"ATTENZIONE: file mancante (saltato): {path}")
            continue
        n = 0
        with open(path, encoding="utf-8") as f:
            while True:
                offset = f.tell()
                raw = f.readline()
                if not raw:
                    break
                parsed = parse_line(raw, schema)
                if parsed is None:
                    continue
                _, labs, lang = parsed
                tset = tags_of(normalize_labels(labs))
                rows.append(dict(fi=fi, off=offset, src=source, lang=lang, tags=tset))
                pool_by_stratum[(source, lang)] += 1
                for t in tset:
                    pool_tagrows[t] += 1
                n += 1
        print(f"  indicizzate {n:>7} righe da {path}")
    return rows, pool_tagrows, pool_by_stratum


# --------------------------------------------------------------------------- #
# Selezione stratificata (proporzionale + floor)
# --------------------------------------------------------------------------- #
def select(rows, N, stratify_by_lang, rare_floor, default_floor):
    total = len(rows)
    N = min(N, total)
    by_stratum = defaultdict(list)
    by_tag = defaultdict(list)
    for i, r in enumerate(rows):
        key = (r["src"], r["lang"]) if stratify_by_lang else (r["src"],)
        by_stratum[key].append(i)
        for t in r["tags"]:
            by_tag[t].append(i)

    selected = set()

    # --- 1) FLOOR: prima i tag piu' rari, cosi' non vengono "rubati" dai comuni ---
    all_tags = sorted(by_tag, key=lambda t: len(by_tag[t]))   # rarest first
    for t in all_tags:
        need = rare_floor.get(t, default_floor)
        have = sum(1 for i in by_tag[t] if i in selected)
        if have >= need:
            continue
        pool = [i for i in by_tag[t] if i not in selected]
        random.shuffle(pool)
        for i in pool[: need - have]:
            if len(selected) >= N:
                break
            selected.add(i)

    # --- 2) RIEMPIMENTO proporzionale per stratum fino a N ---
    quotas = {k: round(N * len(v) / total) for k, v in by_stratum.items()}
    for key in sorted(by_stratum, key=lambda k: -len(by_stratum[k])):
        if len(selected) >= N:
            break
        have = sum(1 for i in by_stratum[key] if i in selected)
        deficit = quotas[key] - have
        if deficit <= 0:
            continue
        pool = [i for i in by_stratum[key] if i not in selected]
        random.shuffle(pool)
        for i in pool[:deficit]:
            if len(selected) >= N:
                break
            selected.add(i)

    # --- 3) Top-up finale (arrotondamenti) con righe casuali non ancora prese ---
    if len(selected) < N:
        rest = [i for i in range(total) if i not in selected]
        random.shuffle(rest)
        for i in rest[: N - len(selected)]:
            selected.add(i)

    return selected


# --------------------------------------------------------------------------- #
# PASSATA 2: rilegge (per seek) solo le righe selezionate e scrive l'output
# --------------------------------------------------------------------------- #
def write_subset(rows, selected, files, out_path):
    by_file = defaultdict(list)
    for i in selected:
        by_file[rows[i]["fi"]].append(i)
    written = 0
    with open(out_path, "w", encoding="utf-8") as out:
        for fi, idxs in by_file.items():
            cfg = files[fi]
            schema, source = cfg["schema"], cfg["source"]
            idxs.sort(key=lambda i: rows[i]["off"])
            with open(cfg["path"], encoding="utf-8") as f:
                for i in idxs:
                    f.seek(rows[i]["off"])
                    parsed = parse_line(f.readline(), schema)
                    if parsed is None:
                        continue
                    toks, labs, lang = parsed
                    out.write(json.dumps({
                        "tokens": toks,
                        "bio_labels": normalize_labels(labs),
                        "language": lang,
                        "source": source,
                    }, ensure_ascii=False) + "\n")
                    written += 1
    return written


# --------------------------------------------------------------------------- #
# Report di verifica
# --------------------------------------------------------------------------- #
def report(name, rows, selected, pool_tagrows, pool_by_stratum):
    sub = [rows[i] for i in selected]
    n = len(sub)
    print(f"\n{'='*72}\n{name}: {n} righe\n{'='*72}")

    print("\nRighe per (fonte, lingua)   [subset%  vs  pool%]")
    sub_str = Counter((r["src"], r["lang"]) for r in sub)
    tot_pool = sum(pool_by_stratum.values())
    for key in sorted(pool_by_stratum, key=lambda k: -pool_by_stratum[k]):
        sp = 100 * sub_str.get(key, 0) / n
        pp = 100 * pool_by_stratum[key] / tot_pool
        print(f"  {str(key):<28} {sub_str.get(key,0):>6}  {sp:5.1f}%   vs {pp:5.1f}%")

    print("\nRighe che contengono il tag [subset%  vs  pool%]")
    sub_tagrows = Counter()
    for r in sub:
        for t in r["tags"]:
            sub_tagrows[t] += 1
    tot_pool_rows = sum(pool_by_stratum.values())
    for t in sorted(pool_tagrows, key=lambda t: -pool_tagrows[t]):
        sp = 100 * sub_tagrows.get(t, 0) / n
        pp = 100 * pool_tagrows[t] / tot_pool_rows
        flag = "  <-- 0!" if sub_tagrows.get(t, 0) == 0 else ""
        print(f"  {t:<18} {sub_tagrows.get(t,0):>6}  {sp:5.1f}%   vs {pp:5.1f}%{flag}")


# --------------------------------------------------------------------------- #
# MAIN
# --------------------------------------------------------------------------- #
def build(files, N, out_path, stratify_by_lang, rare_floor, default_floor, name):
    print(f"\n### {name} -> {out_path} (target {N} righe) ###")
    rows, pool_tagrows, pool_by_stratum = index_files(files)
    print(f"  pool totale: {len(rows)} righe valide")
    selected = select(rows, N, stratify_by_lang, rare_floor, default_floor)
    written = write_subset(rows, selected, files, out_path)
    report(name, rows, selected, pool_tagrows, pool_by_stratum)
    print(f"\nScritte {written} righe in {out_path}")


if __name__ == "__main__":
    (DATA / "subsets").mkdir(parents=True, exist_ok=True)
    build(
        TRAIN_FILES, 10000, str(DATA / "subsets" / "train_subset_10k.jsonl"),
        stratify_by_lang=True, rare_floor=TRAIN_RARE_FLOOR,
        default_floor=TRAIN_DEFAULT_FLOOR, name="TRAIN subset",
    )
    build(
        VAL_FILES, 5000, str(DATA / "subsets" / "val_subset_5k.jsonl"),
        stratify_by_lang=False, rare_floor={}, default_floor=0,
        name="VALIDATION subset",
    )
    print("\nFatto. Avvia il training di prova con:  python src/training/train_pii.py --type subset")
