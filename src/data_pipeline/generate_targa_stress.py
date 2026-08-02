#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Genera uno stress test TARGA sintetico, separato dai dati di training.

Le famiglie di template definite qui non compaiono in generate_targa_it.py.
Il file risultante serve solo per regression test e confronto tra modello puro
e pipeline applicativa modello+regex; non deve entrare nel training.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PIPELINE_DIR = ROOT / "src" / "data_pipeline"
sys.path.insert(0, str(PIPELINE_DIR))

import generate_synthetic_pii as base  # noqa: E402
import generate_targa_it as train_gen  # noqa: E402


DEFAULT_SEED = 20260705
DEFAULT_OUT = ROOT / "dataset" / "validation" / "validation_targa_it_stress_300.jsonl"
GENERATOR_VERSION = "targa-stress-1.0.0"

# Nessuna frase è condivisa con POSITIVE_TEMPLATES del generatore di training.
POSITIVE_TEMPLATES = [
    "La telecamera al varco ha registrato {TARGA} alle ore 23:14.",
    "Sul parabrezza del mezzo {TARGA} era esposto il contrassegno scaduto.",
    "Dalla banca dati della motorizzazione emerge {TARGA} come identificativo del mezzo.",
    "Nel filmato si distingue l'automobile {TARGA} mentre lascia il parcheggio.",
    "Il carro attrezzi ha rimosso {TARGA} dall'area riservata.",
    "Alla frontiera è stato controllato il mezzo {TARGA}.",
    "Il perito ha fotografato cofano e contrassegno posteriore di {TARGA}.",
    "Nel box numero sette era parcheggiata {TARGA}.",
    "La vettura sostava in doppia fila: {TARGA}.",
    "Dal portale dell'assicuratore risulta ancora attivo il mezzo {TARGA}.",
    "Rilevazione automatica\nmezzo: {TARGA}\nvelocità: 87 km/h.",
    "Nel registro di ingresso compare {TARGA}, entrata alle 08:42.",
    "Il numero impresso sulla placca posteriore era {TARGA}.",
    "L'autorimessa conferma di avere custodito {TARGA} per tre notti.",
    "La vettura riconsegnata al locatore riportava {TARGA}.",
]

HARD_NEGATIVE_TEMPLATES = [
    "La password monouso comunicata al cliente è {CODE}.",
    "Il pacco contrassegnato {CODE} è arrivato al centro logistico.",
    "Per aprire la segnalazione indicare il ticket {CODE}.",
    "Il buono sconto {CODE} può essere utilizzato una sola volta.",
    "La commessa di lavorazione {CODE} termina venerdì.",
    "Il codice del componente sostituito è {CODE}.",
    "L'ordine web {CODE} è stato rimborsato.",
    "La prenotazione alberghiera {CODE} comprende due camere.",
    "Inserire {CODE} nel campo codice di verifica.",
    "Il lotto farmaceutico {CODE} è stato richiamato.",
    "La pratica software usa come chiave interna {CODE}.",
    "La spedizione internazionale {CODE} è ferma in dogana.",
    "Il coupon {CODE} è riservato ai nuovi clienti.",
    "Il numero della richiesta tecnica è {CODE}.",
    "La sessione {CODE} è terminata per inattività.",
]

MIXED_TEMPLATES = [
    "{FULLNAME} ha parcheggiato {TARGA} davanti alla sede di {ORG}.",
    "Il mezzo {TARGA} è stato consegnato a {FULLNAME} presso {ADDRESS}.",
    "La segnalazione di {FULLNAME}, inviata da {EMAIL}, riguarda {TARGA}.",
    "{ORG}, P. IVA {PIVA}, ha noleggiato il mezzo {TARGA}.",
    "Per informazioni su {TARGA}, scrivere a {EMAIL} indicando {FULLNAME}.",
    "Il custode {FULLNAME} ha registrato {TARGA} all'indirizzo {ADDRESS}.",
    "La manutenzione di {TARGA} è stata fatturata da {ORG}, P. IVA {PIVA}.",
    "{FULLNAME} ha ritirato {TARGA}; recapito telefonico {PHONE}.",
    "Il mezzo {TARGA} è assegnato a {ORG} e affidato a {FULLNAME}.",
    "La comunicazione spedita a {ADDRESS} menziona il veicolo {TARGA}.",
]


def stress_plate_value() -> str:
    """Varianti plausibili ma distribuite diversamente dal training."""
    code = train_gen.plate_code()
    style = random.choices(("compact", "spaces", "hyphens"), (45, 40, 15), k=1)[0]
    if style == "spaces":
        value = f"{code[:2]} {code[2:5]} {code[5:]}"
    elif style == "hyphens":
        value = f"{code[:2]}-{code[2:5]}-{code[5:]}"
    else:
        value = code
    # Una piccola quota riproduce trascrizioni manuali in minuscolo.
    return value.lower() if random.random() < 0.08 else value


def mixed_slots():
    return {
        "TARGA": [(stress_plate_value(), "TARGA")],
        "FULLNAME": train_gen.full_name_piece(),
        "ORG": base.org_piece(),
        "PIVA": base.piva_piece(),
        "ADDRESS": base.address(),
        "EMAIL": base.email_piece(),
        "PHONE": base.phone_piece(),
    }


def make_record(kind: str, index: int, seed: int):
    if kind == "positive":
        template_id = random.randrange(len(POSITIVE_TEMPLATES))
        text, entities = train_gen.render(
            POSITIVE_TEMPLATES[template_id],
            {"TARGA": [(stress_plate_value(), "TARGA")]},
        )
    elif kind == "hard_negative":
        template_id = random.randrange(len(HARD_NEGATIVE_TEMPLATES))
        # Compatto e formalmente identico a una targa: il contesto è l'unico segnale.
        text, entities = train_gen.render(
            HARD_NEGATIVE_TEMPLATES[template_id],
            {"CODE": [(train_gen.plate_code(), None)]},
        )
    elif kind == "mixed_pii":
        template_id = random.randrange(len(MIXED_TEMPLATES))
        text, entities = train_gen.render(MIXED_TEMPLATES[template_id], mixed_slots())
    else:
        raise ValueError(f"categoria sconosciuta: {kind}")

    tokens, bio_labels = base.to_bio(text, entities)
    record = {
        "source_text": text,
        "language": "it",
        "tokens": tokens,
        "bio_labels": bio_labels,
        "entities": entities,
        "meta": {
            "contributor": "rizzoaiacademy",
            "seed": seed,
            "generator_version": GENERATOR_VERSION,
            "synthetic": True,
            "split": "stress",
            "category": kind,
            "template_id": template_id,
            "row_index": index,
        },
    }
    train_gen.validate_record(record)
    return record


def generate(per_category: int, seed: int):
    random.seed(seed)
    kinds = [
        kind
        for kind in ("positive", "hard_negative", "mixed_pii")
        for _ in range(per_category)
    ]
    random.shuffle(kinds)

    rows = []
    seen = set()
    while len(rows) < len(kinds):
        record = make_record(kinds[len(rows)], len(rows), seed)
        if record["source_text"] in seen:
            continue
        seen.add(record["source_text"])
        rows.append(record)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-category", type=int, default=100)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    if args.per_category < 1:
        parser.error("--per-category deve essere positivo")

    rows = generate(args.per_category, args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="\n") as output:
        for row in rows:
            output.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")

    categories = Counter(row["meta"]["category"] for row in rows)
    print(f"Scritti {len(rows)} record stress validi in {args.out}")
    print("Categorie:", dict(categories))


if __name__ == "__main__":
    main()
