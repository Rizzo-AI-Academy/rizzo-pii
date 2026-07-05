#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Genera esempi sintetici italiani mirati alla disambiguazione di TARGA.

Il dataset contiene tre famiglie:
  * positive: una targa in contesto automobilistico;
  * hard_negative: un codice con forma da targa usato con un altro significato;
  * mixed_pii: una targa insieme ad altre PII.

L'output segue docs/FORMATO_DATI.md e viene validato prima della scrittura.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "data_pipeline"))

import generate_synthetic_pii as base  # noqa: E402


GENERATOR_VERSION = "targa-it-1.0.0"
DEFAULT_SEED = 20260704
DEFAULT_OUT = ROOT / "dataset" / "synthetic" / "synthetic_targa_it_3k.jsonl"

POSITIVE_TEMPLATES = [
    "Il veicolo con targa {TARGA} è stato sottoposto a sequestro.",
    "Dal verbale risulta il transito dell'autovettura targata {TARGA}.",
    "La copertura assicurativa del mezzo {TARGA} scade il prossimo mese.",
    "Nel sinistro è rimasto coinvolto il veicolo identificato dalla targa {TARGA}.",
    "Targa del veicolo: {TARGA}.",
    "Il contratto di leasing riguarda l'automobile {TARGA}.",
    "La Polizia Locale ha identificato il mezzo targato {TARGA}.",
    "Il fermo amministrativo è stato iscritto sul veicolo {TARGA}.",
    "Nel registro PRA il mezzo risulta associato alla targa {TARGA}.",
    "L'officina ha preso in consegna l'autovettura {TARGA} per la riparazione.",
    "Si richiede la restituzione del veicolo recante targa {TARGA}.",
    "Il conducente viaggiava a bordo del mezzo con targa {TARGA}.",
    "Dati identificativi del veicolo — {TARGA}.",
    "La fotografia allegata mostra chiaramente la targa {TARGA}.",
    "Il bene pignorato è l'autoveicolo targato {TARGA}.",
    "Automezzo aziendale {TARGA}, immatricolato in Italia.",
    "La targa rilevata dal dispositivo elettronico è {TARGA}.",
    "Per il veicolo {TARGA} è stata presentata denuncia di furto.",
    "Il passaggio di proprietà interessa la vettura {TARGA}.",
    "Mezzo sostitutivo consegnato: targa {TARGA}.",
]

NEGATIVE_O_TEMPLATES = [
    "Il codice dell'ordine è {CODE}.",
    "Usare la password temporanea {CODE} al primo accesso.",
    "Il ticket di assistenza {CODE} è stato chiuso.",
    "Il coupon promozionale {CODE} non è più valido.",
    "La spedizione è identificata dal codice {CODE}.",
    "Il lotto di produzione {CODE} ha superato il collaudo.",
    "Inserire il codice sessione {CODE} per continuare.",
    "La richiesta interna {CODE} è in attesa di approvazione.",
    "Il codice articolo {CODE} non è disponibile a magazzino.",
    "La chiave di recupero {CODE} scade tra dieci minuti.",
    "Il riferimento del pagamento è {CODE}.",
    "La prenotazione {CODE} è stata annullata.",
]

NEGATIVE_DOCID_TEMPLATES = [
    "La pratica amministrativa {CODE} è stata archiviata.",
    "Il fascicolo interno {CODE} è assegnato all'ufficio competente.",
    "Il protocollo {CODE} è richiamato nella comunicazione.",
    "Il procedimento identificato come {CODE} risulta ancora pendente.",
]

MIXED_TEMPLATES = [
    "Il veicolo con targa {TARGA} è intestato a {FULLNAME}.",
    "L'automezzo {TARGA} appartiene a {ORG}, P. IVA {PIVA}.",
    "Il mezzo targato {TARGA} è custodito presso {ADDRESS}.",
    "{FULLNAME} ha denunciato il furto dell'autovettura {TARGA}.",
    "La società {ORG} utilizza il veicolo aziendale {TARGA}.",
    "Per il veicolo {TARGA} contattare {EMAIL} o il numero {PHONE}.",
    "Il contratto tra {FULLNAME} e {ORG} riguarda il mezzo {TARGA}.",
    "La fattura di {ORG}, P. IVA {PIVA}, riporta la targa {TARGA}.",
    "Il veicolo {TARGA}, intestato a {FULLNAME}, risulta domiciliato in {ADDRESS}.",
    "{ORG} ha consegnato a {FULLNAME} l'automezzo targato {TARGA}.",
    "Nel verbale relativo a {TARGA} compare il nominativo {FULLNAME}.",
    "La richiesta inviata da {EMAIL} riguarda il veicolo {TARGA}.",
]


def plate_code() -> str:
    letters = base.PLATE_LETTERS
    return (
        "".join(random.choice(letters) for _ in range(2))
        + f"{random.randint(100, 999)}"
        + "".join(random.choice(letters) for _ in range(2))
    )


def plate_value() -> str:
    code = plate_code()
    style = random.choices(("compact", "spaces", "hyphens"), (50, 35, 15), k=1)[0]
    if style == "spaces":
        return f"{code[:2]} {code[2:5]} {code[5:]}"
    if style == "hyphens":
        return f"{code[:2]}-{code[2:5]}-{code[5:]}"
    return code


def full_name_piece():
    given, surname, _ = base._person()
    return [(f"{given} {surname}", "FULLNAME")]


def piva_ok(value: str) -> bool:
    digits = "".join(character for character in value if character.isdigit())
    if len(digits) != 11:
        return False
    total = 0
    for index, character in enumerate(digits[:10]):
        number = int(character)
        if index % 2:
            number *= 2
            if number > 9:
                number -= 9
        total += number
    return (10 - total % 10) % 10 == int(digits[-1])


def render(template: str, slots: dict[str, list[tuple[str, str | None]]]):
    text = ""
    entities = []
    for part in re.split(r"(\{\w+\})", template):
        match = base.SLOT_RE.fullmatch(part)
        if not match:
            text += part
            continue
        for value, label in slots[match.group(1)]:
            start = len(text)
            text += value
            if label:
                entities.append(
                    {"value": value, "label": label, "start": start, "end": len(text)}
                )
    return text, entities


def mixed_slots():
    return {
        "TARGA": [(plate_value(), "TARGA")],
        "FULLNAME": full_name_piece(),
        "ORG": base.org_piece(),
        "PIVA": base.piva_piece(),
        "ADDRESS": base.address(),
        "EMAIL": base.email_piece(),
        "PHONE": base.phone_piece(),
    }


def make_record(kind: str, index: int, seed: int):
    if kind == "positive":
        template = random.choice(POSITIVE_TEMPLATES)
        text, entities = render(template, {"TARGA": [(plate_value(), "TARGA")]})
    elif kind == "hard_negative":
        code = plate_code()
        if random.random() < 0.25:
            template = random.choice(NEGATIVE_DOCID_TEMPLATES)
            label = "DOCID"
        else:
            template = random.choice(NEGATIVE_O_TEMPLATES)
            label = None
        text, entities = render(template, {"CODE": [(code, label)]})
    elif kind == "mixed_pii":
        template = random.choice(MIXED_TEMPLATES)
        text, entities = render(template, mixed_slots())
    else:
        raise ValueError(f"categoria sconosciuta: {kind}")

    tokens, bio = base.to_bio(text, entities)
    return {
        "source_text": text,
        "language": "it",
        "tokens": tokens,
        "bio_labels": bio,
        "entities": entities,
        "meta": {
            "contributor": "rizzoaiacademy",
            "seed": seed,
            "generator_version": GENERATOR_VERSION,
            "synthetic": True,
            "category": kind,
            "row_index": index,
        },
    }


def validate_record(record: dict) -> None:
    text = record["source_text"]
    expected_tokens = base.TOKEN_RE.findall(text)
    if record["tokens"] != expected_tokens:
        raise ValueError("tokenizzazione non coerente")
    if len(record["tokens"]) != len(record["bio_labels"]):
        raise ValueError("tokens e bio_labels hanno lunghezze diverse")
    for entity in record["entities"]:
        if text[entity["start"] : entity["end"]] != entity["value"]:
            raise ValueError(f"offset non valido: {entity}")
    expected_bio = base.to_bio(text, record["entities"])[1]
    if record["bio_labels"] != expected_bio:
        raise ValueError("label BIO non coerenti con le entità")
    for entity in record["entities"]:
        if entity["label"] == "PIVA" and not piva_ok(entity["value"]):
            raise ValueError(f"PIVA non valida: {entity['value']}")


def generate(n: int, seed: int):
    random.seed(seed)
    counts = {
        "positive": int(n * 0.40),
        "hard_negative": int(n * 0.30),
    }
    counts["mixed_pii"] = n - sum(counts.values())
    kinds = [kind for kind, count in counts.items() for _ in range(count)]
    random.shuffle(kinds)

    rows = []
    seen = set()
    attempts = 0
    while len(rows) < n:
        attempts += 1
        if attempts > n * 20:
            raise RuntimeError("troppi duplicati durante la generazione")
        kind = kinds[len(rows)]
        record = make_record(kind, len(rows), seed)
        if record["source_text"] in seen:
            continue
        validate_record(record)
        seen.add(record["source_text"])
        rows.append(record)
    return rows


def summarize(rows):
    categories = Counter(row["meta"]["category"] for row in rows)
    labels = Counter(
        entity["label"] for row in rows for entity in row["entities"]
    )
    return categories, labels


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-n", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    if args.n < 1:
        parser.error("-n deve essere positivo")

    rows = generate(args.n, args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="\n") as output:
        for row in rows:
            output.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")

    categories, labels = summarize(rows)
    print(f"Scritti {len(rows)} record validi in {args.out}")
    print("Categorie:", dict(categories))
    print("Entità:", dict(labels))


if __name__ == "__main__":
    main()
