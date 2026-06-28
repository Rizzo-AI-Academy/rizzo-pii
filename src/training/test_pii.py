# -*- coding: utf-8 -*-
"""
Test del modello PII addestrato.

Carica il modello da models/rizzo-pii-0.3B (fallback: models/pii_model_legacy) e
individua le entita' sensibili in alcuni esempi (o in un testo passato da riga di
comando), stampando le entita' trovate e una versione anonimizzata con [TIPO].

Uso:
  python src/training/test_pii.py                       # esempi predefiniti
  python src/training/test_pii.py "Mi chiamo Mario..."  # testa un testo tuo
"""

import io
import sys
from pathlib import Path

import torch
from transformers import pipeline

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

# Modello: ULTIMA versione models/rizzo-pii-0.3B-v* (storico versioni); fallback al vecchio
# models/rizzo-pii-0.3B non versionato, poi al legacy. Override puntuale: env PII_MODEL_DIR.
_ROOT = Path(__file__).resolve().parents[2]


def resolve_model_dir(models_dir):
    import os
    import re
    if os.environ.get("PII_MODEL_DIR"):
        return os.environ["PII_MODEL_DIR"]
    versioned = [p for p in models_dir.glob("rizzo-pii-0.3B-v*") if p.is_dir()]
    if versioned:
        def _key(p):
            m = re.search(r"-v([0-9][0-9.]*)$", p.name)
            return tuple(int(x) for x in m.group(1).split(".")) if m else ()
        return str(max(versioned, key=_key))
    base = models_dir / "rizzo-pii-0.3B"
    return str(base if base.exists() else models_dir / "pii_model_legacy")


MODEL_DIR = resolve_model_dir(_ROOT / "models")

# pipeline di token-classification; aggregation_strategy raggruppa i subword
# in entita' intere (unisce B-/I-). device=0 -> GPU, -1 -> CPU.
device = 0 if torch.cuda.is_available() else -1
nlp = pipeline(
    "token-classification",
    model=MODEL_DIR,
    tokenizer=MODEL_DIR,
    aggregation_strategy="simple",
    device=device,
)
print(f"Modello caricato da {MODEL_DIR} | device: {'GPU' if device == 0 else 'CPU'}\n")

EXAMPLES = [
    "Mi chiamo Mario Rossi e la mia email e' mario.rossi@gmail.com, "
    "telefono +39 333 1234567.",
    "Il sottoscritto, nato a Milano il 12/06/1985, residente in Via Garibaldi 24, "
    "chiede la restituzione della somma.",
    "Per il pagamento usare l'IBAN IT60X0542811101000000123456 intestato "
    "alla societa'.",
    "L'avvocato ha depositato la comparsa presso il Tribunale di Roma in data "
    "3 marzo 2024.",
]


def anonymize(text, ents):
    """Sostituisce le entita' con [TIPO], lavorando da destra per non sfasare gli offset."""
    out = text
    for e in sorted(ents, key=lambda x: x["start"], reverse=True):
        out = out[: e["start"]] + f"[{e['entity_group']}]" + out[e["end"]:]
    return out


def run(text):
    ents = nlp(text)
    print("TESTO   :", text)
    if ents:
        print("ENTITA' :")
        for e in ents:
            print(f"   [{e['entity_group']:<14}] '{e['word']}'  (score {e['score']:.2f})")
    else:
        print("ENTITA' : nessuna trovata")
    print("ANONIMO :", anonymize(text, ents))
    print("-" * 80)


texts = [" ".join(sys.argv[1:])] if len(sys.argv) > 1 else EXAMPLES
for t in texts:
    run(t)
