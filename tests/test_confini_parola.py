#!/usr/bin/env python3
"""
Test di regressione: nessun segnaposto deve restare attaccato a una parola.

Il difetto (issue #11): il modello etichetta i sotto-token e a volte ne copre solo
una parte ("No" di "Novara"). Sostituendo la span cosi' com'e', l'anonymized_text
conservava frammenti leggibili -- "Direzione Provinciale di [CITY_2]vara" -- da cui
il valore originale e' banalmente ricostruibile.

L'asserzione e' volutamente **indipendente dal richiamo del modello**: non pretende
che una certa entita' venga riconosciuta, ma che *quando* viene riconosciuta la
sostituzione copra parole intere. Cosi' il test non diventa rosso se una versione
futura del modello riconosce piu' o meno entita' di oggi.

Richiede il backend avviato (`python src/app/serve.py`).

Uso:
    python tests/test_confini_parola.py
    python tests/test_confini_parola.py --endpoint http://127.0.0.1:5005/analyze
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

DEFAULT_ENDPOINT = "http://127.0.0.1:5005/analyze"
FIXTURES = Path(__file__).with_name("fixtures_economico_giuridiche.jsonl")

# Un segnaposto incollato a un carattere di parola, da un lato o dall'altro:
# "[CITY_2]vara", "Fin[ORG_2]". E' esattamente il sintomo del mascheramento parziale.
ADIACENZA = re.compile(r"\[[A-Z_]+_\d+\]\w|\w\[[A-Z_]+_\d+\]")


def analizza(endpoint: str, testo: str, timeout: int = 180) -> dict:
    payload = json.dumps({"text": testo}).encode("utf-8")
    req = urllib.request.Request(endpoint, data=payload,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def contesto(testo: str, m: re.Match) -> str:
    a, b = max(0, m.start() - 30), min(len(testo), m.end() + 30)
    return "..." + testo[a:b].replace("\n", " ") + "..."


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    parser.add_argument("--fixtures", type=Path, default=FIXTURES)
    args = parser.parse_args()

    righe = [json.loads(l) for l in args.fixtures.read_text(encoding="utf-8").splitlines() if l.strip()]
    print(f"Fixture: {args.fixtures.name} ({len(righe)} documenti sintetici)")

    fallimenti = 0
    for n, riga in enumerate(righe, start=1):
        try:
            risposta = analizza(args.endpoint, riga["source_text"])
        except urllib.error.URLError as exc:
            print(f"Backend non raggiungibile su {args.endpoint}: {exc}", file=sys.stderr)
            print("Avviare `python src/app/serve.py` prima del test.", file=sys.stderr)
            return 2

        anonimo = risposta["anonymized_text"]
        difetti = list(ADIACENZA.finditer(anonimo))

        # informativo: quante delle entita' annotate sono state effettivamente coperte
        coperte = sum(1 for e in riga["entities"] if e["value"] not in anonimo)

        stato = "OK  " if not difetti else "FAIL"
        print(f"  [{stato}] doc {n}: {coperte}/{len(riga['entities'])} entita' annotate mascherate")
        for m in difetti:
            fallimenti += 1
            print(f"          segnaposto attaccato a una parola: {contesto(anonimo, m)}")

    if fallimenti:
        print(f"\nFALLITO: {fallimenti} segnaposto parziali.")
        return 1

    print("\nOK: nessun segnaposto parziale.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
