# -*- coding: utf-8 -*-
"""`TIME`: la regex dell'ora, e il motivo per cui sta in `SOFT_REGEX_LABELS`.

Il modello taglia i minuti (`09:50` -> `09:`) e la metà che resta finisce in chiaro
accanto al segnaposto: da qui la regex. Ma una data **include** spesso l'ora, e in un
timestamp ISO (`2026-03-15T10:30:00`) la data la trova solo il modello, in un'unica span.
Se `TIME` avesse la priorità della rete regex scalzerebbe quella span in fusione e
lascerebbe `2026-03-` **in chiaro**: la data del documento, sotto un placeholder che dice
il contrario. Da soft perde contro il modello e vince solo dove nessuno reclama l'ora.

Due punti fra due numeri non sono comunque una prova, e i falsi positivi qui sotto sono
dichiarati, non nascosti: per un anonimizzatore mascherare in più è l'errore reversibile.

Tutti i valori sono SINTETICI.
"""

import os
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "app"))

import detectors  # noqa: E402


def ore(testo):
    return [testo[e["start"]:e["end"]]
            for e in detectors.detect_regex(testo) if e["label"] == "TIME"]


def _carica_app():
    """Importa app coi pesi finti: qui il modello vero non serve.

    Tre accorgimenti, tutti necessari:
    - il finto transformers va messo in sys.modules, non con patch("transformers.
      pipeline"): un `from transformers import pipeline` (app.py riga 66) su un
      _LazyModule NON passa dall'attributo patchato e prende la funzione vera;
    - la voce si sostituisce e ripristina A MANO: patch.dict(sys.modules) al
      ripristino fa clear()+update() dell'intero sys.modules, buttando via anche
      l'app appena importata (e ricaricando numpy);
    - PII_MODEL_DIR (l'override che app.py stesso documenta) puntato a una cartella
      esistente: senza, il controllo del modello mancante fa sys.exit(2) durante
      l'import - un SystemExit che l'except qui sotto non cattura.
    """
    if "app" in sys.modules:
        return sys.modules["app"]
    finto = MagicMock()
    finto.pipeline.return_value = MagicMock(return_value=[])
    vero = sys.modules.get("transformers")
    sys.modules["transformers"] = finto
    prima = os.environ.get("PII_MODEL_DIR")
    os.environ["PII_MODEL_DIR"] = str(ROOT / "tests")
    try:
        import app as pii_app
    finally:
        if vero is not None:
            sys.modules["transformers"] = vero
        else:
            del sys.modules["transformers"]
        if prima is not None:
            os.environ["PII_MODEL_DIR"] = prima
        else:
            del os.environ["PII_MODEL_DIR"]
    return pii_app


try:                                    # torch/transformers/flask/fitz presenti?
    APP = _carica_app()
except Exception:                       # noqa: BLE001 - qualunque import mancante
    APP = None


class Ora(unittest.TestCase):
    def test_riconosce_la_forma_coi_due_punti(self):
        self.assertEqual(ore("Ingresso ore 18:28, uscita 19:30."), ["18:28", "19:30"])
        self.assertEqual(ore("Accesso alle 09:50:12."), ["09:50:12"])
        self.assertEqual(ore("Riunione dalle 8:00 alle 16:45."), ["8:00", "16:45"])

    def test_il_punto_non_e_accettato(self):
        """`10.30` ha la stessa forma di `versione 1.30` ed `euro 10.30`."""
        for frase in ("versione 1.30 del programma", "euro 10.30 di spesa",
                      "capitolo 3.15", "coordinate 45.4642, 9.1900"):
            self.assertEqual(ore(frase), [], frase)

    def test_ore_e_minuti_fuori_scala(self):
        for frase in ("codice 24:00", "codice 25:30", "codice 10:60", "codice 99:99"):
            self.assertEqual(ore(frase), [], frase)

    def test_time_e_soft(self):
        """La guardia che impedisce all'ora di scalzare una data del modello."""
        self.assertIn("TIME", detectors.SOFT_REGEX_LABELS)


@unittest.skipIf(APP is None, "app.py non importabile (torch/flask/fitz assenti)")
class OraControDataInFusione(unittest.TestCase):
    """Il caso che conta: la data la vede solo il modello, l'ora anche la regex."""

    def test_la_time_non_scalza_la_date_del_modello(self):
        testo = "Deposito telematico: 2026-03-15T10:30:00 (ricevuta PEC)."
        i = testo.index("2026")
        data = {"label": "DATE", "start": i, "end": i + len("2026-03-15T10:30:00"),
                "score": 1.0, "validated": False, "source": "modello"}
        tenute = APP._merge([data] + detectors.detect_regex(testo), testo)
        span = [(e["label"], testo[e["start"]:e["end"]]) for e in tenute]
        self.assertIn(("DATE", "2026-03-15T10:30:00"), span,
                      "la data del modello e' stata spezzata dall'ora: %r" % span)
        self.assertNotIn("TIME", [l for l, _ in span])


if __name__ == "__main__":
    unittest.main()
