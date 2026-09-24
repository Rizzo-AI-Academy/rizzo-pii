# -*- coding: utf-8 -*-
"""Valori lasciati in chiaro dall'utente (`keep_values`).

Il modello a volte anonimizza qualcosa che non e' un dato personale ("Tribunale di
Milano" preso per un'ORG): dall'UI lo si toglie dall'anonimizzazione con un clic.
Qui si verifica il contratto lato server, quello che usa anche il PDF censurato:

- ogni occorrenza del valore (confronto su _norm: maiuscole e spazi non contano)
  resta in chiaro e sparisce dal dizionario;
- i placeholder rimasti hanno gli STESSI numeri che avrebbero senza keep_values
  (l'UI applica il filtro lato client sul risultato completo: il PDF deve coincidere);
- un valore diverso, anche se contiene quello tenuto, resta anonimizzato.

Come test_tsv_anonymize, importa `app.py` con la pipeline HF finta: senza
torch/transformers/flask/fitz il test si salta. Nomi SINTETICI.
"""

import re
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "app"))


def _carica_app():
    if "app" in sys.modules:
        return sys.modules["app"]
    with patch("transformers.pipeline", return_value=MagicMock(return_value=[])):
        import app as pii_app
    return pii_app


try:
    APP = _carica_app()
except Exception:                       # noqa: BLE001 - qualunque import mancante
    APP = None

TESTO = ("Mario Rossi ha citato Luca Bianchi davanti al Tribunale di Milano. "
         "Il tribunale di  milano ha fissato l'udienza; Mario Rossi era presente.")


def _entita(text):
    ents = []
    for label, pattern in (("FULLNAME", r"Mario Rossi|Luca Bianchi"),
                           ("ORG", r"(?i)tribunale di\s+milano")):
        for m in re.finditer(pattern, text):
            ents.append({"label": label, "start": m.start(), "end": m.end(),
                         "score": 0.9, "validated": False, "source": "modello"})
    return ents


@unittest.skipIf(APP is None, "serve l'ambiente dell'app (torch/transformers/flask/fitz)")
class KeepValues(unittest.TestCase):

    def _analyze(self, **kw):
        with patch.object(APP, "detect_model", return_value=(_entita(TESTO), 1)), \
             patch.object(APP, "detect_regex", return_value=[]):
            return APP.analyze(TESTO, **kw)

    def test_valore_tenuto_resta_in_chiaro_ovunque(self):
        r = self._analyze(keep_values=["tribunale di milano"])
        self.assertIn("Tribunale di Milano", r["anonymized_text"])
        self.assertIn("tribunale di  milano", r["anonymized_text"])
        self.assertNotIn("[ORG_", r["anonymized_text"])
        self.assertNotIn("Tribunale di Milano", r["mapping"].values())
        self.assertEqual(r["n_kept"], 2)
        self.assertNotIn("ORG", r["by_label"])

    def test_numerazione_invariata(self):
        base = self._analyze()
        r = self._analyze(keep_values=["Mario Rossi"])
        self.assertEqual(base["mapping"]["[FULLNAME_2]"], "Luca Bianchi")
        self.assertEqual(r["mapping"], {"[FULLNAME_2]": "Luca Bianchi",
                                        "[ORG_1]": "Tribunale di Milano"})
        self.assertEqual(r["n_unique"], 2)
        self.assertEqual(r["n_entities"], 3)

    def test_valore_diverso_resta_anonimizzato(self):
        r = self._analyze(keep_values=["Rossi", "Milano"])
        self.assertEqual(r["n_kept"], 0)
        self.assertNotIn("Mario Rossi", r["anonymized_text"])

    def test_senza_dizionario(self):
        r = self._analyze(keep_values=["Luca Bianchi"], mapping_enabled=False)
        self.assertIn("Luca Bianchi", r["anonymized_text"])
        self.assertEqual(r["mapping"], {})
        self.assertTrue(all("t" not in s for s in r["segments"] if "label" in s))

    def test_parse_keep_values(self):
        self.assertEqual(APP.parse_keep_values(None), [])
        self.assertEqual(APP.parse_keep_values('["Via Roma, 10"]'), ["Via Roma, 10"])
        self.assertEqual(APP.parse_keep_values(["a"]), ["a"])
        for bad in ("Via Roma", '{"a": 1}', [1], "[1]"):
            with self.assertRaises(ValueError):
                APP.parse_keep_values(bad)


if __name__ == "__main__":
    unittest.main()
