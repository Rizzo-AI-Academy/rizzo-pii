# -*- coding: utf-8 -*-
"""`POST /analyze`: il testo originale torna SOLO col dizionario attivo (issue #118).

In modalita' definitiva (`include_mapping: false`) la risposta non deve contenere
nessun valore in chiaro: niente `mapping`, niente testo nei segmenti, e niente
`source_text` - che da solo restituiva l'intero documento originale.

Tutti i valori sono SINTETICI.
"""

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))

from test_ora import APP  # noqa: E402  (stesso caricamento di app coi pesi finti)

TESTO = "Mario Rossi, IBAN IT60X0542811101000000123456"


@unittest.skipIf(APP is None, "app.py non importabile (torch/flask/fitz assenti)")
class SourceText(unittest.TestCase):
    def _post(self, **extra):
        with patch.object(APP, "nlp", lambda chunks: [[] for _ in chunks]):
            r = APP.app.test_client().post("/analyze", json={"text": TESTO, **extra})
        self.assertEqual(r.status_code, 200)
        return r.get_json()

    def test_definitiva_senza_testo_originale(self):
        out = self._post(include_mapping=False)
        self.assertNotIn("source_text", out)
        self.assertEqual(out["mapping"], {})
        self.assertNotIn("IT60X0542811101000000123456", str(out))

    def test_reversibile_col_testo_originale(self):
        out = self._post(include_mapping=True)
        self.assertEqual(out["source_text"], TESTO)
        self.assertIn("[IBAN_1]", out["mapping"])


if __name__ == "__main__":
    unittest.main()
