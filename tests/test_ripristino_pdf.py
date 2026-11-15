# -*- coding: utf-8 -*-
"""Ripristino di un PDF che contiene i placeholder (pdf_export.restore_pdf e POST /restore/pdf).

Caso d'uso: l'LLM produce un PDF con [FULLNAME_1], [IBAN_1], ...; l'utente lo ripristina
con il dizionario della sessione senza perdere l'impaginazione.

Tutti i valori sono SINTETICI.
"""

import io
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "app"))
sys.path.insert(0, str(ROOT / "tests"))

try:
    import fitz
    import pdf_export
except Exception:  # pragma: no cover - dipende dall'ambiente
    fitz = pdf_export = None

MAPPA = {
    "[FULLNAME_1]": "Mario Rossi",
    "[IBAN_1]": "IT60X0542811101000000123456",
    "[CF_1]": "RSSMRA80A01H501U",
}


def _pdf_con_placeholder():
    doc = fitz.open()
    p = doc.new_page()
    p.insert_text((72, 100), "Gentile [FULLNAME_1],", fontsize=11)
    p.insert_text((72, 120), "accrediteremo l'importo sull'IBAN [IBAN_1].", fontsize=11)
    p.insert_text((72, 140), "Codice fiscale: [CF_1]. Riferimento interno [DOCID_9].", fontsize=11)
    p2 = doc.new_page()
    p2.insert_text((72, 100), "Seconda pagina: di nuovo [FULLNAME_1].", fontsize=11)
    return doc.tobytes()


def _testo(pdf):
    with fitz.open(stream=pdf, filetype="pdf") as d:
        return "\n".join(pg.get_text() for pg in d)


@unittest.skipIf(pdf_export is None, "PyMuPDF assente")
class RestorePdf(unittest.TestCase):
    def test_valori_reali_e_niente_placeholder_noti(self):
        out, rep = pdf_export.restore_pdf(_pdf_con_placeholder(), MAPPA)
        txt = _testo(out)
        for ph, val in MAPPA.items():
            self.assertNotIn(ph, txt)
            self.assertIn(val, txt.replace("\n", " "))
        self.assertEqual(rep["occurrences"], 4)
        self.assertEqual(rep["by_placeholder"]["[FULLNAME_1]"], 2)
        self.assertEqual(rep["not_found"], [])

    def test_placeholder_sconosciuto_resta_e_viene_segnalato(self):
        out, rep = pdf_export.restore_pdf(_pdf_con_placeholder(), MAPPA)
        self.assertIn("[DOCID_9]", _testo(out))
        self.assertEqual(rep["unknown"], ["[DOCID_9]"])

    def test_valore_lungo_non_troncato(self):
        out, rep = pdf_export.restore_pdf(_pdf_con_placeholder(), MAPPA)
        self.assertIn("IT60X0542811101000000123456", _testo(out))
        # il resto della riga c'e' ancora e non e' sovrapposto: si legge in ordine
        self.assertIn("RSSMRA80A01H501U. Riferimento interno [DOCID_9].", _testo(out))

    def test_pagine_conservate(self):
        out, _ = pdf_export.restore_pdf(_pdf_con_placeholder(), MAPPA)
        with fitz.open(stream=out, filetype="pdf") as d:
            self.assertEqual(d.page_count, 2)

    def test_dizionario_vuoto_o_invalido(self):
        with self.assertRaises(pdf_export.PdfError):
            pdf_export.restore_pdf(_pdf_con_placeholder(), {})
        with self.assertRaises(pdf_export.PdfError):
            pdf_export.restore_pdf(_pdf_con_placeholder(), {"[X_1]": 3})

    def test_senza_placeholder_restituisce_lo_stesso_file(self):
        doc = fitz.open()
        doc.new_page().insert_text((72, 100), "Nessun dato.")
        data = doc.tobytes()
        out, rep = pdf_export.restore_pdf(data, MAPPA)
        self.assertEqual(out, data)
        self.assertEqual(rep["occurrences"], 0)


try:
    from test_ora import APP  # stesso caricamento di app coi pesi finti
except Exception:  # pragma: no cover
    APP = None


@unittest.skipIf(APP is None, "app.py non importabile (torch/flask/fitz assenti)")
class RestorePdfRoute(unittest.TestCase):
    def _post(self, dati, mapping):
        with patch.object(APP, "nlp", lambda chunks: [[] for _ in chunks]):
            return APP.app.test_client().post(
                "/restore/pdf",
                data={"file": (io.BytesIO(dati), "risposta.pdf"), "mapping": json.dumps(mapping)},
                content_type="multipart/form-data")

    def test_route(self):
        r = self._post(_pdf_con_placeholder(), MAPPA)
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.mimetype, "application/pdf")
        self.assertEqual(r.headers["X-PII-Restored"], "4")
        self.assertIn("Mario Rossi", _testo(r.data))

    def test_route_dizionario_vuoto(self):
        r = self._post(_pdf_con_placeholder(), {})
        self.assertEqual(r.status_code, 400)


if __name__ == "__main__":
    unittest.main()
