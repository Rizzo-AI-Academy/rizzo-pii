# -*- coding: utf-8 -*-
"""/preview e le sorgenti delle richieste (file caricato, doc_id, immagini).

- /preview tiene il documento e ne da' le proporzioni delle pagine;
- /analyze, /pdf/preview e /ocr accettano il doc_id restituito da /preview, senza un
  secondo upload; un doc_id sconosciuto risponde 410 con code "doc_gone";
- un'immagine caricata diventa un PDF (una pagina) e /preview dice che serve l'OCR;
- /ocr rifiuta i file di testo.

Il modello NER e' finto (come in test_pdf_protetto); l'OCR vero non gira: le
pagine di testo non ne hanno bisogno e /preview non lo esegue mai.
Tutti i valori sono SINTETICI.
"""

import io
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))

from notarial_test_support import APP  # noqa: E402  (stesso caricamento di app coi pesi finti)


def _pdf():
    import fitz
    doc = fitz.open()
    doc.new_page().insert_text((72, 100), "Il sig. Mario Rossi, IBAN IT60X0542811101000000123456.")
    return doc.tobytes()


def _png():
    import fitz
    pix = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 60, 40), False)
    pix.clear_with(255)
    return pix.tobytes("png")


@unittest.skipIf(APP is None, "app.py non importabile (flask/fitz assenti)")
class Sorgenti(unittest.TestCase):
    def setUp(self):
        self._p = patch.object(APP, "nlp", lambda chunks: [[] for _ in chunks])
        self._p.start()
        self.c = APP.app.test_client()
        self.c.environ_base["HTTP_X_PII_TOKEN"] = APP.SESSION_TOKEN

    def tearDown(self):
        self._p.stop()

    def _upload(self, route, data, name):
        return self.c.post(route, data={"file": (io.BytesIO(data), name)},
                           content_type="multipart/form-data")

    def test_preview_poi_doc_id(self):
        r = self._upload("/preview", _pdf(), "atto.pdf")
        self.assertEqual(r.status_code, 200, r.get_json())
        d = r.get_json()
        self.assertEqual(d["kind"], "pdf")
        self.assertEqual(d["n_pages"], 1)
        self.assertEqual(len(d["page_sizes"]), 1)
        self.assertEqual(d["ocr"]["needed"], [])
        self.assertIn("Mario Rossi", d["text"])
        r = self.c.post("/analyze", data={"doc_id": d["doc_id"]})
        self.assertEqual(r.status_code, 200, r.get_json())
        self.assertIn("[IBAN_1]", r.get_json()["anonymized_text"])   # rete regex/checksum
        self.assertEqual(r.get_json()["source_kind"], "pdf")
        r = self.c.post("/pdf/preview", data={"doc_id": d["doc_id"]})
        self.assertEqual(r.status_code, 200, r.get_json())
        self.assertGreaterEqual(r.get_json()["redactions"], 1)

    def test_doc_id_scaduto(self):
        r = self.c.post("/analyze", data={"doc_id": "non-esiste"})
        self.assertEqual(r.status_code, 410)
        self.assertEqual(r.get_json()["code"], "doc_gone")

    def test_immagine_diventa_pdf_da_leggere_con_ocr(self):
        r = self._upload("/preview", _png(), "foto.png")
        self.assertEqual(r.status_code, 200, r.get_json())
        d = r.get_json()
        self.assertEqual(d["kind"], "image")
        self.assertEqual(d["ocr"]["needed"], [1])
        f = self.c.get(f"/doc/{d['doc_id']}/file.pdf")
        self.assertEqual(f.status_code, 403)  # sorgente consultabile, mai output verificato

    def test_ocr_rifiuta_il_testo(self):
        r = self._upload("/ocr", "Mario Rossi".encode(), "nota.txt")
        self.assertEqual(r.status_code, 400)


if __name__ == "__main__":
    unittest.main()
