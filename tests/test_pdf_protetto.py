# -*- coding: utf-8 -*-
"""Un PDF protetto da password deve dirlo, non rispondere "document closed or encrypted".

pdf_export aveva gia' il messaggio chiaro, ma l'estrazione del testo fallisce prima e
tutte le vie (/preview, /analyze, /pdf, /pdf/preview) rimandavano l'errore grezzo di
PyMuPDF. Un PDF con la sola password del proprietario invece si apre, e deve restare cosi'.

Tutti i valori sono SINTETICI.
"""

import io
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))

from test_ora import APP  # noqa: E402  (stesso caricamento di app coi pesi finti)

MSG = "PDF protetto da password: rimuovi la protezione e riprova."


def _pdf(**cifratura):
    import fitz
    doc = fitz.open()
    doc.new_page().insert_text((72, 100), "Il Sig. Mario Rossi, residente a Milano.")
    return doc.tobytes(encryption=fitz.PDF_ENCRYPT_AES_256, **cifratura)


@unittest.skipIf(APP is None, "app.py non importabile (torch/flask/fitz assenti)")
class PdfProtetto(unittest.TestCase):
    def _post(self, route, dati):
        with patch.object(APP, "nlp", lambda chunks: [[] for _ in chunks]):
            return APP.app.test_client().post(
                route, data={"file": (io.BytesIO(dati), "atto.pdf")},
                content_type="multipart/form-data")

    def test_con_password_di_apertura_lo_dice(self):
        dati = _pdf(user_pw="apertura", owner_pw="proprietario")
        for route in ("/preview", "/analyze", "/pdf", "/pdf/preview"):
            r = self._post(route, dati)
            self.assertEqual(r.status_code, 400, route)
            self.assertIn(MSG, r.get_json()["error"], route)

    def test_con_la_sola_password_del_proprietario_si_legge(self):
        r = self._post("/analyze", _pdf(user_pw="", owner_pw="proprietario", permissions=0))
        self.assertEqual(r.status_code, 200)
        self.assertIn("Mario Rossi", r.get_json()["source_text"])


if __name__ == "__main__":
    unittest.main()
