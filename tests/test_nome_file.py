# -*- coding: utf-8 -*-
"""Il nome del PDF anonimizzato non deve riportare i dati che il documento nasconde.

Issue #97: il Garante ha contestato a una scuola un provvedimento anonimizzato il cui
nome file conteneva ancora il cognome dell'alunno. Qui "Rossi_Mario_sospensione.pdf"
usciva "Rossi_Mario_sospensione_anonimizzato.pdf" da tutte e tre le vie di download.

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

NOME = "Rossi_Mario_sospensione.pdf"
ATTESO = "FULLNAME_1_sospensione_anonimizzato.pdf"


def _nlp(chunks):
    """Modello finto: marca "Mario Rossi" come FULLNAME."""
    out = []
    for c in chunks:
        i = c.find("Mario Rossi")
        out.append([] if i < 0 else [{"entity_group": "FULLNAME", "start": i,
                                      "end": i + 11, "score": 0.99, "word": "Mario Rossi"}])
    return out


def _pdf():
    import fitz
    doc = fitz.open()
    doc.new_page().insert_text((72, 100), "Provvedimento di sospensione dell'alunno "
                                          "Mario Rossi, classe terza.", fontsize=11)
    return doc.tobytes()


@unittest.skipIf(APP is None, "app.py non importabile (torch/flask/fitz assenti)")
class NomeFile(unittest.TestCase):
    def _post(self, route):
        with patch.object(APP, "nlp", _nlp):
            return APP.app.test_client().post(
                route, data={"file": (io.BytesIO(_pdf()), NOME)},
                content_type="multipart/form-data")

    def test_download_diretto(self):
        r = self._post("/pdf")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.headers["Content-Disposition"], f"attachment; filename={ATTESO}")

    def test_anteprima_e_suo_download(self):
        d = self._post("/pdf/preview").get_json()
        self.assertEqual(d["filename"], ATTESO)
        r = APP.app.test_client().get(f"/doc/{d['doc_id']}/file.pdf")
        self.assertEqual(r.headers["Content-Disposition"], f"attachment; filename={ATTESO}")

    def test_ordine_maiuscole_accenti(self):
        m = {"[FULLNAME_1]": "Niccolò Rossi", "[CF_1]": "RSSNCC85H12F205X",
             "[FULLNAME_2]": "Fran-\ncesco Cordella", "[FULLNAME_3]": "Hans Müller"}
        for nome, atteso in (("ROSSI_NICCOLO.pdf", "FULLNAME_1"),
                             ("Niccolò Rossi ricorso.pdf", "FULLNAME_1 ricorso"),
                             ("RossiNiccolo.pdf", "FULLNAME_1"),
                             ("Rossi2024.pdf", "FULLNAME_1_2024"),
                             ("Cordella_Francesco.pdf", "FULLNAME_2"),
                             ("Müller_Hans.pdf", "FULLNAME_3"),     # accento scomposto (macOS)
                             ("RSSNCC85H12F205X.pdf", "CF_1"),
                             ("Rossiello.pdf", "Rossiello"),              # non e' "Rossi"
                             ("grossi.pdf", "grossi"),
                             ("", "documento")):
            self.assertEqual(APP._nome_anonimizzato(nome, m), f"{atteso}_anonimizzato.pdf")


if __name__ == "__main__":
    unittest.main()
