# -*- coding: utf-8 -*-
"""`hard_split()` non deve mai disegnare oltre il bordo della pagina.

La finestra `cap` serve a non misurare tutta la coda a ogni giro, ma va calibrata sul
glifo PIU' STRETTO del repertorio: con "l" come riferimento una parola di apostrofi (che
in helv sono piu' stretti) sta dentro la finestra, il ciclo esce subito e accoda tutta la
coda su una riga sola. La riga finisce fuori pagina e nel PDF quei caratteri SPARISCONO,
senza nessun errore: il documento scaricato sembra completo e non lo e'.
"""

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "app"))

try:
    import fitz
    import pdf_export
except ImportError:  # PyMuPDF assente: e' una dipendenza dell'app, non del training
    fitz = None


@unittest.skipIf(fitz is None, "PyMuPDF non installato")
class HardSplit(unittest.TestCase):
    def pagine(self, testo):
        return fitz.open("pdf", pdf_export.text_to_pdf(testo))

    def test_nessun_carattere_perde_la_pagina(self):
        # l'apostrofo e' il glifo piu' stretto di helv in latin-1
        doc = self.pagine("'" * 300)
        letti = "".join(p.get_text() for p in doc).count("'")
        doc.close()
        self.assertEqual(letti, 300, "caratteri persi fuori pagina")

    def test_nessuna_riga_sfora_il_bordo(self):
        for parola in ("'" * 300, "'ijl" * 225, "l" * 300):
            doc = self.pagine("Verbale.\n" + parola + "\nFirmato.")
            fuori = [l["bbox"][2] for p in doc
                     for b in p.get_text("rawdict")["blocks"]
                     for l in b.get("lines", []) if l["bbox"][2] > p.rect.width + 0.5]
            doc.close()
            self.assertFalse(fuori, "righe oltre il bordo pagina: %r" % fuori)


if __name__ == "__main__":
    unittest.main()
