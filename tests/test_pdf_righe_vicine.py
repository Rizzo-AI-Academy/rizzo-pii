# -*- coding: utf-8 -*-
"""`redact_pdf()` non deve cancellare le righe sopra e sotto un'entita' (issue #117).

Il box di rawdict e' piu' alto del corpo: con un'interlinea stretta il
rettangolo di redazione toccava la riga vicina e apply_redactions() ne toglieva i
caratteri, mentre gli header dicevano residui 0. E un nome sillabato a fine riga,
unito "per sovrapposizione" in un solo rettangolo, cancellava righe intere.

Il testo e' quello della issue, con dati SINTETICI.
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

MAPPA = {"[FULLNAME_1]": "Mario Rossi", "[CF_1]": "RSSMRA85H12F205Z",
         "[STREET_1]": "Via Garibaldi", "[CITY_1]": "Milano", "[PROVINCE_1]": "MI",
         "[IBAN_1]": "IT60X0542811101000000123456", "[PIVA_1]": "12345678901"}


def _pdf(righe, font, corpo, interlinea):
    doc = fitz.open()
    page = doc.new_page(width=700, height=400)
    for i, r in enumerate(righe):
        page.insert_text((36, 72 + i * interlinea), r, fontname=font, fontsize=corpo)
    return doc.tobytes()


def _testo(pdf):
    with fitz.open(stream=pdf, filetype="pdf") as doc:
        return doc[0].get_text()


@unittest.skipIf(fitz is None, "PyMuPDF non installato")
class RigheVicine(unittest.TestCase):
    def test_le_parole_non_entita_restano(self):
        righe = ["Il Sig. Mario Rossi, C.F. RSSMRA85H12F205Z, residente in Via Garibaldi 24,",
                 "20121 Milano (MI), IBAN IT60X0542811101000000123456, tel. +39 333 1234567,",
                 "P.IVA 12345678901, proprietario dell'immobile."]
        # cupsfilter (Courier 12, 6 righe per pollice) e l'interlinea singola di Word
        for font, interlinea in (("cour", 12.0), ("helv", 13.8), ("tiro", 13.8)):
            red, rep = pdf_export.redact_pdf(_pdf(righe, font, 12.0, interlinea), MAPPA)
            testo = _testo(red)
            for parola in ("Sig.", "C.F.", "residente", "+39 333 1234567,",
                           "proprietario", "dell'immobile."):
                self.assertIn(parola, testo, f"{font} 12/{interlinea}: persa {parola!r}")
            for valore in MAPPA.values():
                if len(valore) > 2:
                    self.assertNotIn(valore, testo, f"{font}: {valore!r} in chiaro")
            self.assertEqual(rep["residual"], [])

    def test_il_nome_sillabato_non_cancella_le_righe(self):
        righe = ["Il ricorrente dichiara di aver ricevuto la notifica, assistito da Fran-",
                 "cesco Cordella, suo difensore di fiducia presso il foro competente.",
                 "Nessuna altra parte risulta costituita nel presente giudizio."]
        red, rep = pdf_export.redact_pdf(_pdf(righe, "helv", 12.0, 13.8),
                                         {"[FULLNAME_1]": "Francesco Cordella"})
        testo = _testo(red)
        for pezzo in ("Il ricorrente dichiara", "suo difensore di fiducia",
                      "Nessuna altra parte risulta costituita"):
            self.assertIn(pezzo, testo)
        self.assertNotIn("Cordella", testo)
        self.assertEqual(rep["occurrences"], 1)

    def test_scansione_con_ocr_l_immagine_della_pii_sparisce(self):
        """Sopra una scansione l'OCR stende testo invisibile: la PII e' nell'immagine, e
        apply_redactions() ne cancella i pixel solo sotto il rettangolo, che qui deve
        restare intero anche a interlinea stretta."""
        righe = ["Il Sig. Mario Rossi, C.F. RSSMRA85H12F205Z, residente in Via Garibaldi 24,",
                 "20121 Milano (MI), IBAN IT60X0542811101000000123456, tel. +39 333 1234567,",
                 "P.IVA 12345678901, proprietario dell'immobile."]

        def pagina(testo, invisibile=False, sfondo=None):
            doc = fitz.open()
            page = doc.new_page(width=700, height=400)
            if sfondo is not None:
                page.insert_image(page.rect, pixmap=sfondo)
            for i, r in enumerate(testo):
                page.insert_text((36, 72 + i * 10.8), r, fontname="helv", fontsize=12,
                                 render_mode=3 if invisibile else 0)
            return doc

        def neri(pix):
            s, n = pix.samples, pix.n              # samples e' una copia: leggerlo una volta
            return {i for i in range(pix.width * pix.height) if max(s[i * n:i * n + 3]) < 60}

        scansione = pagina(righe)[0].get_pixmap(dpi=100)
        pdf = pagina(righe, invisibile=True, sfondo=scansione).tobytes()
        solo_pii = fitz.open()                     # i soli valori, ciascuno al suo posto
        p = solo_pii.new_page(width=700, height=400)
        for i, r in enumerate(righe):
            for v in MAPPA.values():
                j = r.find(v)
                if j >= 0 and len(v) > 2:
                    x = 36 + fitz.get_text_length(r[:j], fontname="helv", fontsize=12)
                    p.insert_text((x, 72 + i * 10.8), v, fontname="helv", fontsize=12)
        maschera = neri(p.get_pixmap(dpi=100))
        red, _ = pdf_export.redact_pdf(pdf, MAPPA)
        with fitz.open(stream=red, filetype="pdf") as doc:
            rimasti = maschera & neri(doc[0].get_pixmap(dpi=100))
        self.assertEqual(len(rimasti), 0, f"{len(rimasti)} pixel della PII ancora visibili")

    def test_il_valore_contenuto_non_si_conta_due_volte(self):
        """"Rossi" dentro "Mario Rossi" e' gia' redatto: il rettangolo piu' corto puo'
        uscire piu' alto (sopra "Rossi" non c'e' niente da evitare), non per questo e'
        una redazione nuova."""
        righe = ["Visto", "Mario Rossi dichiara quanto segue, e Rossi conferma."]
        red, rep = pdf_export.redact_pdf(
            _pdf(righe, "helv", 12.0, 13.8),
            {"[FULLNAME_1]": "Mario Rossi", "[FULLNAME_2]": "Rossi"})
        self.assertIn("Visto", _testo(red))
        self.assertEqual(rep["by_placeholder"], {"[FULLNAME_1]": 1, "[FULLNAME_2]": 1})


if __name__ == "__main__":
    unittest.main()
