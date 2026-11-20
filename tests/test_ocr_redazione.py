#!/usr/bin/env python3
"""OCR delle scansioni: le parti che non hanno bisogno dei modelli.

- `pdf_export.redact_pdf(..., ocr_pages=...)`: su una pagina fatta solo di
  un'immagine i PIXEL della PII vanno sovrascritti, l'immagine originale deve
  sparire dal file, il placeholder deve comparire come testo. La pagina OCR qui
  e' finta (righe e box per carattere scritti a mano): si verifica la redazione,
  non il riconoscimento.
- `pdf_text.collect_readable_text(page_text=...)`: il testo OCR prende il posto
  del layer vuoto della pagina scansionata.
- `server_config.parse_ocr_mode/parse_ocr_preset`.
- `docmd.text._char_spans`: dai passi CTC agli intervalli dei caratteri.

Senza fitz/numpy (la CI gira sulla sola libreria standard) le prove che li usano
vengono saltate, come per gli altri test sui PDF.
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "app"))

import pdf_text  # noqa: E402
import server_config  # noqa: E402

try:
    import fitz  # noqa: F401
    import numpy as np
    import pdf_export
except Exception:          # pragma: no cover - CI senza dipendenze
    fitz = np = pdf_export = None

try:                       # docmd importa opencv: senza, le prove CTC si saltano
    from docmd.text import _char_spans
except Exception:          # pragma: no cover
    _char_spans = None


def _scanned_pdf(w=400, h=200):
    """PDF di una pagina w x h punti con dentro SOLO un'immagine grigio chiaro."""
    img = np.full((h * 2, w * 2, 3), 230, dtype=np.uint8)       # 144 dpi
    pix = fitz.Pixmap(fitz.csRGB, w * 2, h * 2, img.tobytes(), 0)
    doc = fitz.open()
    page = doc.new_page(width=w, height=h)
    page.insert_image(page.rect, stream=pix.tobytes("png"))
    return doc.tobytes()


def _line(text, x0, y0, char_w=10.0, h=14.0):
    chars = [[x0 + i * char_w, y0, x0 + (i + 1) * char_w, y0 + h] for i in range(len(text))]
    return {"text": text, "score": 0.99, "box": [x0, y0, x0 + len(text) * char_w, y0 + h],
            "chars": chars}


@unittest.skipIf(pdf_export is None, "fitz/numpy non installati")
class RedazioneScansione(unittest.TestCase):
    def setUp(self):
        self.pdf = _scanned_pdf()
        # "Il sig. Mario Rossi abita qui" su una riga, il CF sulla seguente
        self.ocr = {0: {"page": 0, "dpi": 144, "angle": 0.0, "seals": [[300, 150, 380, 190]],
                        "lines": [_line("Il sig. Mario Rossi abita qui", 20, 30),
                                  _line("CF RSSMRA85H12F839X", 20, 60)]}}
        self.mapping = {"[FULLNAME_1]": "Mario Rossi", "[CF_1]": "RSSMRA85H12F839X"}

    def test_pixel_coperti_e_immagine_originale_sparita(self):
        out, rep = pdf_export.redact_pdf(self.pdf, self.mapping, ocr_pages=self.ocr)
        self.assertEqual(rep["occurrences"], 2)
        self.assertEqual(rep["ocr_pages"], [0])
        self.assertEqual(rep["not_found"], [])
        self.assertEqual(rep["seals"], 1)
        doc = fitz.open(stream=out, filetype="pdf")
        page = doc[0]
        imgs = page.get_images(full=True)
        self.assertEqual(len(imgs), 1, "deve restare solo l'immagine ricodificata")
        self.assertEqual(imgs[0][8], "DCTDecode", "l'immagine e' quella nuova (JPEG), non il PNG originale")
        # nessun oggetto immagine orfano nel file (l'originale era FlateDecode)
        flate_imgs = [x for x in range(1, doc.xref_length())
                      if doc.xref_get_key(x, "Subtype")[1] == "/Image"
                      and "Flate" in doc.xref_get_key(x, "Filter")[1]]
        self.assertEqual(flate_imgs, [])
        # pixel dentro la casella di "Mario Rossi" (x = 20 + 8..19 caratteri), sul bordo
        # alto: al centro c'e' il placeholder scritto in bianco
        pix = page.get_pixmap(dpi=72)
        r, g, b = pix.pixel(int(20 + 13.5 * 10), 31)
        self.assertLess(abs(r - 124) + abs(g - 58) + abs(b - 158), 60, (r, g, b))
        # fuori dalla PII lo sfondo resta chiaro ("Il" a inizio riga)
        r, g, b = pix.pixel(25, 37)
        self.assertGreater(min(r, g, b), 180, (r, g, b))
        # il timbro e' coperto (angolo della casella: al centro c'e' "[TIMBRO]")
        r, g, b = pix.pixel(304, 153)
        self.assertLess(abs(r - 124) + abs(g - 58) + abs(b - 158), 60, (r, g, b))
        # i placeholder sono testo vero sopra le caselle
        text = page.get_text()
        self.assertIn("[CF_1]", text)
        self.assertIn("[TIMBRO]", text)
        self.assertNotIn("Mario Rossi", text)

    def test_timbri_lasciati_se_richiesto(self):
        out, rep = pdf_export.redact_pdf(self.pdf, self.mapping, ocr_pages=self.ocr,
                                         redact_seals=False)
        self.assertEqual(rep["seals"], 0)
        pix = fitz.open(stream=out, filetype="pdf")[0].get_pixmap(dpi=72)
        self.assertGreater(min(pix.pixel(304, 153)), 180)

    def test_confini_di_parola_anche_sull_ocr(self):
        # "Rossi" non deve essere coperto dentro "Rossini"
        ocr = {0: {"dpi": 144, "lines": [_line("il maestro Rossini", 20, 30)]}}
        out, rep = pdf_export.redact_pdf(self.pdf, {"[FULLNAME_1]": "Rossi"}, ocr_pages=ocr)
        self.assertEqual(rep["occurrences"], 0)
        self.assertEqual(rep["not_found"], ["[FULLNAME_1]"])

    def test_valore_a_cavallo_di_due_righe(self):
        ocr = {0: {"dpi": 144, "lines": [_line("difeso dall'avv. Giulia", 20, 30),
                                         _line("Bianchi del Foro", 20, 60)]}}
        out, rep = pdf_export.redact_pdf(self.pdf, {"[FULLNAME_1]": "Giulia Bianchi"},
                                         ocr_pages=ocr)
        self.assertEqual(rep["occurrences"], 1)
        pix = fitz.open(stream=out, filetype="pdf")[0].get_pixmap(dpi=72)
        for x, y in ((20 + 20 * 10, 31), (20 + 3 * 10, 61)):    # "Giulia" e "Bianchi"
            r, g, b = pix.pixel(x, y)
            self.assertLess(abs(r - 124) + abs(g - 58) + abs(b - 158), 60, (x, y, r, g, b))


class TestoPerPagina(unittest.TestCase):
    class _Page:
        def __init__(self, text):
            self._t = text

        def get_text(self):
            return self._t

        def annots(self):
            return []

        def widgets(self):
            return []

    class _Doc(list):
        def get_toc(self, simple=True):
            return []

    def test_ocr_al_posto_del_layer_vuoto(self):
        doc = self._Doc([self._Page("pagina nativa"), self._Page("")])
        out = pdf_text.collect_readable_text(doc, page_text={1: "testo letto con l'OCR"})
        self.assertEqual(out, "pagina nativa\ntesto letto con l'OCR")

    def test_senza_override_resta_come_prima(self):
        doc = self._Doc([self._Page("a"), self._Page("b")])
        self.assertEqual(pdf_text.collect_readable_text(doc), "a\nb")


class OpzioniOcr(unittest.TestCase):
    def test_mode(self):
        p = server_config.parse_ocr_mode
        self.assertEqual(p("FORCE"), "force")
        self.assertEqual(p("0"), "off")
        self.assertEqual(p("si"), "force")
        self.assertEqual(p(None), "auto")
        self.assertEqual(p("boh", "off"), "off")

    def test_preset(self):
        p = server_config.parse_ocr_preset
        self.assertEqual(p("Fast"), "fast")
        self.assertEqual(p("ultra"), "balanced")


@unittest.skipIf(np is None or _char_spans is None, "numpy/opencv non installati")
class PosizioniCtc(unittest.TestCase):
    def test_intervalli_monotoni_dentro_il_ritaglio(self):
        crop = np.zeros((48, 480, 3), dtype=np.uint8)      # 10:1, nessun padding
        spans = _char_spans([2, 10, 20, 30], n_steps=40, batch_w=480, crop=crop, img_h=48)
        self.assertEqual(len(spans), 4)
        for (a0, a1), (b0, b1) in zip(spans, spans[1:]):
            self.assertLessEqual(a0, a1)
            self.assertAlmostEqual(a1, b0)
        self.assertGreaterEqual(spans[0][0], 0)
        self.assertLessEqual(spans[-1][1], 480)
        # il passo t=20 su 40 cade a meta' del ritaglio
        self.assertTrue(spans[2][0] < 246 < spans[2][1])

    def test_riga_verticale_senza_posizioni(self):
        self.assertEqual(_char_spans([1, 2], 10, 48, np.zeros((200, 40, 3), np.uint8), 48), [])


if __name__ == "__main__":
    unittest.main()
