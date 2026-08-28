# -*- coding: utf-8 -*-
"""`pdf_text.py` — crop verticale e fallback OCR sui PDF.

Il modulo dipende SOLO da PyMuPDF (+ pytesseract/Pillow, importati solo quando
servono): niente torch/transformers. Se fitz manca il test si salta; l'OCR vero
non si esegue mai, si simula `_ocr_page` o si mettono nel sys.modules due finti
`pytesseract`/`PIL` per verificare che l'IMMAGINE venga ritagliata prima del
passaggio a Tesseract.

Nomi e dati nei fixture sono SINTETICI.
"""

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

try:
    import fitz
except ImportError:                      # noqa: E402
    fitz = None

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "app"))

try:
    import pdf_text  # noqa: E402
except Exception:                        # noqa: BLE001 - manca fitz (o PyMuPDF rotto)
    pdf_text = None

REASON = "serve PyMuPDF"

# testo piu' lungo di OCR_MIN_CHARS, cosi' la pagina conta come "nativa"
CORPO = "Corpo del documento. " + "parole ripetute per riempire la pagina. " * 8


@unittest.skipIf(fitz is None or pdf_text is None, REASON)
class ParseCrop(unittest.TestCase):
    def test_default_se_non_valido(self):
        self.assertEqual(pdf_text.parse_crop(None), 0.0)
        self.assertEqual(pdf_text.parse_crop("abc"), 0.0)
        self.assertEqual(pdf_text.parse_crop(None, 5.0), 5.0)

    def test_manca_il_default(self):
        self.assertEqual(pdf_text.parse_crop("", default=2.5), 2.5)

    def test_bounds(self):
        self.assertEqual(pdf_text.parse_crop("-3"), 0.0)
        self.assertEqual(pdf_text.parse_crop("150"), 100.0)
        self.assertEqual(pdf_text.parse_crop("12.5"), 12.5)


@unittest.skipIf(fitz is None or pdf_text is None, REASON)
class CropSuTestoNativo(unittest.TestCase):
    """Con crop 0 l'estrazione deve essere identica a page.get_text();
    col crop le fasce spariscono dal testo estratto."""

    @classmethod
    def setUpClass(cls):
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((40, 30), "INTESTAZIONE", fontsize=12)   # alto
        page.insert_text((40, 400), CORPO, fontsize=12)           # centro
        page.insert_text((40, 790), "PIEDIPAGINA", fontsize=12)   # basso
        cls.bytes = doc.tobytes()
        doc.close()

    def _testo(self, top=0, bottom=0):
        t, info = pdf_text.extract_pdf_text(self.bytes, crop_top=top,
                                            crop_bottom=bottom)
        return t, info

    def test_senza_crop_equivale_a_get_text(self):
        t, info = self._testo()
        self.assertEqual(info["pages"], 1)
        self.assertEqual(info["ocr_pages"], 0)
        self.assertIn("INTESTAZIONE", t)
        self.assertIn("PIEDIPAGINA", t)
        self.assertIn("Corpo", t)

    def test_crop_top_toglie_l_intestazione(self):
        t, info = self._testo(top=10)
        self.assertNotIn("INTESTAZIONE", t)
        self.assertIn("PIEDIPAGINA", t)
        self.assertIn("Corpo", t)

    def test_crop_bottom_toglie_il_piedipagina(self):
        t, info = self._testo(bottom=10)
        self.assertNotIn("PIEDIPAGINA", t)
        self.assertIn("INTESTAZIONE", t)
        self.assertIn("Corpo", t)

    def test_crop_doppio_lascia_solo_il_corpo(self):
        t, info = self._testo(top=10, bottom=10)
        self.assertNotIn("INTESTAZIONE", t)
        self.assertNotIn("PIEDIPAGINA", t)
        self.assertIn("Corpo", t)


@unittest.skipIf(fitz is None or pdf_text is None, REASON)
class FallbackOcr(unittest.TestCase):
    """La pagina vuota (scannerizzata) passa da `_ocr_page`; quella con testo
    nativo no. Il flusso con OCR e' simulato: nessun tesseract reale."""

    @classmethod
    def setUpClass(cls):
        doc = fitz.open()
        doc.new_page()                                  # pagina vuota -> OCR
        cls.blank = doc.tobytes()
        doc.close()

    def test_pagina_vuota_usa_ocr(self):
        with patch.object(pdf_text, "_ocr_page", return_value="TESTO DA SCANSIONE") as ocr:
            t, info = pdf_text.extract_pdf_text(self.blank)
        self.assertEqual(info["pages"], 1)
        self.assertEqual(info["ocr_pages"], 1)
        self.assertEqual(info["page_methods"], ["ocr"])
        self.assertEqual(t, "TESTO DA SCANSIONE")
        ocr.assert_called_once()

    def test_errore_ocr_si_propaga_come_errore_utente(self):
        with patch.object(pdf_text, "_ocr_page",
                          side_effect=pdf_text.OcrUnavailableError("niente tesseract")):
            with self.assertRaises(pdf_text.OcrUnavailableError):
                pdf_text.extract_pdf_text(self.blank)


@unittest.skipIf(fitz is None or pdf_text is None, REASON)
class PdfMisto(unittest.TestCase):
    """Pagina 1 nativa (testo lungo) + pagina 2 scannerizzata: metodo per
    pagina, e il crop si applica alla pagina nativa."""

    @classmethod
    def setUpClass(cls):
        doc = fitz.open()
        p1 = doc.new_page()
        p1.insert_text((40, 400), CORPO, fontsize=12)
        p2 = doc.new_page()
        pix = p2.get_pixmap(dpi=120, alpha=False)       # "scansione" di p2
        p2.insert_image(p2.rect, pixmap=pix)
        cls.bytes = doc.tobytes()
        doc.close()

    def test_metodi_per_pagina(self):
        with patch.object(pdf_text, "_ocr_page", return_value="OCR") as ocr:
            t, info = pdf_text.extract_pdf_text(self.bytes)
        self.assertEqual(info["pages"], 2)
        self.assertEqual(info["page_methods"], ["native", "ocr"])
        self.assertEqual(info["ocr_pages"], 1)
        self.assertIn("Corpo", t)
        self.assertIn("OCR", t)
        ocr.assert_called_once()


@unittest.skipIf(fitz is None or pdf_text is None, REASON)
class OcrCropImmagine(unittest.TestCase):
    """`_ocr_page` deve ritagliare le fasce dall'IMMAGINE prima di Tesseract:
    finti pytesseract/PIL nel sys.modules, si controlla il box del crop."""

    @classmethod
    def setUpClass(cls):
        cls.doc = fitz.open()
        cls.page = cls.doc.new_page()

    @classmethod
    def tearDownClass(cls):
        cls.doc.close()

    def test_ritaglia_le_fasce_prima_dell_ocr(self):
        seen = {}

        class FakeImg:
            def __init__(self, size):
                self.size = size

            @classmethod
            def frombytes(cls, mode, size, samples):
                return cls(size)

            def crop(self, box):
                seen["box"] = box
                return self

        class FakePyTesseract:
            class TesseractNotFoundError(Exception):
                pass

            class TesseractError(Exception):
                pass

            def image_to_string(self, img, lang=None):
                seen["img_size"] = img.size
                return "TESTO OCR"

        class FakePIL:
            Image = FakeImg

        with patch.dict(sys.modules, {
                "pytesseract": FakePyTesseract(),
                "PIL": FakePIL,
                "PIL.Image": FakeImg}):
            out = pdf_text._ocr_page(self.page, 0.2, 0.1, "ita")

        self.assertEqual(out, "TESTO OCR")
        w, h = seen["img_size"]
        top, bottom = round(h * 0.2), round(h * 0.9)
        self.assertEqual(seen["box"], (0, top, w, bottom))
        # il crop ha tolto il 30% dell'altezza: resta il 70%
        self.assertAlmostEqual((bottom - top) / h, 0.7, places=1)

    def test_pagina_completamente_tagliata_resta_vuota(self):
        seen = {}

        class FakeImg:
            def __init__(self, size):
                self.size = size

            @classmethod
            def frombytes(cls, mode, size, samples):
                return cls(size)

            def crop(self, box):
                seen["box"] = box
                return self

        class FakePyTesseract:
            class TesseractNotFoundError(Exception):
                pass

            class TesseractError(Exception):
                pass

            def image_to_string(self, img, lang=None):
                raise AssertionError("immagine vuota: Tesseract non va chiamato")

        class FakePIL:
            Image = FakeImg

        with patch.dict(sys.modules, {
                "pytesseract": FakePyTesseract(),
                "PIL": FakePIL,
                "PIL.Image": FakeImg}):
            out = pdf_text._ocr_page(self.page, 100.0, 0, "ita")   # tutto tagliato
        self.assertEqual(out, "")


@unittest.skipIf(fitz is None or pdf_text is None, REASON)
class InputNonValido(unittest.TestCase):
    def test_pdf_corrotto(self):
        with self.assertRaises(ValueError):
            pdf_text.extract_pdf_text(b"questo non e' un pdf")

    def test_pdf_protetto(self):
        doc = fitz.open()
        doc.new_page().insert_text((40, 400), CORPO, fontsize=12)
        data = doc.tobytes(encryption=fitz.PDF_ENCRYPT_AES_256,
                           owner_pw="segreta", user_pw="segreta")
        doc.close()
        with self.assertRaises(ValueError):
            pdf_text.extract_pdf_text(data)


if __name__ == "__main__":
    unittest.main()