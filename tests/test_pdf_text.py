# -*- coding: utf-8 -*-
"""Decisione 'questa pagina va passata all'OCR' e ricomposizione del documento."""

import unittest

from . import _path  # noqa: F401  (sys.path)

from pdf_text import (merge_ocr_pages, page_needs_ocr, pages_needing_ocr,
                      unreadable_pages)


class PageNeedsOcr(unittest.TestCase):

    def test_empty_page_needs_ocr(self):
        self.assertTrue(page_needs_ocr(""))

    def test_whitespace_only_page_needs_ocr(self):
        self.assertTrue(page_needs_ocr("  \n\t \n "))

    def test_page_with_real_text_does_not_need_ocr(self):
        self.assertFalse(page_needs_ocr("Il sottoscritto Mario Rossi, nato a Padova "
                                        "il 12/03/1980, dichiara quanto segue."))

    def test_page_with_only_a_page_number_needs_ocr(self):
        """Tipico della scansione: il numero di pagina e' testo, il resto e' immagine."""
        self.assertTrue(page_needs_ocr("- 3 -"))

    def test_threshold_is_configurable(self):
        self.assertFalse(page_needs_ocr("abc", min_chars=2))


class PagesNeedingOcr(unittest.TestCase):

    def test_returns_indexes_of_scanned_pages_only(self):
        pages = ["testo nativo lungo abbastanza da superare la soglia dei caratteri",
                 "",
                 "  ",
                 "altro testo nativo lungo abbastanza da superare la soglia minima"]
        self.assertEqual(pages_needing_ocr(pages), [1, 2])

    def test_document_fully_native_returns_empty(self):
        pages = ["x" * 100, "y" * 100]
        self.assertEqual(pages_needing_ocr(pages), [])


class UnreadablePages(unittest.TestCase):
    """Quando l'OCR collassa (scansione pessima) restituisce poche righe sparse.

    Il documento anonimizzato sembra a posto ma ha perso quasi tutto il contenuto,
    e le PII non lette non sono state nemmeno viste: vanno segnalate, non ignorate.
    """

    def test_page_that_produced_almost_nothing_is_flagged(self):
        self.assertEqual(unreadable_pages(["007ki110157, PREMESSO"]), [0])

    def test_page_with_a_full_text_is_not_flagged(self):
        self.assertEqual(unreadable_pages(["parola " * 50]), [])

    def test_page_that_produced_nothing_is_flagged(self):
        self.assertEqual(unreadable_pages([""]), [0])

    def test_flags_only_the_bad_pages(self):
        self.assertEqual(unreadable_pages(["parola " * 50, "x", "parola " * 50]), [1])


class MergeOcrPages(unittest.TestCase):

    def test_ocr_text_replaces_only_the_scanned_pages(self):
        native = ["pagina uno", "", "pagina tre"]
        out = merge_ocr_pages(native, [1], ["testo letto via OCR"])
        self.assertEqual(out, ["pagina uno", "testo letto via OCR", "pagina tre"])

    def test_page_order_is_preserved_with_multiple_ocr_pages(self):
        native = ["", "nativa", ""]
        out = merge_ocr_pages(native, [0, 2], ["prima", "terza"])
        self.assertEqual(out, ["prima", "nativa", "terza"])

    def test_missing_ocr_results_leave_pages_untouched(self):
        """Il backend ha ritornato meno pagine del richiesto: non si perde il resto."""
        native = ["", "nativa", ""]
        out = merge_ocr_pages(native, [0, 2], ["prima"])
        self.assertEqual(out, ["prima", "nativa", ""])

    def test_does_not_mutate_the_input_list(self):
        native = ["", "nativa"]
        merge_ocr_pages(native, [0], ["letta"])
        self.assertEqual(native, ["", "nativa"])


if __name__ == "__main__":
    unittest.main()
