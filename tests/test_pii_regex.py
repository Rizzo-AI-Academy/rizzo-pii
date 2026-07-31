# -*- coding: utf-8 -*-
"""Rete regex + checksum: l'ultima linea di difesa quando il modello sbaglia.

I valori usati come fixture sono esempi pubblici canonici (nessuna PII reale).
"""

import unittest

from . import _path  # noqa: F401  (sys.path)

from pii_regex import cf_ok, detect_regex, iban_ok, luhn_ok, piva_ok

IBAN_OK = "IT60X0542811101000000123456"
CF_OK = "RSSMRA80A01H501U"
PIVA_OK = "00743110157"
CARD_OK = "4111111111111111"

# stessi valori con una cifra alterata: la FORMA resta valida, il checksum no.
# E' esattamente cio' che produce un OCR che confonde 0/O, 1/l, 5/S, 8/B.
IBAN_BROKEN = "IT60X0542811101000000123457"
CF_BROKEN = "RSSMRA80A01H501X"
PIVA_BROKEN = "00743110158"


def labels(ents):
    return {e["label"] for e in ents}


def by_label(ents, label):
    return [e for e in ents if e["label"] == label]


class Checksums(unittest.TestCase):

    def test_valid_iban_passes(self):
        self.assertTrue(iban_ok(IBAN_OK))

    def test_iban_with_altered_digit_fails(self):
        self.assertFalse(iban_ok(IBAN_BROKEN))

    def test_iban_accepts_internal_spaces(self):
        self.assertTrue(iban_ok("IT60 X054 2811 1010 0000 0123 456"))

    def test_valid_cf_passes(self):
        self.assertTrue(cf_ok(CF_OK))

    def test_cf_with_wrong_control_char_fails(self):
        self.assertFalse(cf_ok(CF_BROKEN))

    def test_cf_is_case_insensitive(self):
        self.assertTrue(cf_ok(CF_OK.lower()))

    def test_valid_piva_passes(self):
        self.assertTrue(piva_ok(PIVA_OK))

    def test_piva_with_altered_digit_fails(self):
        self.assertFalse(piva_ok(PIVA_BROKEN))

    def test_valid_card_passes_luhn(self):
        self.assertTrue(luhn_ok(CARD_OK))

    def test_card_with_altered_digit_fails_luhn(self):
        self.assertFalse(luhn_ok("4111111111111112"))


class DetectRegexOnCleanText(unittest.TestCase):
    """Comportamento storico: da non cambiare."""

    def test_finds_email(self):
        ents = detect_regex("Scrivere a mario.rossi@studiolegale.it entro il termine.")
        self.assertIn("EMAIL", labels(ents))

    def test_finds_valid_iban(self):
        ents = detect_regex(f"Bonifico su {IBAN_OK} intestato al ricorrente.")
        self.assertTrue(by_label(ents, "IBAN")[0]["validated"])

    def test_iban_with_broken_checksum_is_not_redacted(self):
        """strict: senza checksum valido troppi falsi positivi su testo pulito."""
        ents = detect_regex(f"Codice pratica {IBAN_BROKEN} da verificare.")
        self.assertEqual(by_label(ents, "IBAN"), [])

    def test_piva_with_broken_checksum_is_not_redacted(self):
        ents = detect_regex(f"Protocollo n. {PIVA_BROKEN} del 2024.")
        self.assertEqual(by_label(ents, "PIVA"), [])

    def test_cf_with_broken_checksum_is_redacted_anyway(self):
        """La forma del CF e' talmente specifica che conviene nasconderlo comunque."""
        ents = detect_regex(f"C.F. {CF_BROKEN} del dichiarante.")
        cf = by_label(ents, "CF")
        self.assertEqual(len(cf), 1)
        self.assertFalse(cf[0]["validated"])


class DetectRegexOnOcrText(unittest.TestCase):
    """Su testo OCR gli errori di riconoscimento rompono i checksum.

    Un IBAN non redatto che finisce su un LLM cloud e' il fallimento che questo
    prodotto esiste per impedire: sulle pagine OCR la forma basta.
    """

    def test_iban_with_broken_checksum_is_redacted(self):
        ents = detect_regex(f"Bonifico su {IBAN_BROKEN} intestato al ricorrente.",
                            relax_strict=True)
        self.assertEqual(len(by_label(ents, "IBAN")), 1)

    def test_piva_with_broken_checksum_is_redacted(self):
        ents = detect_regex(f"P.IVA {PIVA_BROKEN} della societa'.", relax_strict=True)
        self.assertEqual(len(by_label(ents, "PIVA")), 1)

    def test_card_with_broken_checksum_is_redacted(self):
        ents = detect_regex("Carta 4111111111111112 in atti.", relax_strict=True)
        self.assertEqual(len(by_label(ents, "CREDITCARDNUMBER")), 1)

    def test_unvalidated_entity_is_flagged_as_ocr_sourced(self):
        """L'UI deve poterla mostrare come 'da verificare'."""
        ents = detect_regex(f"IBAN {IBAN_BROKEN}.", relax_strict=True)
        iban = by_label(ents, "IBAN")[0]
        self.assertEqual(iban["source"], "regex-ocr")
        self.assertFalse(iban["validated"])

    def test_valid_entity_keeps_its_normal_source_and_validation(self):
        ents = detect_regex(f"IBAN {IBAN_OK}.", relax_strict=True)
        iban = by_label(ents, "IBAN")[0]
        self.assertEqual(iban["source"], "regex")
        self.assertTrue(iban["validated"])

    def test_span_offsets_point_at_the_entity(self):
        text = f"Bonifico su {IBAN_BROKEN} entro maggio."
        iban = by_label(detect_regex(text, relax_strict=True), "IBAN")[0]
        self.assertEqual(text[iban["start"]:iban["end"]], IBAN_BROKEN)


if __name__ == "__main__":
    unittest.main()
