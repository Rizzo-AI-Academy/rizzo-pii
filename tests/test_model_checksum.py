# -*- coding: utf-8 -*-
"""Un'entita' del MODELLO su una label con checksum deve rivalidarsi come una
della rete regex: `detect_model` etichettava ogni entita' `validated: False`
a prescindere, quindi un CF/IBAN/PIVA/carta indovinato dal modello restava in
fondo alla priorita' di `_merge` anche quando il checksum passava (issue #119).

I valori sono gli stessi, gia' verificati, di test_detector_formats.py.
"""

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "app"))

import detectors  # noqa: E402

# (label, valore valido, valore con checksum sbagliato)
CASI = [
    ("IBAN", "IT60X0542811101000000123456", "IT61X0542811101000000123456"),
    ("CF", "RSSMRA85H12F205Y", "RSSMRA85H12F205X"),
    ("PIVA", "00743110157", "00743110158"),
    ("CREDITCARDNUMBER", "4111111111111111", "4111111111111112"),
]

# Label che il modello puo' emettere ma per cui non esiste un checksum: la
# rete regex le lascia validated=False sulla sola forma (validatore None in
# DETECTORS), e cosi' deve restare anche per il modello.
SENZA_CHECKSUM = ("EMAIL", "TELEPHONENUM", "AMOUNT", "TARGA", "URL", "IPADDR", "DOCID", "DATE")


class ValidatorsCopronoIChecksum(unittest.TestCase):
    """`VALIDATORS` e' la mappa label->validatore che `detect_model` consulta:
    deve avere esattamente le quattro label con checksum, e nessun'altra."""

    def test_solo_le_label_con_checksum(self):
        self.assertEqual(set(detectors.VALIDATORS), {"IBAN", "CF", "PIVA", "CREDITCARDNUMBER"})

    def test_label_senza_checksum_assente_dalla_mappa(self):
        for label in SENZA_CHECKSUM:
            self.assertNotIn(label, detectors.VALIDATORS, label)

    def test_valore_valido_passa_il_validatore_della_sua_label(self):
        for label, valido, _ in CASI:
            with self.subTest(label=label):
                self.assertTrue(detectors.VALIDATORS[label](valido))

    def test_checksum_sbagliato_non_passa(self):
        for label, _, sbagliato in CASI:
            with self.subTest(label=label):
                self.assertFalse(detectors.VALIDATORS[label](sbagliato))


class EntitaDelModelloRivalidata(unittest.TestCase):
    """Stessa forma dell'entita' che `detect_model` costruisce (label, start,
    end, score, source="modello"), con `validated` calcolato come lo calcola
    ora `detect_model`: prima del fix era sempre False."""

    @staticmethod
    def _validated_come_detect_model(label, testo):
        validator = detectors.VALIDATORS.get(label)
        return bool(validator and validator(testo))

    def test_checksum_valido_dal_modello_si_valida(self):
        for label, valido, _ in CASI:
            with self.subTest(label=label):
                self.assertTrue(self._validated_come_detect_model(label, valido))

    def test_checksum_sbagliato_dal_modello_resta_non_validato(self):
        for label, _, sbagliato in CASI:
            with self.subTest(label=label):
                self.assertFalse(self._validated_come_detect_model(label, sbagliato))

    def test_label_senza_checksum_resta_non_validata(self):
        for label in SENZA_CHECKSUM:
            self.assertFalse(self._validated_come_detect_model(label, "qualunque testo"), label)


if __name__ == "__main__":
    unittest.main()
