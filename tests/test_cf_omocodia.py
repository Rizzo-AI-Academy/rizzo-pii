# -*- coding: utf-8 -*-
"""Codice fiscale omocodico: generazione sintetica e riconoscimento nell'app.

L'omocodia e' la procedura con cui l'Agenzia delle Entrate differenzia due codici
fiscali identici: sostituisce le cifre con una lettera partendo da destra
(0=L 1=M 2=N 3=P 4=Q 5=R 6=S 7=T 8=U 9=V) e ricalcola il carattere di controllo.
Il codice che ne esce e' un codice fiscale valido a tutti gli effetti.

I detector si importano da `src/app/detectors.py`, che non tira dentro
torch/transformers: provare una regex non deve costare 0.3B di pesi scaricati.
"""

import random
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "data_pipeline"))
sys.path.insert(0, str(ROOT / "src" / "app"))

import detectors  # noqa: E402
import generate_synthetic_pii as gen  # noqa: E402


# --------------------------------------------------------------------------- #
# Verificatore di checksum indipendente.
# Riscritto qui dalla specifica e non importato dal generatore: se lo prendessimo
# in prestito da lui, il test direbbe solo che il generatore concorda con se stesso.
# --------------------------------------------------------------------------- #
_ODD = {"0": 1, "1": 0, "2": 5, "3": 7, "4": 9, "5": 13, "6": 15, "7": 17, "8": 19,
        "9": 21, "A": 1, "B": 0, "C": 5, "D": 7, "E": 9, "F": 13, "G": 15, "H": 17,
        "I": 19, "J": 21, "K": 2, "L": 4, "M": 18, "N": 20, "O": 11, "P": 3, "Q": 6,
        "R": 8, "S": 12, "T": 14, "U": 16, "V": 10, "W": 22, "X": 25, "Y": 24, "Z": 23}


def cf_checksum_ok(code):
    code = code.strip().upper()
    if len(code) != 16 or not code.isalnum():
        return False
    total = 0
    for i, ch in enumerate(code[:15]):
        if i % 2 == 0:
            if ch not in _ODD:
                return False
            total += _ODD[ch]
        else:
            total += int(ch) if ch.isdigit() else ord(ch) - 65
    return chr(65 + total % 26) == code[15]


# --------------------------------------------------------------------------- #
# I detector CF veri, importati.
# --------------------------------------------------------------------------- #
def _load_cf_detectors():
    """[(regex compilata, strict), ...] per le voci CF di DETECTORS."""
    return [(rx, strict) for label, rx, _validator, strict in detectors.DETECTORS
            if label == "CF"]


CF_DETECTORS = _load_cf_detectors()


def app_would_redact(text):
    """True se la rete regex dell'app sostituirebbe `text` come CF.

    Riproduce la sola regola di `detect_regex`: con strict=True il match si scarta
    quando il checksum fallisce, con strict=False si redige comunque.
    """
    for rx, strict in CF_DETECTORS:
        match = rx.search(text)
        if not match:
            continue
        if strict and not cf_checksum_ok(match.group(0)):
            continue
        return True
    return False


def make_cf(omocodia=0, seed=0):
    rnd = random.Random(seed)
    city = rnd.choice(gen.CITY_NAMES)
    return gen.codice_fiscale("Mario", "Rossi", "M", 1985, 6, 12, city, omocodia=omocodia)


class OmocodiaGeneration(unittest.TestCase):
    def test_ordinary_code_is_unchanged(self):
        code = make_cf(omocodia=0)
        self.assertTrue(cf_checksum_ok(code))
        self.assertRegex(code, r"^[A-Z]{6}\d{2}[A-Z]\d{2}[A-Z]\d{3}[A-Z]$")

    def test_every_substitution_depth_stays_valid(self):
        for n in range(1, 8):
            with self.subTest(sostituzioni=n):
                code = make_cf(omocodia=n)
                self.assertTrue(cf_checksum_ok(code),
                                f"checksum non ricalcolato per {n} sostituzioni: {code}")

    def test_substitution_runs_right_to_left(self):
        ordinary = make_cf(omocodia=0)
        for n in range(1, 8):
            code = make_cf(omocodia=n)
            changed = [i for i in range(15) if code[i] != ordinary[i]]
            self.assertEqual(sorted(changed), sorted(gen.OMOCODIA_POS[:n]),
                             f"posizioni sbagliate con {n} sostituzioni")

    def test_substituted_characters_follow_the_table(self):
        ordinary = make_cf(omocodia=0)
        code = make_cf(omocodia=7)
        for pos in gen.OMOCODIA_POS:
            self.assertEqual(code[pos], "LMNPQRSTUV"[int(ordinary[pos])])

    def test_generator_emits_some_omocodic_codes(self):
        random.seed(20260803)
        codes = [gen.cf_piece()[0][0] for _ in range(2000)]
        omocodic = [c for c in codes if not c[6:8].isdigit() or not c[12:15].isdigit()]
        self.assertTrue(omocodic, "nessun CF omocodico in 2000 estrazioni")
        for code in codes:
            self.assertTrue(cf_checksum_ok(code), f"CF non valido: {code}")


class OmocodiaDetection(unittest.TestCase):
    def test_app_has_a_detector_for_the_omocodic_shape(self):
        self.assertGreaterEqual(len(CF_DETECTORS), 2,
                                "manca il detector per la forma omocodica")

    def test_ordinary_code_is_still_redacted(self):
        self.assertTrue(app_would_redact(f"C.F. {make_cf(omocodia=0)} residente in Roma"))

    def test_omocodic_code_is_redacted(self):
        for n in range(1, 8):
            code = make_cf(omocodia=n)
            with self.subTest(sostituzioni=n, cf=code):
                self.assertTrue(app_would_redact(f"C.F. {code} residente in Roma"),
                                f"CF omocodico lasciato in chiaro: {code}")

    def test_malformed_code_of_the_same_shape_is_not_redacted(self):
        """La forma omocodica da sola non basta: senza checksum non si redige.

        Serve a tenere onesto l'allargamento della regex — una parola di sedici
        lettere che combacia con la forma non deve diventare un CF.
        """
        code = make_cf(omocodia=7)
        wrong = code[:15] + ("A" if code[15] != "A" else "B")
        self.assertFalse(cf_checksum_ok(wrong))
        self.assertFalse(app_would_redact(f"riferimento {wrong} in atti"))

    def test_ordinary_code_with_broken_checksum_is_still_redacted(self):
        """Comportamento storico invariato: sulla forma ordinaria si redige comunque."""
        code = make_cf(omocodia=0)
        wrong = code[:15] + ("A" if code[15] != "A" else "B")
        self.assertFalse(cf_checksum_ok(wrong))
        self.assertTrue(app_would_redact(f"C.F. {wrong} in atti"))


if __name__ == "__main__":
    unittest.main()
