# -*- coding: utf-8 -*-
"""Codice fiscale scritto come lo si stampa: a gruppi, col trattino, a lettere spaziate,
spezzato da un a-capo.

La rete regex vedeva solo i sedici caratteri attaccati. Su "RSS MRA 85H12 H501V" non
trovava nulla e il codice restava al modello, che su queste forme lo lascia in chiaro,
per intero o a pezzi accanto a un [CF_1].

I detector si importano da `src/app/detectors.py`: niente torch, niente modello.
"""

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "data_pipeline"))
sys.path.insert(0, str(ROOT / "src" / "app"))

import detectors  # noqa: E402
import generate_synthetic_pii as gen  # noqa: E402


def make_cf(omocodia=0):
    return gen.codice_fiscale("Mario", "Rossi", "M", 1985, 6, 12, gen.CITY_NAMES[0],
                              omocodia=omocodia)


# Le forme in cui il codice compare davvero: tesserino, moduli a caselle, intestazioni,
# testo estratto da PDF che va a capo a meta'.
FORME = {
    "gruppi 3-3-5-5": lambda c: f"{c[:3]} {c[3:6]} {c[6:11]} {c[11:]}",
    "trattini 3-3-5-5": lambda c: f"{c[:3]}-{c[3:6]}-{c[6:11]}-{c[11:]}",
    "trattini fra spazi": lambda c: f"{c[:3]} - {c[3:6]} - {c[6:11]} - {c[11:]}",
    "punti 6-5-5":lambda c: f"{c[:6]}.{c[6:11]}.{c[11:]}",
    "6-10": lambda c: f"{c[:6]} {c[6:]}",
    "per campo 6-2-1-2-4-1": lambda c: f"{c[:6]} {c[6:8]} {c[8]} {c[9:11]} {c[11:15]} {c[15]}",
    "lettere spaziate": lambda c: " ".join(c),
    "a capo": lambda c: f"{c[:9]}\n{c[9:]}",
    "a capo dentro i gruppi": lambda c: f"{c[:3]} {c[3:6]} {c[6:10]}\n{c[10]} {c[11:]}",
    "spazio non separabile": lambda c: f"{c[:3]} {c[3:6]} {c[6:11]} {c[11:]}",
}


def cf_spans(text):
    return [(e["start"], e["end"]) for e in detectors.detect_regex(text) if e["label"] == "CF"]


def covered(text, value):
    """Ogni carattere del codice sotto una span CF (i separatori possono restare fuori)."""
    a = text.index(value)
    spans = cf_spans(text)
    return all(any(s <= k < e for s, e in spans)
               for k in range(a, a + len(value)) if text[k].isalnum())


class FormaStampa(unittest.TestCase):
    def test_ogni_forma_stampata_e_coperta(self):
        for omocodia in (0, 1, 2):
            code = make_cf(omocodia)
            for nome, forma in FORME.items():
                scritto = forma(code)
                with self.subTest(forma=nome, omocodia=omocodia):
                    self.assertTrue(covered(f"C.F. {scritto} residente in Roma", scritto),
                                    f"CF in chiaro: {scritto!r}")

    def test_span_validata_e_senza_punteggiatura_finale(self):
        scritto = FORME["gruppi 3-3-5-5"](make_cf())
        text = f"codice fiscale: {scritto}."
        (e,) = [e for e in detectors.detect_regex(text) if e["label"] == "CF"]
        self.assertTrue(e["validated"])
        self.assertEqual(text[e["start"]:e["end"]], scritto)

    def test_minuscolo(self):
        scritto = FORME["gruppi 3-3-5-5"](make_cf()).lower()
        self.assertTrue(covered(f"cf {scritto} in atti", scritto))


class NonECodiceFiscale(unittest.TestCase):
    def test_checksum_sbagliato_non_si_redige(self):
        code = make_cf()
        wrong = code[:15] + ("A" if code[15] != "A" else "B")
        for nome, forma in FORME.items():
            with self.subTest(forma=nome):
                self.assertEqual(cf_spans(f"C.F. {forma(wrong)} in atti"), [])

    def test_prosa_con_la_forma_omocodica(self):
        """Due parole di 6 e 10 lettere, tutte lettere ammesse dall'omocodia, e il checksum
        che passa per caso: e' una frase vera, trovata su Wikipedia."""
        self.assertTrue(detectors.cf_ok("FURONOTRATTENUTE"))
        self.assertEqual(cf_spans("le persone furono trattenute per qualche periodo"), [])

    def test_gruppi_che_nessuno_stampa(self):
        code = make_cf()
        self.assertEqual(cf_spans(f"rif. {code[:4]} {code[4:]} in atti"), [])
        self.assertEqual(cf_spans(f"rif. {code[:7]} {code[7:12]} {code[12:]} in atti"), [])

    def test_due_a_capo_sono_una_colonna(self):
        code = make_cf()
        self.assertEqual(cf_spans(f"{code[:5]}\n{code[5:10]}\n{code[10:]}"), [])

    def test_attaccato_a_una_parola(self):
        scritto = FORME["gruppi 3-3-5-5"](make_cf())
        self.assertEqual(cf_spans(f"X{scritto} in atti"), [])
        self.assertEqual(cf_spans(f"{scritto}9 in atti"), [])


if __name__ == "__main__":
    unittest.main()
