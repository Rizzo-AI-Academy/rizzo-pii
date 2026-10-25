# -*- coding: utf-8 -*-
"""I template tabellari: nome e cognome in colonne separate, come negli export CSV.

Devono dare BIO valide e righe col numero di campi dell'intestazione (l'importo, che ha
la virgola decimale, va fra virgolette), e i loro slot {GIVEN}/{SURNAME} NON devono
finire fra quelli proposti all'LLM: in un testo corrente farebbero di un nome due entita'.

Tutti i valori sono SINTETICI (generati dal codice).
"""

import csv
import io
import random
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "data_pipeline"))

import generate_synthetic_pii as gen  # noqa: E402
import llm_template_bank as tb  # noqa: E402

TABELLE = [i for i, t in enumerate(gen.TEMPLATES) if "{GIVEN}" in t]


class TabelleSintetiche(unittest.TestCase):
    def test_ci_sono_le_tre_tabelle(self):
        self.assertEqual(len(TABELLE), 3)

    def test_bio_e_campi(self):
        for tid in TABELLE:
            for seme in range(200):
                random.seed(seme)
                text, ents = gen.build_example(tid, gen.TEMPLATES)
                _, labels = gen.to_bio(text, ents)
                for prima, dopo in zip(["O"] + labels, labels):
                    if dopo.startswith("I-"):
                        self.assertIn(prima[2:], (dopo[2:],), f"I- incoerente: {text!r}")
                        self.assertNotEqual(prima, "O", f"I- dopo O: {text!r}")
                righe = text.split("\n")[1:]
                sep = ";" if ";" in righe[0] else ","
                campi = [len(r) for r in csv.reader(io.StringIO("\n".join(righe)), delimiter=sep)]
                self.assertEqual(set(campi), {campi[0]}, f"campi diversi: {text!r}")
                self.assertEqual({e["label"] for e in ents} & {"GIVENNAME", "SURNAME"},
                                 {"GIVENNAME", "SURNAME"})

    def test_non_sono_proposti_all_llm(self):
        self.assertFalse({"GIVEN", "SURNAME"} & tb.PROMPT_SLOTS)
        self.assertEqual(tb.PROMPT_SLOTS | {"GIVEN", "SURNAME"}, tb.ALLOWED_SLOTS)


if __name__ == "__main__":
    unittest.main()
