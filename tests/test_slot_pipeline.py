# -*- coding: utf-8 -*-
"""I segnaposto ammessi dalla guardia devono essere quelli che gli iniettori riempiono.

`llm_template_bank.ALLOWED_SLOTS` decide quali template entrano nel dataset;
`generate_synthetic_pii.SLOTS` decide quali il codice sa riempire. Se il primo e'
piu' stretto, si buttano template validi in silenzio - e' successo davvero: la
lista era una copia a mano e le mancavano PROVINCE, NAMELIST, ORGLIST, MIXEDLIST,
quindi ogni template che li usava veniva scartato senza che nessuno se ne
accorgesse. Se invece fosse piu' larga, entrerebbero template con segnaposto che
restano `{COSI}` nel testo generato.

Ora ALLOWED_SLOTS si deriva da SLOTS, quindi questo test verifica che nessuno
torni a scrivere le due liste separatamente.
"""

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "data_pipeline"))

import generate_synthetic_pii as gen  # noqa: E402
import llm_template_bank as tb  # noqa: E402


class SlotAllineati(unittest.TestCase):

    def test_la_guardia_ammette_esattamente_cio_che_si_sa_riempire(self):
        self.assertEqual(set(tb.ALLOWED_SLOTS), set(gen.SLOTS))

    def test_i_template_scritti_a_mano_passano_la_guardia(self):
        """author_templates.py non deve accumulare template che verranno scartati."""
        sys.path.insert(0, str(ROOT / "src" / "data_pipeline"))
        import author_templates as at

        scartati = [(doc_type, errore)
                    for doc_type, text in at.DRAFTS
                    for _, errore in [at.valida(text)] if errore]
        self.assertEqual(scartati, [], "template scartati dalla guardia")


if __name__ == "__main__":
    unittest.main()
