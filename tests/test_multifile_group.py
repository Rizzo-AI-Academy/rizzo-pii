# -*- coding: utf-8 -*-
"""Upload MULTI-FILE: una mappa di placeholder CONDIvISA tra i file del gruppo.

`analyze_group()` condivide `state = {counters, seen, mapping}` tra i testi: lo
stesso valore reale in piu' file riceve lo stesso placeholder, esattamente come
le occorrenze ripetute dentro un singolo documento. `mapping` resta la mappa
piatta dell'intero gruppo (compatibile col restore), e la nuova `provenance`
dice a quale file appartiene ogni placeholder.

Le entita' qui vengono dalla rete regex/checksum (IBAN, che non richiede il
modello): `detect_model` e' finto. Gli IBAN sono SINTETICI, generati con
checksum mod-97 valido.

Questo modulo importa `app.py` (torch/transformers/flask/fitz): se mancano, il
test si salta (stesso schema di test_tsv_anonymize.py).
"""

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "app"))


def _carica_app():
    if "app" in sys.modules:
        return sys.modules["app"]
    with patch("transformers.pipeline", return_value=MagicMock(return_value=[])):
        import app as pii_app
    return pii_app


try:                                    # torch/transformers/flask/fitz presenti?
    APP = _carica_app()
except Exception:                       # noqa: BLE001 - qualunque import mancante
    APP = None


def _iban_italiano(cin, abi, cab, conto):
    """IBAN IT con checksum mod-97 calcolato: SINTETICO ma valido."""
    bban = cin + abi + cab + conto
    spostato = ("IT00" + bban)[4:] + "IT00"
    num = "".join(str(ord(c) - 55) if c.isalpha() else c for c in spostato)
    check = 98 - (int(num) % 97)
    return "IT%02d%s" % (check, bban)


IB1 = _iban_italiano("X", "05428", "11101", "000000123456")
IB2 = _iban_italiano("Y", "03296", "01320", "000000654321")


@unittest.skipIf(APP is None, "serve l'ambiente dell'app (torch/transformers/flask/fitz)")
class GruppoFile(unittest.TestCase):
    """La rete regex trova l'IBAN (checksum) senza il modello."""

    def _gruppo(self, testi, **kw):
        with patch.object(APP, "detect_model", return_value=([], 0)):
            return APP.analyze_group(testi, names=[f"f{i}.pdf" for i in range(len(testi))],
                                     **kw)

    def test_stesso_valore_in_piu_file_stesso_placeholder(self):
        t1 = f"Contratto n.1, conto {IB1}, intestato a Mario Rossi."
        t2 = f"Contratto n.2, stessa posizione {IB1}."
        out = self._gruppo([t1, t2])

        self.assertTrue(out["group"])
        self.assertEqual(len(out["files"]), 2)
        # stesso IBAN nei due file -> stesso placeholder, in entrambi
        for f in out["files"]:
            self.assertIn("[IBAN_1]", f["anonymized_text"])
        # mapping piatta dell'intero gruppo
        self.assertEqual(out["mapping"], {"[IBAN_1]": IB1})
        self.assertEqual(out["n_unique"], 1)
        # provenance: [IBAN_1] proviene da entrambi i file
        self.assertIn("[IBAN_1]", out["provenance"]["f0.pdf"])
        self.assertIn("[IBAN_1]", out["provenance"]["f1.pdf"])

    def test_valori_diversi_placeholder_diversi(self):
        out = self._gruppo([f"Conto {IB1}.", f"Conto {IB2}."])

        self.assertEqual(out["mapping"], {"[IBAN_1]": IB1, "[IBAN_2]": IB2})
        self.assertEqual(out["n_unique"], 2)
        self.assertIn("[IBAN_1]", out["files"][0]["anonymized_text"])
        self.assertIn("[IBAN_2]", out["files"][1]["anonymized_text"])
        # provenance per-file: il primo ha solo IBAN_1, il secondo solo IBAN_2
        self.assertEqual(out["provenance"]["f0.pdf"], ["[IBAN_1]"])
        self.assertEqual(out["provenance"]["f1.pdf"], ["[IBAN_2]"])

    def test_statistiche_aggregate(self):
        t1 = f"Conto {IB1}."
        t2 = f"Conto {IB1} e {IB2}."
        out = self._gruppo([t1, t2])

        # il testo da copiare e' l'intero gruppo
        self.assertIn("[IBAN_1]", out["anonymized_text"])
        self.assertIn("[IBAN_2]", out["anonymized_text"])
        # 2 occorrenze di IB1 + 1 di IB2
        self.assertEqual(out["n_entities"], 3)
        self.assertEqual(out["by_label"]["IBAN"], 3)
        # n_chars = somma dei testi
        self.assertEqual(out["n_chars"], len(t1) + len(t2))

    def test_file_singolo_uguale_al_flusso_classico(self):
        t = f"Conto {IB1}."
        with patch.object(APP, "detect_model", return_value=([], 0)):
            gruppo = APP.analyze_group([t], names=["unico.pdf"])
            solo = APP.analyze(t)

        # il flusso a file singolo NON espone group/files
        self.assertNotIn("group", solo)
        self.assertNotIn("files", solo)
        self.assertEqual(solo["anonymized_text"],
                         gruppo["files"][0]["anonymized_text"])
        self.assertEqual(solo["mapping"], gruppo["mapping"])
        self.assertEqual(solo["n_unique"], gruppo["n_unique"])
        self.assertEqual(solo["excluded_tags"], gruppo["excluded_tags"])

    def test_dizionario_disattivato_nessun_valore_esposto(self):
        out = self._gruppo([f"Conto {IB1}."], mapping_enabled=False)

        self.assertEqual(out["mapping"], {})
        self.assertFalse(out["mapping_enabled"])
        self.assertNotIn(IB1, out["anonymized_text"])
        for s in out["files"][0]["segments"]:
            if s.get("label"):
                self.assertNotIn("t", s)          # niente valore originale

    def test_excluded_tags_lasciano_il_valore_in_chiaro(self):
        out = self._gruppo([f"Conto {IB1}."], excluded={"IBAN"})

        self.assertIn(IB1, out["files"][0]["anonymized_text"])
        self.assertNotIn("[IBAN", out["files"][0]["anonymized_text"])
        self.assertEqual(out["excluded_tags"], ["IBAN"])
        self.assertEqual(out["n_entities"], 0)


if __name__ == "__main__":
    unittest.main()