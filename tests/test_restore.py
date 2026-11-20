# -*- coding: utf-8 -*-
"""Test del ripristino del mapping senza rompere la sintassi JSON.

Il caso d'uso: /analyze restituisce ``anonymized_text`` con i placeholder
(``[AMOUNT_1]``) e un ``mapping`` {placeholder: valore} con i valori originali
come stringhe (``"1.500,00"``). Ripristinare con una sostituzione di
sottostringa su un payload JSON produce ``""1.500,00""`` (sintassi rotta) se il
placeholder e' dentro una stringa quotata, e non normalizza gli importi in
formato italiano.

Qui si verifica il comportamento del helper ``restore.restore``: sintassi JSON
sempre valida, importi AMOUNT normalizzati a numero, placeholder in-stringa
restati stringhe, testo libero sostituito, chiavi mai toccate.

Nomi, importi e date sono SINTETICI.

Solo stdlib: nessuna dipendenza pesante, il test gira anche in CI.
"""

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "app"))

from restore import restore  # noqa: E402


class RestoreJsonTest(unittest.TestCase):
    """Ripristino su payload JSON (stringa o struttura)."""

    def test_mapping_vuoto_lascia_payload_invariato(self):
        payload = {"totale": "[AMOUNT_1]"}
        self.assertEqual(restore(payload, {}), payload)

    def test_stringa_json_placeholder_intero_resta_json_valido(self):
        payload = '{"totale": "[AMOUNT_1]"}'
        out = restore(payload, {"[AMOUNT_1]": "1.500,00"})
        self.assertEqual(json.loads(out)["totale"], 1500.0)

    def test_stringa_json_con_fence_markdown(self):
        payload = '```json\n{"totale": "[AMOUNT_1]"}\n```'
        out = restore(payload, {"[AMOUNT_1]": "1.500,00"})
        self.assertEqual(json.loads(out)["totale"], 1500.0)

    def test_dict_diretto_restituisce_copia_con_numero(self):
        payload = {"imponibile": "[AMOUNT_1]", "iva": "[AMOUNT_2]"}
        out = restore(payload, {"[AMOUNT_1]": "500,00", "[AMOUNT_2]": "110,00"})
        self.assertEqual(out, {"imponibile": 500.0, "iva": 110.0})
        # il payload originale non viene modificato (copia)
        self.assertEqual(payload["imponibile"], "[AMOUNT_1]")

    def test_lista_di_valori(self):
        payload = ["[AMOUNT_1]", "nota", {"x": "[AMOUNT_1]"}]
        out = restore(payload, {"[AMOUNT_1]": "1.500,00"})
        self.assertEqual(out, [1500.0, "nota", {"x": 1500.0}])

    def test_placeholder_in_stringa_resta_stringa(self):
        payload = {"descrizione": "Totale: [AMOUNT_1] IVA inclusa"}
        out = restore(payload, {"[AMOUNT_1]": "1.500,00"})
        self.assertEqual(out["descrizione"], "Totale: 1.500,00 IVA inclusa")

    def test_importo_con_prefisso_euro(self):
        out = restore({"x": "[AMOUNT_1]"}, {"[AMOUNT_1]": "€ 12.500,00"})
        self.assertEqual(out["x"], 12500.0)

    def test_importo_con_suffisso_eur(self):
        out = restore({"x": "[AMOUNT_1]"}, {"[AMOUNT_1]": "12.500,00 EUR"})
        self.assertEqual(out["x"], 12500.0)

    def test_valore_non_numerico_su_amount_resta_stringa(self):
        # valore AMOUNT che non e' un numero (raro): nessuna normalizzazione
        out = restore({"x": "[AMOUNT_1]"}, {"[AMOUNT_1]": "N/D"})
        self.assertEqual(out["x"], "N/D")

    def test_tag_non_monetario_non_viene_normalizzato(self):
        # "[FULLNAME_1]" con valore numerico resta stringa: solo AMOUNT diventa numero
        out = restore({"x": "[FULLNAME_1]"}, {"[FULLNAME_1]": "1.500,00"})
        self.assertEqual(out["x"], "1.500,00")

    def test_placeholder_piu_lunghi_sostituiti_prima(self):
        # [ORG_1] non deve essere sostituito dentro [ORG_10]
        out = restore(
            {"a": "[ORG_10]", "b": "[ORG_1]"},
            {"[ORG_1]": "Beta S.r.l.", "[ORG_10]": "Gamma S.p.A."},
        )
        self.assertEqual(out, {"a": "Gamma S.p.A.", "b": "Beta S.r.l."})

    def test_stessa_occorrenza_stesso_valore(self):
        payload = {"a": "[FULLNAME_1]", "b": "[FULLNAME_1]"}
        out = restore(payload, {"[FULLNAME_1]": "Mario Rossi"})
        self.assertEqual(out, {"a": "Mario Rossi", "b": "Mario Rossi"})


class RestoreTextTest(unittest.TestCase):
    """Fallback su testo libero (non-JSON) e scalari JSON."""

    def test_testo_libero_sostituzione_diretta(self):
        out = restore(
            "Fattura emessa da [ORG_1] il [DATE_1].",
            {"[ORG_1]": "Alfa S.p.A.", "[DATE_1]": "10/03/2026"},
        )
        self.assertEqual(out, "Fattura emessa da Alfa S.p.A. il 10/03/2026.")

    def test_stringa_json_scalare_resta_serializzata(self):
        # la stringa nuda "[AMOUNT_1]" (senza struttura) torna serializzata:
        # stringa o numero, mai sintassi rotta
        out = restore('"[AMOUNT_1]"', {"[AMOUNT_1]": "1.500,00"})
        self.assertEqual(json.loads(out), 1500.0)

    def test_valore_non_stringa_resta_invariato(self):
        self.assertEqual(restore({"x": 5}, {"[AMOUNT_1]": "1.500,00"}), {"x": 5})


if __name__ == "__main__":
    unittest.main()
