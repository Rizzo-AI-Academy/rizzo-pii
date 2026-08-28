# -*- coding: utf-8 -*-
"""Correzioni utente sul risultato: falsi positivi da IGNORARE e falsi negativi
da AGGIUNGERE.

`analyze()` / `analyze_group()` accettano due parametri opzionali:
- ignore  = lista di valori (stringhe) da NON anonimizzare: le entita' con quel
  valore esatto (a normalizzazione pari: spazi collassati + casefold) vengono
  scartate DOPO la fusione e il valore resta in chiaro;
- custom  = lista di {value, tag} da anonimizzare a prescindere: le occorrenze
  letterali entrano tra i candidati PRIMA della fusione, quindi una aggiunta
  esplicita vince anche su un tag escluso, e la numerazione si integra con
  quella di modello e regex (fonte "utente").

Le entita' qui vengono dalla rete regex/checksum (IBAN, che non richiede il
modello): `detect_model` e' finto. Gli IBAN sono SINTETICI ma con checksum
mod-97 valido (stesso schema di test_multifile_group.py).
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


@unittest.skipIf(APP is None, "serve l'ambiente dell'app (torch/transformers/flask/fitz)")
class FalsiPositivi(unittest.TestCase):
    """Un valore marcato come falso positivo resta in chiaro."""

    def _analizza(self, testo, **kw):
        with patch.object(APP, "detect_model", return_value=([], 0)):
            return APP.analyze(testo, **kw)

    def test_valore_ignorato_restain_chiaro(self):
        t = f"Pagamento sul conto {IB1} omonimo."
        out = self._analizza(t, ignore=[IB1])

        self.assertIn(IB1, out["anonymized_text"])
        self.assertNotIn("[IBAN", out["anonymized_text"])
        self.assertEqual(out["mapping"], {})
        self.assertEqual(out["n_entities"], 0)

    def test_il_match_e_normalizzato(self):
        # il valore nel testo e' minuscolo, quello ignorato maiuscolo (o viceversa):
        # conta il valore, non il casing
        t = f"Conto {IB1.lower()} chiuso."
        out = self._analizza(t, ignore=[IB1.upper()])

        self.assertIn(IB1.lower(), out["anonymized_text"])
        self.assertNotIn("[IBAN", out["anonymized_text"])

    def test_senza_ignore_tutto_come_prima(self):
        t = f"Conto {IB1}."
        out = self._analizza(t)

        self.assertIn("[IBAN_1]", out["anonymized_text"])
        self.assertEqual(out["mapping"], {"[IBAN_1]": IB1})

    def test_un_altro_valore_simile_non_viene_toccato(self):
        # ignoro un valore che NON compare nel testo: l'entita' reale resta al suo posto
        ib2 = _iban_italiano("Z", "05428", "11101", "000000999999")
        t = f"Conto {IB1} e nient'altro."
        out = self._analizza(t, ignore=[ib2])

        self.assertEqual(out["mapping"], {"[IBAN_1]": IB1})
        self.assertEqual(out["n_entities"], 1)


@unittest.skipIf(APP is None, "serve l'ambiente dell'app (torch/transformers/flask/fitz)")
class FalsiNegativi(unittest.TestCase):
    """Un valore aggiunto a mano viene anonimizzato anche se nessun detector lo vede."""

    def _analizza(self, testo, **kw):
        with patch.object(APP, "detect_model", return_value=([], 0)):
            return APP.analyze(testo, **kw)

    def test_valore_aggiunto_diventa_placeholder(self):
        t = "Il signor Mario Rossi e' tornato. Saluta Mario Rossi."
        out = self._analizza(t, custom=[{"value": "Mario Rossi", "tag": "FULLNAME"}])

        self.assertNotIn("Mario Rossi", out["anonymized_text"])
        self.assertEqual(out["anonymized_text"].count("[FULLNAME_1]"), 2)
        self.assertEqual(out["mapping"], {"[FULLNAME_1]": "Mario Rossi"})
        self.assertEqual(out["n_entities"], 2)
        self.assertEqual(out["by_source"].get("utente"), 2)
        for s in out["segments"]:
            if s.get("label"):
                self.assertEqual(s["src"], "utente")

    def test_custom_vince_su_tag_escluso(self):
        # FULLNAME e' escluso per tipo, ma la aggiunta esplicita dell'utente passa
        t = f"Conto {IB1} di Mario Rossi."
        out = self._analizza(t, excluded={"IBAN"},
                             custom=[{"value": "Mario Rossi", "tag": "FULLNAME"}])

        self.assertIn(IB1, out["anonymized_text"])          # esclusione per tipo: ok
        self.assertIn("[FULLNAME_1]", out["anonymized_text"])  # aggiunta: vince comunque

    def test_tag_sconosciuto_ricade_su_fullname(self):
        t = "Scrivo a Mario Rossi domani."
        out = self._analizza(t, custom=[{"value": "Mario Rossi", "tag": "TAG_INESISTENTE"}])

        self.assertIn("[FULLNAME_1]", out["anonymized_text"])

    def test_match_case_insensitive(self):
        t = "mario rossi fu chiamato. MARIO ROSSI rispose."
        out = self._analizza(t, custom=[{"value": "Mario Rossi", "tag": "FULLNAME"}])

        self.assertEqual(out["n_entities"], 2)
        self.assertNotIn("rossi", out["anonymized_text"].lower())

    def test_ignore_non_tocca_entita_diversa(self):
        # ignoro "Rossi" ma l'entita' e' "Mario Rossi": il valore intero e' un altro,
        # quindi il placeholder resta (il match del falso positivo e' ESATTO)
        t = "Saluta Mario Rossi."
        out = self._analizza(t, custom=[{"value": "Mario Rossi", "tag": "FULLNAME"}],
                             ignore=["Rossi"])

        self.assertIn("[FULLNAME_1]", out["anonymized_text"])


@unittest.skipIf(APP is None, "serve l'ambiente dell'app (torch/transformers/flask/fitz)")
class CorrezioniNelGruppo(unittest.TestCase):
    """Le correzioni valgono per TUTTO il gruppo, come la mappa condivisa."""

    def _gruppo(self, testi, **kw):
        with patch.object(APP, "detect_model", return_value=([], 0)):
            return APP.analyze_group(testi, names=[f"f{i}.pdf" for i in range(len(testi))],
                                     **kw)

    def test_falso_positivo_in_piu_file(self):
        t1 = f"Conto {IB1} primo atto."
        t2 = f"Secondo atto, conto {IB1} ancora."
        out = self._gruppo([t1, t2], ignore=[IB1])

        for f in out["files"]:
            self.assertIn(IB1, f["anonymized_text"])
            self.assertNotIn("[IBAN", f["anonymized_text"])
        self.assertEqual(out["mapping"], {})
        self.assertEqual(out["provenance"], {"f0.pdf": [], "f1.pdf": []})

    def test_stesso_valore_aggiunto_in_piu_file_stesso_placeholder(self):
        t1 = "Primo atto firmato da Mario Rossi."
        t2 = "Secondo atto firmato da Mario Rossi."
        out = self._gruppo([t1, t2],
                           custom=[{"value": "Mario Rossi", "tag": "FULLNAME"}])

        for f in out["files"]:
            self.assertIn("[FULLNAME_1]", f["anonymized_text"])
        self.assertEqual(out["mapping"], {"[FULLNAME_1]": "Mario Rossi"})
        self.assertIn("[FULLNAME_1]", out["provenance"]["f0.pdf"])
        self.assertIn("[FULLNAME_1]", out["provenance"]["f1.pdf"])


@unittest.skipIf(APP is None, "serve l'ambiente dell'app (torch/transformers/flask/fitz)")
class ParseAdjust(unittest.TestCase):
    """Il parser delle correzioni: body JSON (liste) oppure campi multipart (stringhe)."""

    def test_none_e_malformato_sono_liste_vuote(self):
        self.assertEqual(APP._parse_adjust(None), [])
        self.assertEqual(APP._parse_adjust("non e' json"), [])
        self.assertEqual(APP._parse_adjust("{struttura sbagliata}"), [])
        self.assertEqual(APP._parse_adjust({"value": "x"}), [])

    def test_stringa_json_da_multipart(self):
        self.assertEqual(APP._parse_adjust('["Valore A","Valore B"]'),
                         ["Valore A", "Valore B"])
        self.assertEqual(APP._parse_adjust('[{"value":"V","tag":"CF"}]'),
                         [{"value": "V", "tag": "CF"}])

    def test_pulizia_elementi(self):
        out = APP._parse_adjust([{"value": "  Mario Rossi  ", "tag": " fullname "},
                                 {"value": "", "tag": "CF"},
                                 {"tag": "CF"},
                                 "   ",
                                 12345])
        self.assertEqual(out, [{"value": "Mario Rossi", "tag": "FULLNAME"}, "12345"])

    def test_cap_su_numero_e_lunghezza(self):
        molti = [{"value": "v" * (APP.MAX_ADJUST_LEN + 50), "tag": "CF"}
                 for _ in range(APP.MAX_ADJUST_ITEMS + 10)]
        out = APP._parse_adjust(molti)

        self.assertEqual(len(out), APP.MAX_ADJUST_ITEMS)
        self.assertTrue(all(len(x["value"]) == APP.MAX_ADJUST_LEN for x in out))


@unittest.skipIf(APP is None, "serve l'ambiente dell'app (torch/transformers/flask/fitz)")
class RottaAnalyze(unittest.TestCase):
    """/analyze accetta ignore_values/custom_values dal body JSON."""

    def test_correzioni_via_json_body(self):
        client = APP.app.test_client()
        with patch.object(APP, "detect_model", return_value=([], 0)):
            r = client.post("/analyze", json={
                "text": f"Conto {IB1} di Mario Rossi.",
                "ignore_values": [IB1],
                "custom_values": [{"value": "Mario Rossi", "tag": "FULLNAME"}],
            })

        self.assertEqual(r.status_code, 200)
        d = r.get_json()
        self.assertIn(IB1, d["anonymized_text"])              # falso positivo: in chiaro
        self.assertIn("[FULLNAME_1]", d["anonymized_text"])   # falso negativo: mascherato
        self.assertEqual(d["mapping"], {"[FULLNAME_1]": "Mario Rossi"})
        self.assertNotIn(IB1, (d["mapping"] or {}).values())


if __name__ == "__main__":
    unittest.main()
