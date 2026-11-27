"""Casi residui ricostruiti con valori inventati, senza documenti personali."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src/app"))
import contextual
import review
import span_repairs


class ResidualContext(unittest.TestCase):
    def test_all_residual_structures_have_explicit_context_without_manual_choices(self):
        text = ("Catasto Fabbricati:\n7841 sub. 9 - vani 8,7 - cat. A/3.\n"
                "Articolo 9 - Prezzo\nUn importo di euro 32.000,00 "
                "(a titolo di caparra confirmatoria ed ora imputata a prezzo) in data 9 gennaio 2028 "
                "con accredito sul conto corrente del venditore.\n"
                "Società iscritta presso la C.C.I.A.A. di Mantova al R.E.A. n. MN-48261.\n"
                "Mediazione pagata mediante un bonifico bancario eseguito in data 7 dicembre 2027 "
                "con addebito sul conto del compratore.\n"
                "Altro bonifico bancario eseguito in data 10 dicembre 2027 con accredito sul conto.\n"
                "15.3) Si richiamano il D.P.R. 18 aprile 1994 n. 74 e la L. 23 settembre 2014 n. 97.")
        specifications = ((",7", "BUILDINGNUM", "immobile.catasto"),
                          ("9 gennaio 2028", "DATE", "economici.data"),
                          ("Mantova", "CITY", "luoghi_documentali.camera_commercio_comune"),
                          ("7 dicembre 2027", "DATE", "economici.data"),
                          ("10 dicembre 2027", "DATE", "economici.data"),
                          ("15.", "BUILDINGNUM", "riferimenti.articolo"),
                          (") Si richiamano", "BUILDINGNUM", "riferimenti.articolo"),
                          ("aprile 1994", "DATE", "riferimenti.normativo"),
                          ("settembre 2014", "DATE", "riferimenti.normativo"))
        spans, categories = [], []
        for value, label, category in specifications:
            start = text.index(value)
            stop = start + (1 if value.startswith(") Si") else len(value))
            spans.append({"start": start, "end": stop, "label": label,
                          "source": "synthetic", "score": .8, "validated": False})
            categories.append(category)
        entities = review.assign_placeholders(contextual.classify(text, span_repairs.normalize(text, spans)))
        self.assertEqual([e["category"] for e in entities], categories)
        self.assertFalse(any(e["ambiguous"] for e in entities))
        self.assertEqual(text[entities[0]["start"]:entities[0]["end"]], "8,7")
        result, _ = review.render(text, entities, categories=["immobile.catasto"])
        self.assertEqual(result["unresolved"], [])
        self.assertIn("vani [BUILDINGNUM_1]", result["anonymized_text"])
        self.assertNotIn("vani 8", result["anonymized_text"])
        self.assertIn("C.C.I.A.A. di Mantova", result["anonymized_text"])
        self.assertIn("15.3) Si richiamano", result["anonymized_text"])
        for retained in ("9 gennaio 2028", "7 dicembre 2027", "10 dicembre 2027", "aprile 1994"):
            self.assertIn(retained, result["anonymized_text"])
