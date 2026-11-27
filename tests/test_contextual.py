# -*- coding: utf-8 -*-
"""Classificazione e selezione su sole frasi sintetiche, senza modello o I/O."""

import re
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "app"))
import contextual  # noqa: E402


def spans(text, pattern, label):
    return [{"label": label, "start": m.start(), "end": m.end(), "score": 0.9,
             "validated": False, "source": "modello"} for m in re.finditer(pattern, text)]


class Classification(unittest.TestCase):
    def test_stessa_citta_in_tre_ruoli(self):
        text = ("Il signor Mario Rossi, nato a Padova, residente a Padova in via Roma 15, "
                "proprietario dell'immobile sito in Padova, via Venezia 20.")
        entities = spans(text, "Padova", "CITY")
        out = contextual.classify(text, entities)
        self.assertEqual([e["category"] for e in out],
                         ["nascita.luogo", "residenza.comune", "immobile.comune"])
        self.assertFalse(any(e["ambiguous"] for e in out))
        self.assertEqual(len({e["occurrence_id"] for e in out}), 3)
        self.assertNotIn("category", entities[0])
        chosen = contextual.select(out, ["residenza"])
        self.assertEqual([e["selected"] for e in chosen], [False, True, False])
        parts, pos = [], 0
        for e in chosen:
            if e["selected"]:
                parts += [text[pos:e["start"]], "[RESIDENZA]"]
                pos = e["end"]
        parts.append(text[pos:])
        self.assertEqual("".join(parts), text[:50] + "[RESIDENZA]" + text[56:])

    def test_piu_soggetti_non_ereditano_contesto(self):
        text = "Mario Rossi residente a Padova e Giulia Verdi a Padova."
        entities = spans(text, "Padova", "CITY") + spans(text, "Mario Rossi|Giulia Verdi", "FULLNAME")
        cities = [e for e in contextual.classify(text, entities) if e["label"] == "CITY"]
        self.assertEqual(cities[0]["category"], "residenza.comune")
        self.assertTrue(cities[1]["ambiguous"])
        self.assertEqual(cities[1]["category"], "altri.city")

    def test_residenza_domicilio_e_indirizzo_multilinea(self):
        text = "Residente a Padova,\nvia Roma n. 15 (PD), domiciliato a Verona, via Po 20."
        entities = (spans(text, "Padova|Verona", "CITY") + spans(text, "via Roma|via Po", "STREET")
                    + spans(text, "15|20", "BUILDINGNUM") + spans(text, "PD", "PROVINCE"))
        out = contextual.classify(text, entities)
        self.assertEqual([e["category"] for e in out], ["residenza.comune", "residenza.indirizzo",
            "residenza.civico", "residenza.provincia", "domicilio.comune", "domicilio.indirizzo",
            "domicilio.civico"])
        self.assertFalse(any(e["ambiguous"] for e in out))

    def test_nuova_societa_chiude_contesto_sede(self):
        text = "Alfa con sede legale a Padova e Beta a Verona."
        entities = spans(text, "Alfa|Beta", "ORG") + spans(text, "Padova|Verona", "CITY")
        cities = [e for e in contextual.classify(text, entities) if e["label"] == "CITY"]
        self.assertEqual(cities[0]["category"], "sede_societaria.comune")
        self.assertTrue(cities[1]["ambiguous"])

    def test_frase_e_paragrafo_chiudono_contesto(self):
        for separator in (". ", "; ", "\n\n"):
            text = "Residenza a Padova" + separator + "Verona"
            out = contextual.classify(text, spans(text, "Padova|Verona", "CITY"))
            self.assertFalse(out[0]["ambiguous"])
            self.assertTrue(out[1]["ambiguous"])

    def test_formule_abbreviate_e_errore_ocr_controllato(self):
        text = "Resldente a Padova, dom.to a Verona, nato il 01/02/1980."
        out = contextual.classify(text, spans(text, "Padova|Verona", "CITY")
                                   + spans(text, "01/02/1980", "DATE"))
        self.assertEqual([e["category"] for e in out],
                         ["residenza.comune", "domicilio.comune", "nascita.data"])

    def test_date_documentali_distinte_da_nascita(self):
        text = "Nato il 01/02/1980, atto del 03/04/2025, registrato il 05/04/2025."
        out = contextual.classify(text, spans(text, r"\d{2}/\d{2}/\d{4}", "DATE"))
        self.assertEqual([e["category"] for e in out],
                         ["nascita.data", "date_documentali.data", "date_documentali.data"])

    def test_sede_legale_non_residenza(self):
        text = "Alfa S.r.l. con sede legale in Padova, via Roma 15."
        out = contextual.classify(text, spans(text, "Padova", "CITY")
                                   + spans(text, "via Roma", "STREET") + spans(text, "15", "BUILDINGNUM"))
        self.assertEqual([e["category"] for e in out],
                         ["sede_societaria.comune", "sede_societaria.indirizzo", "sede_societaria.civico"])

    def test_contesti_conflittuali_e_relativi_restano_ambigui(self):
        for text in ("Residenza e domicilio in Padova", "Ivi residente a Padova", "Padova"):
            out = contextual.classify(text, spans(text, "Padova", "CITY"))
            self.assertTrue(out[0]["ambiguous"])
            self.assertEqual(contextual.select(out)[0]["review_status"], "pending")

    def test_overlap_non_scompare(self):
        text = "Padova"
        entities = spans(text, "Padova", "CITY") + spans(text, "dova", "STREET")
        self.assertTrue(all(e["ambiguous"] for e in contextual.classify(text, entities)))

    def test_offset_unicode_invariati_e_input_validato(self):
        text = "😀 Residenza a Padova\r\n"
        entity = spans(text, "Padova", "CITY")[0]
        out = contextual.classify(text, [entity])[0]
        self.assertEqual((out["start"], out["end"]), (entity["start"], entity["end"]))
        self.assertEqual(text[out["start"]:out["end"]], "Padova")
        for bad in ({"label": "CITY", "start": -1, "end": 3},
                    {"label": "CITY", "start": True, "end": 3},
                    {"label": "CITY", "start": 1, "end": 500}):
            with self.assertRaises(ValueError):
                contextual.classify(text, [bad])


class Selection(unittest.TestCase):
    def setUp(self):
        text = "Padova e Padova"
        self.entities = contextual.classify(text, spans(text, "Padova", "CITY"))

    def test_override_indipendente_e_revisione_esplicita(self):
        first, second = [e["occurrence_id"] for e in self.entities]
        out = contextual.select(self.entities, [], overrides={first: True, second: False})
        self.assertEqual([e["selected"] for e in out], [True, False])
        self.assertEqual([e["review_status"] for e in out], ["reviewed", "reviewed"])
        self.assertNotIn("selected", self.entities[0])

    def test_correzione_classificazione_rispetta_selezione(self):
        identifier = self.entities[0]["occurrence_id"]
        out = contextual.select(self.entities, ["residenza"],
                                classifications={identifier: "residenza.comune"})
        self.assertEqual([e["selected"] for e in out], [True, False])
        self.assertEqual(out[0]["review_status"], "reviewed")
        self.assertEqual(out[1]["review_status"], "pending")

    def test_input_non_validi_rifiutati(self):
        identifier = self.entities[0]["occurrence_id"]
        for options in ({"selected_categories": ["inesistente"]},
                        {"selected_categories": [{}]},
                        {"selected_categories": "residenza"},
                        {"overrides": {"e999999": True}},
                        {"overrides": {identifier: "false"}},
                        {"classifications": {identifier: "residenza"}},
                        {"classifications": {identifier: "nascita.data"}}):
            with self.assertRaises(ValueError):
                contextual.select(self.entities, **options)


if __name__ == "__main__":
    unittest.main()
