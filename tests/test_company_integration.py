"""Soggetti e sedi inventati: stessa città, funzioni e scelte indipendenti."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src/app"))
import contextual
import review


def synthetic_company_document():
    text, entities, expected = "", [], []

    def plain(value):
        nonlocal text
        text += value

    def entity(label, value, category, pending=False):
        nonlocal text
        entities.append({"label": label, "start": len(text), "end": len(text) + len(value),
                         "source": "synthetic", "score": .9, "validated": False})
        expected.append((category, pending))
        text += value

    entity("FULLNAME", "Elena Bosco", "identificativi.nome_completo")
    plain(", nata a ")
    entity("CITY", "Trento", "nascita.luogo")
    plain(", residente a ")
    entity("CITY", "Trento", "residenza.comune")
    plain(" in ")
    entity("STREET", "via dei Laghi", "residenza.indirizzo")
    plain(" n. ")
    entity("BUILDINGNUM", "27", "residenza.civico")
    plain(", rappresentante della società\n- ")
    entity("ORG", '"BOSCO LAB S.R.L."', "identificativi.organizzazione")
    plain(" società con unico socio, con sede in\n")
    entity("CITY", "Trento", "sede_societaria.sede_comune")
    plain(" (")
    entity("PROVINCE", "TN", "sede_societaria.sede_provincia")
    plain(") viale ")
    entity("STREET", "Acacie", "sede_societaria.sede_indirizzo")
    plain(" n. ")
    entity("BUILDINGNUM", "42", "sede_societaria.sede_civico")
    plain(", con capitale sociale di Euro 4.700.000,00, iscritta nel Registro delle Imprese di ")
    entity("CITY", "Trento", "luoghi_documentali.registro_imprese_comune")
    plain(" (")
    entity("PROVINCE", "TN", "luoghi_documentali.registro_imprese_provincia")
    plain(").\nL’agenzia ")
    entity("ORG", '"CASA VERDE S.A.S. DI BOSCO & C."', "identificativi.organizzazione")
    plain(", con sede in ")
    entity("CITY", "Trento", "sede_societaria.sede_comune")
    plain(", ")
    entity("STREET", "Viale delle Ortensie", "sede_societaria.sede_indirizzo")
    plain(" n. ")
    entity("BUILDINGNUM", "63", "sede_societaria.sede_civico")
    plain(".\nL'impresa individuale denominata ")
    entity("ORG", "CASA VERDE DI LUCA BOSCO", "identificativi.organizzazione")
    plain(", corrente a ")
    entity("CITY", "Trento", "sede_societaria.sede_comune")
    plain(", in ")
    entity("STREET", "viale delle Ginestre", "sede_societaria.sede_indirizzo")
    plain(" n. ")
    entity("BUILDINGNUM", "75", "sede_societaria.sede_civico")
    plain(", iscritta al Registro delle Imprese di ")
    entity("CITY", "Trento", "luoghi_documentali.registro_imprese_comune")
    plain(".\nImmobile sito in ")
    entity("CITY", "Trento", "immobile.comune")
    plain(", ")
    entity("STREET", "via dei Castagni", "immobile.indirizzo")
    plain(" n. ")
    entity("BUILDINGNUM", "96", "immobile.civico")
    plain(".\nUna nota cita ")
    entity("CITY", "CITTA_1", "altri.city", True)
    plain(" senza specificarne la funzione.")
    return text, entities, expected


class CompanyIntegration(unittest.TestCase):
    def setUp(self):
        self.text, self.spans, self.expected = synthetic_company_document()
        self.entities = review.assign_placeholders(contextual.classify(self.text, self.spans))

    def test_same_city_in_multiple_roles_and_three_company_forms(self):
        self.assertEqual([(e["category"], e["ambiguous"]) for e in self.entities], self.expected)
        self.assertEqual([(e["start"], e["end"], e["label"]) for e in self.entities],
                         [(e["start"], e["end"], e["label"]) for e in self.spans])
        self.assertEqual(len({e["occurrence_id"] for e in self.entities}), len(self.entities))

    def test_company_selection_preserves_register_birth_residence_and_property(self):
        result, chosen = review.render(self.text, self.entities, categories=["sede_societaria"])
        selected = [e for e in chosen if e["selected"]]
        self.assertEqual(len(selected), 10)
        expected = self.text
        for e in reversed(selected):
            expected = expected[:e["start"]] + e["ph"] + expected[e["end"]:]
        self.assertEqual(result["anonymized_text"], expected)
        self.assertEqual(result["anonymized_text"].count("Trento"), self.text.count("Trento") - 3)
        for retained in ("nata a Trento", "residente a Trento", "Registro delle Imprese di Trento",
                         "Immobile sito in Trento", "via dei Castagni n. 96"):
            self.assertIn(retained, result["anonymized_text"])
        self.assertEqual(len(result["unresolved"]), 1)

    def test_individual_override_changes_only_one_company_city(self):
        company_cities = [e for e in self.entities if e["category"] == "sede_societaria.sede_comune"]
        identifier = company_cities[1]["occurrence_id"]
        result, chosen = review.render(self.text, self.entities, categories=["sede_societaria"],
                                       overrides={identifier: False})
        self.assertEqual(result["anonymized_text"].count("Trento"), self.text.count("Trento") - 2)
        self.assertEqual([e["selected"] for e in chosen if e["category"] == "sede_societaria.sede_comune"],
                         [True, False, True])

    def test_irreversible_response_has_no_selected_company_address_or_local_context(self):
        result, _ = review.render(self.text, self.entities, categories=["sede_societaria"])
        self.assertEqual(result["mapping"], {})
        self.assertNotIn("context_excerpt", str(result))
        for value in ("Acacie", "Viale delle Ortensie", "viale delle Ginestre"):
            self.assertNotIn(value, str(result))
