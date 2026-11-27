"""Formule sintetiche realistiche e limiti dell'ereditarietà del contesto.

Gli span sono dati al classificatore: questi test non misurano il NER.
"""
import re
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "app"))
import contextual


def entities(text, patterns):
    found = []
    for pattern, label in patterns:
        found.extend({"start": m.start(), "end": m.end(), "label": label,
                      "score": 0.9, "source": "synthetic"}
                     for m in re.finditer(pattern, text))
    return found


class ContextChallenge(unittest.TestCase):
    def classify(self, text, *patterns):
        return contextual.classify(text, entities(text, patterns))

    def test_address_form_header_survives_blank_line_between_fields(self):
        text = ("RESIDENZA\n\nComune: Padova\nIndirizzo: via delle Viole\n"
                "Civico: 14\nCAP: 35100\nProvincia: PD")
        out = self.classify(text, ("Padova", "CITY"), ("via delle Viole", "STREET"),
                            ("14", "BUILDINGNUM"), ("35100", "ZIPCODE"), ("PD", "PROVINCE"))
        self.assertEqual([e["category"] for e in out],
                         ["residenza.comune", "residenza.indirizzo", "residenza.civico",
                          "residenza.cap", "residenza.provincia"])
        self.assertFalse(any(e["ambiguous"] for e in out))

    def test_address_form_does_not_leak_into_new_section(self):
        text = ("RESIDENZA\n\nComune: Padova\nIndirizzo: via delle Viole\n\n"
                "PROCEDURA GIUDIZIARIA\nTribunale di Verona")
        out = self.classify(text, ("Padova|Verona", "CITY"), ("via delle Viole", "STREET"))
        self.assertEqual(out[0]["category"], "residenza.comune")
        self.assertEqual(out[-1]["category"], "altri.city")
        self.assertTrue(out[-1]["ambiguous"])

    def test_unlabelled_address_lines_under_explicit_heading(self):
        text = "RESIDENZA\n\nPadova\nvia delle Viole n. 14\n35100 (PD)"
        out = self.classify(text, ("Padova", "CITY"), ("via delle Viole", "STREET"),
                            ("14", "BUILDINGNUM"), ("35100", "ZIPCODE"), ("PD", "PROVINCE"))
        self.assertEqual([e["category"] for e in out],
                         ["residenza.comune", "residenza.indirizzo", "residenza.civico",
                          "residenza.cap", "residenza.provincia"])
        self.assertFalse(any(e["ambiguous"] for e in out))

    def test_heading_does_not_extend_past_narrative_sentence(self):
        text = "RESIDENZA\nComune: Padova. Il confronto cita Verona."
        out = self.classify(text, ("Padova|Verona", "CITY"))
        self.assertFalse(out[0]["ambiguous"])
        self.assertTrue(out[1]["ambiguous"])

    def test_bare_header_field_checks_prose_after_entity(self):
        for text in ("RESIDENZA\n\nPadova è citata come esempio e il domicilio è ignoto.",
                     "SEDE LEGALE\n\nPadova è la città del luogo di nascita e non della società."):
            with self.subTest(text=text):
                out = self.classify(text, ("Padova", "CITY"))
                self.assertEqual(out[0]["category"], "altri.city")
                self.assertTrue(out[0]["ambiguous"])

    def test_negations_and_neighbouring_municipality_are_not_current_addresses(self):
        for formula in ("non è più residente a", "non risulta essere residente a",
                        "non è residente né domiciliato a"):
            with self.subTest(formula=formula):
                out = self.classify(f"Ada Fiori {formula} Padova.", ("Padova", "CITY"))
                self.assertEqual(out[0]["category"], "altri.city")
                self.assertTrue(out[0]["ambiguous"])
        text = "Immobile sito a Padova, confinante con il Comune di Verona."
        out = self.classify(text, ("Padova|Verona", "CITY"))
        self.assertEqual(out[0]["category"], "immobile.comune")
        self.assertTrue(out[1]["ambiguous"])

    def test_street_name_does_not_trigger_office_context_change(self):
        text = "Residente a Padova, via del Tribunale 8."
        out = self.classify(text, ("Padova", "CITY"), ("via del Tribunale", "STREET"),
                            ("8", "BUILDINGNUM"))
        self.assertEqual(out[-1]["category"], "residenza.civico")
        self.assertFalse(any(e["ambiguous"] for e in out))

    def test_res_te_abbreviation_is_explicit_residence(self):
        text = "La sig.ra Ada Fiori, res.te in Padova, via delle Viole 14."
        out = self.classify(text, ("Ada Fiori", "FULLNAME"), ("Padova", "CITY"),
                            ("via delle Viole", "STREET"), ("14", "BUILDINGNUM"))
        self.assertEqual([e["category"] for e in out[1:]],
                         ["residenza.comune", "residenza.indirizzo", "residenza.civico"])
        self.assertFalse(any(e["ambiguous"] for e in out))

    def test_elected_domicile_does_not_change_residence(self):
        text = ("Ada Fiori, residente in Padova, con domicilio eletto in Verona, "
                "via dei Tigli 8.")
        out = self.classify(text, ("Padova|Verona", "CITY"), ("via dei Tigli", "STREET"),
                            ("8", "BUILDINGNUM"))
        self.assertEqual([e["category"] for e in out],
                         ["residenza.comune", "domicilio.comune", "domicilio.indirizzo",
                          "domicilio.civico"])

    def test_property_formulas_are_anchored_by_a_property_noun(self):
        for formula in ("bene sito in", "unità immobiliare sita in"):
            with self.subTest(formula=formula):
                text = f"Oggetto della stima: {formula} Padova, via dei Tigli 8."
                out = self.classify(text, ("Padova", "CITY"), ("via dei Tigli", "STREET"),
                                    ("8", "BUILDINGNUM"))
                self.assertEqual([e["category"] for e in out],
                                 ["immobile.comune", "immobile.indirizzo", "immobile.civico"])
                self.assertFalse(any(e["ambiguous"] for e in out))

    def test_transcription_date_not_transcription_office_location(self):
        text = "Atto trascritto a Padova il 14/05/2024."
        out = self.classify(text, ("Padova", "CITY"), (r"14/05/2024", "DATE"))
        self.assertEqual(out[0]["category"], "luoghi_documentali.ufficio_comune")
        self.assertFalse(out[0]["ambiguous"])
        self.assertEqual(out[1]["category"], "date_documentali.data")
        self.assertFalse(out[1]["ambiguous"])

    def test_documentary_date_headings(self):
        for formula in ("Iscrizione del", "Data sopralluogo:", "Data del sopralluogo:"):
            with self.subTest(formula=formula):
                text = f"{formula} 14/05/2024"
                out = self.classify(text, (r"14/05/2024", "DATE"))
                self.assertEqual(out[0]["category"], "date_documentali.data")
                self.assertFalse(out[0]["ambiguous"])

    def test_relative_and_conflicting_contexts_remain_pending(self):
        for formula in ("Ivi residente in", "Residenza e domicilio in"):
            with self.subTest(formula=formula):
                out = self.classify(f"{formula} Padova", ("Padova", "CITY"))
                self.assertTrue(out[0]["ambiguous"])
                self.assertEqual(contextual.select(out)[0]["review_status"], "pending")

    def test_new_subject_does_not_inherit_other_subject_residence(self):
        text = "Ada Fiori residente in Padova e Bruno Neri in Verona."
        out = self.classify(text, ("Ada Fiori|Bruno Neri", "FULLNAME"),
                            ("Padova|Verona", "CITY"))
        cities = [e for e in out if e["label"] == "CITY"]
        self.assertEqual(cities[0]["category"], "residenza.comune")
        self.assertEqual(cities[1]["category"], "altri.city")
        self.assertTrue(cities[1]["ambiguous"])

    def test_birth_and_documentary_dates_in_same_sentence(self):
        text = "Ada Fiori nata il 06/03/1982, iscrizione del 14/05/2024."
        out = self.classify(text, (r"\d{2}/\d{2}/\d{4}", "DATE"))
        self.assertEqual([e["category"] for e in out],
                         ["nascita.data", "date_documentali.data"])
        self.assertFalse(any(e["ambiguous"] for e in out))

    def test_unlabelled_narrative_paragraph_has_no_inherited_role(self):
        text = "Residente in Padova, via delle Viole 14.\n\nIl sopralluogo è avvenuto a Verona."
        out = self.classify(text, ("Padova|Verona", "CITY"))
        self.assertEqual(out[0]["category"], "residenza.comune")
        self.assertEqual(out[1]["category"], "altri.city")
        self.assertTrue(out[1]["ambiguous"])

    def test_judicial_office_does_not_become_property_location(self):
        text = "Immobile sito in Padova, procedura presso il Tribunale di Verona."
        out = self.classify(text, ("Padova|Verona", "CITY"))
        self.assertEqual(out[0]["category"], "immobile.comune")
        self.assertEqual(out[1]["category"], "altri.city")
        self.assertTrue(out[1]["ambiguous"])

    def test_same_city_occurrences_keep_their_roles_and_offsets(self):
        text = ("Ada Fiori nata a Padova, res.te in Padova, proprietaria del bene sito "
                "in Padova, con domicilio eletto in Verona.")
        original = entities(text, (("Padova|Verona", "CITY"),))
        out = contextual.classify(text, original)
        self.assertEqual([e["category"] for e in out],
                         ["nascita.luogo", "residenza.comune", "immobile.comune", "domicilio.comune"])
        self.assertEqual([(e["start"], e["end"]) for e in out],
                         [(e["start"], e["end"]) for e in original])
        self.assertEqual([e["selected"] for e in contextual.select(out, ["residenza"])],
                         [False, True, False, False])


if __name__ == "__main__":
    unittest.main()
