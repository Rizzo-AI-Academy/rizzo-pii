"""Sedi e registri su sole formule inventate, senza modello o I/O."""

import re
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src/app"))
from company_context import classify_company


def entity(text, value, label, occurrence=0):
    match = list(re.finditer(r"(?<!\w)" + re.escape(value) + r"(?!\w)", text))[occurrence]
    return {"start": match.start(), "end": match.end(), "label": label}


class CompanyContext(unittest.TestCase):
    def classify(self, text, value, label, entities=(), occurrence=0):
        return classify_company(text, entity(text, value, label, occurrence), entities)

    def test_society_seat_without_legal_qualifier_and_wrapped_address(self):
        text = ('La società "RADICE SERVIZI S.R.L.", società con unico socio, con sede in\n'
                'Ravenna (RA) viale delle Felci n. 72, CAP 48100, capitale sociale versato.')
        for value, label, suffix in (("Ravenna", "CITY", "comune"), ("RA", "PROVINCE", "provincia"),
                                     ("delle Felci", "STREET", "indirizzo"), ("72", "BUILDINGNUM", "civico"),
                                     ("48100", "ZIPCODE", "cap")):
            with self.subTest(label=label):
                result = self.classify(text, value, label)
                self.assertIsNotNone(result)
                self.assertEqual(result[0], "sede_societaria.sede_" + suffix)

    def test_legal_form_without_society_noun_is_sufficient(self):
        for form in ("S.A.S.", "S.N.C.", "S.R.L.", "S.P.A.", "SRL", "SAS"):
            text = f'"GIRASOLE {form}", con sede a Imola (BO), contrà delle Rose n. 27.'
            with self.subTest(form=form):
                self.assertEqual(self.classify(text, "Imola", "CITY")[0], "sede_societaria.sede_comune")
                self.assertEqual(self.classify(text, "delle Rose", "STREET")[0], "sede_societaria.sede_indirizzo")
                self.assertEqual(self.classify(text, "27", "BUILDINGNUM")[0], "sede_societaria.sede_civico")

    def test_generic_seat_never_asserts_legal_seat(self):
        text = "La società Quercia con sede in CITTA_1."
        self.assertEqual(self.classify(text, "CITTA_1", "CITY")[0], "sede_societaria.sede_comune")
        text = "La società Quercia con sede legale nel Comune di Ravenna."
        self.assertEqual(self.classify(text, "Ravenna", "CITY")[0], "sede_societaria.comune")

    def test_individual_business_corrente_is_a_company_seat(self):
        text = ("Impresa individuale denominata BOTTEGA DEI GIGLI, corrente a Imola (BO), "
                "in viale dei Pioppi n. 96, iscritta al Registro delle Imprese di Ravenna.")
        for value, label, suffix in (("Imola", "CITY", "comune"), ("BO", "PROVINCE", "provincia"),
                                     ("viale dei Pioppi", "STREET", "indirizzo"), ("96", "BUILDINGNUM", "civico")):
            with self.subTest(label=label):
                self.assertEqual(self.classify(text, value, label)[0], "sede_societaria.sede_" + suffix)
        self.assertEqual(self.classify(text, "Ravenna", "CITY")[0], "luoghi_documentali.registro_imprese_comune")

    def test_ner_org_is_used_without_changing_input(self):
        text = '"Faggio Verde", con sede in Ravenna, via delle Primule n. 24.'
        org = entity(text, "Faggio Verde", "ORG")
        city = entity(text, "Ravenna", "CITY")
        copy = dict(city)
        self.assertEqual(classify_company(text, city, [org])[0], "sede_societaria.sede_comune")
        self.assertEqual(city, copy)
        self.assertEqual(org, entity(text, "Faggio Verde", "ORG"))

    def test_registry_place_is_distinct_and_same_city_occurrences_independent(self):
        text = ("La società Faggio con sede in Ravenna (RA), via dei Tigli n. 51, "
                "iscritta nel Registro delle Imprese di Ravenna (RA).")
        self.assertEqual(self.classify(text, "Ravenna", "CITY", occurrence=0)[0], "sede_societaria.sede_comune")
        self.assertEqual(self.classify(text, "Ravenna", "CITY", occurrence=1)[0], "luoghi_documentali.registro_imprese_comune")
        self.assertEqual(self.classify(text, "RA", "PROVINCE", occurrence=1)[0], "luoghi_documentali.registro_imprese_provincia")

    def test_new_notary_or_company_subject_wins_over_older_context(self):
        for text in ("La società Faggio incontra il notaio, con sede in Ravenna.",
                     "Società Faggio, poi il signor Ada Neri con sede in Ravenna.",
                     "Notaio del Collegio Notarile con sede a Ravenna."):
            self.assertIsNone(self.classify(text, "Ravenna", "CITY"))
        text = "Il notaio riceve la società Faggio S.A.S., con sede in Ravenna."
        self.assertEqual(self.classify(text, "Ravenna", "CITY")[0], "sede_societaria.sede_comune")
        text = "La società Faggio S.R.L. e Ada Neri con sede in Imola."
        self.assertIsNone(self.classify(text, "Imola", "CITY", [entity(text, "Ada Neri", "FULLNAME")]))
        text = "Impresa individuale denominata BOTTEGA DEL FAGGIO DI ADA NERI, corrente a Imola."
        self.assertEqual(self.classify(text, "Imola", "CITY", [entity(text, "ADA NERI", "FULLNAME")])[0],
                         "sede_societaria.sede_comune")

    def test_plain_seat_and_personal_corrente_do_not_imply_company(self):
        for text in ("Con sede in Ravenna.", "Ada Neri corrente a Ravenna.", "Ufficio con sede a Ravenna."):
            self.assertIsNone(self.classify(text, "Ravenna", "CITY"))
        text = "Comune di Ravenna con sede in Imola."
        self.assertIsNone(self.classify(text, "Imola", "CITY", [entity(text, "Comune di Ravenna", "ORG")]))
        for text, org in (("Agenzia delle Entrate con sede in Imola.", "Agenzia delle Entrate"),
                          ("Il notaio opera nello Studio Faggio, con sede in Imola.", "Studio Faggio")):
            self.assertIsNone(self.classify(text, "Imola", "CITY", [entity(text, org, "ORG")]))

    def test_new_person_role_and_sentence_stop_company_context(self):
        for text in ("Società Faggio con sede a Ravenna, residente a Imola.",
                     "Società Faggio con sede a Ravenna; annotazione Imola.",
                     "Società Faggio con sede a Ravenna. Via delle Rose a Imola.",
                     "Società Faggio con sede a Ravenna, immobile sito a Imola.",
                     "Società Faggio con sede a Ravenna\n\nImola"):
            self.assertIsNone(self.classify(text, "Imola", "CITY"))
        text = "Società Faggio con sede a Ravenna e Ada Neri in via dei Gelsi n. 46."
        subjects = [entity(text, "Ada Neri", "FULLNAME")]
        self.assertIsNone(self.classify(text, "via dei Gelsi", "STREET", subjects))
        self.assertIsNone(self.classify(text, "46", "BUILDINGNUM", subjects))
        self.assertIsNone(self.classify(text, "via dei Gelsi", "STREET"))

    def test_capital_identifiers_and_registry_end_address_scope(self):
        for suffix in ("capitale sociale con riferimento n. 64", "cod. fisc. n. 64", "partita IVA n. 64"):
            text = "Società Faggio con sede a Ravenna, via dei Gelsi n. 46, " + suffix
            self.assertIsNone(self.classify(text, "64", "BUILDINGNUM"))
        text = "Società Faggio con sede a Ravenna, Registro delle Imprese di Imola, via delle Rose n. 83."
        self.assertIsNone(self.classify(text, "via delle Rose", "STREET"))

    def test_negated_seats_and_unrelated_types_are_not_accepted(self):
        for text in ("Società Faggio non con sede in Ravenna.",
                     "Società Faggio non è stata con sede in Ravenna.",
                     "Società Faggio mai corrente a Ravenna."):
            self.assertIsNone(self.classify(text, "Ravenna", "CITY"))
        self.assertIsNone(self.classify("2028", "2028", "DATE"))
        with self.assertRaises(ValueError):
            classify_company("abc", {"start": -1, "end": 3, "label": "CITY"})

    def test_chamber_of_commerce_is_distinct_from_company_seat(self):
        for institution in ("C.C.I.A.A.", "C. C. I. A. A.", "CCIAA", "Camera di Commercio"):
            text = ("Società Faggio con sede a Ravenna, via delle Rose n. 46, "
                    "iscritta presso la " + institution + " di Imola (BO) al R.E.A. n. BO-84261.")
            with self.subTest(institution=institution):
                self.assertEqual(self.classify(text, "Ravenna", "CITY")[0], "sede_societaria.sede_comune")
                self.assertEqual(self.classify(text, "Imola", "CITY")[0],
                                 "luoghi_documentali.camera_commercio_comune")
                self.assertEqual(self.classify(text, "BO", "PROVINCE", occurrence=0)[0],
                                 "luoghi_documentali.camera_commercio_provincia")
        for text in ("Società Faggio con sede a Ravenna; nota sul Comune di Imola.",
                     "Camera di Commercio di Ravenna; residenza a Imola."):
            self.assertIsNone(self.classify(text, "Imola", "CITY"))
