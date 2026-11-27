"""Formule di luogo esclusivamente sintetiche, senza documenti o modello."""

import re
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "app"))
from notarial_places import classify_place


class NotarialPlaces(unittest.TestCase):
    def classify(self, text, value, label, occurrence=0):
        matches = list(re.finditer(re.escape(value), text))
        match = matches[occurrence]
        return classify_place(text, match.start(), match.end(), label)

    def category(self, text, value, label, occurrence=0):
        result = self.classify(text, value, label, occurrence)
        return result[0] if result else None

    def test_stipula_requires_complete_opening_formula(self):
        text = ("REPUBBLICA ITALIANA\nIl giorno 13 (tredici) gennaio 2032;\n"
                "in Ravenna (RA), in via della Quercia n. 62, CAP 48100;\n"
                "davanti a me dottor ELIO CORTI, Notaio del Collegio Notarile.")
        for value, label, suffix in (("Ravenna", "CITY", "comune"), ("RA", "PROVINCE", "provincia"),
                                     ("via della Quercia", "STREET", "indirizzo"),
                                     ("62", "BUILDINGNUM", "civico"), ("48100", "ZIPCODE", "cap")):
            with self.subTest(label=label):
                self.assertEqual(self.category(text, value, label), "luoghi_documentali.stipula_" + suffix)

    def test_opening_does_not_capture_birthplace_or_later_address(self):
        text = ("REPUBBLICA ITALIANA\nIl giorno 13 gennaio 2032;\nin Ravenna;\n"
                "davanti a me ELIO CORTI, Notaio; è presente Neri, nato a Ferrara e residente a Parma.")
        for city in ("Ferrara", "Parma"):
            self.assertIsNone(self.classify(text, city, "CITY"))
        self.assertIsNone(self.classify("Il giorno 13 gennaio 2032 è nato a Ravenna.", "Ravenna", "CITY"))

    def test_opening_without_notary_or_republic_is_not_enough(self):
        for text in ("REPUBBLICA ITALIANA\nIl giorno 13 gennaio 2032; in Ravenna; davanti a me il testimone.",
                     "Il giorno 13 gennaio 2032; in Ravenna; davanti a me ELIO CORTI, Notaio."):
            self.assertIsNone(self.classify(text, "Ravenna", "CITY"))

    def test_district_multiple_places_and_linewrap(self):
        text = ("Notaio del Collegio Notarile dei Distretti Riuniti di Ravenna e\n"
                "San Lupo del Monte, con sede ad Imola;")
        for city in ("Ravenna", "San Lupo del Monte"):
            self.assertEqual(self.category(text, city, "CITY"), "luoghi_documentali.distretto_notarile")
        self.assertEqual(self.category(text, "Imola", "CITY"), "luoghi_documentali.sede_notaio_comune")

    def test_notary_seat_address(self):
        text = "ELIO CORTI, notaio con sede in Imola, via delle Felci n. 73 (BO);"
        for value, label, suffix in (("Imola", "CITY", "comune"), ("via delle Felci", "STREET", "indirizzo"),
                                     ("73", "BUILDINGNUM", "civico"), ("BO", "PROVINCE", "provincia")):
            with self.subTest(label=label):
                self.assertEqual(self.category(text, value, label), "luoghi_documentali.sede_notaio_" + suffix)

    def test_company_seat_does_not_become_notary_seat(self):
        for text in ("Il notaio riceve la società Bosco S.r.l. con sede a Imola.",
                     "Bosco S.r.l. con sede legale a Imola.",
                     "Il notaio ha sede a Ravenna; la società Bosco con sede a Imola."):
            self.assertIsNone(self.classify(text, "Imola", "CITY"))
        text = "Il notaio ha ricevuto la società ed è con sede a Imola, poi visita Ravenna."
        self.assertIsNone(self.classify(text, "Ravenna", "CITY"))
        text = "Il notaio con sede a Imola, testo 73 del repertorio."
        self.assertIsNone(self.classify(text, "73", "BUILDINGNUM"))

    def test_district_does_not_leak_into_new_context(self):
        text = ("Collegio Notarile dei Distretti Riuniti di Ravenna;\n"
                "Ada Cerri, nata a Ferrara, è residente in Parma.")
        for city in ("Ferrara", "Parma"):
            self.assertIsNone(self.classify(text, city, "CITY"))

    def test_registration_and_transcription_offices(self):
        for formula in ("registrato a", "trascritto in", "registrata ad"):
            text = f"Atto {formula} Ravenna in data 18 gennaio 2032, al n. 7254."
            self.assertEqual(self.category(text, "Ravenna", "CITY"), "luoghi_documentali.ufficio_comune")
            self.assertIsNone(self.classify(text, "7254", "BUILDINGNUM"))
        text = "Atto registrato a Ravenna (RA) il 18 gennaio 2032."
        self.assertEqual(self.category(text, "RA", "PROVINCE"), "luoghi_documentali.ufficio_provincia")

    def test_registration_does_not_absorb_later_city(self):
        text = "Atto registrato a Ravenna in data 18 gennaio 2032; residente a Ferrara."
        self.assertIsNone(self.classify(text, "Ferrara", "CITY"))
        text = "Atto registrato a Ravenna in data 18 gennaio 2032, annotazione RA."
        self.assertIsNone(self.classify(text, "RA", "PROVINCE"))
        text = "La studentessa è iscritta a Ravenna all'università."
        self.assertIsNone(self.classify(text, "Ravenna", "CITY"))

    def test_administrative_authority_not_generic_municipality(self):
        for formula in ("permesso rilasciato dal Comune di", "domanda presentata al Comune di",
                        "istanza depositata presso il Comune di"):
            text = f"Il documento è una {formula} Ferrara in data 18 gennaio 2032."
            self.assertEqual(self.category(text, "Ferrara", "CITY"), "luoghi_documentali.ufficio_comune")
        for text in ("Immobile sito nel Comune di Ferrara, via delle Rose.",
                     "Catasto Fabbricati: Comune di Ferrara, foglio 43."):
            self.assertIsNone(self.classify(text, "Ferrara", "CITY"))

    def test_professionals_have_explicit_role_and_bounded_name(self):
        for formula in ("notaio ELIO CORTI di", "notaio Elio Bruno Corti di",
                        "perito industriale Lea Neri di"):
            text = f"Relazione redatta dal {formula} Modena, allegata all'atto."
            self.assertEqual(self.category(text, "Modena", "CITY"), "luoghi_documentali.sede_professionista_comune")
        text = "L'atto richiama il notaio Elio Corti di Modena (MO)."
        self.assertEqual(self.category(text, "MO", "PROVINCE"), "luoghi_documentali.sede_professionista_provincia")
        for text in ("Elio Corti di Modena redige la relazione.",
                     "Notaio Elio Corti, nato a Modena.",
                     "Il notaio Elio Corti riferisce del residente di Modena."):
            self.assertIsNone(self.classify(text, "Modena", "CITY"))

    def test_negation_and_unrelated_types_are_not_classified(self):
        for text in ("Atto non registrato a Ravenna.", "Notaio non con sede a Ravenna."):
            self.assertIsNone(self.classify(text, "Ravenna", "CITY"))
        self.assertIsNone(classify_place("2032", 0, 4, "DATE"))
        with self.assertRaises(ValueError):
            classify_place("Ravenna", -1, 7, "CITY")

    def test_compound_negations_do_not_assert_office_location(self):
        for formula in ("non è stato registrato a", "non risulta essere stato registrato a",
                        "non fu mai trascritto a", "né è stato registrato a",
                        "non è stato presentato al Comune di"):
            with self.subTest(formula=formula):
                self.assertIsNone(self.classify(f"Il documento {formula} Ravenna.", "Ravenna", "CITY"))
        text = "Atto non registrato a Imola, ma poi trascritto a Ravenna."
        self.assertEqual(self.category(text, "Ravenna", "CITY"), "luoghi_documentali.ufficio_comune")

    def test_notary_seat_does_not_override_later_roles_or_subjects(self):
        for formula in ("Ada Neri res. a Ferrara,", "Ada Neri dom.to a Ferrara,",
                        "il bene sito a Ferrara,", "l'appartamento sito a Ferrara,",
                        "e Ada Neri in"):
            with self.subTest(formula=formula):
                text = f"Notaio con sede a Imola, {formula} via delle Felci n. 73."
                self.assertIsNone(self.classify(text, "via delle Felci", "STREET"))
                self.assertIsNone(self.classify(text, "73", "BUILDINGNUM"))


if __name__ == "__main__":
    unittest.main()
