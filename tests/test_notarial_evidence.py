"""Solo strutture e valori inventati: nessun estratto di documenti utente."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "app"))
import notarial_evidence as rules


def entity(text, value, label="BUILDINGNUM", occurrence=0):
    start = -1
    for _ in range(occurrence + 1):
        start = text.index(value, start + 1)
    return {"label": label, "start": start, "end": start + len(value)}


class NumericEvidence(unittest.TestCase):
    def test_particella_marker_and_repeated_rows_are_independent(self):
        text = ("Catasto Fabbricati:\nComune di Rovigo - foglio 17 - particelle n.ri:\n"
                "6789 sub. 4 - Via delle Betulle n. 22 - cat. A/3;\n"
                "6789 sub. 7 - Via delle Betulle n. SNC - cat. C/7.")
        first, second = entity(text, "6789"), entity(text, "6789", occurrence=1)
        for original in (first, second, entity(text, "4")):
            snapshot = dict(original)
            result = rules.evidence(text, original)
            self.assertEqual(result["category"], "immobile.catasto")
            self.assertFalse(result["ambiguous"])
            self.assertEqual(original, snapshot)
        self.assertNotEqual(first["start"], second["start"])

    def test_direct_particella_is_explicit_even_without_section(self):
        for marker in ("particella n.", "particelle n.ri:", "particella n°"):
            text = marker + " 7846"
            self.assertEqual(rules.evidence(text, entity(text, "7846"))["category"], "immobile.catasto")

    def test_section_ending_stops_cadastral_inheritance(self):
        text = "Catasto Fabbricati:\n6789 sub. 4 - Via delle Betulle n. 22;\nArticolo 12 - Altro\n6789 sub. 7"
        self.assertIsNone(rules.evidence(text, entity(text, "6789", occurrence=1)))

    def test_false_civic_section_number(self):
        text = "Articolo 7 - Condizioni\n7.2) Le parti concordano. Via delle Betulle 7.2"
        out = rules.evidence(text, entity(text, "7.2"))
        self.assertEqual(out["category"], "riferimenti.articolo")
        self.assertFalse(out["ambiguous"])
        self.assertIsNone(rules.evidence(text, entity(text, "7.2", occurrence=1)))

    def test_section_fragments_are_classified_before_decimal_fragment_guard(self):
        text = "Il testo precedente termina.\n  15.3) Le parti dichiarano."
        for value in ("15.", "3", ")", "15.3", "15.3)"):
            with self.subTest(value=value):
                span = entity(text, value)
                snapshot = dict(span)
                result = rules.evidence(text, span)
                self.assertEqual(result["category"], "riferimenti.articolo")
                self.assertFalse(result["ambiguous"])
                self.assertEqual(span, snapshot)

    def test_section_fragment_rule_does_not_accept_unrelated_parenthesis_or_prose(self):
        for text in ("Le parti (delegato) dichiarano.", "Via del Lago 15.3) è citata.",
                     "15) Una lista senza sottoparagrafo."):
            self.assertIsNone(rules.evidence(text, entity(text, ")")))
        text = "15.3) Le parti dichiarano."
        self.assertIsNone(rules.evidence(text, entity(text, ") Le")))
        text = "Superficie 15,3."
        self.assertTrue(rules.evidence(text, entity(text, ",3"))["ambiguous"])

    def test_law_year_wholly_inside_explicit_law_date(self):
        text = "Ai sensi del D. Lgs. 18 aprile 2006, n. 88, si dichiara quanto segue."
        out = rules.evidence(text, entity(text, "2006"))
        self.assertEqual(out["category"], "riferimenti.normativo")
        self.assertFalse(out["ambiguous"])
        text = "Via delle Betulle n. 2006, indicata nel contratto."
        self.assertIsNone(rules.evidence(text, entity(text, "2006")))

    def test_full_law_date_is_normative_and_birth_date_is_not(self):
        text = "Ai sensi del D. Lgs. 18 aprile 2006, n. 88, si dichiara quanto segue."
        span = entity(text, "18 aprile 2006", "DATE")
        snapshot = dict(span)
        result = rules.evidence(text, span)
        self.assertEqual(result["category"], "riferimenti.normativo")
        self.assertFalse(result["ambiguous"])
        self.assertEqual(span, snapshot)
        for text in ("Ada Fanti è nata il 18 aprile 2006.",
                     "Il contratto è stato stipulato il 18 aprile 2006."):
            self.assertIsNone(rules.evidence(text, entity(text, "18 aprile 2006", "DATE")))

    def test_decimal_fragment_remains_pending_and_never_expands(self):
        text = "Vani 8,7 e superficie 143,25."
        for value in (",7", "8", ",25"):
            span = entity(text, value)
            snapshot = dict(span)
            out = rules.evidence(text, span)
            self.assertTrue(out["ambiguous"])
            self.assertEqual(out["evidence_type"], "numeric_fragment")
            self.assertEqual(span, snapshot)
        self.assertIsNone(rules.evidence(text, entity(text, "143,25")))

    def test_cadastral_decimal_bounds_require_a_row_and_immediate_metric_field(self):
        text = ("Catasto Fabbricati:\n6789 sub. 4 - Via delle Betulle n. SNC - "
                "cat. A/3 - vani 8,7 - sup. cat. totale m.q. 143,25;")
        for fragment, full in ((",7", "8,7"), ("8", "8,7"), ("8,7", "8,7"),
                               (",25", "143,25"), ("143,25", "143,25")):
            with self.subTest(fragment=fragment):
                expected = entity(text, full)
                start = expected["start"] + full.index(fragment)
                span = {"label": "BUILDINGNUM", "start": start, "end": start + len(fragment)}
                self.assertEqual(rules.cadastral_decimal_bounds(text, span["start"], span["end"]),
                                 (expected["start"], expected["end"]))
                result = rules.evidence(text, span)
                if fragment == full:
                    self.assertEqual(result["category"], "immobile.catasto")
                    self.assertFalse(result["ambiguous"])
                else:
                    self.assertTrue(result["ambiguous"])
                    self.assertEqual(result["evidence_type"], "numeric_fragment")

    def test_cadastral_decimal_proof_does_not_extend_to_other_contexts(self):
        for text in ("Vani 8,7 e superficie 143,25.",
                     "Catasto Fabbricati: vani 8,7.",
                     "Catasto Fabbricati:\n6789 sub. 4 - importo 8,7 Euro.",
                     "Catasto Fabbricati:\n6789 sub. 4\nArticolo 9 - Altro\nvani 8,7."):
            span = entity(text, ",7")
            self.assertIsNone(rules.cadastral_decimal_bounds(text, span["start"], span["end"]))
            self.assertTrue(rules.evidence(text, span)["ambiguous"])

    def test_equal_cadastral_decimals_keep_their_own_bounds(self):
        text = ("Catasto Fabbricati:\n6789 sub. 4 - vani 8,7;\n"
                "6789 sub. 7 - vani 8,7;")
        results = []
        for occurrence in (0, 1):
            span = entity(text, ",7", occurrence=occurrence)
            results.append(rules.cadastral_decimal_bounds(text, span["start"], span["end"]))
        self.assertNotEqual(results[0], results[1])
        self.assertEqual([text[start:end] for start, end in results], ["8,7", "8,7"])
        text = "Catasto Fabbricati:\n6789 sub. 4 - vani 8,7."
        span = entity(text, ",7")
        self.assertEqual(text[slice(*rules.cadastral_decimal_bounds(text, span["start"], span["end"]))], "8,7")

    def test_separate_date_day_and_year_inside_explicit_normative_date(self):
        for code in ("D.P.R.", "D. L.", "L."):
            text = f"La dichiarazione richiama il {code} 17 aprile 1994, n. 74."
            for value in ("17", "1994", "17 aprile 1994"):
                with self.subTest(code=code, value=value):
                    span = entity(text, value, "DATE")
                    snapshot = dict(span)
                    out = rules.evidence(text, span)
                    self.assertEqual(out["category"], "riferimenti.normativo")
                    self.assertFalse(out["ambiguous"])
                    self.assertEqual(span, snapshot)
        text = "Nata il 17 aprile 1994; il precedente D.P.R. è citato a parte."
        for value in ("17", "1994"):
            self.assertIsNone(rules.evidence(text, entity(text, value, "DATE")))

    def test_normative_month_and_year_fragment_uses_the_complete_legal_formula(self):
        for code in ("D.P.R.", "L."):
            text = f"Si applica il {code} 9 maggio 1993, n. 86."
            span = entity(text, "maggio 1993", "DATE")
            snapshot = dict(span)
            result = rules.evidence(text, span)
            self.assertEqual(result["category"], "riferimenti.normativo")
            self.assertFalse(result["ambiguous"])
            self.assertEqual(span, snapshot)
        text = "Si richiama la L. 9 maggio\n  1993, n. 86."
        self.assertFalse(rules.evidence(text, entity(text, "maggio\n  1993", "DATE"))["ambiguous"])
        for text in ("Ada nacque nel maggio 1993.",
                     "Richiamato un D.P.R.; Ada nacque nel maggio 1993.",
                     "L. 9\n\nmaggio 1993, n. 86."):
            self.assertIsNone(rules.evidence(text, entity(text, "maggio 1993", "DATE")))

    def test_normative_identifiers_require_legal_formula(self):
        for text, value in (("Ai sensi del D.P.R. 17 aprile 1994 n. 74.", "74"),
                            ("Ai sensi del D.L. 74/1994 si dichiara.", "74/1994"),
                            ("La L. 91/2010 è richiamata.", "91/2010")):
            with self.subTest(text=text):
                out = rules.evidence(text, entity(text, value, "DOCID"))
                self.assertEqual(out["category"], "riferimenti.normativo")
                self.assertFalse(out["ambiguous"])
        for text in ("Contratto n. 74/1994.", "D.P.R. citato a parte; contratto n. 74/1994.",
                     "D.P.R.\n\n74/1994"):
            self.assertIsNone(rules.evidence(text, entity(text, "74/1994", "DOCID")))

    def test_cadastral_class_slash_is_not_a_civic_and_keeps_its_span(self):
        text = "Catasto Fabbricati:\n6789 sub. 4 - contrà delle Rose n. SNC - cat. A/3."
        span = entity(text, "/")
        snapshot = dict(span)
        result = rules.evidence(text, span)
        self.assertEqual(result["category"], "immobile.catasto")
        self.assertFalse(result["ambiguous"])
        self.assertEqual(span, snapshot)
        self.assertIsNone(rules.evidence("Via delle Rose n. /", entity("Via delle Rose n. /", "/")))
        text = "Catasto Fabbricati:\nArticolo 8\ncat. A/3"
        self.assertIsNone(rules.evidence(text, entity(text, "/")))

    def test_construction_threshold_date_is_documentary_without_birth_inference(self):
        for formula in ("costruito in data anteriore all'", "edificata in data anteriore all’"):
            text = "Il fabbricato è stato " + formula + "9 febbraio 1958."
            out = rules.evidence(text, entity(text, "9 febbraio 1958", "DATE"))
            self.assertEqual(out["category"], "date_documentali.data")
            self.assertFalse(out["ambiguous"])
        for text in ("Ada è nata il 9 febbraio 1958.",
                     "Data anteriore al 9 febbraio 1958.",
                     "Il fabbricato non è stato costruito in data anteriore all'9 febbraio 1958."):
            self.assertIsNone(rules.evidence(text, entity(text, "9 febbraio 1958", "DATE")))


class PropertyEvidence(unittest.TestCase):
    def test_named_building_and_address_despite_interposed_name(self):
        text = ('Unità facenti parte del fabbricato plurifamiliare denominato Residence '
                '"Le Betulle", sito in Comune di Rovigo (RO), Via delle Acacie n. 22.')
        for value, label, category in (("Betulle", "STREET", "immobile.denominazione"),
                                       ('Residence "Le Betulle', "STREET", "immobile.denominazione"),
                                       ("Rovigo", "CITY", "immobile.comune"),
                                       ("RO", "PROVINCE", "immobile.provincia"),
                                       ("Via delle Acacie", "STREET", "immobile.indirizzo"),
                                       ("22", "BUILDINGNUM", "immobile.civico")):
            with self.subTest(value=value):
                self.assertEqual(rules.property_context(text, entity(text, value, label))[0], category)
        self.assertEqual(rules.property_context(text, entity(text, "Betulle", "CITY"))[0],
                         "immobile.denominazione")
        self.assertIsNone(rules.property_context(text, entity(text, "Residence", "CITY")))
        self.assertIsNone(rules.property_context(text, entity(text, 'denominato Residence "Le Betulle', "STREET")))

    def test_arbitrary_sito_has_no_property_inference(self):
        text = "Studio notarile sito in Rovigo, Via delle Acacie n. 22."
        self.assertIsNone(rules.property_context(text, entity(text, "Rovigo", "CITY")))
        text = "Il fabbricato non ha indirizzo. Uno studio contabile è sito in Belluno."
        self.assertIsNone(rules.property_context(text, entity(text, "Belluno", "CITY")))

    def test_other_subject_role_and_neighbouring_city_stop_property(self):
        for suffix in ("residente a Belluno", "confinante con il Comune di Belluno",
                       "Notaio con sede in Belluno"):
            text = "Fabbricato sito in Rovigo, " + suffix + "."
            self.assertIsNone(rules.property_context(text, entity(text, "Belluno", "CITY")))
        text = "Fabbricato non è sito in Rovigo."
        self.assertIsNone(rules.property_context(text, entity(text, "Rovigo", "CITY")))
        text = "Fabbricato non risulta essere ubicato a Rovigo."
        self.assertIsNone(rules.property_context(text, entity(text, "Rovigo", "CITY")))
        text = "Fabbricato sito a Rovigo e Davide Fanti a Belluno."
        target = entity(text, "Belluno", "CITY")
        self.assertIsNone(rules.property_context(text, target, [entity(text, "Davide Fanti", "FULLNAME")]))
        text = "Non è un fabbricato sito in Rovigo, si tratta di un ufficio esterno."
        self.assertIsNone(rules.property_context(text, entity(text, "Rovigo", "CITY")))

    def test_cadastral_municipality_streets_and_explicit_civics(self):
        text = ("Catasto Fabbricati:\nComune di Rovigo (RO) - foglio 17 - particelle n.ri:\n"
                "6789 sub. 4 - Via delle Betulle n. 22 - cat. A/3;\n"
                "6789 sub. 7 - Via delle Betulle n. SNC - cat. C/7.")
        for value, label, category, occurrence in (
                ("Rovigo", "CITY", "immobile.comune", 0),
                ("RO", "PROVINCE", "immobile.provincia", 0),
                ("Via delle Betulle", "STREET", "immobile.indirizzo", 0),
                ("Via delle Betulle", "STREET", "immobile.indirizzo", 1),
                ("22", "BUILDINGNUM", "immobile.civico", 0),
                ("SNC", "BUILDINGNUM", "immobile.civico", 0)):
            with self.subTest(value=value, occurrence=occurrence):
                self.assertEqual(rules.property_context(text, entity(text, value, label, occurrence))[0], category)
        self.assertIsNone(rules.property_context(text, entity(text, "3", occurrence=0)))

    def test_no_cadastral_address_outside_section(self):
        text = "6789 sub. 4 - Via delle Betulle n. 22 - cat. A/3."
        self.assertIsNone(rules.property_context(text, entity(text, "Via delle Betulle", "STREET")))

    def test_explicit_personal_or_company_role_closes_cadastral_scope(self):
        for role in ("Residenza", "Domicilio", "Sede legale"):
            text = ("Catasto Fabbricati:\n6789 sub. 4 - Via delle Betulle n. 22\n"
                    + role + ": Via delle Rose n. 13")
            self.assertIsNone(rules.property_context(text, entity(text, "Via delle Rose", "STREET")))
            self.assertIsNone(rules.property_context(text, entity(text, "13")))

    def test_contra_street_name_without_introducer_in_ner_span(self):
        for street_type in ("Contrà", "Contra'", "Contrada"):
            text = ("Catasto Fabbricati:\n6789 sub. 4 - " + street_type +
                    " delle Rose Vecchie n. SNC - cat. C/7;\n6789 sub. 7 - " +
                    street_type + " delle Rose Vecchie n. SNC - cat. C/7.")
            for occurrence in (0, 1):
                out = rules.property_context(text, entity(text, "delle Rose Vecchie", "STREET", occurrence))
                self.assertEqual(out[0], "immobile.indirizzo")

    def test_cadastral_boundaries_have_their_own_address_function(self):
        text = ("Confini:\n- del garage: particella n. 6789 sub 12, contrà S. Lupo\n"
                "Vecchio, particella n. 6789 sub 9 e particella n. 6788.")
        for value in ("S. Lupo\nVecchio", ". Lupo\nVecchio"):
            span = entity(text, value, "STREET")
            snapshot = dict(span)
            self.assertEqual(rules.property_context(text, span)[0], "immobile.confine_indirizzo")
            self.assertEqual(span, snapshot)

    def test_boundary_address_does_not_leak_to_other_roles_or_bare_roads(self):
        for text in ("Particella n. 6789 sub 12, contrà delle Rose.",
                     "Confini: contrà delle Rose.",
                     "Confini: particella n. 6789 sub 12.\nResidenza: contrà delle Rose."):
            self.assertIsNone(rules.property_context(text, entity(text, "delle Rose", "STREET")))
        text = "Confini: particella n. 6789 sub 12, Ada Fanti in contrà delle Rose."
        self.assertIsNone(rules.property_context(text, entity(text, "delle Rose", "STREET"),
                                                [entity(text, "Ada Fanti", "FULLNAME")]))


if __name__ == "__main__":
    unittest.main()
