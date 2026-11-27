"""Formule inventate: non contiene estratti o identificativi del documento utente."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "app"))
from notarial_rules import documentary


def classify(text, value, label):
    start = text.index(value)
    return documentary(text, start, start + len(value), label)


class DocumentaryRules(unittest.TestCase):
    def test_reference_prefix_number_symbols_and_line_wraps(self):
        for prefix in ("Repertorio n°", "Raccolta n.º", "Prot. n.", "Rep. nº", "Racc. n.ri",
                       "Repertorio\n n°", "Protocollo numero", "Prot. n°.", "prot. pratica n."):
            with self.subTest(prefix=prefix):
                result = classify(prefix + " 71234", "71234", "DOCID")
                self.assertEqual(result[0], "riferimenti.documentale")

    def test_reference_suffix_and_numeric_slash_identifier(self):
        for text, value in (("Atto in data 7 aprile 2027 n. 6281 rep. notaio Ada Fiori", "6281"),
                            ("Identificativo 781/582 repertorio", "781/582")):
            self.assertEqual(classify(text, value, "DOCID")[0], "riferimenti.documentale")

    def test_registration_numbers_require_explicit_bounded_clause(self):
        for text, value in (("registrato a Trento in data 7 aprile 2027 al n. 87219 serie 2T", "87219"),
                            ("trascritto a Rovigo in data 8 aprile 2027 ai n.ri 31926/22901", "31926/22901")):
            self.assertEqual(classify(text, value, "DOCID")[0], "riferimenti.documentale")
        for text in ("Numero isolato al n. 87219", "registrato a Trento; altro dato al n. 87219",
                     "registrato a Trento. Un elenco indica al n. 87219",
                     "registrato a Trento\n\naltro dato al n. 87219",
                     "registrato a Trento ma non al n. 87219"):
            self.assertIsNone(classify(text, "87219", "DOCID"))

    def test_date_formulas_with_modifiers_and_wrapped_locations(self):
        for prefix in ("contratto\npreliminare in data", "registrato a Trento in\ndata",
                       "registrata a San Donato Milanese il", "trascritto a Rovigo in data",
                       "permesso di costruire in data", "SCIA in variante in data",
                       "presentata al Comune di Trento in data", "protocollata in data",
                       "redatto in data", "depositata in data", "certificato di nascita del"):
            with self.subTest(prefix=prefix):
                result = classify(prefix + " 7 aprile 2027", "7 aprile 2027", "DATE")
                self.assertEqual(result[0], "date_documentali.data")

    def test_opening_notarial_formula_needs_future_notary_cue(self):
        text = ("COMPRAVENDITA\nREPUBBLICA ITALIANA\nIl giorno 7 (sette) aprile 2027;\n"
                "a Trento, via delle Viole n. 9;\ndavanti a me dottor Ada Fiori, Notaio, sono presenti...")
        self.assertEqual(classify(text, "7 (sette) aprile 2027", "DATE")[0], "date_documentali.data")
        for text in ("REPUBBLICA ITALIANA\nIl giorno 7 aprile 2027 si verifica un evento.",
                     "Il giorno 7 aprile 2027 davanti a me Ada Fiori, Notaio.",
                     "REPUBBLICA ITALIANA\nAda è nata il giorno 7 aprile 2027; davanti a me Ada Fiori, Notaio."):
            self.assertIsNone(classify(text, "7 aprile 2027", "DATE"))

    def test_written_year_is_part_of_the_same_bounded_opening_date(self):
        text = ("REPUBBLICA ITALIANA\nIl giorno 7 (sette) aprile 2027 (duemilaventisette);\n"
                "a Trento, via delle Viole n. 9; davanti a me Ada Fiori, Notaio.")
        self.assertEqual(classify(text, "duemilaventisette", "DATE")[0], "date_documentali.data")
        for text in ("REPUBBLICA ITALIANA\nAda è nata il giorno 7 aprile 2027 (duemilaventisette); "
                     "davanti a me Ada Fiori, Notaio.",
                     "REPUBBLICA ITALIANA\nIl giorno 7 aprile 2027; testo diverso duemilaventisette "
                     "davanti a me Ada Fiori, Notaio.",
                     "REPUBBLICA ITALIANA\nIl giorno del confronto risale al duemilaventisette; "
                     "davanti a me Ada Fiori, Notaio."):
            self.assertIsNone(classify(text, "duemilaventisette", "DATE"))

    def test_numbered_permit_and_protocol_require_an_explicit_date_clause(self):
        for prefix in ("permesso di costruire in data 7 aprile 2027 n.",
                       "protocollata in data 9 aprile 2027 al n."):
            self.assertEqual(classify(prefix + " 89125", "89125", "DOCID")[0], "riferimenti.documentale")
        for prefix in ("un riferimento n.", "permesso di costruire abitazione al n.",
                       "protocollata in data ignota al n.", "non protocollata in data 9 aprile 2027 al n.",
                       "protocollata in data di al n.", "protocollata in data    al n.",
                       "permesso di costruire in data 7 aprile 2027; altro riferimento n.",
                       "protocollata in data 9 aprile 2027 residente a Parma al n."):
            with self.subTest(prefix=prefix):
                self.assertIsNone(classify(prefix + " 89125", "89125", "DOCID"))

    def test_isolated_dates_birth_dates_and_negations_remain_unassigned(self):
        for prefix in ("in data", "nata il", "non protocollata in data", "non è stato registrato a Trento il",
                       "non fu registrato a Trento il", "non viene trascritto a Rovigo in data",
                       "contratto. In data", "contratto\n\nin data"):
            with self.subTest(prefix=prefix):
                self.assertIsNone(classify(prefix + " 7 aprile 2027", "7 aprile 2027", "DATE"))
        for text in ("non è il Repertorio n° 71234", "non registrato a Trento al n. 71234"):
            self.assertIsNone(classify(text, "71234", "DOCID"))

    def test_location_slots_do_not_consume_new_personal_context(self):
        for prefix in ("registrato a Trento residente a Parma in data",
                       "trascritto a Trento nato a Parma in data",
                       "presentata al Comune di Trento domiciliata a Parma in data",
                       "registrato a Trento non a Parma in data",
                       "trascritto a Trento matrimonio celebrato in data"):
            with self.subTest(prefix=prefix):
                self.assertIsNone(classify(prefix + " 7 aprile 2027", "7 aprile 2027", "DATE"))
        self.assertIsNone(classify("registrato a Trento in via delle Lanterne al n. 81927",
                                   "81927", "DOCID"))

    def test_other_ner_types_are_never_retyped(self):
        self.assertIsNone(classify("Repertorio n° 71234", "71234", "BUILDINGNUM"))

    def test_numbered_building_permit_chain_and_work_date(self):
        text = ("Permesso di costruire n. 7518/19 U.T. n. 82614/19 P.G. in\n"
                "data 17 aprile 2019 con inizio lavori in data 23 maggio 2020;")
        for value in ("7518/19", "82614/19"):
            self.assertEqual(classify(text, value, "DOCID")[0], "riferimenti.documentale")
        for value in ("17 aprile 2019", "23 maggio 2020"):
            self.assertEqual(classify(text, value, "DATE")[0], "date_documentali.data")
        text = "DIA in variante in data 8 giugno 2021 n. 9641/21 Progr. n. 83142/21 Prot. Gener."
        for value in ("9641/21", "83142/21"):
            self.assertEqual(classify(text, value, "DOCID")[0], "riferimenti.documentale")
        self.assertEqual(classify(text, "8 giugno 2021", "DATE")[0], "date_documentali.data")

    def test_request_protocol_rea_and_suffixes(self):
        text = ("Presentata al Comune di Parma richiesta in data 12 gennaio 2022 "
                "n. 7158/22 Prot. Gener. n. 812/22 progr.")
        for value in ("7158/22", "812/22"):
            self.assertEqual(classify(text, value, "DOCID")[0], "riferimenti.documentale")
        self.assertEqual(classify(text, "12 gennaio 2022", "DATE")[0], "date_documentali.data")
        for prefix in ("REA con n.", "R.E.A. n.", "Prot. Gener. n.", "Progr. n."):
            self.assertEqual(classify(prefix + " 817531", "817531", "DOCID")[0], "riferimenti.documentale")

    def test_invoice_dates_and_numeric_identifiers_are_separate_occurrences(self):
        text = ("Fatture emesse in data 17 marzo 2029 n. 8/29 e in data 21 ottobre 2029 n. 17/29, "
                "per attività professionali.")
        for value in ("17 marzo 2029", "21 ottobre 2029"):
            self.assertEqual(classify(text, value, "DATE")[0], "date_documentali.data")
        for value in ("8/29", "17/29"):
            for label in ("DATE", "DOCID"):
                self.assertEqual(classify(text, value, label)[0], "riferimenti.documentale")
        text = "Fattura emessa in data odierna n. 918/29"
        self.assertEqual(classify(text, "918/29", "DOCID")[0], "riferimenti.documentale")

    def test_payment_dates_compravendita_and_numbered_registry_office(self):
        text = ("Atto di compravendita in data 8 ottobre 2021, registrato a Parma 3 "
                "il 18 novembre 2021; bonifico bancario in data 12 dicembre 2021.")
        for value in ("8 ottobre 2021", "18 novembre 2021"):
            self.assertEqual(classify(text, value, "DATE")[0], "date_documentali.data")
        self.assertEqual(classify(text, "12 dicembre 2021", "DATE")[0], "economici.data")

    def test_executed_bank_transfer_dates_allow_only_explicit_modifiers(self):
        for modifier in ("eseguito", "disposto", "effettuato"):
            text = f"Compenso pagato mediante bonifico bancario {modifier} in data 14 maggio 2031 con addebito su c/c."
            with self.subTest(modifier=modifier):
                self.assertEqual(classify(text, "14 maggio 2031", "DATE")[0], "economici.data")
        for text in ("Bonifico bancario non eseguito in data 14 maggio 2031.",
                     "Non è stato disposto il bonifico bancario in data 14 maggio 2031.",
                     "Bonifico bancario eseguito; in data 14 maggio 2031.",
                     "Bonifico bancario eseguito. Nato in data 14 maggio 2031.",
                     "Bonifico bancario eseguito, residente a Parma in data 14 maggio 2031."):
            with self.subTest(text=text):
                self.assertIsNone(classify(text, "14 maggio 2031", "DATE"))

    def test_caparra_date_requires_amount_and_following_credit_evidence(self):
        for phrase in ("caparra confirmatoria", "caparra confirmatoria ed ora imputata a prezzo",
                       "caparra, ora imputata in conto prezzo"):
            text = (f"Importo di euro 18.400,00 (a titolo di {phrase}) in data 14 maggio 2031"
                    " con accredito sul conto corrente del beneficiario.")
            with self.subTest(phrase=phrase):
                self.assertEqual(classify(text, "14 maggio 2031", "DATE")[0], "economici.data")
        for text in ("Importo di euro 18.400,00, Ada è nata in data 14 maggio 2031 con accredito sul conto corrente.",
                     "Caparra confirmatoria in data 14 maggio 2031 con accredito sul conto corrente.",
                     "Importo di euro 18.400,00 (a titolo di caparra) in data 14 maggio 2031.",
                     "Importo di euro 18.400,00 (a titolo di caparra); in data 14 maggio 2031 con accredito sul conto corrente.",
                     "Importo non versato di euro 18.400,00 (a titolo di caparra) in data 14 maggio 2031 con accredito sul conto corrente.",
                     "Importo di euro 18.400,00 (a titolo di caparra), nata in data 14 maggio 2031 con accredito sul conto corrente."):
            with self.subTest(text=text):
                self.assertIsNone(classify(text, "14 maggio 2031", "DATE"))

    def test_expanded_document_context_never_crosses_personal_or_narrative_boundaries(self):
        for text, value, label in (
            ("Permesso di costruire; n. 7518/19", "7518/19", "DOCID"),
            ("Permesso di costruire riferito alla casa al n. 7518/19", "7518/19", "DOCID"),
            ("Permesso di costruire n. 7518/19, residente a Parma in data 17 aprile 2019", "17 aprile 2019", "DATE"),
            ("non è stato emesso il permesso di costruire n. 7518/19", "7518/19", "DOCID"),
            ("non è stato emesso il permesso di costruire n. 7518/19 Prot. Gener.", "7518/19", "DOCID"),
            ("non è la fattura emessa in data 17 marzo 2029 n. 8/29", "8/29", "DATE"),
            ("fatture emesse in data 17 marzo 2029; in data 21 ottobre 2029", "21 ottobre 2029", "DATE"),
            ("fatture emesse in data 17 marzo 2029, nata in data 21 ottobre 2029", "21 ottobre 2029", "DATE"),
            ("fatture emesse in data 17 marzo 2029, un altro riferimento n. 8/29", "8/29", "DATE"),
            ("riferimento isolato n. 8/29", "8/29", "DATE"),
            ("non è stato eseguito il bonifico bancario in data 12 dicembre 2021", "12 dicembre 2021", "DATE"),
        ):
            with self.subTest(text=text):
                self.assertIsNone(classify(text, value, label))


if __name__ == "__main__":
    unittest.main()
