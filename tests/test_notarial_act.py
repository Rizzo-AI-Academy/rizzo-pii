"""Struttura notarile interamente inventata: NER, contesto e scelte distinti."""

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src/app"))
import contextual
import review


class NotarialAct(unittest.TestCase):
    def setUp(self):
        self.case = json.loads((ROOT / "tests/fixtures/notarial_act_synthetic.jsonl").read_text(encoding="utf-8"))
        self.text = self.case["text"]
        self.gold = self.case["entities"]
        # Le categorie gold non devono essere disponibili al classificatore.
        self.spans = [{k: e[k] for k in ("start", "end", "label")} |
                      {"source": "synthetic", "validated": False, "score": .9}
                      for e in self.gold]
        self.entities = review.assign_placeholders(contextual.classify(self.text, self.spans))

    def test_explicit_functions_and_true_abstentions_across_the_act(self):
        self.assertTrue(self.case["synthetic"])
        self.assertEqual(len(self.entities), 60)
        for original, actual, expected in zip(self.spans, self.entities, self.gold):
            with self.subTest(offset=original["start"]):
                self.assertEqual((actual["category"], actual["ambiguous"]),
                                 (expected["category"], expected["ambiguous"]))
                self.assertEqual(tuple(actual[k] for k in ("start", "end", "label")),
                                 tuple(original[k] for k in ("start", "end", "label")))
                self.assertIn(actual["category"], contextual._LEAVES)
                self.assertNotIn("category", original)
        self.assertEqual(sum(e["ambiguous"] for e in self.entities), 2)

    def test_only_residence_is_replaced_and_all_other_characters_remain(self):
        output, chosen = review.render(self.text, self.entities, categories=["residenza"])
        selected = [e for e in chosen if e["selected"]]
        self.assertEqual(len(selected), 3)
        expected = self.text
        for e in reversed(selected):
            expected = expected[:e["start"]] + e["ph"] + expected[e["end"]:]
        self.assertEqual(output["anonymized_text"], expected)
        self.assertIn("Comune di Parma", output["anonymized_text"])
        self.assertIn("via delle Magnolie n. 81", output["anonymized_text"])
        self.assertEqual(len(output["unresolved"]), 2)
        self.assertFalse(output["verification"]["complete_anonymization"])

    def test_cadastral_numbers_do_not_inherit_the_legacy_civic_exclusion(self):
        output, chosen = review.render(self.text, self.entities, excluded=["BUILDINGNUM"])
        parcels = [e for e in chosen if e["category"] == "immobile.catasto"]
        civics = [e for e in chosen if e["category"].endswith((".civico", "_civico"))]
        self.assertEqual(len(parcels), 3)
        self.assertTrue(all(e["selected"] for e in parcels))
        self.assertTrue(all(not e["selected"] for e in civics))
        self.assertNotIn("5678", output["anonymized_text"])
        self.assertIn("73", output["anonymized_text"])
        oid = parcels[0]["occurrence_id"]
        _, chosen = review.render(self.text, self.entities, excluded=["BUILDINGNUM"], overrides={oid: False})
        self.assertFalse(next(e for e in chosen if e["occurrence_id"] == oid)["selected"])
        self.assertTrue(all(e["selected"] for e in chosen if e["category"] == "immobile.catasto"
                            and e["occurrence_id"] != oid))

    def test_manual_correction_accepts_cadastral_semantics_with_original_ner_type(self):
        parcel = next(e for e in self.entities if e["category"] == "immobile.catasto")
        selected = contextual.select(self.entities, ["immobile.catasto"],
                                     classifications={parcel["occurrence_id"]: "immobile.catasto"})
        e = next(e for e in selected if e["occurrence_id"] == parcel["occurrence_id"])
        self.assertEqual(e["label"], "BUILDINGNUM")
        self.assertEqual(e["review_status"], "reviewed")
        self.assertTrue(e["selected"])

    def test_no_review_context_or_original_selected_values_in_irreversible_output(self):
        result, _ = review.render(self.text, self.entities)
        serialized = json.dumps(result)
        self.assertNotIn("context_excerpt", serialized)
        self.assertNotIn("review_text", serialized)
        self.assertEqual(result["mapping"], {})
        self.assertNotIn("DARIO FIORI", serialized)
        self.assertNotIn("via delle Magnolie", serialized)

    def test_specialized_rules_do_not_override_negative_context(self):
        for text in ("Non è stato un fabbricato sito in Mantova.",
                     "Fabbricato non è stato ubicato in Mantova.",
                     "Atto non è stato registrato a Mantova il 8 giugno 2024."):
            start = text.index("Mantova")
            result = contextual.classify(text, [{"start": start, "end": start + 7, "label": "CITY"}])
            self.assertTrue(result[0]["ambiguous"])

    def test_office_unit_and_engineer_location_have_explicit_non_address_roles(self):
        text = ("Atto registrato a Rovereto 3 il 7 aprile 2024. Relazione redatta "
                "dall'ingegnere Lea Cerri di Ravenna.")
        inputs = []
        for value, label in (("3", "BUILDINGNUM"), ("Ravenna", "CITY")):
            start = text.index(value)
            inputs.append({"start": start, "end": start + len(value), "label": label})
        out = contextual.classify(text, inputs)
        self.assertEqual([e["category"] for e in out],
                         ["luoghi_documentali.ufficio_unita", "luoghi_documentali.sede_professionista_comune"])
        self.assertFalse(any(e["ambiguous"] for e in out))

    def test_invoice_numbers_typed_as_dates_follow_document_reference_preferences(self):
        text = "Fatture emesse in data 7 aprile 2024 n. 8/24 e in data 9 maggio 2024 n. 17/24."
        inputs = []
        for value in ("7 aprile 2024", "8/24", "9 maggio 2024", "17/24"):
            start = text.index(value)
            inputs.append({"start": start, "end": start + len(value), "label": "DATE",
                           "source": "synthetic", "score": .9, "validated": False})
        entities = review.assign_placeholders(contextual.classify(text, inputs))
        result, selected = review.render(text, entities, excluded=["DATE"])
        self.assertEqual([e["selected"] for e in selected], [False, True, False, True])
        self.assertEqual(result["unresolved"], [])
        self.assertIn("7 aprile 2024", result["anonymized_text"])
        self.assertNotIn("8/24", result["anonymized_text"])
