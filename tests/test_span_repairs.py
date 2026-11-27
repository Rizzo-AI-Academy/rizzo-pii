"""Riparazioni di span su soli numeri e strutture catastali inventati."""

import copy
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src/app"))
import contextual
import review
from span_repairs import normalize

try:
    import fitz
    import pdf_export
    import pdf_text
except ImportError:
    fitz = pdf_export = pdf_text = None


def cadastral_text(value="8,7"):
    return ("Catasto Fabbricati:\nComune di Belluno - foglio 23 - particelle n.ri:\n"
            "7841 sub. 9 - Via delle Camelie n. SNC - cat. A/3 - vani " + value +
            " - sup. cat. totale m.q. 145.")


def entity(start, end, label="BUILDINGNUM"):
    return {"start": start, "end": end, "label": label, "source": "modello",
            "score": .73, "validated": False}


def fragment(text, value=",7", *, occurrence=0, label="BUILDINGNUM"):
    start = -1
    for _ in range(occurrence + 1):
        start = text.index(value, start + 1)
    return entity(start, start + len(value), label)


class SpanRepairs(unittest.TestCase):
    def test_full_cadastral_decimal_replaces_fragment_and_preserves_recognition_evidence(self):
        text = cadastral_text()
        original = fragment(text)
        inputs = [original]
        snapshot = copy.deepcopy(inputs)
        result = normalize(text, inputs)
        self.assertEqual(inputs, snapshot)
        self.assertIsNot(result[0], original)
        self.assertEqual(text[result[0]["start"]:result[0]["end"]], "8,7")
        for key in ("label", "source", "score", "validated"):
            self.assertEqual(result[0][key], original[key])
        self.assertEqual(result[0]["span_repair"], "cadastral_decimal")
        self.assertEqual(result[0]["recognition_spans"], [
            {key: original[key] for key in ("start", "end", "label")}])
        self.assertTrue(all(set(s) == {"start", "end", "label"}
                            for s in result[0]["recognition_spans"]))

    def test_compatible_parts_of_one_numeric_value_become_one_occurrence(self):
        text = cadastral_text()
        start = text.index("8,7")
        inputs = [entity(start + 1, start + 3), entity(start, start + 1)]
        snapshot = copy.deepcopy(inputs)
        result = normalize(text, inputs)
        self.assertEqual(inputs, snapshot)
        self.assertEqual(len(result), 1)
        self.assertEqual((result[0]["start"], result[0]["end"]), (start, start + 3))
        self.assertEqual(result[0]["recognition_spans"], [
            {key: e[key] for key in ("start", "end", "label")}
            for e in sorted(inputs, key=lambda e: e["start"])])

    def test_complete_values_are_not_rewritten_and_normalization_is_idempotent(self):
        text = cadastral_text()
        full = fragment(text, "8,7")
        unchanged = normalize(text, [full])
        self.assertEqual(unchanged, [full])
        repaired = normalize(text, [fragment(text)])
        self.assertEqual(normalize(text, repaired), repaired)

    def test_surface_decimal_has_a_cadastral_measure_context(self):
        text = ("Catasto Fabbricati:\n7841 sub. 9 - Via delle Camelie n. SNC - "
                "sup. cat. totale m.q. 143,25 - cat. C/7.")
        result = normalize(text, [fragment(text, ",25")])
        self.assertEqual(text[result[0]["start"]:result[0]["end"]], "143,25")

    def test_unqualified_decimals_and_new_sections_are_never_expanded(self):
        for text in ("Vani 8,7 e superficie 143,25.",
                     "Catasto Fabbricati:\n7841 sub. 9 - vani 8,7,9 - cat. A/3.",
                     "Catasto Fabbricati:\n7841 sub. 9 - cat. A/3.\nArticolo 8 - Altro\nVani 8,7.",
                     "Catasto Fabbricati:\n7841 sub. 9 - cat. A/3 - Società con sede a Belluno, vani 8,7.",
                     "Catasto Fabbricati:\n7841 sub. 9 - cat. A/3.\nResidenza: Via delle Camelie n. 8,7."):
            with self.subTest(text=text):
                original = fragment(text)
                result = normalize(text, [original])
                self.assertEqual(result, [original])
                self.assertNotIn("span_repair", result[0])

    def test_incompatible_entity_blocks_expansion_without_dropping_either_span(self):
        text = cadastral_text()
        start = text.index("8,7")
        for label in ("AMOUNT", "DATE"):
            with self.subTest(label=label):
                inputs = [entity(start, start + 1, label), entity(start + 1, start + 3)]
                snapshot = copy.deepcopy(inputs)
                result = normalize(text, inputs)
                self.assertEqual(result, snapshot)
                self.assertEqual(inputs, snapshot)
                self.assertTrue(contextual.classify(text, result)[1]["ambiguous"])

    def test_invalid_ranges_types_and_overlapping_input_are_rejected(self):
        text = cadastral_text()
        valid = fragment(text)
        cases = [[{**valid, "start": -1}], [{**valid, "end": len(text) + 1}],
                 [{**valid, "start": True}], [{**valid, "end": valid["start"]}],
                 [valid, {**valid, "label": "DATE"}]]
        for inputs in cases:
            with self.subTest(inputs=inputs), self.assertRaises(ValueError):
                normalize(text, inputs)

    def test_unicode_offsets_and_only_one_selected_equal_value(self):
        text = "😀 Valore da mantenere 8,7.\n" + cadastral_text()
        inputs = [fragment(text, occurrence=1), fragment(text)]
        entities = review.assign_placeholders(contextual.classify(text, normalize(text, inputs)))
        self.assertEqual([e["category"] for e in entities], ["altri.buildingnum", "immobile.catasto"])
        first, repaired = entities
        self.assertEqual(text[repaired["start"]:repaired["end"]], "8,7")
        self.assertFalse(repaired["ambiguous"])
        result, chosen = review.render(text, entities, categories=["immobile.catasto"],
                                       overrides={first["occurrence_id"]: False})
        self.assertEqual([e["selected"] for e in chosen], [False, True])
        self.assertEqual(result["unresolved"], [])
        self.assertEqual(result["anonymized_text"],
                         text[:repaired["start"]] + repaired["ph"] + text[repaired["end"]:])
        self.assertIn("😀 Valore da mantenere 8,7.", result["anonymized_text"])
        self.assertIn("vani " + repaired["ph"], result["anonymized_text"])
        self.assertNotIn("vani 8", result["anonymized_text"])


@unittest.skipIf(fitz is None, "PyMuPDF non installato")
class RepairedPdf(unittest.TestCase):
    def test_native_pdf_removes_complete_selected_decimal_only(self):
        text = ("Valore da mantenere 8,7.\nCatasto Fabbricati:\n"
                "7841 sub. 9 - vani 8,7 - cat. A/3.")
        with fitz.open() as doc:
            page = doc.new_page(width=1000, height=300)
            page.insert_text((25, 50), text, fontsize=11)
            data = doc.tobytes()
        with fitz.open(stream=data, filetype="pdf") as doc:
            snapshot = pdf_text.collect_readable_document(doc)
        original_text = snapshot["text"]
        original_sources = copy.deepcopy(snapshot["sources"])
        inputs = [fragment(original_text), fragment(original_text, occurrence=1)]
        entities = review.assign_placeholders(contextual.classify(original_text, normalize(original_text, inputs)))
        first, selected = entities
        _, chosen = review.render(original_text, entities, categories=["immobile.catasto"],
                                  overrides={first["occurrence_id"]: False})
        self.assertEqual(original_text[selected["start"]:selected["end"]], "8,7")
        output, report = pdf_export.redact_occurrences(data, chosen, snapshot)
        self.assertEqual(report["errors"], [])
        self.assertEqual(report["residual"], [])
        self.assertEqual(report["occurrences"], 1)
        self.assertEqual(snapshot["text"], original_text)
        self.assertEqual(snapshot["sources"], original_sources)
        with fitz.open(stream=output, filetype="pdf") as doc:
            exported_text = doc[0].get_text()
        self.assertEqual(exported_text.count("8,7"), 1)
        self.assertIn("Valore da mantenere 8,7.", exported_text)
        self.assertNotIn("vani 8,7", exported_text)
