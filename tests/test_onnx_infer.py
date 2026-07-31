#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""aggregate_entities: la logica pura del backend ONNX (Fase 4). Nessun modello richiesto.

Raggruppa i token BIO in entita' come aggregation_strategy="simple", con offset carattere
sul testo originale e spazi ai bordi rifilati.
"""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "app"))

from onnx_infer import aggregate_entities  # noqa: E402


class TestAggregateEntities(unittest.TestCase):
    def test_multi_token_entity_merged(self):
        text = "Mario Rossi paga."
        per_token = [
            {"label": "B-FULLNAME", "score": 0.9, "start": 0, "end": 5},   # Mario
            {"label": "I-FULLNAME", "score": 0.8, "start": 6, "end": 11},  # Rossi
            {"label": "O", "score": 0.99, "start": 12, "end": 16},         # paga
        ]
        ents = aggregate_entities(per_token, text)
        self.assertEqual(len(ents), 1)
        self.assertEqual(ents[0]["entity_group"], "FULLNAME")
        self.assertEqual((ents[0]["start"], ents[0]["end"]), (0, 11))
        self.assertEqual(ents[0]["word"], "Mario Rossi")

    def test_o_breaks_entities(self):
        text = "CF poi IBAN"
        per_token = [
            {"label": "B-CF", "score": 0.9, "start": 0, "end": 2},
            {"label": "O", "score": 0.9, "start": 3, "end": 6},
            {"label": "B-IBAN", "score": 0.9, "start": 7, "end": 11},
        ]
        ents = aggregate_entities(per_token, text)
        self.assertEqual([e["entity_group"] for e in ents], ["CF", "IBAN"])

    def test_type_change_splits(self):
        text = "aa bb"
        per_token = [
            {"label": "B-CF", "score": 1.0, "start": 0, "end": 2},
            {"label": "B-IBAN", "score": 1.0, "start": 3, "end": 5},
        ]
        ents = aggregate_entities(per_token, text)
        self.assertEqual([e["entity_group"] for e in ents], ["CF", "IBAN"])

    def test_same_type_consecutive_merge_like_simple(self):
        # "simple" fonde token adiacenti dello stesso tipo (anche oltre un B-)
        text = "aa bb"
        per_token = [
            {"label": "B-ORG", "score": 1.0, "start": 0, "end": 2},
            {"label": "B-ORG", "score": 1.0, "start": 3, "end": 5},
        ]
        ents = aggregate_entities(per_token, text)
        self.assertEqual(len(ents), 1)
        self.assertEqual((ents[0]["start"], ents[0]["end"]), (0, 5))

    def test_whitespace_trimmed_at_edges(self):
        text = "x  Mario  y"
        per_token = [
            {"label": "B-FULLNAME", "score": 1.0, "start": 1, "end": 8},   # "  Mario" con spazi
        ]
        ents = aggregate_entities(per_token, text)
        self.assertEqual(ents[0]["word"], "Mario")
        self.assertEqual((ents[0]["start"], ents[0]["end"]), (3, 8))

    def test_trailing_separator_trimmed(self):
        # la virgola finale del modello non deve entrare nell'entita' (ma il '.' sì: S.r.l.)
        text = "Tecnova S.r.l., paga"
        per_token = [{"label": "B-ORG", "score": 1.0, "start": 0, "end": 14}]  # "Tecnova S.r.l.,"
        ents = aggregate_entities(per_token, text)
        self.assertEqual(ents[0]["word"], "Tecnova S.r.l.")

    def test_score_is_mean(self):
        text = "Mario Rossi"
        per_token = [
            {"label": "B-FULLNAME", "score": 1.0, "start": 0, "end": 5},
            {"label": "I-FULLNAME", "score": 0.6, "start": 6, "end": 11},
        ]
        ents = aggregate_entities(per_token, text)
        self.assertAlmostEqual(ents[0]["score"], 0.8)

    def test_empty_input(self):
        self.assertEqual(aggregate_entities([], "qualsiasi"), [])

    def test_all_O(self):
        text = "solo contesto"
        per_token = [{"label": "O", "score": 1.0, "start": 0, "end": 4},
                     {"label": "O", "score": 1.0, "start": 5, "end": 13}]
        self.assertEqual(aggregate_entities(per_token, text), [])


if __name__ == "__main__":
    unittest.main()
