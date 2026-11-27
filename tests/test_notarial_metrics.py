"""La classificazione su span gold non deve essere spacciata per accuratezza NER."""
import importlib.util
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("notarial_metrics", ROOT / "scripts/evaluate_notarial.py")
metrics = importlib.util.module_from_spec(spec)
spec.loader.exec_module(metrics)


class Metrics(unittest.TestCase):
    def corpus(self):
        return [json.loads(line) for line in (ROOT / "tests/fixtures/notarial_synthetic.jsonl").read_text(encoding="utf-8").splitlines()]

    def test_gold_classification_and_abstention_are_separate_from_recognition(self):
        result = metrics.evaluate(self.corpus())
        self.assertFalse(result["recognition_measured"])
        self.assertIsNone(result["recognition_exact_spans"])
        contextual = result["contextual_on_gold_spans"]
        self.assertEqual(contextual["macro_f1"], 1.0)
        self.assertEqual(contextual["ambiguity_accuracy"], 1.0)
        self.assertLess(contextual["automatic_coverage"], 1.0)

    def test_missed_entities_lower_recognition_without_changing_gold_classification(self):
        result = metrics.evaluate(self.corpus(), {"missing": lambda text: []})
        self.assertEqual(result["recognition_exact_spans"]["missing"]["recall"], 0.0)
        self.assertEqual(result["contextual_on_gold_spans"]["macro_f1"], 1.0)

    def test_only_synthetic_corpus_is_accepted(self):
        with self.assertRaises(ValueError):
            metrics.evaluate([{"synthetic": False}])

    def test_simulated_ner_errors_cannot_be_used_as_recognition_gold(self):
        case = {"synthetic": True, "recognition_gold_available": False}
        with self.assertRaises(ValueError):
            metrics.evaluate([case], {"model": lambda text: []})

    def test_notarial_challenge_covers_formulas_and_conservative_abstention(self):
        corpus = [json.loads(line) for line in
                  (ROOT / "tests/fixtures/notarial_challenge.jsonl").read_text(encoding="utf-8").splitlines()]
        result = metrics.evaluate(corpus)["contextual_on_gold_spans"]
        self.assertEqual(result["entities"], 68)
        self.assertEqual(result["macro_f1"], 1.0)
        self.assertEqual(result["automatic_precision"], 1.0)
        self.assertEqual(result["ambiguity_accuracy"], 1.0)
        self.assertLess(result["automatic_coverage"], 1.0)
