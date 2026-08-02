import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "data_pipeline"))

import generate_targa_it as train_gen
import generate_targa_stress as stress_gen


class TargaGeneratorTests(unittest.TestCase):
    def test_training_mix_and_schema(self):
        rows = train_gen.generate(30, seed=123)
        counts = train_gen.summarize(rows)[0]
        self.assertEqual(
            counts, {"positive": 12, "hard_negative": 9, "mixed_pii": 9}
        )
        self.assertEqual(len({row["source_text"] for row in rows}), 30)
        for row in rows:
            train_gen.validate_record(row)

    def test_stress_mix_has_exact_categories(self):
        rows = stress_gen.generate(per_category=5, seed=456)
        counts = {
            category: sum(row["meta"]["category"] == category for row in rows)
            for category in ("positive", "hard_negative", "mixed_pii")
        }
        self.assertEqual(counts, {"positive": 5, "hard_negative": 5, "mixed_pii": 5})
        for row in rows:
            train_gen.validate_record(row)

    def test_stress_templates_are_disjoint_from_training_templates(self):
        training = set(
            train_gen.POSITIVE_TEMPLATES
            + train_gen.NEGATIVE_O_TEMPLATES
            + train_gen.NEGATIVE_DOCID_TEMPLATES
            + train_gen.MIXED_TEMPLATES
        )
        stress = set(
            stress_gen.POSITIVE_TEMPLATES
            + stress_gen.HARD_NEGATIVE_TEMPLATES
            + stress_gen.MIXED_TEMPLATES
        )
        self.assertFalse(training & stress)

    def test_plate_alphabet_excludes_ambiguous_letters(self):
        allowed = set(train_gen.base.PLATE_LETTERS)
        self.assertFalse(allowed & set("IOQU"))
        for _ in range(100):
            code = train_gen.plate_code()
            self.assertEqual(len(code), 7)
            self.assertTrue(set(code[:2] + code[-2:]) <= allowed)
            self.assertTrue(code[2:5].isdigit())

    def test_hard_negatives_have_no_targa_label(self):
        rows = stress_gen.generate(per_category=10, seed=789)
        negatives = [
            row for row in rows if row["meta"]["category"] == "hard_negative"
        ]
        self.assertTrue(negatives)
        for row in negatives:
            self.assertNotIn("B-TARGA", row["bio_labels"])
            self.assertFalse(
                any(entity["label"] == "TARGA" for entity in row["entities"])
            )


if __name__ == "__main__":
    unittest.main()
