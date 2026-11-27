"""Metriche aggregate distinte; corpus esclusivamente sintetico, nessun download.

Default: valuta il classificatore sugli span gold, senza caricare il modello.
--recognize aggiunge NER e pipeline combinata usando i pesi locali dell'app.
Non stampa testo, valori, offset o nomi dei documenti.
"""
import argparse
import json
import sys
from collections import Counter
from contextlib import redirect_stdout
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "app"))
import contextual


def scores(correct, predicted, expected):
    precision = correct / predicted if predicted else 0.0
    recall = correct / expected if expected else 0.0
    return {"precision": precision, "recall": recall,
            "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0}


def evaluate(corpus, recognizers=None):
    tp, predicted, expected = Counter(), Counter(), Counter()
    recognition = {name: Counter() for name in (recognizers or {})}
    total = automatic = automatic_correct = ambiguity_correct = 0
    for case in corpus:
        if case.get("synthetic") is not True:
            raise ValueError("Sono ammessi soltanto esempi dichiarati sintetici.")
        if recognizers and case.get("recognition_gold_available") is False:
            raise ValueError("Gli span NER simulati non sono un riferimento per misurare il riconoscimento.")
        text, gold = case["text"], case["entities"]
        spans = [{**e, "source": "gold", "score": 1.0, "validated": False} for e in gold]
        classified = contextual.classify(text, spans)
        for actual, target in zip(classified, gold):
            category = actual["category"]
            predicted[category] += 1
            expected[target["category"]] += 1
            correct = category == target["category"]
            if correct:
                tp[category] += 1
            total += 1
            ambiguity_correct += actual["ambiguous"] == target["ambiguous"]
            if not actual["ambiguous"]:
                automatic += 1
                automatic_correct += correct and not target["ambiguous"]
        reference = {(e["label"], e["start"], e["end"]) for e in gold}
        for name, detect in (recognizers or {}).items():
            found = {(e["label"], e["start"], e["end"]) for e in detect(text)}
            recognition[name].update(correct=len(found & reference), predicted=len(found), expected=len(reference))
    categories = sorted(expected.keys() | predicted.keys())
    per_category = {c: scores(tp[c], predicted[c], expected[c]) for c in categories}
    return {
        "contextual_on_gold_spans": {
            "entities": total, "macro_f1": sum(v["f1"] for v in per_category.values()) / len(categories) if categories else 0.0,
            "automatic_precision": automatic_correct / automatic if automatic else 0.0,
            "automatic_coverage": automatic / total if total else 0.0,
            "ambiguity_accuracy": ambiguity_correct / total if total else 0.0,
            "per_category": per_category},
        "recognition_exact_spans": {name: scores(**counts) for name, counts in recognition.items()} or None,
        "recognition_measured": bool(recognizers),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=ROOT / "tests/fixtures/notarial_synthetic.jsonl")
    parser.add_argument("--recognize", action="store_true", help="Valuta anche il modello locale invariato")
    args = parser.parse_args()
    recognizers = None
    if args.recognize:
        with redirect_stdout(sys.stderr):
            import app
        def combined(text):
            entities, _ = app.detect_model(text)
            app.complete_time(entities, text)
            return app._merge(entities + app.detect_regex(text), text)
        def model_postprocessed(text):
            entities, _ = app.detect_model(text)
            app.complete_time(entities, text)
            return app._merge(entities, text)
        recognizers = {"model_raw": lambda text: app.detect_model(text)[0],
                       "model_postprocessed": model_postprocessed, "combined": combined}
    try:
        corpus = [json.loads(line) for line in args.corpus.read_text(encoding="utf-8").splitlines() if line.strip()]
        result = evaluate(corpus, recognizers)
    except (OSError, ValueError, KeyError, TypeError):
        parser.error("Corpus sintetico non valido o non disponibile.")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
