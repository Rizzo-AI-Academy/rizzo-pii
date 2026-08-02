#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Valuta TARGA sullo stress test per modello puro e pipeline app completa."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
APP_DIR = ROOT / "src" / "app"
DEFAULT_DATA = ROOT / "dataset" / "validation" / "validation_targa_it_stress_300.jsonl"
DEFAULT_OUT = ROOT / "reports" / "targa_stress_baseline.json"


def exact_spans(entities):
    return {
        (int(entity["start"]), int(entity["end"]))
        for entity in entities
        if entity["label"] == "TARGA"
    }


def trim_span(text: str, start: int, end: int):
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


def calculate(rows, predictions):
    totals = {
        "tp": 0,
        "fp": 0,
        "fn": 0,
        "documents": len(rows),
        "documents_exact": 0,
        "documents_with_fp": 0,
    }
    by_category = defaultdict(
        lambda: {
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "documents": 0,
            "documents_exact": 0,
            "documents_with_fp": 0,
        }
    )

    for row, predicted in zip(rows, predictions):
        expected = exact_spans(row["entities"])
        category = row["meta"]["category"]
        stats = by_category[category]
        stats["documents"] += 1

        tp = len(expected & predicted)
        fp = len(predicted - expected)
        fn = len(expected - predicted)
        for target in (totals, stats):
            target["tp"] += tp
            target["fp"] += fp
            target["fn"] += fn
            if expected == predicted:
                target["documents_exact"] += 1
            if fp:
                target["documents_with_fp"] += 1

    def finalize(stats):
        precision = stats["tp"] / (stats["tp"] + stats["fp"]) if stats["tp"] + stats["fp"] else 0.0
        recall = stats["tp"] / (stats["tp"] + stats["fn"]) if stats["tp"] + stats["fn"] else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        result = dict(stats)
        result.update(
            {
                "precision": round(precision, 6),
                "recall": round(recall, 6),
                "f1": round(f1, 6),
                "document_exact_rate": round(
                    stats["documents_exact"] / stats["documents"], 6
                ),
                "false_positive_document_rate": round(
                    stats["documents_with_fp"] / stats["documents"], 6
                ),
            }
        )
        return result

    return {
        "overall": finalize(totals),
        "by_category": {
            category: finalize(stats) for category, stats in sorted(by_category.items())
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--model-dir", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    if args.model_dir:
        os.environ["PII_MODEL_DIR"] = str(args.model_dir.resolve())
    sys.path.insert(0, str(APP_DIR))
    import app as pii_app  # noqa: E402

    rows = [
        json.loads(line)
        for line in args.data.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    texts = [row["source_text"] for row in rows]
    results = pii_app.nlp(texts, batch_size=args.batch_size)

    raw_predictions = []
    app_predictions = []
    examples = {"raw_model": [], "model_plus_regex": []}
    for row, result in zip(rows, results):
        text = row["source_text"]
        model_entities = []
        raw_spans = set()
        for entity in result:
            start, end = trim_span(text, int(entity["start"]), int(entity["end"]))
            candidate = {
                "label": entity["entity_group"],
                "start": start,
                "end": end,
                "score": float(entity["score"]),
                "validated": False,
                "source": "modello",
            }
            model_entities.append(candidate)
            if candidate["label"] == "TARGA":
                raw_spans.add((start, end))

        merged = pii_app._merge(
            [dict(entity) for entity in model_entities] + pii_app.detect_regex(text),
            text,
        )
        merged_spans = exact_spans(merged)
        expected = exact_spans(row["entities"])
        raw_predictions.append(raw_spans)
        app_predictions.append(merged_spans)

        for name, predicted in (
            ("raw_model", raw_spans),
            ("model_plus_regex", merged_spans),
        ):
            if predicted != expected and len(examples[name]) < 12:
                examples[name].append(
                    {
                        "category": row["meta"]["category"],
                        "text": text,
                        "expected": sorted(expected),
                        "predicted": sorted(predicted),
                    }
                )

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": Path(pii_app.MODEL_DIR).name,
        "dataset": (
            args.data.resolve().relative_to(ROOT).as_posix()
            if args.data.resolve().is_relative_to(ROOT)
            else str(args.data)
        ),
        "rows": len(rows),
        "metric": "exact character-span match for TARGA",
        "raw_model": calculate(rows, raw_predictions),
        "model_plus_regex": calculate(rows, app_predictions),
        "error_examples": examples,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print(f"Stress test: {len(rows)} righe")
    for name in ("raw_model", "model_plus_regex"):
        overall = report[name]["overall"]
        negative = report[name]["by_category"]["hard_negative"]
        print(
            f"{name}: P={overall['precision']:.3f} R={overall['recall']:.3f} "
            f"F1={overall['f1']:.3f} document_exact={overall['document_exact_rate']:.3f} "
            f"hard_negative_FP_docs={negative['documents_with_fp']}/{negative['documents']}"
        )
    print(f"Report -> {args.out}")


if __name__ == "__main__":
    main()
