#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Creates a mock model at models/rizzo-pii-0.3B-v1.2.0/ that matches the real
model's interface (ModernBERT token-classification, 22 entity types, 45 BIO labels)
but with random weights. Good enough for build verification and pipeline testing.

Usage:  python create_mock_model.py
"""

import json
import sys
from pathlib import Path

# The 22 entity types from TASSONOMIA_TAG.md
ENTITY_TYPES = [
    "AGE", "AMOUNT", "BUILDINGNUM", "CATASTO", "CF", "CITY",
    "CREDITCARDNUMBER", "DATE", "DOCID", "EMAIL", "FULLNAME",
    "GENDER", "IBAN", "ID_DOC", "ORG", "PIVA", "PROVINCE",
    "STREET", "TARGA", "TELEPHONENUM", "TIME", "ZIPCODE",
]

# Build BIO label list: O + B-/I- for each type, sorted
label_list = ["O"]
for t in sorted(ENTITY_TYPES):
    label_list.append(f"B-{t}")
    label_list.append(f"I-{t}")

label2id = {l: i for i, l in enumerate(label_list)}
id2label = {i: l for i, l in enumerate(label_list)}

print(f"Label count: {len(label_list)} (O + {len(ENTITY_TYPES)} types × 2 BIO)")

# Try to create the model using transformers
try:
    from transformers import AutoModelForTokenClassification, AutoTokenizer
except ImportError:
    print("ERROR: transformers not installed. Run: pip install transformers torch")
    sys.exit(1)

MODEL_NAME = "jhu-clsp/mmBERT-base"   # same backbone used by train_pii.py
OUT_DIR = Path(__file__).resolve().parent / "models" / "rizzo-pii-0.3B-v1.2.0"
OUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"Downloading tokenizer from {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print(f"Creating token-classification model ({len(label_list)} labels)...")
model = AutoModelForTokenClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(label_list),
    id2label=id2label,
    label2id=label2id,
    ignore_mismatched_sizes=True,   # the classifier head size won't match
)

print(f"Saving mock model to {OUT_DIR}/ ...")
model.save_pretrained(OUT_DIR)
tokenizer.save_pretrained(OUT_DIR)

# Mark it as a mock so nobody confuses it with a trained model
(OUT_DIR / "MOCK_MODEL.txt").write_text(
    "This is a MOCK model with random classifier weights.\n"
    "It matches the real model's interface but produces garbage predictions.\n"
    "Created by create_mock_model.py for build verification only.\n"
)

print(f"\nDone. Mock model saved to: {OUT_DIR}")
print(f"Files: {[f.name for f in sorted(OUT_DIR.iterdir())]}")
