# -*- coding: utf-8 -*-
"""OnnxTokenClassifier a batch: stesse entita' dell'inferenza uno alla volta.

Niente modello vero: tokenizer e sessione ONNX sono finti (un token per parola, i
logits dipendono solo dall'id del token). La sessione finta mette logits "sbagliati"
sulle posizioni di padding: se il batch non tagliasse ogni riga alla sua lunghezza, le
entita' cambierebbero. Serve solo numpy; si salta se manca (CI senza pip install).
"""
import re
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "app"))

try:
    import numpy as np
    from onnx_ner import OnnxTokenClassifier
except ImportError:                      # CI senza numpy
    np = None

LABELS = {0: "O", 1: "B-FULLNAME", 2: "I-FULLNAME", 3: "B-CITY"}
VOCAB = {"<pad>": 0, "Mario": 10, "Rossi": 11, "Milano": 12}


class _Enc:
    def __init__(self, text):
        words = list(re.finditer(r"\S+", text))
        self.ids = [VOCAB.get(w.group(), 5) for w in words]
        self.offsets = [(w.start(), w.end()) for w in words]
        self.attention_mask = [1] * len(words)
        self.special_tokens_mask = [0] * len(words)


class _Tok:
    def encode(self, text):
        return _Enc(text)

    def encode_batch(self, texts):
        return [_Enc(t) for t in texts]


class _Session:
    def __init__(self):
        self.calls = []

    def run(self, names, feeds):
        ids, mask = feeds["input_ids"], feeds["attention_mask"]
        self.calls.append(ids.shape)
        logits = np.zeros(ids.shape + (len(LABELS),), dtype=np.float32)
        best = np.where(ids == 10, 1, np.where(ids == 11, 2, np.where(ids == 12, 3, 0)))
        best = np.where(mask == 0, 3, best)          # padding: CITY, deve sparire tagliando
        np.put_along_axis(logits, best[..., None], 5.0, axis=-1)
        return [logits]


def _classifier():
    clf = OnnxTokenClassifier.__new__(OnnxTokenClassifier)
    clf.id2label = LABELS
    clf.tokenizer = _Tok()
    clf.pad_id = 0
    clf.input_names = {"input_ids", "attention_mask"}
    clf.session = _Session()
    return clf


@unittest.skipIf(np is None, "numpy non installato")
class OnnxBatch(unittest.TestCase):
    TEXTS = ["Mario Rossi abita a Milano", "x", "una frase lunga senza nomi di nessuno qui",
             "Milano", "ciao Mario Rossi", "Mario"]

    def key(self, ents):
        return [(e["entity_group"], e["start"], e["end"]) for e in ents]

    def test_batch_uguale_a_uno_alla_volta(self):
        clf = _classifier()
        one = [clf(t) for t in self.TEXTS]
        bat = clf(self.TEXTS, batch_size=4)
        self.assertEqual([self.key(e) for e in bat], [self.key(e) for e in one])
        self.assertEqual(self.key(bat[0]), [("FULLNAME", 0, 11), ("CITY", 20, 26)])
        self.assertEqual(bat[1], [])

    def test_una_inferenza_per_gruppo(self):
        clf = _classifier()
        clf(self.TEXTS, batch_size=4)
        self.assertEqual([s[0] for s in clf.session.calls], [4, 2])

    def test_senza_batch_size_uno_alla_volta(self):
        clf = _classifier()
        clf(self.TEXTS)
        self.assertEqual(len(clf.session.calls), len(self.TEXTS))


if __name__ == "__main__":
    unittest.main()
