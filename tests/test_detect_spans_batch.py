# -*- coding: utf-8 -*-
"""detect_spans_batch (rilevamento a batch usato dai job CSV), con il modello finto.

Come test_tsv_anonymize.py: importa app.py (torch/transformers/flask/fitz) e si salta
se mancano. Il modello e' sostituito da un finto `nlp` che riconosce "Mario Rossi" e
"Anna Bianchi" come FULLNAME; il resto lo fa la rete regex (e-mail, CF...).
"""
import re
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "app"))


def _carica_app():
    if "app" in sys.modules:
        return sys.modules["app"]
    with patch("transformers.pipeline", return_value=MagicMock(return_value=[])):
        import app as pii_app
    return pii_app


try:
    APP = _carica_app()
except Exception:                       # noqa: BLE001
    APP = None

NAMES = re.compile(r"Mario Rossi|Anna Bianchi")
CALLS = []


def fake_nlp(chunks, batch_size=None):
    CALLS.append((len(chunks), batch_size))
    return [[{"entity_group": "FULLNAME", "start": m.start(), "end": m.end(), "score": 0.99}
             for m in NAMES.finditer(c)] for c in chunks]


@unittest.skipIf(APP is None, "torch/transformers/flask/fitz non installati")
class DetectSpansBatch(unittest.TestCase):
    def setUp(self):
        p = patch.object(APP, "nlp", fake_nlp)
        p.start()
        self.addCleanup(p.stop)
        CALLS.clear()

    def test_analyze_invariato(self):
        r = APP.analyze("Mario Rossi e Anna Bianchi, ancora Mario Rossi.")
        self.assertEqual(r["anonymized_text"], "[FULLNAME_1] e [FULLNAME_2], ancora [FULLNAME_1].")
        self.assertEqual(r["mapping"], {"[FULLNAME_1]": "Mario Rossi", "[FULLNAME_2]": "Anna Bianchi"})

    def test_spans_batch_uguali_ad_analyze(self):
        texts = ["Mario Rossi, mario@example.com.", "", "niente da vedere", "Anna Bianchi"]
        got = APP.detect_spans_batch(texts)
        for t, spans in zip(texts, got):
            kept = APP._merge(APP._candidates(t, APP.detect_model(t)[0], set()), t)
            self.assertEqual(spans, [(e["start"], e["end"], e["label"]) for e in kept])
        self.assertEqual(got[1], [])
        self.assertTrue(any(l == "FULLNAME" for _, _, l in got[0]))

    def test_un_solo_batch_e_batch_size(self):
        APP.detect_spans_batch(["Mario Rossi", "Anna Bianchi", ""])
        self.assertEqual(len(CALLS), 1)
        self.assertEqual(CALLS[0], (2, APP.MODEL_BATCH_SIZE))

    def test_testo_piu_lungo_di_un_chunk(self):
        words = ["parola"] * 300
        words[5] = "Mario Rossi"
        words[290] = "Anna Bianchi"
        t = " ".join(words)
        spans = APP.detect_spans_batch([t, "Mario Rossi"])
        self.assertGreater(len(APP.chunk_text(t)), 1)
        names = [(t[s:e]) for s, e, l in spans[0] if l == "FULLNAME"]
        self.assertEqual(names, ["Mario Rossi", "Anna Bianchi"])
        self.assertEqual(spans[1], [(0, 11, "FULLNAME")])

    def test_tag_esclusi(self):
        t = "Mario Rossi, mario@example.com"
        s = APP.detect_spans_batch([t], excluded=["FULLNAME"])[0]
        self.assertEqual([l for *_, l in s], ["EMAIL"])
        self.assertEqual(len(CALLS), 1)              # il modello gira comunque
        s = APP.detect_spans_batch([t], excluded=["EMAIL", "URL"])[0]
        self.assertEqual([l for *_, l in s], ["FULLNAME"])

if __name__ == "__main__":
    unittest.main()
