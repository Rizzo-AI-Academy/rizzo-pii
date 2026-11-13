# -*- coding: utf-8 -*-
"""Scelta del dispositivo del modello: CUDA -> MPS -> CPU, override PII_DEVICE e
ripiego su CPU se l'inferenza su MPS fallisce.

Come test_tsv_anonymize.py: importa app.py (torch/transformers/flask/fitz) e si salta
se mancano. Il modello e' finto.
"""
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


def fake_nlp(chunks, **kw):
    return [[{"entity_group": "FULLNAME", "start": 0, "end": 11, "score": 0.99}]
            if c.startswith("Mario Rossi") else [] for c in chunks]


@unittest.skipIf(APP is None, "torch/transformers/flask/fitz non installati")
class Device(unittest.TestCase):
    def test_pii_device_forza_il_dispositivo(self):
        self.assertEqual(APP._pick_device("cpu"), "cpu")
        self.assertEqual(APP._pick_device(" CPU "), "cpu")
        self.assertIn(APP._pick_device("auto"), ("cuda", "mps", "cpu"))
        with patch.object(APP.torch.cuda, "is_available", return_value=False):
            # non disponibile o sconosciuto -> scelta automatica (con avviso)
            self.assertNotEqual(APP._pick_device("cuda"), "cuda")
            self.assertNotEqual(APP._pick_device("gpu"), "gpu")

    def test_ordine_automatico(self):
        mps = MagicMock(is_available=MagicMock(return_value=True))
        with patch.object(APP.torch.cuda, "is_available", return_value=False), \
             patch.object(APP.torch.backends, "mps", mps, create=True):
            self.assertEqual(APP._pick_device("auto"), "mps")
        with patch.object(APP.torch.cuda, "is_available", return_value=True):
            self.assertEqual(APP._pick_device("auto"), "cuda")

    def test_fallback_mps_su_cpu(self):
        def boom(chunks, **kw):
            raise RuntimeError("mps")
        cpu = MagicMock(side_effect=fake_nlp)
        with patch.object(APP, "nlp", boom), patch.object(APP, "DEVICE_NAME", "mps"), \
             patch.object(APP, "_build_pipeline", return_value=cpu) as build:
            ents, _ = APP.detect_model("Mario Rossi scrive.")
            self.assertEqual([(e["start"], e["end"], e["label"]) for e in ents], [(0, 11, "FULLNAME")])
            self.assertEqual(APP.DEVICE_NAME, "cpu")
            build.assert_called_once_with("cpu")

    def test_su_cpu_gli_errori_non_si_nascondono(self):
        def boom(chunks, **kw):
            raise RuntimeError("cpu")
        with patch.object(APP, "nlp", boom), patch.object(APP, "DEVICE_NAME", "cpu"):
            with self.assertRaises(RuntimeError):
                APP.detect_model("Mario Rossi")

    def test_health_riporta_il_dispositivo(self):
        with patch.object(APP, "DEVICE_NAME", "mps"):
            r = APP.app.test_client().get("/health")
            self.assertEqual(r.get_json()["device"], "mps")


if __name__ == "__main__":
    unittest.main()
