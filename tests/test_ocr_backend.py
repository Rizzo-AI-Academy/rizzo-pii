# -*- coding: utf-8 -*-
"""Selezione del backend e conversione del risultato RapidOCR in testo.

Non richiede rapidocr/onnxruntime installati: cio' che si testa qui e' la
logica di orchestrazione, non l'engine.
"""

import os
import tempfile
import unittest
from pathlib import Path

from . import _path  # noqa: F401  (sys.path)

import ocr
from ocr.rapid_ocr import RapidOcr, ocr_result_to_text


class FakeResult:
    def __init__(self, boxes=None, txts=None, scores=None):
        self.boxes, self.txts, self.scores = boxes, txts, scores


def box(x0, y0, x1, y1):
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]


class OcrResultToText(unittest.TestCase):

    def test_none_result_becomes_empty_string(self):
        """L'engine ritorna None quando non trova nulla: pagina bianca, non crash."""
        self.assertEqual(ocr_result_to_text(None), "")

    def test_result_without_recognitions_becomes_empty_string(self):
        self.assertEqual(ocr_result_to_text(FakeResult(boxes=[], txts=None)), "")

    def test_recognized_lines_are_rebuilt_in_reading_order(self):
        res = FakeResult(boxes=[box(100, 125, 400, 145), box(100, 100, 400, 120)],
                         txts=["seconda", "prima"], scores=[0.9, 0.9])
        self.assertEqual(ocr_result_to_text(res), "prima\nseconda")


class BackendAvailability(unittest.TestCase):

    def setUp(self):
        self._saved = {k: os.environ.get(k) for k in ("PII_OCR", "PII_OCR_MODEL_DIR")}

    def tearDown(self):
        for k, v in self._saved.items():
            os.environ.pop(k, None)
            if v is not None:
                os.environ[k] = v

    def test_models_are_missing_in_an_empty_directory(self):
        os.environ["PII_OCR_MODEL_DIR"] = os.path.join(os.path.dirname(__file__),
                                                       "no-such-dir")
        self.assertFalse(RapidOcr().models_present())

    def test_models_are_present_with_detection_classification_recognition(self):
        """PP-OCRv6 usa un unico modello di riconoscimento multilingua: i file
        sono tre e NON serve un dizionario di caratteri separato."""
        with tempfile.TemporaryDirectory() as d:
            for f in ("det.onnx", "cls.onnx", "rec.onnx"):
                Path(d, f).write_bytes(b"")
            os.environ["PII_OCR_MODEL_DIR"] = d
            self.assertTrue(RapidOcr().models_present())

    def test_models_are_missing_when_one_file_is_absent(self):
        with tempfile.TemporaryDirectory() as d:
            for f in ("det.onnx", "cls.onnx"):
                Path(d, f).write_bytes(b"")
            os.environ["PII_OCR_MODEL_DIR"] = d
            self.assertFalse(RapidOcr().models_present())

    def test_backend_is_unavailable_without_models_even_if_library_is_installed(self):
        os.environ["PII_OCR_MODEL_DIR"] = os.path.join(os.path.dirname(__file__),
                                                       "no-such-dir")
        self.assertFalse(RapidOcr().available())

    def test_backend_reports_its_name(self):
        self.assertEqual(RapidOcr().name, "rapidocr")

    def test_ocr_disabled_by_env_returns_no_backend(self):
        os.environ["PII_OCR"] = "off"
        self.assertIsNone(ocr.get_backend())

    def test_ocr_enabled_by_default(self):
        os.environ.pop("PII_OCR", None)
        self.assertIsNotNone(ocr.get_backend())


if __name__ == "__main__":
    unittest.main()
