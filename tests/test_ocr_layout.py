# -*- coding: utf-8 -*-
"""Ricostruzione dell'ordine di lettura dai box dell'OCR."""

import unittest

from . import _path  # noqa: F401  (sys.path)

from ocr.layout import lines_to_text


def box(x0, y0, x1, y1):
    """Box nel formato RapidOCR: 4 vertici (x, y) in senso orario."""
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]


class LinesToText(unittest.TestCase):

    def test_empty_input_returns_empty_string(self):
        self.assertEqual(lines_to_text([], []), "")

    def test_none_boxes_returns_empty_string(self):
        self.assertEqual(lines_to_text(None, None), "")

    def test_boxes_on_same_line_are_joined_left_to_right(self):
        boxes = [box(300, 100, 400, 120), box(100, 102, 200, 122)]
        txts = ["ROSSI", "Mario"]
        self.assertEqual(lines_to_text(boxes, txts), "Mario ROSSI")

    def test_consecutive_lines_are_separated_by_single_newline(self):
        boxes = [box(100, 100, 400, 120), box(100, 125, 400, 145)]
        txts = ["prima riga", "seconda riga"]
        self.assertEqual(lines_to_text(boxes, txts), "prima riga\nseconda riga")

    def test_large_vertical_gap_opens_a_paragraph(self):
        boxes = [box(100, 100, 400, 120), box(100, 300, 400, 320)]
        txts = ["fine paragrafo", "nuovo paragrafo"]
        self.assertEqual(lines_to_text(boxes, txts),
                         "fine paragrafo\n\nnuovo paragrafo")

    def test_lines_are_ordered_top_to_bottom_regardless_of_detection_order(self):
        boxes = [box(100, 125, 400, 145), box(100, 100, 400, 120)]
        txts = ["seconda", "prima"]
        self.assertEqual(lines_to_text(boxes, txts), "prima\nseconda")

    def test_blank_recognitions_are_dropped(self):
        boxes = [box(100, 100, 400, 120), box(100, 125, 400, 145)]
        txts = ["testo", "   "]
        self.assertEqual(lines_to_text(boxes, txts), "testo")

    def test_low_confidence_lines_are_kept(self):
        """Una riga scartata e' una PII che sfugge: si tiene tutto."""
        boxes = [box(100, 100, 400, 120), box(100, 125, 400, 145)]
        txts = ["IT60X0542811101000000123456", "Mario Rossi"]
        out = lines_to_text(boxes, txts, scores=[0.11, 0.99])
        self.assertIn("IT60X0542811101000000123456", out)


if __name__ == "__main__":
    unittest.main()
