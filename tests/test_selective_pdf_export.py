"""Occorrenze sintetiche: mai oscurare valori uguali mantenuti dall'utente."""
import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "app"))
import pdf_text

try:
    import fitz
    import pdf_export
except ImportError:
    fitz = pdf_export = None


class _Page:
    def __init__(self, text):
        self.text = text

    def get_text(self, kind=None):
        if kind is None:
            return self.text
        return {"blocks": [{"type": 0, "lines": [{"spans": [{"chars": [
            {"c": c, "bbox": [i * 8, 10, i * 8 + 7, 20]}
            for i, c in enumerate(self.text.rstrip("\n"))]}]}]}]}

    def get_images(self, full=True):
        return []

    def annots(self):
        return []

    def widgets(self):
        return []


class _Doc(list):
    def get_toc(self, simple=True):
        return []

    def embfile_names(self):
        return []


class SourceMap(unittest.TestCase):
    def test_repeated_values_keep_distinct_offsets(self):
        doc = _Doc([_Page("Padova Padova\n"), _Page("Padova\n")])
        snapshot = pdf_text.collect_readable_document(doc)
        self.assertEqual(snapshot["text"], pdf_text.collect_readable_text(doc))
        self.assertEqual([s["start"] for s in snapshot["sources"]], [0, 15])
        self.assertEqual(snapshot["sources"][0]["boxes"][7], [56, 10, 63, 20])

    def test_non_whitespace_mismatch_is_not_guessed(self):
        page = _Page("Padova\n")
        self.assertIsNone(pdf_text._native_boxes(page, "Pad0va\n"))

    def test_ocr_canonical_text_is_not_markdown(self):
        ocr = {0: {"text": "# Padova", "lines": [
            {"text": "Padova", "box": [0, 0, 60, 10],
             "chars": [[i * 10, 0, (i + 1) * 10, 10] for i in range(6)]}]}}
        snapshot = pdf_text.collect_readable_document(_Doc([_Page("")]), ocr_pages=ocr)
        self.assertEqual(snapshot["text"], "Padova\n")
        self.assertEqual(snapshot["sources"][0]["origin"], "ocr")

    def test_missing_ocr_character_geometry_is_explicit(self):
        snapshot = pdf_text.collect_readable_document(_Doc([_Page("")]), ocr_pages={
            0: {"lines": [{"text": "Padova", "box": [0, 0, 60, 10]}]}})
        self.assertTrue(snapshot["sources"][0]["mapping_error"])


@unittest.skipIf(fitz is None, "PyMuPDF non installato")
class SelectivePdf(unittest.TestCase):
    def pdf(self, text="nato a Padova, residente a Padova, immobile a Padova. civico 15"):
        doc = fitz.open()
        page = doc.new_page(width=1000, height=300)
        page.insert_text((25, 50), text, fontsize=11)
        data = doc.tobytes()
        doc.close()
        return data

    def snapshot(self, data, **kwargs):
        with fitz.open(stream=data, filetype="pdf") as doc:
            return pdf_text.collect_readable_document(doc, **kwargs)

    def occurrence(self, snapshot, start, value, n=1):
        return {"occurrence_id": f"e{n}", "start": start, "end": start + len(value),
                "ph": f"[RESIDENZA_{n}]", "label": "CITY"}

    def assert_verified(self, report):
        self.assertEqual(report["errors"], [])
        self.assertEqual(report["residual"], [])

    def test_only_second_city_is_removed_and_short_civic_is_supported(self):
        data = self.pdf()
        snapshot = self.snapshot(data)
        start = snapshot["text"].index("Padova", snapshot["text"].index("Padova") + 1)
        occurrences = [self.occurrence(snapshot, start, "Padova"),
                       self.occurrence(snapshot, snapshot["text"].index("15"), "15", 2)]
        out, report = pdf_export.redact_occurrences(data, occurrences, snapshot)
        self.assert_verified(report)
        with fitz.open(stream=out, filetype="pdf") as doc:
            text = doc[0].get_text()
        self.assertEqual(text.count("Padova"), 2)
        self.assertNotIn("15", text)
        self.assertEqual(report["occurrences"], 2)

    def test_stale_snapshot_and_overlaps_are_rejected(self):
        data = self.pdf()
        snapshot = self.snapshot(data)
        start = snapshot["text"].index("Padova")
        occurrence = self.occurrence(snapshot, start, "Padova")
        with self.assertRaises(pdf_export.PdfError):
            pdf_export.redact_occurrences(self.pdf("altro documento"), [occurrence], snapshot)
        with self.assertRaises(pdf_export.PdfError):
            pdf_export.redact_occurrences(data, [occurrence, {**occurrence, "occurrence_id": "e2"}], snapshot)

    def test_empty_selection_preserves_text_and_removes_metadata(self):
        data = self.pdf()
        snapshot = self.snapshot(data)
        out, report = pdf_export.redact_occurrences(data, [], snapshot)
        self.assert_verified(report)
        with fitz.open(stream=out, filetype="pdf") as doc:
            self.assertEqual(doc[0].get_text(), snapshot["text"])
        self.assertEqual(report["occurrences"], 0)

    def test_independent_annotation_widget_and_toc_occurrences(self):
        doc = fitz.open(stream=self.pdf("corpo"), filetype="pdf")
        page = doc[0]
        annot = page.add_text_annot((50, 90), "Padova Padova")
        widget = fitz.Widget()
        widget.field_name = "residenza"
        widget.field_type = fitz.PDF_WIDGET_TYPE_TEXT
        widget.rect = fitz.Rect(50, 120, 300, 150)
        widget.field_value = "Padova Padova"
        page.add_widget(widget)
        doc.set_toc([[1, "Padova Padova", 1]])
        data = doc.tobytes()
        doc.close()
        snapshot = self.snapshot(data)
        occurrences = []
        for source in snapshot["sources"]:
            if source["kind"] in {"annotation", "widget", "toc"}:
                value = snapshot["text"][source["start"]:source["end"]]
                if value == "Padova Padova":
                    occurrences.append(self.occurrence(snapshot, source["start"] + 7,
                                                       "Padova", len(occurrences) + 1))
        self.assertEqual(len(occurrences), 3)
        widget_occurrence = occurrences[1]
        with self.assertRaisesRegex(pdf_export.PdfError, "Scelte incompatibili"):
            pdf_export.redact_occurrences(data, [widget_occurrence], snapshot)
        page_source = next(s for s in snapshot["sources"] if s["kind"] == "page")
        page_value = snapshot["text"][page_source["start"]:page_source["end"]]
        appearance = self.occurrence(snapshot, page_source["start"] + page_value.index("Padova") + 7,
                                     "Padova", 4)
        with self.assertRaisesRegex(pdf_export.PdfError, "Scelte incompatibili"):
            pdf_export.redact_occurrences(data, [appearance], snapshot)
        occurrences.append(appearance)
        with self.assertRaisesRegex(pdf_export.PdfError, "Segnaposti incoerenti"):
            pdf_export.redact_occurrences(data, occurrences, snapshot)
        original_marker = appearance["ph"]
        occurrences = pdf_export.align_occurrence_aliases(occurrences, snapshot)
        self.assertEqual(appearance["ph"], original_marker, "l'input non viene mutato")
        self.assertEqual(occurrences[-1]["ph"], widget_occurrence["ph"])
        self.assertEqual(occurrences[-1]["alias_of"], widget_occurrence["occurrence_id"])
        planned_text = pdf_export._replace_spans(snapshot["text"], [
            (e["start"], e["end"], e["ph"]) for e in occurrences])
        self.assertEqual(planned_text.count(widget_occurrence["ph"]), 2)
        out, report = pdf_export.redact_occurrences(data, occurrences, snapshot)
        self.assert_verified(report)
        after = self.snapshot(out)
        self.assertEqual(after["text"].count(widget_occurrence["ph"]), 2)
        for source in after["sources"]:
            if source["kind"] in {"annotation", "widget", "toc"}:
                value = after["text"][source["start"]:source["end"]]
                if value:
                    self.assertEqual(value.count("Padova"), 1)
        self.assertEqual((report["annots"], report["widgets"], report["toc"]), (1, 1, 1))

    def test_multiline_span_and_unselected_subject_are_preserved(self):
        data = self.pdf("Residente in via Roma\n15. Immobile in via Roma 15.")
        snapshot = self.snapshot(data)
        start = snapshot["text"].index("via Roma")
        value = "via Roma\n15"
        self.assertEqual(snapshot["text"][start:start + len(value)], value)
        out, report = pdf_export.redact_occurrences(
            data, [self.occurrence(snapshot, start, value)], snapshot)
        self.assert_verified(report)
        with fitz.open(stream=out, filetype="pdf") as doc:
            self.assertIn("Immobile in via Roma 15.", doc[0].get_text())

    def test_ocr_uses_selected_range_and_preserves_other_equal_values(self):
        try:
            import numpy as np
        except ImportError:
            self.skipTest("NumPy non installato")

        def _line(value, x, y):
            return {"text": value, "box": [x, y, x + len(value) * 10, y + 14],
                    "chars": [[x + i * 10, y, x + (i + 1) * 10, y + 14]
                              for i in range(len(value))]}

        pixels = np.full((400, 800, 3), 230, dtype=np.uint8)
        image = fitz.Pixmap(fitz.csRGB, 800, 400, pixels.tobytes(), 0)
        doc = fitz.open()
        page = doc.new_page(width=400, height=200)
        page.insert_image(page.rect, stream=image.tobytes("png"))
        data = doc.tobytes()
        doc.close()
        ocr = {0: {"dpi": 144, "lines": [
            _line("nato a Padova", 20, 30),
            _line("residente a Padova", 20, 70),
            _line("immobile a Padova", 20, 110)]}}
        snapshot = self.snapshot(data, ocr_pages=ocr)
        start = snapshot["text"].index("Padova", snapshot["text"].index("Padova") + 1)
        out, report = pdf_export.redact_occurrences(
            data, [self.occurrence(snapshot, start, "Padova")], snapshot, ocr_pages=ocr)
        self.assert_verified(report)
        self.assertEqual(report["occurrences"], 1)
        self.assertEqual(report["ocr_pages"], [0])
        self.assertIn({"code": "raster_content", "page": 0}, report["unverified"])
        with fitz.open(stream=out, filetype="pdf") as doc:
            pix = doc[0].get_pixmap(dpi=72)
            self.assertGreater(min(pix.pixel(110, 31)), 200)
            r, g, b = pix.pixel(150, 71)
            self.assertLess(abs(r - 124) + abs(g - 58) + abs(b - 158), 70)
            self.assertGreater(min(pix.pixel(150, 111)), 200)

    def test_attachments_removed_are_reported_before_export(self):
        doc = fitz.open(stream=self.pdf(), filetype="pdf")
        doc.embfile_add("sintetico.txt", b"contenuto sintetico non verificato")
        doc.set_metadata({"author": "Autore Sintetico"})
        data = doc.tobytes()
        doc.close()
        snapshot = self.snapshot(data)
        out, report = pdf_export.redact_occurrences(data, [], snapshot)
        self.assert_verified(report)
        self.assertEqual(report["embedded"], 1)
        self.assertIn({"code": "attachments_removed"}, report["unverified"])
        with fitz.open(stream=out, filetype="pdf") as verified:
            self.assertEqual(verified.embfile_names(), [])
            self.assertFalse(verified.metadata["author"])

    def test_freetext_appearance_cannot_contradict_annotation_content(self):
        doc = fitz.open(stream=self.pdf("corpo"), filetype="pdf")
        page = doc[0]
        page.add_freetext_annot(fitz.Rect(25, 100, 400, 150), "Padova Padova")
        data = doc.tobytes()
        doc.close()
        snapshot = self.snapshot(data)
        annotation = next(s for s in snapshot["sources"]
                          if s["kind"] == "annotation" and s["field"] == "content")
        page_source = next(s for s in snapshot["sources"] if s["kind"] == "page")
        content = self.occurrence(snapshot, annotation["start"] + 7, "Padova")
        page_value = snapshot["text"][page_source["start"]:page_source["end"]]
        appearance = self.occurrence(snapshot, page_value.index("Padova") + 7, "Padova", 2)
        with self.assertRaisesRegex(pdf_export.PdfError, "Scelte incompatibili"):
            pdf_export.redact_occurrences(data, [content], snapshot)
        occurrences = pdf_export.align_occurrence_aliases([appearance, content], snapshot)
        self.assertEqual(occurrences[0]["ph"], occurrences[1]["ph"])
        out, report = pdf_export.redact_occurrences(data, occurrences, snapshot)
        self.assert_verified(report)
        after = self.snapshot(out)
        values = [after["text"][s["start"]:s["end"]] for s in after["sources"]
                  if s["kind"] == "annotation" and s["field"] == "content"]
        self.assertEqual(values, ["Padova [RESIDENZA_1]"])

    def test_geometric_aliases_do_not_merge_other_cities_or_incompatible_labels(self):
        doc = fitz.open(stream=self.pdf("nato a Padova, immobile a Padova"), filetype="pdf")
        page = doc[0]
        widget = fitz.Widget()
        widget.field_name = "residenza"
        widget.field_type = fitz.PDF_WIDGET_TYPE_TEXT
        widget.rect = fitz.Rect(25, 100, 300, 130)
        widget.field_value = "Padova"
        page.add_widget(widget)
        data = doc.tobytes()
        doc.close()
        snapshot = self.snapshot(data)
        page_source = next(s for s in snapshot["sources"] if s["kind"] == "page")
        field = next(s for s in snapshot["sources"] if s["kind"] == "widget")
        text = snapshot["text"][page_source["start"]:page_source["end"]]
        starts, offset = [], 0
        while True:
            offset = text.find("Padova", offset)
            if offset < 0:
                break
            starts.append(offset)
            offset += 6
        entities = [self.occurrence(snapshot, start, "Padova", i + 1)
                    for i, start in enumerate(starts)]
        entities.append(self.occurrence(snapshot, field["start"], "Padova", 4))
        self.assertEqual(len(entities), 4)
        aligned = pdf_export.align_occurrence_aliases(entities, snapshot)
        self.assertEqual([e["ph"] for e in aligned[:2]], ["[RESIDENZA_1]", "[RESIDENZA_2]"])
        self.assertEqual(aligned[2]["ph"], aligned[3]["ph"])
        self.assertNotIn("alias_of", aligned[0])
        incompatible = [*entities[:3], {**entities[3], "label": "ORG"}]
        unchanged = pdf_export.align_occurrence_aliases(incompatible, snapshot)
        self.assertNotEqual(unchanged[2]["ph"], unchanged[3]["ph"])
        with self.assertRaisesRegex(pdf_export.PdfError, "Segnaposti incoerenti"):
            pdf_export.redact_occurrences(data, unchanged, snapshot)

    def test_real_ocr_geometry_clips_only_padding_before_kept_comma(self):
        """Geometria catturata su documento sintetico con OCR/NER locali reali.

        Il box del civico finisce dove inizia la virgola: solo il padding del
        vecchio esportatore la intersecava. Nessun modello serve a questo test.
        """
        try:
            import numpy  # noqa: F401 - rasterizzazione del percorso esistente
        except ImportError:
            self.skipTest("NumPy non installato")
        fixture = json.loads((Path(__file__).parent / "fixtures" /
                              "notarial_ocr_geometry.json").read_text(encoding="utf-8"))
        render = fixture["render"]
        doc = fitz.open()
        page = doc.new_page(width=render["page"][0], height=render["page"][1])
        page.insert_textbox(fitz.Rect(render["textbox"]), fixture["input_text"],
                            fontsize=render["font_size"])
        pix = page.get_pixmap(dpi=render["dpi"])
        scanned = fitz.open()
        target = scanned.new_page(width=page.rect.width, height=page.rect.height)
        target.insert_image(target.rect, stream=pix.tobytes("png"))
        data = scanned.tobytes()
        doc.close()
        scanned.close()
        ocr = {int(k): v for k, v in fixture["ocr_pages"].items()}
        snapshot = self.snapshot(data, ocr_pages=ocr)
        self.assertEqual(snapshot["text"], fixture["text"])
        source = snapshot["sources"][0]
        ranges = [(e["ph"], e["start"], e["end"],
                   pdf_export._span_rects(source["boxes"], e["start"], e["end"]))
                  for e in fixture["selected"]]
        masks = pdf_export._bounded_ocr_masks(snapshot["text"], source["boxes"], ranges,
                                             fitz.Rect(0, 0, *render["page"]))
        for _, start, stop, rects in ranges:
            for base, mask in zip(rects, masks[(start, stop)]):
                self.assertTrue(mask.contains(base), "non restringere il box della PII")
        comma = fitz.Rect(source["boxes"][71])
        self.assertTrue(any((pdf_export._pad(base, fitz.Rect(0, 0, *render["page"])) & comma).get_area() > 0
                            for _, _, _, rects in ranges for base in rects))
        self.assertTrue(all((mask & comma).get_area() == 0 for values in masks.values() for mask in values))
        out, report = pdf_export.redact_occurrences(
            data, fixture["selected"], snapshot, ocr_pages=ocr)
        self.assert_verified(report)
        self.assertEqual(report["occurrences"], 3)
        with fitz.open(stream=out, filetype="pdf") as checked:
            self.assertEqual(len(checked[0].get_images(full=True)), 1)

    def test_ocr_overlap_in_original_character_boxes_still_blocks(self):
        boxes = [[0, 0, 10, 10], [9, 0, 20, 10]]
        ranges = [("[CITY_1]", 0, 1, [fitz.Rect(boxes[0])])]
        with self.assertRaisesRegex(pdf_export.PdfError, "testo non selezionato"):
            pdf_export._bounded_ocr_masks("AB", boxes, ranges, fitz.Rect(0, 0, 100, 100))


if __name__ == "__main__":
    unittest.main()
