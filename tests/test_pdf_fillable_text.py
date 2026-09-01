# -*- coding: utf-8 -*-
"""Regressione issue #85: il testo dei campi modulo deve entrare in /analyze.

`_text_from_bytes` leggeva solo `page.get_text()`. Nei PDF fillable (AcroForm)
il valore sta nel widget, non nel content stream: get_text() torna vuoto, lo
strip in /analyze produce 400 \"Nessun testo da analizzare\", e i campi restano
in chiaro. `pdf_export._readable_text` gia' raccoglie i widget: l'estrazione
per l'analisi deve fare lo stesso.

Nessun import di fitz: le pagine sono duck-typed, come i detector. Gira in CI
sulla sola libreria standard. I valori sono SINTETICI.
"""

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "app"))

import pdf_text  # noqa: E402


class FakeWidget:
    def __init__(self, value):
        self.field_value = value


class FakeAnnot:
    def __init__(self, content="", subject="", title="", type_id=0):
        self.type = (type_id, "Text")
        self.info = {"content": content, "subject": subject, "title": title}


class FakePage:
    def __init__(self, text="", widgets=(), annots=()):
        self._text = text
        self._widgets = list(widgets)
        self._annots = list(annots)

    def get_text(self):
        return self._text

    def widgets(self):
        return iter(self._widgets)

    def annots(self):
        return iter(self._annots)


class FakeDoc(list):
    def __init__(self, pages, toc=None):
        super().__init__(pages)
        self._toc = toc or []

    def get_toc(self, simple=True):
        return self._toc


class FillablePdfText(unittest.TestCase):
    def test_widget_value_is_collected_when_page_stream_is_empty(self):
        """Il caso della issue: modulo pagoPA / AcroForm, get_text() vuoto."""
        doc = FakeDoc([
            FakePage(text="", widgets=[FakeWidget("Mario Rossi")]),
        ])
        out = pdf_text.collect_readable_text(doc)
        self.assertIn("Mario Rossi", out)
        self.assertTrue(out.strip(), "senza il widget /analyze risponderebbe 400")

    def test_page_stream_text_is_still_collected(self):
        doc = FakeDoc([FakePage(text="Il sottoscritto dichiara.")])
        out = pdf_text.collect_readable_text(doc)
        self.assertIn("Il sottoscritto dichiara.", out)

    def test_empty_and_non_string_widgets_are_ignored(self):
        doc = FakeDoc([
            FakePage(text="corpo", widgets=[
                FakeWidget(""),
                FakeWidget(None),
                FakeWidget(True),
                FakeWidget("CF RSSMRA80A01H501U"),
            ]),
        ])
        out = pdf_text.collect_readable_text(doc)
        self.assertIn("corpo", out)
        self.assertIn("CF RSSMRA80A01H501U", out)

    def test_annot_content_is_collected_redact_marks_are_not(self):
        doc = FakeDoc([
            FakePage(
                text="",
                annots=[
                    FakeAnnot(content="nota con PII", type_id=0),
                    FakeAnnot(content="redazione interna", type_id=12),
                ],
            ),
        ])
        out = pdf_text.collect_readable_text(doc)
        self.assertIn("nota con PII", out)
        self.assertNotIn("redazione interna", out)

    def test_bookmark_titles_are_collected(self):
        doc = FakeDoc(
            [FakePage(text="pagina")],
            toc=[[1, "Atto Rossi Mario", 1]],
        )
        out = pdf_text.collect_readable_text(doc)
        self.assertIn("Atto Rossi Mario", out)

    def test_truly_empty_pdf_stays_empty(self):
        doc = FakeDoc([FakePage(text="", widgets=[FakeWidget("")])])
        self.assertFalse(pdf_text.collect_readable_text(doc).strip())


if __name__ == "__main__":
    unittest.main()
