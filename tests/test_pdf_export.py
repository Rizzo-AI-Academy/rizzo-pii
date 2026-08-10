# -*- coding: utf-8 -*-
"""Regressioni per PII persistente fuori dal normale testo di pagina.

Le fixture sono interamente sintetiche e non caricano il modello neurale.
"""

import importlib.util
import io
import sys
import types
import unittest
from pathlib import Path
from urllib.parse import unquote
from unittest import mock

import fitz

ROOT = Path(__file__).resolve().parents[1]
APP_DIR = ROOT / "src" / "app"
sys.path.insert(0, str(APP_DIR))

import pdf_export as px  # noqa: E402


CF = "RSSMRA85H12H501V"
EMAIL = "mario.rossi@example.com"


def annotation_only_pdf():
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text(
        (50, 80),
        "Documento di prova senza identificativi personali.",
        fontsize=11,
    )
    annot = page.add_text_annot((200, 200), f"Codice fiscale {CF}")
    annot.set_info(content=f"Codice fiscale {CF}")
    annot.update()
    out = doc.tobytes()
    doc.close()
    return out


def link_pdf():
    return uri_link_pdf(f"mailto:{EMAIL}", visible_email=True)


def uri_link_pdf(uri, visible_email=False):
    doc = fitz.open()
    page = doc.new_page()
    if visible_email:
        page.insert_text((50, 80), EMAIL, fontsize=11)
    page.insert_text((50, 130), "Invia un messaggio", fontsize=11)
    page.insert_link({
        "kind": fitz.LINK_URI,
        "from": fitz.Rect(45, 115, 180, 135),
        "uri": uri,
    })
    out = doc.tobytes()
    doc.close()
    return out


def link_only_pdf():
    """Target noto al verificatore, ma nessuna copia nel page text."""
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((50, 80), "Invia un messaggio", fontsize=11)
    page.insert_link({
        "kind": fitz.LINK_URI,
        "from": fitz.Rect(45, 65, 180, 85),
        "uri": f"mailto:{EMAIL}",
    })
    out = doc.tobytes()
    doc.close()
    return out


def all_auxiliary_surfaces_pdf():
    values = {
        "content": "content.person@example.com",
        "subject": "subject.person@example.com",
        "title": "title.person@example.com",
        "widget": "widget.person@example.com",
        "toc": "toc.person@example.com",
        "link": "link.person@example.com",
    }
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((50, 80), "Documento sintetico.", fontsize=11)

    annot = page.add_text_annot((200, 200), values["content"])
    annot.set_info(
        content=values["content"],
        subject=values["subject"],
        title=values["title"],
    )
    annot.update()

    widget = fitz.Widget()
    widget.rect = fitz.Rect(50, 250, 300, 275)
    widget.field_name = "contatto"
    widget.field_type = fitz.PDF_WIDGET_TYPE_TEXT
    widget.field_value = values["widget"]
    page.add_widget(widget)

    doc.set_toc([[1, values["toc"], 1]])
    page.insert_link({
        "kind": fitz.LINK_URI,
        "from": fitz.Rect(45, 300, 180, 320),
        "uri": f"mailto:{values['link']}",
    })
    out = doc.tobytes()
    doc.close()
    return out, values


def image_only_pdf():
    doc = fitz.open()
    page = doc.new_page()
    pixmap = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 16, 16), False)
    pixmap.clear_with(255)
    page.insert_image(fitz.Rect(50, 50, 150, 150), pixmap=pixmap)
    out = doc.tobytes()
    doc.close()
    return out


def split_iban_annotation_pdf(single_field=False):
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((50, 80), "Documento sintetico senza PII.", fontsize=11)
    first = "IT60 X054 2811"
    second = "1010 0000 0123 456"
    content = first + ("\n" + second if single_field else "")
    annot = page.add_text_annot((200, 200), content)
    annot.set_info(content=content, subject="" if single_field else second, title="")
    annot.update()
    out = doc.tobytes()
    doc.close()
    return out


def plain_email_pdf():
    """Page text con PII, ma nessuna superficie ausiliaria presente."""
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((50, 80), EMAIL, fontsize=11)
    out = doc.tobytes()
    doc.close()
    return out


def auxiliary_failure_pdf(failed_surface):
    """PII nella superficie che fallira' + una seconda superficie sanificabile."""
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((50, 80), "Documento sintetico.", fontsize=11)

    if failed_surface in ("annotations", "toc"):
        annot = page.add_text_annot((200, 200), EMAIL)
        annot.set_info(content=EMAIL)
        annot.update()
    if failed_surface == "links":
        page.insert_link({
            "kind": fitz.LINK_URI,
            "from": fitz.Rect(45, 115, 180, 135),
            "uri": f"mailto:{EMAIL}",
        })
    if failed_surface == "widgets":
        widget = fitz.Widget()
        widget.rect = fitz.Rect(50, 250, 300, 275)
        widget.field_name = "contatto"
        widget.field_type = fitz.PDF_WIDGET_TYPE_TEXT
        widget.field_value = EMAIL
        page.add_widget(widget)
    doc.set_toc([[1, EMAIL, 1]])

    out = doc.tobytes()
    doc.close()
    return out


def load_app_without_model():
    """Importa app.py con stub minimi: il test usa detector regex + Flask reali."""
    fake_torch = types.ModuleType("torch")
    fake_torch.cuda = types.SimpleNamespace(is_available=lambda: False)
    fake_transformers = types.ModuleType("transformers")
    fake_transformers.pipeline = lambda *args, **kwargs: (
        lambda texts: [[] for _ in texts]
    )

    spec = importlib.util.spec_from_file_location("_rizzo_pdf_test_app", APP_DIR / "app.py")
    module = importlib.util.module_from_spec(spec)
    with mock.patch.dict(
        sys.modules,
        {"torch": fake_torch, "transformers": fake_transformers},
    ), mock.patch.dict(
        "os.environ",
        {"PII_MODEL_DIR": str(APP_DIR)},
    ):
        spec.loader.exec_module(module)
    return module


class PdfAuxiliarySurfaceTests(unittest.TestCase):
    def _assert_enumeration_failure_is_closed(
        self, pdf, owner, method, message, other_counter,
    ):
        _out, baseline = px.redact_pdf(pdf, {"[EMAIL_1]": EMAIL})
        self.assertGreater(baseline[other_counter], 0)

        with mock.patch.object(
            owner,
            method,
            side_effect=RuntimeError("forced enumeration failure"),
        ):
            with self.assertRaisesRegex(px.PdfError, message):
                px.redact_pdf(pdf, {"[EMAIL_1]": EMAIL})

    def test_auxiliary_extraction_covers_supported_surfaces(self):
        pdf, values = all_auxiliary_surfaces_pdf()
        surfaces = px.extract_detection_surfaces(pdf)
        for surface, value in values.items():
            with self.subTest(surface):
                self.assertTrue(any(value in text for text in surfaces))

    def test_annotation_only_pii_is_discovered_and_sanitized_by_pdf_route(self):
        pdf = annotation_only_pdf()
        with fitz.open(stream=pdf, filetype="pdf") as doc:
            self.assertNotIn(CF, doc[0].get_text())

        app_module = load_app_without_model()
        client = app_module.app.test_client()
        analyze_response = client.post(
            "/analyze",
            data={"pdf": (io.BytesIO(pdf), "annotazione.pdf")},
            content_type="multipart/form-data",
        )
        self.assertEqual(200, analyze_response.status_code)
        analysis = analyze_response.get_json()
        self.assertNotIn(CF, analysis["source_text"])
        self.assertNotIn(CF, analysis["anonymized_text"])

        response = client.post(
            "/pdf",
            data={"pdf": (io.BytesIO(pdf), "annotazione.pdf")},
            content_type="multipart/form-data",
        )

        self.assertEqual(200, response.status_code, response.get_data()[:200])
        self.assertEqual("0", response.headers["X-PII-Redactions"])
        with fitz.open(stream=response.data, filetype="pdf") as doc:
            readable = px._readable_text(doc)
            info = list(doc[0].annots())[0].info
        self.assertNotIn(CF, readable)
        self.assertNotIn(CF, info["content"])
        self.assertIn("[CF_1]", info["content"])
        self.assertEqual("0", response.headers["X-PII-Residual"])

        preview_response = client.post(
            "/pdf/preview",
            data={"pdf": (io.BytesIO(pdf), "annotazione.pdf")},
            content_type="multipart/form-data",
        )
        self.assertEqual(200, preview_response.status_code)
        self.assertEqual(0, preview_response.get_json()["redactions"])

    def test_link_target_is_included_in_residual_verification(self):
        self.assertEqual(
            ["[EMAIL_1]"],
            px._verify_residuals(link_only_pdf(), [("[EMAIL_1]", EMAIL)]),
        )

    def test_mailto_target_is_scrubbed_without_removing_link(self):
        out, report = px.redact_pdf(link_pdf(), {"[EMAIL_1]": EMAIL})
        with fitz.open(stream=out, filetype="pdf") as doc:
            visible = doc[0].get_text()
            links = doc[0].get_links()
            readable = px._readable_text(doc)

        self.assertNotIn(EMAIL, visible)
        self.assertEqual(1, len(links))
        self.assertEqual("mailto:%5BEMAIL_1%5D", links[0]["uri"])
        self.assertNotIn(EMAIL, readable)
        self.assertGreater(report["links"], 0)
        self.assertEqual([], report["residual"])

    def _assert_encoded_mailto_is_neutralized(self, uri):
        encoded_only_pdf = uri_link_pdf(uri, visible_email=False)
        surfaces = px.extract_detection_surfaces(encoded_only_pdf)
        self.assertTrue(any(EMAIL in surface for surface in surfaces))
        self.assertEqual(
            ["[EMAIL_1]"],
            px._verify_residuals(encoded_only_pdf, [("[EMAIL_1]", EMAIL)]),
        )

        pdf = uri_link_pdf(uri, visible_email=True)
        out, report = px.redact_pdf(pdf, {"[EMAIL_1]": EMAIL})
        with fitz.open(stream=out, filetype="pdf") as doc:
            visible = doc[0].get_text()
            links = doc[0].get_links()
            readable = px._readable_text(doc)
        final_uris = [link.get("uri", "") for link in links]

        self.assertNotIn(EMAIL, visible)
        self.assertEqual([], links)
        self.assertTrue(all(EMAIL not in unquote(final) for final in final_uris))
        self.assertTrue(all(EMAIL not in final for final in final_uris))
        self.assertTrue(all(uri not in final for final in final_uris))
        self.assertNotIn(EMAIL, readable)
        self.assertNotIn(uri, readable)
        self.assertGreater(report["links"], 0)
        self.assertEqual([], report["residual"])

    def test_percent_encoded_mailto_target_is_neutralized(self):
        self._assert_encoded_mailto_is_neutralized(
            "mailto:mario.rossi%40example.com",
        )

    def test_more_encoded_mailto_target_is_neutralized(self):
        self._assert_encoded_mailto_is_neutralized(
            "mailto:mario%2Erossi%40example%2Ecom",
        )

    def test_short_literal_mapping_cannot_mask_encoded_email(self):
        uri = "mailto:mario%2Erossi%40example%2Ecom"
        out, report = px.redact_pdf(
            uri_link_pdf(uri, visible_email=False),
            {"[EMAIL_1]": EMAIL, "[FULLNAME_1]": "mario"},
        )
        with fitz.open(stream=out, filetype="pdf") as doc:
            links = doc[0].get_links()
            readable = px._readable_text(doc)
        self.assertEqual([], links)
        self.assertNotIn("rossi@example.com", readable)
        self.assertGreater(report["links"], 0)
        self.assertEqual([], report["residual"])

    def test_unrelated_percent_encoded_uri_is_unchanged(self):
        uri = "urn:example:chapter%201+appendix"
        out, report = px.redact_pdf(
            uri_link_pdf(uri, visible_email=True),
            {"[EMAIL_1]": EMAIL},
        )
        with fitz.open(stream=out, filetype="pdf") as doc:
            links = doc[0].get_links()
        self.assertEqual(uri, links[0]["uri"])
        self.assertEqual(0, report["links"])

    def test_link_enumeration_failure_is_controlled_end_to_end(self):
        app_module = load_app_without_model()
        client = app_module.app.test_client()
        for route in ("/pdf", "/pdf/preview"):
            with self.subTest(route), mock.patch.object(
                fitz.Page,
                "get_links",
                side_effect=RuntimeError("forced enumeration failure"),
            ):
                response = client.post(
                    route,
                    data={"pdf": (io.BytesIO(link_pdf()), "link.pdf")},
                    content_type="multipart/form-data",
                )
            self.assertEqual(400, response.status_code)
            self.assertEqual("application/json", response.mimetype)
            error = response.get_json()["error"]
            self.assertIn("link", error.lower())
            self.assertNotIn(EMAIL, error)

    def test_post_discovery_verification_failure_is_controlled_end_to_end(self):
        app_module = load_app_without_model()
        client = app_module.app.test_client()
        original_get_links = fitz.Page.get_links
        calls = 0

        def fail_after_discovery(page, *args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 1:
                return original_get_links(page, *args, **kwargs)
            raise RuntimeError("forced post-discovery enumeration failure")

        with mock.patch.object(fitz.Page, "get_links", new=fail_after_discovery):
            response = client.post(
                "/pdf",
                data={
                    "pdf": (
                        io.BytesIO(auxiliary_failure_pdf("links")),
                        "link_e_indice.pdf",
                    ),
                },
                content_type="multipart/form-data",
            )

        self.assertGreaterEqual(calls, 3)
        self.assertEqual(400, response.status_code)
        self.assertEqual("application/json", response.mimetype)
        self.assertIn("link", response.get_json()["error"].lower())

    def test_discovery_fails_closed_for_auxiliary_enumeration_errors(self):
        all_aux_pdf, _values = all_auxiliary_surfaces_pdf()
        cases = (
            (annotation_only_pdf(), fitz.Page, "annots", "annotazioni"),
            (all_aux_pdf, fitz.Page, "widgets", "campi modulo"),
            (link_pdf(), fitz.Page, "get_links", "link"),
            (all_aux_pdf, fitz.Document, "get_toc", "indice"),
        )
        for pdf, owner, method, message in cases:
            with self.subTest(method), mock.patch.object(
                owner,
                method,
                side_effect=RuntimeError("forced enumeration failure"),
            ):
                with self.assertRaisesRegex(px.PdfError, message):
                    px.extract_detection_surfaces(pdf)

    def test_link_enumeration_failure_aborts_residual_verification(self):
        self._assert_enumeration_failure_is_closed(
            auxiliary_failure_pdf("links"),
            fitz.Page,
            "get_links",
            "link",
            "toc",
        )

    def test_annotation_enumeration_failure_aborts_residual_verification(self):
        self._assert_enumeration_failure_is_closed(
            auxiliary_failure_pdf("annotations"),
            fitz.Page,
            "annots",
            "annotazioni",
            "toc",
        )

    def test_widget_enumeration_failure_aborts_residual_verification(self):
        self._assert_enumeration_failure_is_closed(
            auxiliary_failure_pdf("widgets"),
            fitz.Page,
            "widgets",
            "campi modulo",
            "toc",
        )

    def test_toc_enumeration_failure_aborts_residual_verification(self):
        self._assert_enumeration_failure_is_closed(
            auxiliary_failure_pdf("toc"),
            fitz.Document,
            "get_toc",
            "indice",
            "annots",
        )

    def test_absent_auxiliary_surfaces_are_not_inspection_failures(self):
        pdf = plain_email_pdf()
        with fitz.open(stream=pdf, filetype="pdf") as doc:
            self.assertEqual([], list(doc[0].annots() or ()))
            self.assertEqual([], list(doc[0].widgets() or ()))
            self.assertEqual([], doc[0].get_links())
            self.assertEqual([], doc.get_toc(simple=True))

        app_module = load_app_without_model()
        response = app_module.app.test_client().post(
            "/pdf",
            data={"pdf": (io.BytesIO(pdf), "senza_aux.pdf")},
            content_type="multipart/form-data",
        )
        self.assertEqual(200, response.status_code, response.get_data()[:200])
        self.assertEqual("1", response.headers["X-PII-Redactions"])
        self.assertEqual("0", response.headers["X-PII-Residual"])

    def test_pdf_surface_mapping_uses_global_placeholder_namespace(self):
        app_module = load_app_without_model()
        mapping = app_module._mapping_for_surfaces([
            "primo@example.com",
            "secondo@example.com",
            "PRIMO@example.com",
            "altro valore secondo@example.com",
            CF,
        ])
        self.assertEqual({
            "[EMAIL_1]": "primo@example.com",
            "[EMAIL_2]": "secondo@example.com",
            "[CF_1]": CF,
        }, mapping)

    def test_detection_does_not_cross_annotation_fields(self):
        pdf = split_iban_annotation_pdf(single_field=False)
        app_module = load_app_without_model()
        surfaces = px.extract_detection_surfaces(pdf)
        mapping = app_module._mapping_for_surfaces(surfaces)
        self.assertEqual({}, mapping)

        client = app_module.app.test_client()
        response = client.post(
            "/pdf",
            data={"pdf": (io.BytesIO(pdf), "iban_artificiale.pdf")},
            content_type="multipart/form-data",
        )
        self.assertEqual(422, response.status_code)

        with fitz.open(stream=pdf, filetype="pdf") as doc:
            info = list(doc[0].annots())[0].info
        self.assertEqual("IT60 X054 2811", info["content"])
        self.assertEqual("1010 0000 0123 456", info["subject"])

    def test_multiline_iban_inside_one_annotation_is_sanitized(self):
        pdf = split_iban_annotation_pdf(single_field=True)
        app_module = load_app_without_model()
        mapping = app_module._mapping_for_surfaces(px.extract_detection_surfaces(pdf))
        self.assertEqual(
            {"[IBAN_1]": "IT60 X054 2811\n1010 0000 0123 456"},
            mapping,
        )

        client = app_module.app.test_client()
        response = client.post(
            "/pdf",
            data={"pdf": (io.BytesIO(pdf), "iban_multilinea.pdf")},
            content_type="multipart/form-data",
        )
        self.assertEqual(200, response.status_code, response.get_data()[:200])
        with fitz.open(stream=response.data, filetype="pdf") as doc:
            info = list(doc[0].annots())[0].info
        self.assertEqual("[IBAN_1]", info["content"])
        self.assertEqual("0", response.headers["X-PII-Redactions"])
        self.assertEqual("0", response.headers["X-PII-Residual"])

    def test_image_only_pdf_without_sanitized_surface_is_still_rejected(self):
        app_module = load_app_without_model()
        client = app_module.app.test_client()
        response = client.post(
            "/pdf",
            data={"pdf": (io.BytesIO(image_only_pdf()), "scansione.pdf")},
            content_type="multipart/form-data",
        )
        self.assertEqual(400, response.status_code)
        self.assertEqual("application/json", response.mimetype)


if __name__ == "__main__":
    unittest.main()
