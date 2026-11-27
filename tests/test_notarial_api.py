# -*- coding: utf-8 -*-
"""Integrazione notarile Flask/PDF con soli dati sintetici e NER controllato.

Il riconoscimento viene sostituito per isolare classificazione, decisioni,
geometria PDF e privacy: questi test non misurano l'accuratezza del modello.
"""

import io
import re
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
from notarial_test_support import APP

TEXT = ("Il signor Mario Rossi, nato a Padova, residente a Padova in via Roma 15, "
        "proprietario dell'immobile sito in Padova, via Venezia 20.")


def synthetic_entities(text):
    entities = []
    for label, expression in (("FULLNAME", r"\bMario Rossi\b"),
                              ("CITY", r"\bPadova\b"),
                              ("STREET", r"\bvia (?:Roma|Venezia)\b"),
                              ("BUILDINGNUM", r"\b(?:15|20)\b")):
        for match in re.finditer(expression, text):
            entities.append({"label": label, "start": match.start(), "end": match.end(),
                             "score": 0.99, "validated": False, "source": "modello"})
    return entities, 1


def synthetic_pdf(with_image=False):
    import fitz
    with fitz.open() as document:
        page = document.new_page()
        page.insert_textbox(fitz.Rect(72, 72, 520, 230), TEXT, fontsize=11)
        if with_image:
            pix = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 20, 20), False)
            pix.clear_with(230)
            page.insert_image(fitz.Rect(72, 260, 92, 280), stream=pix.tobytes("png"))
        return document.tobytes()


@unittest.skipIf(APP is None, "serve l'ambiente Flask/PyMuPDF (nessun peso ML)")
class NotarialApi(unittest.TestCase):
    def setUp(self):
        APP._clear_documents()
        self.addCleanup(APP._clear_documents)
        self.patches = [patch.object(APP, "detect_model", side_effect=synthetic_entities),
                        patch.object(APP, "MAPPING_ENABLED", False),
                        patch.object(APP, "EXCLUDED_TAGS", []),
                        patch.object(APP, "OCR_MODE", "off")]
        for item in self.patches:
            item.start()
            self.addCleanup(item.stop)
        self.client = APP.app.test_client()
        self.client.environ_base["HTTP_X_PII_TOKEN"] = APP.SESSION_TOKEN

    def response_json(self, response, expected=200):
        self.assertEqual(response.status_code, expected, response.get_json())
        self.assertIn("no-store", response.headers.get("Cache-Control", ""))
        return response.get_json()

    def analyze(self, text=TEXT, pdf=None):
        if pdf is None:
            response = self.client.post("/analyze", json={"text": text, "review": True})
        else:
            response = self.client.post("/analyze", data={
                "file": (io.BytesIO(pdf), "perizia_sintetica.pdf"), "review": "1", "ocr": "off"},
                content_type="multipart/form-data")
        return self.response_json(response)

    def select_residence_city(self, analysis):
        response = self.client.post("/review/" + analysis["analysis_id"], json={
            "revision": analysis["revision"], "categories": ["residenza.comune"]})
        return self.response_json(response)

    def export_text(self, analysis, **extra):
        return self.client.post("/review/" + analysis["analysis_id"] + "/text", json={
            "revision": analysis["revision"], **extra})

    def export_pdf(self, analysis, **extra):
        return self.client.post("/pdf/preview", json={
            "analysis_id": analysis["analysis_id"], "revision": analysis["revision"], **extra})

    def test_solo_la_residenza_nel_testo_conserva_le_altre_padova(self):
        source = "  " + TEXT + "\n\t"
        analysis = self.select_residence_city(self.analyze(text=source))
        result = self.response_json(self.export_text(analysis))
        output = result["anonymized_text"]
        self.assertEqual(output, source.replace("residente a Padova", "residente a [CITY_2]"))
        self.assertEqual(output.count("Padova"), 2)
        self.assertIn("via Roma 15", output)
        self.assertIn("via Venezia 20", output)
        self.assertFalse(result["verification"]["complete_anonymization"])

    def test_solo_la_residenza_nel_pdf_preserva_nascita_e_immobile(self):
        import fitz
        analysis = self.select_residence_city(self.analyze(pdf=synthetic_pdf()))
        result = self.response_json(self.export_pdf(analysis))
        self.assertEqual(result["redactions"], 1)
        self.assertEqual((result["residual"], result["skipped"], result["not_found"]), (0, 0, 0))
        response = self.client.get("/doc/" + result["doc_id"] + "/file.pdf")
        self.assertEqual(response.status_code, 200)
        self.assertIn("no-store", response.headers["Cache-Control"])
        with fitz.open(stream=response.data, filetype="pdf") as document:
            output = "\n".join(page.get_text() for page in document)
        self.assertEqual(output.count("Padova"), 2)
        self.assertIn("via Roma 15", output)
        self.assertIn("via Venezia 20", output)
        self.assertIn("Mario Rossi", output)

    def test_categoria_esplicita_prevale_sui_tag_legacy_e_override_prevale_su_entrambi(self):
        with patch.object(APP, "EXCLUDED_TAGS", ["CITY"]):
            analysis = self.select_residence_city(self.analyze())
        selected = [e for e in analysis["occurrences"] if e["selected"]]
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0]["category"], "residenza.comune")
        self.assertEqual(self.response_json(self.export_text(analysis))["anonymized_text"],
                         TEXT.replace("residente a Padova", "residente a [CITY_2]"))
        updated = self.response_json(self.client.post("/review/" + analysis["analysis_id"], json={
            "revision": analysis["revision"],
            "overrides": {selected[0]["occurrence_id"]: False}}))
        self.assertFalse(any(e["selected"] for e in updated["occurrences"]))
        self.assertEqual(self.response_json(self.export_text(updated, ack_unverified=True))[
            "anonymized_text"], TEXT)

    def test_ambiguita_blocca_entrambi_export_e_override_la_risolve(self):
        analysis = self.analyze(text="Comune indicato senza ruolo: Padova.")
        self.assertTrue(analysis["unresolved"])
        for response in (self.export_text(analysis), self.export_pdf(analysis)):
            data = self.response_json(response, 422)
            self.assertEqual(data["code"], "review_required")
        self.assertFalse(APP._DOCS)
        oid = analysis["occurrences"][0]["occurrence_id"]
        corrected = self.response_json(self.client.post("/review/" + analysis["analysis_id"], json={
            "revision": analysis["revision"], "overrides": {oid: True}}))
        self.assertEqual(corrected["unresolved"], [])
        self.assertNotIn("Padova", self.response_json(self.export_text(corrected))["anonymized_text"])

    def test_correzione_manuale_categoria_e_revisione_obsoleta(self):
        analysis = self.analyze(text="Comune indicato senza ruolo: Padova.")
        oid = analysis["occurrences"][0]["occurrence_id"]
        changed = self.response_json(self.client.post("/review/" + analysis["analysis_id"], json={
            "revision": analysis["revision"], "classifications": {oid: "immobile.comune"}}))
        self.assertEqual(changed["occurrences"][0]["category"], "immobile.comune")
        self.assertEqual(changed["unresolved"], [])
        self.assertEqual(changed["revision"], analysis["revision"] + 1)
        stale = self.response_json(self.export_text(analysis), 409)
        self.assertEqual(stale["code"], "stale_revision")
        self.response_json(self.export_text(changed))

    def test_sorgente_non_scaricabile_e_anteprima_non_memorizzabile(self):
        response = self.client.post("/preview", data={
            "file": (io.BytesIO(synthetic_pdf()), "perizia_sintetica.pdf")},
            content_type="multipart/form-data")
        source = self.response_json(response)
        self.response_json(self.client.get("/doc/" + source["doc_id"] + "/file.pdf"), 403)
        page = self.client.get("/doc/" + source["doc_id"] + "/page/0.png")
        self.assertEqual(page.status_code, 200)
        self.assertIn("no-store", page.headers["Cache-Control"])

    def test_api_rifiuta_token_origine_host_e_client_remoto(self):
        anonymous = APP.app.test_client()
        self.assertEqual(anonymous.post("/analyze", json={"text": TEXT}).status_code, 403)
        for headers, base_url, environment in (
                ({"X-PII-Token": "errato"}, None, {}),
                ({"Origin": "https://esterno.invalid"}, None, {}),
                ({"Sec-Fetch-Site": "cross-site"}, None, {}),
                ({}, "http://esterno.invalid", {}),
                ({}, None, {"REMOTE_ADDR": "192.0.2.10"})):
            with self.subTest(headers=headers, base_url=base_url, environment=environment):
                kwargs = {"base_url": base_url} if base_url else {}
                response = self.client.post("/analyze", json={"text": TEXT}, headers=headers,
                                            environ_overrides=environment, **kwargs)
                self.assertEqual(response.status_code, 403)

    def test_legacy_irreversibile_non_restituisce_i_valori_selezionati(self):
        response = self.client.post("/analyze", json={"text": TEXT})
        data = self.response_json(response)
        self.assertFalse(data["mapping_enabled"])
        self.assertEqual(data["mapping"], {})
        self.assertNotIn("source_text", data)
        self.assertNotIn("review_text", data)
        self.assertNotIn("context_excerpt", response.get_data(as_text=True))
        for value in ("Mario Rossi", "Padova", "via Roma", "via Venezia"):
            self.assertNotIn(value, response.get_data(as_text=True))
        self.assertFalse(data["verification"]["complete_anonymization"])

    def test_revisione_esplicita_locale_non_crea_dizionario(self):
        data = self.analyze()
        self.assertEqual(data["review_text"], TEXT)
        self.assertTrue(any(e.get("value") == "Padova" for e in data["occurrences"]))
        self.assertTrue(all(e["value"] in e["context_excerpt"] for e in data["occurrences"]))
        self.assertFalse(data["mapping_enabled"])
        self.assertEqual(data["mapping"], {})

    def test_chiusura_invalida_analisi_output_e_cache_ocr(self):
        analysis = self.select_residence_city(self.analyze())
        output = self.response_json(self.export_pdf(analysis))
        generation = APP.ocr._cache_generation
        APP.ocr._cache_put("sintetico:fast", 0, {"text": "SENTINELLA"})
        APP.ocr._img_pdf["immagine_sintetica"] = b"PDF_SINTETICO"
        self.response_json(self.client.delete("/session"))
        self.assertFalse(APP._ANALYSES)
        self.assertFalse(APP._DOCS)
        self.assertFalse(APP.ocr._cache)
        self.assertFalse(APP.ocr._img_pdf)
        self.assertEqual(self.export_text(analysis).status_code, 410)
        self.assertEqual(self.client.get("/doc/" + output["doc_id"] + "/file.pdf").status_code, 404)
        APP.ocr._cache_put("sintetico:fast", 0, {"text": "SENTINELLA"}, generation)
        self.assertFalse(APP.ocr._cache)

    def test_nuovo_documento_invalida_output_precedente(self):
        analysis = self.select_residence_city(self.analyze())
        old = self.response_json(self.export_pdf(analysis))
        self.analyze(text=TEXT + "\nSeconda perizia sintetica.")
        self.assertEqual(self.export_text(analysis).status_code, 410)
        self.assertEqual(self.client.get("/doc/" + old["doc_id"] + "/file.pdf").status_code, 404)

    def test_upload_oltre_soglia_spooling_resta_in_bytesio(self):
        uploaded = APP._uploaded_file
        streams = []

        def capture_upload():
            file = uploaded()
            if file is not None:
                streams.append(file.stream)
            return file

        payload = (TEXT + "\n" + " " * 600_000).encode("utf-8")
        with patch.object(APP, "_uploaded_file", side_effect=capture_upload):
            response = self.client.post("/analyze", data={
                "file": (io.BytesIO(payload), "testo_sintetico.txt")},
                content_type="multipart/form-data")
        # Il client di test Werkzeug, separato dal parser dell'app verificato,
        # puo' usare un TemporaryFile per costruire il corpo multipart sintetico.
        self.addCleanup(response.request.environ["wsgi.input"].close)
        data = self.response_json(response)
        self.assertEqual(data["n_chars"], len(payload.decode("utf-8")))
        self.assertTrue(streams)
        self.assertTrue(all(isinstance(stream, io.BytesIO) for stream in streams))

    def test_immagini_richiedono_presa_visione_prima_di_ogni_export(self):
        analysis = self.select_residence_city(self.analyze(pdf=synthetic_pdf(with_image=True)))
        self.assertTrue(analysis["warnings"])
        for response in (self.export_text(analysis), self.export_pdf(analysis)):
            self.assertEqual(self.response_json(response, 422)["code"], "unverified_content")
        self.assertFalse(any(d["role"] == "output" for d in APP._DOCS.values()))
        text = self.response_json(self.export_text(analysis, ack_unverified=True))
        self.assertFalse(text["verification"]["complete_anonymization"])
        self.response_json(self.export_pdf(analysis, ack_unverified=True))

    def test_verifica_pdf_fallita_non_rende_scaricabile_artefatto(self):
        analysis = self.select_residence_city(self.analyze(pdf=synthetic_pdf()))
        with patch.object(APP.pdf_export, "redact_occurrences", return_value=(
                b"ARTEFATTO_SINTETICO_NON_VERIFICATO", {"residual": ["e000003"]})):
            response = self.export_pdf(analysis)
        self.assertEqual(self.response_json(response, 422)["code"], "verification_failed")
        self.assertFalse(any(d["role"] == "output" for d in APP._DOCS.values()))

    def test_analisi_annullata_durante_inferenza_non_riappare(self):
        def cancelled(text):
            APP._clear_documents()
            return synthetic_entities(text)

        with patch.object(APP, "detect_model", side_effect=cancelled):
            response = self.client.post("/analyze", json={"text": TEXT, "review": True})
        self.assertEqual(response.status_code, 422)
        self.assertFalse(APP._ANALYSES)
        self.assertFalse(APP._DOCS)

    def test_decimal_fragment_is_repaired_before_review_and_pdf_export(self):
        import fitz
        text = ("Valore da mantenere 8,7.\nCatasto Fabbricati:\n"
                "7841 sub. 9 - vani 8,7 - cat. A/3.")
        with fitz.open() as document:
            page = document.new_page(width=1000, height=300)
            page.insert_text((25, 50), text, fontsize=11)
            source = document.tobytes()

        def fragmented_detector(value):
            return [{"start": m.start(), "end": m.end(), "label": "BUILDINGNUM",
                     "source": "modello", "validated": False, "score": .9}
                    for m in re.finditer(",7", value)], 1

        with patch.object(APP, "detect_model", side_effect=fragmented_detector), \
                patch.object(APP, "detect_regex", return_value=[]):
            analysis = self.analyze(pdf=source)
        self.assertEqual([e["value"] for e in analysis["occurrences"]], [",7", "8,7"])
        original, repaired = analysis["occurrences"]
        self.assertEqual(repaired["category"], "immobile.catasto")
        self.assertEqual(repaired["span_repair"], "cadastral_decimal")
        self.assertFalse(repaired["ambiguous"])
        self.response_json(self.export_text(analysis), 422)
        updated = self.response_json(self.client.post("/review/" + analysis["analysis_id"], json={
            "revision": analysis["revision"], "categories": ["immobile.catasto"],
            "overrides": {original["occurrence_id"]: False}}))
        exported = self.response_json(self.export_text(updated))
        self.assertEqual(updated["unresolved"], [])
        self.assertIn("Valore da mantenere 8,7.", exported["anonymized_text"])
        self.assertNotIn("vani 8", exported["anonymized_text"])
        self.assertNotIn("context_excerpt", str(exported))
        preview = self.response_json(self.export_pdf(updated))
        response = self.client.get("/doc/" + preview["doc_id"] + "/file.pdf")
        self.assertEqual(response.status_code, 200)
        with fitz.open(stream=response.data, filetype="pdf") as document:
            result = document[0].get_text()
        self.assertEqual(result.count("8,7"), 1)
        self.assertIn("Valore da mantenere 8,7.", result)


if __name__ == "__main__":
    unittest.main()
