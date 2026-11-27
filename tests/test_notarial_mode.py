"""Optional mode integration and legacy isolation, using only synthetic data."""

import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from notarial_test_support import APP, STANDARD_APP


@unittest.skipIf(APP is None, "Flask and PyMuPDF are required")
class NotarialMode(unittest.TestCase):
    def setUp(self):
        from flask import Flask
        self.original = Flask("synthetic-standard")
        self.original.add_url_rule("/", endpoint="standard_index", view_func=lambda: "standard untouched")
        self.original.add_url_rule("/health", endpoint="standard_health", view_func=lambda: {"profile": "standard"})
        APP.install_notarial_mode(self.original)
        self.client = self.original.test_client()
        self.addCleanup(APP._clear_documents)

    def test_mount_is_idempotent_and_standard_routes_keep_their_response(self):
        original_wsgi = self.original.wsgi_app
        APP.install_notarial_mode(self.original)
        self.assertIs(self.original.wsgi_app, original_wsgi)
        self.assertEqual(self.client.get("/").data, b"standard untouched")
        self.assertEqual(self.client.get("/health").json, {"profile": "standard"})
        self.assertNotIn("Cache-Control", self.client.get("/").headers)
        self.assertEqual(self.client.get("/notarial/health").json["profile"], "notarial-selective")

    def test_recognizer_and_ocr_weights_are_shared_but_document_stores_are_not(self):
        import ocr
        self.assertIs(APP.nlp, STANDARD_APP.nlp)
        self.assertEqual(APP.MODEL_DIR, STANDARD_APP.MODEL_DIR)
        self.assertIs(APP.ocr._pipelines, ocr._pipelines)
        self.assertIs(APP.ocr._lock, ocr._lock)
        self.assertIsNot(APP.ocr._cache, ocr._cache)
        self.assertIsNot(APP._DOCS, STANDARD_APP._DOCS)
        with patch.dict(ocr._cache, {"synthetic-standard": {0: "synthetic"}}, clear=True), \
                patch.dict(APP.ocr._cache, {"synthetic-notarial": {0: "synthetic"}}, clear=True):
            APP.ocr.clear_cache()
            self.assertIn("synthetic-standard", ocr._cache)
            self.assertFalse(APP.ocr._cache)

    def test_cookie_and_api_bootstrap_are_scoped_to_the_mode(self):
        self.assertEqual(self.client.get("/notarial/settings").status_code, 403)
        bootstrap = self.client.get("/notarial/")
        self.assertEqual(bootstrap.status_code, 200)
        self.assertIn("Path=/notarial/", bootstrap.headers["Set-Cookie"])
        self.assertIn("HttpOnly", bootstrap.headers["Set-Cookie"])
        self.assertIn("SameSite=Strict", bootstrap.headers["Set-Cookie"])
        self.assertIn(b'window.PII_BASE="/notarial"', bootstrap.data)
        self.assertIn(b"Nader Deghaili", bootstrap.data)
        self.assertIn("no-store", bootstrap.headers["Cache-Control"])
        response = self.client.get("/notarial/settings")
        self.assertEqual(response.status_code, 200)
        self.assertFalse(response.json["mapping_enabled"])
        self.assertIn("no-store", response.headers["Cache-Control"])
        rejected = self.client.post("/notarial/analyze", json={"text": "synthetic"})
        self.assertEqual(rejected.status_code, 403)  # HttpOnly preview cookie is not a POST capability.

    def test_prefixed_routes_apply_same_origin_guard_and_export_gate(self):
        headers = {"X-PII-Token": APP.SESSION_TOKEN}
        rejected = self.client.post("/notarial/analyze", json={"text": "synthetic"},
                                    headers={**headers, "Origin": "https://synthetic.invalid"})
        self.assertEqual(rejected.status_code, 403)
        analysis = self.client.post("/notarial/analyze", json={"text": "Testo sintetico", "review": True},
                                   headers=headers)
        self.assertEqual(analysis.status_code, 200)
        state = analysis.json
        output = self.client.post("/notarial/review/" + state["analysis_id"] + "/text",
                                  json={"revision": state["revision"]}, headers=headers)
        self.assertEqual(output.status_code, 422)
        self.assertEqual(output.json["code"], "unverified_content")
        output = self.client.post("/notarial/review/" + state["analysis_id"] + "/text",
                                  json={"revision": state["revision"], "ack_unverified": True}, headers=headers)
        self.assertEqual(output.status_code, 200)
        self.assertEqual(output.json["anonymized_text"], "Testo sintetico")
        self.assertFalse(output.json["verification"]["complete_anonymization"])

    def test_notarial_preferences_do_not_read_or_overwrite_standard_mapping(self):
        import server_config
        import notarial_server_config
        with tempfile.TemporaryDirectory() as temporary, \
                patch.object(server_config, "config_dir", return_value=Path(temporary)), \
                patch.dict(os.environ, {"PII_MAPPING": "1"}, clear=True):
            standard_path = Path(temporary) / "prefs.json"
            standard_bytes = json.dumps({"mapping_enabled": True, "excluded_tags": ["AGE"]}).encode()
            standard_path.write_bytes(standard_bytes)
            self.assertTrue(server_config.load_prefs()["mapping_enabled"])
            self.assertFalse(notarial_server_config.load_prefs()["mapping_enabled"])
            self.assertEqual(notarial_server_config.load_prefs()["excluded_tags"], [])
            notarial_server_config.save_prefs(excluded_tags=["ZIPCODE"], mapping_enabled=True)
            self.assertEqual(standard_path.read_bytes(), standard_bytes)
            self.assertEqual(notarial_server_config.prefs_path(), Path(temporary) / "notarial/prefs.json")
            self.assertFalse(notarial_server_config.load_prefs()["mapping_enabled"])
            self.assertEqual(notarial_server_config.load_prefs()["excluded_tags"], ["ZIPCODE"])


if __name__ == "__main__":
    unittest.main()
