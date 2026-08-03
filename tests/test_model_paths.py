#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Risoluzione della cartella del modello (src/common/model_paths.py).

Il caso che ha motivato questo modulo: `export_onnx.py` scrive il bundle in
`models/<nome>-onnx/`, che il glob `rizzo-pii-0.3B-v*` raccoglie ma il regex della
versione non riconosce. La copia inline dentro app.py non filtrava sul match e faceva
`AttributeError` sul `.group(1)` di None — all'IMPORT, quindi l'app non partiva piu'.
"""
import os
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "common"))

from model_paths import resolve_model_dir, versioned_dirs, _version_key  # noqa: E402


class TestVersionKey(unittest.TestCase):
    def test_plain_version(self):
        self.assertEqual(_version_key("rizzo-pii-0.3B-v1.3.0"), (1, 3, 0))

    def test_two_components(self):
        self.assertEqual(_version_key("rizzo-pii-0.3B-v2.0"), (2, 0))

    def test_onnx_bundle_is_not_a_version(self):
        # il cuore del bug: il nome non FINISCE con la versione
        self.assertIsNone(_version_key("rizzo-pii-0.3B-v1.3.0-onnx"))

    def test_other_siblings_ignored(self):
        for name in ("rizzo-pii-0.3B-v1.3.0-backup", "rizzo-pii-0.3B-vecchio",
                     "rizzo-pii-0.3B-v1.3.0.tar"):
            self.assertIsNone(_version_key(name), name)


class TestResolve(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.models = Path(self._tmp.name)
        self._saved = os.environ.pop("PII_MODEL_DIR", None)

    def tearDown(self):
        self._tmp.cleanup()
        if self._saved is not None:
            os.environ["PII_MODEL_DIR"] = self._saved

    def _mk(self, *names):
        for n in names:
            (self.models / n).mkdir(parents=True)

    def test_picks_highest_version(self):
        self._mk("rizzo-pii-0.3B-v1.1.0", "rizzo-pii-0.3B-v1.3.0", "rizzo-pii-0.3B-v1.2.0")
        self.assertTrue(resolve_model_dir(self.models).endswith("v1.3.0"))

    def test_numeric_not_lexicographic(self):
        # "v1.10.0" > "v1.9.0" solo se si confrontano numeri, non stringhe
        self._mk("rizzo-pii-0.3B-v1.9.0", "rizzo-pii-0.3B-v1.10.0")
        self.assertTrue(resolve_model_dir(self.models).endswith("v1.10.0"))

    def test_onnx_sibling_does_not_crash(self):
        # LA regressione: prima sollevava AttributeError
        self._mk("rizzo-pii-0.3B-v1.3.0", "rizzo-pii-0.3B-v1.3.0-onnx")
        self.assertTrue(resolve_model_dir(self.models).endswith("v1.3.0"))

    def test_onnx_alone_is_not_selected(self):
        # se resta solo il bundle non lo si spaccia per un checkpoint: si ripiega
        self._mk("rizzo-pii-0.3B-v1.3.0-onnx")
        got = resolve_model_dir(self.models)
        self.assertFalse(got.endswith("-onnx"))

    def test_pinned_version_wins(self):
        self._mk("rizzo-pii-0.3B-v1.2.0", "rizzo-pii-0.3B-v1.3.0")
        self.assertTrue(resolve_model_dir(self.models, pinned="1.2.0").endswith("v1.2.0"))

    def test_pinned_missing_falls_back_to_latest(self):
        self._mk("rizzo-pii-0.3B-v1.3.0")
        self.assertTrue(resolve_model_dir(self.models, pinned="9.9.9").endswith("v1.3.0"))

    def test_env_override_wins_over_everything(self):
        self._mk("rizzo-pii-0.3B-v1.3.0")
        os.environ["PII_MODEL_DIR"] = "/percorso/scelto/a/mano"
        try:
            self.assertEqual(resolve_model_dir(self.models, pinned="1.3.0"),
                             "/percorso/scelto/a/mano")
        finally:
            os.environ.pop("PII_MODEL_DIR")

    def test_unversioned_fallback(self):
        self._mk("rizzo-pii-0.3B")
        self.assertTrue(resolve_model_dir(self.models).endswith("rizzo-pii-0.3B"))

    def test_legacy_last_resort(self):
        self.assertTrue(resolve_model_dir(self.models).endswith("pii_model_legacy"))

    def test_files_are_not_candidates(self):
        self._mk("rizzo-pii-0.3B-v1.1.0")
        (self.models / "rizzo-pii-0.3B-v9.9.9").write_text("non e' una cartella")
        self.assertTrue(resolve_model_dir(self.models).endswith("v1.1.0"))

    def test_versioned_dirs_sorted(self):
        self._mk("rizzo-pii-0.3B-v1.3.0", "rizzo-pii-0.3B-v1.1.0", "rizzo-pii-0.3B-v1.3.0-onnx")
        got = [p.name for p, _ in versioned_dirs(self.models)]
        self.assertEqual(got, ["rizzo-pii-0.3B-v1.1.0", "rizzo-pii-0.3B-v1.3.0"])


if __name__ == "__main__":
    unittest.main()
