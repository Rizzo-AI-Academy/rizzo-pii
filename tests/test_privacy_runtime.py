# -*- coding: utf-8 -*-
"""Lifecycle e impostazioni privacy, con soli contenuti sintetici.

Le cache si provano senza PyMuPDF/ONNX: qui non si misura il riconoscimento.
"""

import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[1]
APP_DIR = ROOT / "src" / "app"
sys.path.insert(0, str(APP_DIR))
import notarial_server_config as server_config


def load_isolated(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class PrivateDefaults(unittest.TestCase):
    def test_irreversibile_anche_con_preferenze_legacy(self):
        with patch.object(server_config, "_read_prefs_file", return_value={"mapping_enabled": True}), \
                patch.dict(os.environ, {}, clear=True):
            self.assertFalse(server_config.load_prefs()["mapping_enabled"])
            self.assertFalse(server_config.load_prefs()["ocr_redact_seals"])

    def test_override_esplicito_di_esecuzione(self):
        with patch.object(server_config, "_read_prefs_file", return_value={}), \
                patch.dict(os.environ, {"PII_NOTARIAL_MAPPING": "1"}, clear=True):
            self.assertTrue(server_config.load_prefs()["mapping_enabled"])

    def test_mapping_reversibile_solo_nella_sessione(self):
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(server_config, "config_dir", return_value=Path(directory)):
            active = server_config.save_prefs(mapping_enabled=True)
            persisted = json.loads((Path(directory) / "prefs.json").read_text(encoding="utf-8"))
            self.assertTrue(active["mapping_enabled"])
            self.assertFalse(persisted["mapping_enabled"])

    def test_nessun_probe_o_salvataggio_verso_host_esterni(self):
        with patch.object(server_config.socket, "socket") as sock, \
                patch.object(server_config, "config_dir") as directory:
            with self.assertRaises(ValueError):
                server_config.port_available("192.0.2.1", 5005)
            with self.assertRaises(ValueError):
                server_config.save_config("0.0.0.0", 5005)
            sock.assert_not_called()
            directory.assert_not_called()


class CacheLifecycle(unittest.TestCase):
    def setUp(self):
        with patch.dict(sys.modules, {"fitz": MagicMock()}):
            self.ocr = load_isolated("privacy_ocr", APP_DIR / "notarial_ocr.py")

    def test_rilascia_anche_conversioni_e_varianti_preset(self):
        ocr = self.ocr
        pdf = b"PDF_SINTETICO"
        other = b"ALTRO_SINTETICO"
        key = ocr.sha(pdf)
        ocr._cache[key + ":fast"] = {0: {"text": "SENTINELLA"}}
        ocr._cache[key + ":balanced"] = {0: {"text": "SENTINELLA"}}
        ocr._cache[ocr.sha(other) + ":fast"] = {0: {"text": "ALTRA"}}
        ocr._img_pdf["immagine_sintetica"] = pdf
        ocr.clear_cache(pdf)
        self.assertEqual(list(ocr._cache), [ocr.sha(other) + ":fast"])
        self.assertEqual(ocr._img_pdf, {})
        ocr.clear_cache()
        self.assertEqual(ocr._cache, {})

    def test_risultati_annullati_non_ripopolano_la_cache(self):
        ocr = self.ocr
        generation = ocr._cache_generation
        ocr.clear_cache()
        ocr._cache_put("synthetic:fast", 0, {"text": "SENTINELLA"}, generation)
        self.assertEqual(ocr._cache, {})

    def test_budget_ram_e_documento_unico(self):
        ocr = self.ocr
        ocr._cache_put("uno:fast", 0, {"text": "SENTINELLA_UNO"})
        ocr._cache_put("due:fast", 0, {"text": "SENTINELLA_DUE"})
        self.assertEqual(list(ocr._cache), ["due:fast"])
        ocr._CACHE_MAX_BYTES = 512
        ocr._cache_put("grande:fast", 0, {"text": "x" * 4096})
        self.assertEqual(ocr._cache, {})


class Telemetry(unittest.TestCase):
    def test_import_reale_ort_senza_inizializzazione_telemetrica(self):
        if importlib.util.find_spec("onnxruntime") is None:
            self.skipTest("ONNX Runtime non installato: prova nativa separata dai mock")
        for module_file in (APP_DIR / "onnx_ner.py", APP_DIR / "docmd" / "common.py"):
            with self.subTest(module=module_file.name), tempfile.TemporaryDirectory() as directory:
                # Processo nuovo: una disattivazione API dopo import non nasconde
                # eventuali messaggi della fase nativa di inizializzazione.
                script = (
                    "import importlib.util, os\n"
                    "from pathlib import Path\n"
                    f"spec=importlib.util.spec_from_file_location('privacy_probe', {str(module_file)!r})\n"
                    "module=importlib.util.module_from_spec(spec)\n"
                    "spec.loader.exec_module(module)\n"
                    "assert os.environ['ORT_DISABLE_TELEMETRY']=='1'\n"
                    "import onnxruntime as ort\n"
                    "ort.disable_telemetry_events()\n"
                    "print('ORT_IMPORT_OFFLINE_OK')\n"
                )
                # Nessuna modifica di HOME o del profilo reale. Cache e temporanei
                # opzionali della prova puntano a una directory sintetica isolata.
                environment = dict(os.environ, ORT_DISABLE_TELEMETRY="0",
                                   XDG_CACHE_HOME=directory, TMPDIR=directory,
                                   PYTHONDONTWRITEBYTECODE="1")
                completed = subprocess.run([sys.executable, "-B", "-c", script],
                                           cwd=directory, env=environment,
                                           capture_output=True, text=True, timeout=30)
                self.assertEqual(completed.returncode, 0, completed.stderr)
                self.assertEqual(completed.stdout.strip(), "ORT_IMPORT_OFFLINE_OK")
                self.assertNotIn("telemetry", completed.stderr.casefold())
                self.assertEqual(list(Path(directory).rglob("*")), [],
                                 "L'import non deve creare cache o identificatori")

    def test_disabilitata_prima_della_sessione_ner(self):
        events = []
        runtime = MagicMock()
        runtime.disable_telemetry_events.side_effect = lambda: events.append("disable")
        runtime.InferenceSession.side_effect = lambda *a, **k: events.append("session") or MagicMock()
        with tempfile.TemporaryDirectory() as directory, \
                patch.dict(sys.modules, {"numpy": MagicMock(), "onnxruntime": runtime,
                                         "tokenizers": MagicMock()}):
            model = Path(directory)
            (model / "config.json").write_text('{"id2label":{"0":"O"}}')
            module = load_isolated("privacy_ner", APP_DIR / "onnx_ner.py")
            module.OnnxTokenClassifier(directory, onnx_path="synthetic.onnx")
        self.assertEqual(events, ["disable", "session"])

    def test_disabilitata_prima_della_sessione_ocr(self):
        events = []
        runtime = MagicMock()
        runtime.disable_telemetry_events.side_effect = lambda: events.append("disable")
        runtime.InferenceSession.side_effect = lambda *a, **k: events.append("session")
        with patch.dict(sys.modules, {"numpy": MagicMock(), "onnxruntime": runtime,
                                     "yaml": MagicMock()}):
            common = load_isolated("privacy_common", APP_DIR / "docmd" / "common.py")
            common.create_session(Path("synthetic.onnx"))
        self.assertEqual(events, ["disable", "session"])


if __name__ == "__main__":
    unittest.main()
