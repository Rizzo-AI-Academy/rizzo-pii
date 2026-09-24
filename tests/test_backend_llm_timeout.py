# -*- coding: utf-8 -*-
"""Un modello lento non deve interrompere la generazione dei template.

Il timeout di lettura di urllib scade DENTRO getresponse() e solleva TimeoutError, che
deriva da OSError e NON da urllib.error.URLError. Intercettando la sola URLError, una
generazione lenta -- normale con un modello locale su CPU o molto quantizzato -- fermava
l'intera esecuzione invece di consumare un tentativo e passare al successivo, buttando i
template gia' scritti.

Il caso e' stato segnalato da @p3pp01 su Gemma servito da Ollama.

Qui non si parla con nessun server: urlopen viene sostituito da una funzione che solleva
l'eccezione da provare, e si verifica che la chiamata RITORNI None (nessun template) invece
di propagare. Nessun dato personale e' coinvolto.
"""

import sys
import unittest
import urllib.error
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "data_pipeline"))

import llm_template_bank as tb  # noqa: E402


class ErroriDiReteNonInterrompono(unittest.TestCase):
    """Ogni errore di rete deve diventare "nessun template", non un'eccezione."""

    # (nome del caso, eccezione sollevata da urlopen)
    CASI = [
        ("timeout di lettura (il caso segnalato)", TimeoutError("timed out")),
        ("timeout come socket.timeout", OSError("timed out")),
        ("connessione rifiutata: server non avviato", ConnectionRefusedError(111, "rifiutata")),
        ("host irraggiungibile", urllib.error.URLError("nodename nor servname provided")),
    ]

    def test_backend_locale(self):
        with mock.patch.object(tb, "LLM_BASE_URL", "http://127.0.0.1:8080/v1"), \
             mock.patch.object(tb.time, "sleep"):            # niente attese nei test
            for nome, exc in self.CASI:
                with self.subTest(nome):
                    with mock.patch.object(tb.urllib.request, "urlopen", side_effect=exc):
                        self.assertIsNone(tb.call_local_openai("prompt", retries=2))

    def test_gemini(self):
        with mock.patch.object(tb, "API_KEY", "chiave-finta"), \
             mock.patch.object(tb.time, "sleep"):
            for nome, exc in self.CASI:
                with self.subTest(nome):
                    with mock.patch.object(tb.urllib.request, "urlopen", side_effect=exc):
                        self.assertIsNone(tb.call_gemini("prompt", retries=2))

    def test_call_llm_smista_sul_backend_giusto(self):
        """call_llm() e' il punto d'ingresso: anche da li' il timeout non deve propagare."""
        with mock.patch.object(tb, "LLM_BASE_URL", "http://127.0.0.1:8080/v1"), \
             mock.patch.object(tb.time, "sleep"), \
             mock.patch.object(tb.urllib.request, "urlopen", side_effect=TimeoutError()):
            self.assertIsNone(tb.call_llm("prompt", retries=1))


if __name__ == "__main__":
    unittest.main()
