# -*- coding: utf-8 -*-
"""Il log del backend (`serve.py`) deve sopravvivere al riavvio del sidecar.

Quando il backend non parte, lo splash di Tauri dice all'utente di guardare
`backend.log`. Ma il riavvio del sidecar (`retry_backend`, o l'utente che riapre
l'app) riapre quel file: se lo apre in troncamento, cancella proprio il log
dell'avvio fallito che si voleva leggere.

`serve.py` non e' importabile in un test (a import time carica il modello), quindi
qui si esegue il suo PREAMBOLO vero, ritagliato dal file, in un LOCALAPPDATA usa e
getta: si testa il codice che viene spedito, non una copia.
"""

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SERVE = ROOT / "src" / "app" / "serve.py"


def preambolo():
    """Le righe di serve.py fino a prima dell'import di server_config."""
    src = SERVE.read_text(encoding="utf-8")
    testa = src[:src.index("import server_config")]
    assert "backend.log" in testa, "il preambolo del log non e' piu' li'"
    return testa


def avvia(testa, home, marcatore):
    """Un avvio del sidecar: esegue il preambolo e scrive un marcatore nel log."""
    script = Path(home) / "_avvio.py"
    script.write_text(testa + "\nprint(%r)\n" % marcatore, encoding="utf-8")
    env = dict(os.environ, LOCALAPPDATA=home, PYTHONIOENCODING="utf-8")
    return subprocess.run([sys.executable, str(script)], env=env,
                          capture_output=True, text=True)


def log(home, nome="backend.log"):
    p = Path(home) / "rizzo-pii" / nome
    return p.read_text(encoding="utf-8", errors="replace") if p.exists() else None


class LogBackend(unittest.TestCase):
    def setUp(self):
        self.testa = preambolo()
        self.tmp = tempfile.TemporaryDirectory()
        self.home = self.tmp.name
        self.addCleanup(self.tmp.cleanup)

    def test_il_riavvio_non_cancella_il_log_precedente(self):
        avvia(self.testa, self.home, "AVVIO-FALLITO")
        avvia(self.testa, self.home, "RIAVVIO")
        testo = log(self.home) or ""
        self.assertIn("AVVIO-FALLITO", testo, "il log del crash e' stato cancellato")
        self.assertIn("RIAVVIO", testo)

    def test_ruota_oltre_il_massimo_invece_di_crescere(self):
        d = Path(self.home) / "rizzo-pii"
        d.mkdir(parents=True)
        (d / "backend.log").write_text("VECCHIO\n" + "x" * 1_100_000, encoding="utf-8")
        avvia(self.testa, self.home, "DOPO-ROTAZIONE")
        self.assertIn("VECCHIO", log(self.home, "backend.log.1") or "")
        nuovo = log(self.home) or ""
        self.assertIn("DOPO-ROTAZIONE", nuovo)
        self.assertLess(len(nuovo), 10_000, "il log non e' ripartito dopo la rotazione")
        # lo splash manda l'utente su backend.log: se la traccia e' finita in .1,
        # il file che il messaggio nomina deve almeno dirglielo
        self.assertIn("backend.log.1", nuovo)

    def test_tre_avvii_di_fila_nessuno_perso(self):
        for k in range(3):
            avvia(self.testa, self.home, "AVVIO-%d" % k)
        testo = log(self.home) or ""
        for k in range(3):
            self.assertIn("AVVIO-%d" % k, testo)

    def test_se_la_rotazione_fallisce_si_logga_lo_stesso(self):
        # .1 occupato da una cartella non vuota -> os.replace non puo' riuscire
        d = Path(self.home) / "rizzo-pii"
        (d / "backend.log.1" / "occupata").mkdir(parents=True)
        (d / "backend.log").write_text("y" * 1_100_000, encoding="utf-8")
        r = avvia(self.testa, self.home, "ROTAZIONE-FALLITA")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("ROTAZIONE-FALLITA", log(self.home) or "")

    def test_primo_avvio_crea_la_cartella(self):
        r = avvia(self.testa, self.home, "PRIMO-AVVIO")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("PRIMO-AVVIO", log(self.home) or "")


if __name__ == "__main__":
    unittest.main()
