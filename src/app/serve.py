# -*- coding: utf-8 -*-
"""
Entry headless del backend per l'app desktop Tauri.

Avvia SOLO il server Flask in locale: la finestra nativa la apre Tauri (WebView2),
quindi qui NON si apre alcun browser. Pensato per girare come processo figlio (sidecar)
dell'app Tauri, impacchettato con PyInstaller in modalita' windowed (console=False).

In modalita' windowed sys.stdout/sys.stderr sono None: li reindirizziamo su un file di
log PRIMA di importare 'app' (che al caricamento stampa e carica il modello), cosi'
nessun print() fa crashare il processo e i problemi restano diagnosticabili.

Porta: 5005 (5000 e' occupata da AirPlay Receiver su macOS; override con PII_PORT).
Log:   %LOCALAPPDATA%\\rizzo-pii\\backend.log
"""

import os
import sys

PORT = int(os.environ.get("PII_PORT", "5005"))

# --- log su file (windowed mode -> niente console) ------------------------- #
_logdir = os.path.join(os.environ.get("LOCALAPPDATA", os.path.expanduser("~")), "rizzo-pii")
try:
    os.makedirs(_logdir, exist_ok=True)
    _log = open(os.path.join(_logdir, "backend.log"), "w", encoding="utf-8", buffering=1)
    sys.stdout = _log
    sys.stderr = _log
except Exception:
    pass  # se non si puo' loggare, si prosegue comunque

# l'import carica il modello (puo' richiedere alcuni secondi)
from app import app  # noqa: E402

if __name__ == "__main__":
    print(f"[serve] avvio server su 127.0.0.1:{PORT}")
    app.run(host="127.0.0.1", port=PORT, threaded=True)
