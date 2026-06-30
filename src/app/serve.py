# -*- coding: utf-8 -*-
"""
Entry headless del backend per l'app desktop Tauri.

Avvia SOLO il server Flask in locale: la finestra nativa la apre Tauri (WebView2),
quindi qui NON si apre alcun browser. Pensato per girare come processo figlio (sidecar)
dell'app Tauri, impacchettato con PyInstaller in modalita' windowed (console=False).

In modalita' windowed sys.stdout/sys.stderr sono None: li reindirizziamo su un file di
log PRIMA di importare 'app' (che al caricamento stampa e carica il modello), cosi'
nessun print() fa crashare il processo e i problemi restano diagnosticabili.

Configurazione host/porta:
  1) CLI args (--host / --port)   -\u003e  precedenza massima
  2) env PII_HOST / PII_PORT     -\u003e  passati da Tauri (lib.rs)
  3) config.json                 -\u003e  salvato da Tauri o dall'UI Flask
  4) default 127.0.0.1:5005

NB: la 5000 su macOS e' occupata da AirPlay Receiver (ControlCenter) -\u003e pagina bianca.
Log:   %LOCALAPPDATA%\\\\rizzo-pii\\\\backend.log

Il processo esce con codice 76 (EX_PROTOCOL) se la porta e' occupata: Tauri lo
riconosce e mostra il form di configurazione nello splash screen.
"""

import os
import sys

# --- log su file (windowed mode -> niente console) ------------------------- #
_logdir = os.path.join(os.environ.get("LOCALAPPDATA", os.path.expanduser("~")), "rizzo-pii")
try:
    os.makedirs(_logdir, exist_ok=True)
    _log = open(os.path.join(_logdir, "backend.log"), "w", encoding="utf-8", buffering=1)
    sys.stdout = _log
    sys.stderr = _log
except Exception:
    pass  # se non si puo' loggare, si prosegue comunque

import server_config  # noqa: E402

HOST, PORT = server_config.resolve()

# --- pre-check porta PRIMA di caricare il modello (che richiede secondi) --- #
if not server_config.port_available(HOST, PORT):
    print(f"[serve] ERRORE: porta {PORT} occupata su {HOST}")
    sys.exit(server_config.EXIT_PORT_CONFLICT)

# l'import carica il modello (puo' richiedere alcuni secondi)
from app import app  # noqa: E402

if __name__ == "__main__":
    print(f"[serve] avvio server su {HOST}:{PORT}")
    try:
        app.run(host=HOST, port=PORT, threaded=True)
    except OSError as e:
        # sicurezza extra: se il bind fallisce nonostante il pre-check (race condition)
        print(f"[serve] ERRORE bind: {e}")
        sys.exit(server_config.EXIT_PORT_CONFLICT)
