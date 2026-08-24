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
Log:   %LOCALAPPDATA%\\\\rizzo-pii\\\\backend.log  (in append; oltre 1 MB ruota su backend.log.1)

Il processo esce con codice 76 (EX_PROTOCOL) se la porta e' occupata: Tauri lo
riconosce e mostra il form di configurazione nello splash screen.
"""

import os
import sys
import time

# --- log su file (windowed mode -> niente console) ------------------------- #
# In APPEND: con "w" il rilancio del sidecar (retry_backend di Tauri, o l'utente che
# riapre l'app) cancellava il log dell'avvio fallito, cioe' proprio quello che lo
# splash chiede di andare a leggere quando il backend non parte. Il lato Rust (tlog)
# gia' scrive in append. Rotazione a 1 MB su .1 perche' il file non cresca all'infinito.
_LOG_MAX = 1_000_000
_logdir = os.path.join(os.environ.get("LOCALAPPDATA", os.path.expanduser("~")), "rizzo-pii")
_logpath = os.path.join(_logdir, "backend.log")
try:
    os.makedirs(_logdir, exist_ok=True)
    _ruotato = False
    try:
        if os.path.getsize(_logpath) > _LOG_MAX:
            os.replace(_logpath, _logpath + ".1")
            _ruotato = True
    except OSError:
        pass  # log assente, o non spostabile: si accoda al file esistente
    _log = open(_logpath, "a", encoding="utf-8", buffering=1)
    sys.stdout = _log
    sys.stderr = _log
    print(f"\n===== avvio {time.strftime('%Y-%m-%d %H:%M:%S')} (pid {os.getpid()}) =====")
    if _ruotato:
        # senza questa riga, chi segue lo splash ("vedi backend.log") non trova la
        # traccia dell'avvio fallito: la rotazione l'ha appena spostata in .1.
        print("(il log precedente e' in backend.log.1)")
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
