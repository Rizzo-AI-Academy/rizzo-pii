# -*- coding: utf-8 -*-
"""
Entry point dell'app desktop (PyInstaller). Avvia il server Flask in locale e
apre il browser. Chiudere questa finestra termina l'applicazione.

Configurazione host/porta:
  1) CLI args (--host / --port)   ->  precedenza massima
  2) env PII_HOST / PII_PORT
  3) config.json  (vedi server_config.py)
  4) default 127.0.0.1:5005
"""

import multiprocessing
import os
import sys
import threading
import time
import webbrowser

import server_config

HOST, PORT = server_config.resolve()
URL = f"http://{HOST}:{PORT}/"

# --- pre-check porta PRIMA di caricare il modello (che richiede secondi) --- #
if not server_config.port_available(HOST, PORT):
    print(f"ERRORE: porta {PORT} occupata su {HOST}")
    sys.exit(server_config.EXIT_PORT_CONFLICT)


def _open_browser():
    time.sleep(2.5)            # attende che il modello sia caricato e il server su
    webbrowser.open(URL)


if __name__ == "__main__":
    multiprocessing.freeze_support()   # necessario per gli exe congelati su Windows
    print("Avvio Anonimizzatore PII... (il primo avvio carica il modello, attendi)")
    from app import app                # l'import carica il modello
    threading.Thread(target=_open_browser, daemon=True).start()
    print(f"In esecuzione su {URL}  --  chiudi questa finestra per terminare.")
    try:
        app.run(host=HOST, port=PORT, threaded=True)
    except OSError as e:
        print(f"ERRORE bind: {e}")
        sys.exit(server_config.EXIT_PORT_CONFLICT)
