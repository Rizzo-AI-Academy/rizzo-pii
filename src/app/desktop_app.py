# -*- coding: utf-8 -*-
"""
Entry point dell'app desktop (PyInstaller). Avvia il server Flask in locale e
apre il browser. Chiudere questa finestra termina l'applicazione.
"""

import multiprocessing
import threading
import time
import webbrowser

PORT = 5000
URL = f"http://127.0.0.1:{PORT}/"


def _open_browser():
    time.sleep(2.5)            # attende che il modello sia caricato e il server su
    webbrowser.open(URL)


if __name__ == "__main__":
    multiprocessing.freeze_support()   # necessario per gli exe congelati su Windows
    print("Avvio Anonimizzatore PII... (il primo avvio carica il modello, attendi)")
    from app import app                # l'import carica il modello
    threading.Thread(target=_open_browser, daemon=True).start()
    print(f"In esecuzione su {URL}  --  chiudi questa finestra per terminare.")
    app.run(host="127.0.0.1", port=PORT, threaded=True)
