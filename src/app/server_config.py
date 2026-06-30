# -*- coding: utf-8 -*-
"""
Configurazione host/porta del server Flask (condivisa tra tutti gli entry point).

Risoluzione con precedenza:  CLI args  >  env vars  >  config.json  >  default.

Il file config.json e' condiviso con l'app Tauri (che lo legge/scrive dal lato Rust
e passa host/porta al sidecar via env PII_HOST/PII_PORT):

  Windows:  %LOCALAPPDATA%\\rizzo-pii\\config.json
  Linux:    ~/.local/share/rizzo-pii/config.json
  macOS:    ~/Library/Application Support/rizzo-pii/config.json

Formato:  {"host": "127.0.0.1", "port": 5005}

Il codice di uscita 76 (EX_PROTOCOL) segnala "porta occupata" ed e' riconosciuto
dall'app Tauri per mostrare il form di configurazione nello splash.
"""

import json
import os
import socket
import sys
from pathlib import Path

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 5005
EXIT_PORT_CONFLICT = 76  # riconosciuto da Tauri (lib.rs) come "porta occupata"


def config_dir() -> Path:
    """Directory di configurazione (platform-specific, coerente con serve.py e Tauri)."""
    if sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home()))
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
    else:
        base = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share"))
    return base / "rizzo-pii"


def config_path() -> Path:
    return config_dir() / "config.json"


def load_config() -> dict:
    """Legge config.json; ritorna {} se mancante o corrotto."""
    p = config_path()
    if p.exists():
        try:
            return json.loads(p.read_text("utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def save_config(host: str, port: int):
    """Scrive config.json (crea la directory se necessario)."""
    d = config_dir()
    d.mkdir(parents=True, exist_ok=True)
    (d / "config.json").write_text(
        json.dumps({"host": host, "port": port}, indent=2), "utf-8"
    )


def resolve(cli_host=None, cli_port=None):
    """Risolve host/porta con la catena: CLI > env > config.json > default.

    Ritorna (host: str, port: int).
    """
    cfg = load_config()
    host = cli_host or os.environ.get("PII_HOST") or cfg.get("host") or DEFAULT_HOST
    port = cli_port or os.environ.get("PII_PORT") or cfg.get("port") or DEFAULT_PORT
    return str(host), int(port)


def port_available(host: str, port: int) -> bool:
    """True se la porta e' libera (tenta un bind effimero)."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind((host, port))
            return True
    except OSError:
        return False
