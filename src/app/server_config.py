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

Nella stessa directory vivono anche le preferenze di anonimizzazione, prefs.json:

  {"excluded_tags": ["AGE", "GENDER"], "mapping_enabled": true}

E' un file separato di proposito: Tauri riscrive config.json per intero quando si
cambia la porta dallo splash, e sovrascriverebbe qualunque altra chiave.
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


# --------------------------------------------------------------------------- #
# Preferenze di anonimizzazione (prefs.json, accanto a config.json)
#
#   {"excluded_tags": ["AGE"], "mapping_enabled": true}
#
#   excluded_tags   -> tag rilevati ma NON sostituiti (restano in chiaro)
#   mapping_enabled -> false = niente dizionario placeholder->valore, quindi
#                      l'anonimizzazione e' definitiva e non reversibile
# --------------------------------------------------------------------------- #
PREFS_FILE = "prefs.json"
LEGACY_PREFS_FILE = "tags.json"   # nome usato prima che il file contenesse anche il mapping
DEFAULT_MAPPING_ENABLED = True


def prefs_path() -> Path:
    return config_dir() / PREFS_FILE


def parse_tag_list(value) -> list:
    """Normalizza una lista di tag: accetta stringa CSV o iterabile. -> UPPERCASE, senza duplicati."""
    if value is None:
        return []
    if isinstance(value, str):
        items = value.replace(";", ",").split(",")
    else:
        items = list(value)
    out = []
    for it in items:
        t = str(it).strip().upper()
        if t and t not in out:
            out.append(t)
    return out


def parse_bool(value, default=True) -> bool:
    """Accetta bool, 0/1, "true"/"false"/"on"/"off"/"si"/"no". Altro -> default."""
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    s = str(value).strip().lower()
    if s in ("1", "true", "yes", "y", "on", "si", "sì"):
        return True
    if s in ("0", "false", "no", "n", "off"):
        return False
    return default


def _read_prefs_file() -> dict:
    """Contenuto grezzo di prefs.json ({} se manca o e' corrotto)."""
    for p in (prefs_path(), config_dir() / LEGACY_PREFS_FILE):
        if p.exists():
            try:
                data = json.loads(p.read_text("utf-8"))
                if isinstance(data, dict):
                    return data
            except (json.JSONDecodeError, OSError):
                pass
    return {}


def load_prefs() -> dict:
    """Preferenze effettive. Catena: env > prefs.json > default.

    env: PII_EXCLUDE_TAGS="AGE,GENDER"   PII_MAPPING=0|1
    """
    data = _read_prefs_file()
    tags = data.get("excluded_tags")
    mapping = data.get("mapping_enabled", DEFAULT_MAPPING_ENABLED)
    if "PII_EXCLUDE_TAGS" in os.environ:
        tags = os.environ["PII_EXCLUDE_TAGS"]
    if "PII_MAPPING" in os.environ:
        mapping = os.environ["PII_MAPPING"]
    return {
        "excluded_tags": parse_tag_list(tags),
        "mapping_enabled": parse_bool(mapping, DEFAULT_MAPPING_ENABLED),
    }


def save_prefs(excluded_tags=None, mapping_enabled=None) -> dict:
    """Scrive prefs.json unendo le chiavi passate a quelle gia' sul file (i None non toccano nulla)."""
    data = _read_prefs_file()
    if excluded_tags is not None:
        data["excluded_tags"] = parse_tag_list(excluded_tags)
    if mapping_enabled is not None:
        data["mapping_enabled"] = parse_bool(mapping_enabled, DEFAULT_MAPPING_ENABLED)
    data.setdefault("excluded_tags", [])
    data.setdefault("mapping_enabled", DEFAULT_MAPPING_ENABLED)
    d = config_dir()
    d.mkdir(parents=True, exist_ok=True)
    (d / PREFS_FILE).write_text(json.dumps(data, indent=2), "utf-8")
    return data


def resolve(cli_host=None, cli_port=None):
    """Risolve host/porta con la catena: CLI > env > config.json > default.

    Ritorna (host: str, port: int).
    """
    cfg = load_config()
    host = cli_host or os.environ.get("PII_HOST") or cfg.get("host") or DEFAULT_HOST
    port = (cli_port if cli_port is not None else
            os.environ.get("PII_PORT") if "PII_PORT" in os.environ else
            cfg.get("port") if cfg.get("port") is not None else
            DEFAULT_PORT)
    return str(host), int(port)


def port_available(host: str, port: int) -> bool:
    """True se la porta e' libera.

    Il bind del solo host non basta: se un altro programma ascolta su 0.0.0.0:PORT,
    il bind su 127.0.0.1:PORT riesce comunque e da quel momento non e' definito chi
    serve le connessioni - l'utente puo' finire sul server dell'altro programma.
    Quindi si prova a legare anche l'indirizzo jolly; se e' occupato si guarda, con
    una connessione, se qualcuno risponde davvero su host:port (potrebbe ascoltare
    su un'altra interfaccia, e allora la nostra e' libera).
    """
    def _bind(addr, riusa):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                if riusa:
                    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                s.bind((addr, port))
                return True
        except OSError:
            return False

    if not _bind(host, True):
        return False
    # il jolly si prova SENZA SO_REUSEADDR: quando ce l'ha anche il socket dell'altro
    # programma (Werkzeug la imposta di default) su Windows il bind riesce lo stesso,
    # ed e' proprio il caso da rilevare.
    if _bind("", False):
        return True
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.3)
        return s.connect_ex((host, port)) != 0
