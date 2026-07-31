# -*- coding: utf-8 -*-
"""Mette src/app sul sys.path.

I moduli dell'app si importano fra loro in piatto (`import server_config`) perche'
PyInstaller li impacchetta tutti nella stessa directory: i test devono replicare
quel layout, non inventarne un altro.
"""

import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parents[1] / "src" / "app"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))
