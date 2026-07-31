# -*- coding: utf-8 -*-
"""Backend OCR per i PDF scansionati.

L'OCR e' opzionale: se il backend non e' installato o i modelli non ci sono,
l'app si comporta come prima (le pagine senza testo nativo restano vuote), ma
lo dice invece di fallire in silenzio.

Selezione con env PII_OCR:  auto (default) | off
"""

import os


def get_backend():
    """Ritorna il backend OCR configurato, o None se disattivato."""
    if os.environ.get("PII_OCR", "auto").strip().lower() == "off":
        return None
    from .rapid_ocr import RapidOcr
    return RapidOcr()
