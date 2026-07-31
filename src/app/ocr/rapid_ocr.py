# -*- coding: utf-8 -*-
"""Backend OCR su CPU: RapidOCR (PP-OCR portato su ONNXRuntime).

Nessuna GPU, nessun torch: ~25 MB di modelli ONNX e qualche centinaio di MB di
RAM. Si impacchetta dentro l'installer, quindi l'app resta offline anche al
primo uso di un PDF scansionato.

I modelli NON si scaricano a runtime: i path sono espliciti (vedi _models_dir),
altrimenti RapidOCR andrebbe in rete alla prima chiamata - inaccettabile in
un'app che promette 100% locale.
"""

import os
import sys
from pathlib import Path

from .layout import lines_to_text

# PP-OCRv6 usa un UNICO modello di riconoscimento multilingua (multi_PP-OCRv6_rec_*),
# valido per tutte le 52 lingue supportate: nessun modello per-lingua, nessun
# dizionario di caratteri separato. Tre file in tutto, ~30 MB.
MODEL_FILES = ("det.onnx", "cls.onnx", "rec.onnx")


def _models_dir() -> Path:
    """Dentro l'exe (PyInstaller) o in src/app/ocr_models/ in sviluppo."""
    base = getattr(sys, "_MEIPASS", None)
    if base:
        return Path(base) / "ocr_models"
    override = os.environ.get("PII_OCR_MODEL_DIR")
    if override:
        return Path(override)
    return Path(__file__).resolve().parent.parent / "ocr_models"


def ocr_result_to_text(res) -> str:
    """RapidOCROutput -> prosa. Funzione pura, testabile senza engine."""
    if res is None or getattr(res, "txts", None) is None or len(res.txts) == 0:
        return ""
    return lines_to_text(res.boxes, res.txts, getattr(res, "scores", None))


class RapidOcr:
    name = "rapidocr"

    def __init__(self):
        self._engine = None

    def models_present(self) -> bool:
        """I file ONNX impacchettati ci sono? Controllo puro, senza importare nulla."""
        d = _models_dir()
        return all((d / f).is_file() for f in MODEL_FILES)

    def available(self) -> bool:
        """Libreria installata E modelli presenti."""
        try:
            import rapidocr  # noqa: F401
        except ImportError:
            return False
        return self.models_present()

    def _load(self):
        if self._engine is not None:
            return
        from rapidocr import RapidOCR

        d = _models_dir()
        self._engine = RapidOCR(params={
            # path espliciti = nessuna risoluzione online, nessun download
            "Det.model_path": str(d / "det.onnx"),
            "Cls.model_path": str(d / "cls.onnx"),
            "Rec.model_path": str(d / "rec.onnx"),
            "Rec.lang_type": "it",         # valida la lingua; il modello v6 e' multilingua
            "Global.use_cls": True,        # raddrizza le righe capovolte
            "Det.limit_side_len": 1280,    # default 736: alzato, altrimenti il corpo
                                           # 8pt dei riferimenti catastali sparisce
        })

    def read_pages(self, images):
        """PNG in, un testo per pagina out. Stesso ordine, stessa lunghezza."""
        import cv2       # arriva con rapidocr (opencv-python-headless)
        import numpy as np

        self._load()
        out = []
        for png in images:
            img = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_COLOR)
            out.append(ocr_result_to_text(self._engine(img)) if img is not None else "")
        return out
