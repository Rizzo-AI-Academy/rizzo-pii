# -*- coding: utf-8 -*-
"""Popola src/app/ocr_models/ con i modelli ONNX che l'app impacchetta.

Si esegue UNA VOLTA in fase di sviluppo/build, mai a runtime:

    cd src/app && python -m ocr.fetch_models

Strategia: si istanzia RapidOCR una volta lasciandogli scaricare i modelli nella
sua cartella (site-packages/rapidocr/models/), poi si copiano da li' con nomi
stabili. Nessun URL hardcoded: la fonte resta quella ufficiale della libreria
anche quando cambia versione.

PP-OCRv6 usa un unico modello di riconoscimento MULTILINGUA (52 lingue, italiano
incluso): non serve scegliere una lingua ne' un dizionario di caratteri.
"""

import shutil
import sys
from pathlib import Path

DEST = Path(__file__).resolve().parent.parent / "ocr_models"

# nome di destinazione -> sottostringhe che identificano il file scaricato
WANTED = {
    "det.onnx": ("det",),
    "cls.onnx": ("cls",),
    "rec.onnx": ("rec",),
}


def main():
    try:
        from rapidocr import RapidOCR
    except ImportError:
        sys.exit("rapidocr non installato:  pip install rapidocr onnxruntime")

    import rapidocr

    # istanziazione = risoluzione + download dei modelli nella cartella della libreria
    RapidOCR(params={"Rec.lang_type": "it"})

    cache = Path(rapidocr.__file__).parent / "models"
    onnx = list(cache.glob("*.onnx"))
    if not onnx:
        sys.exit(f"Nessun modello trovato in {cache}. "
                 f"Copiare i file a mano (vedi src/app/ocr_models/README.md)")

    DEST.mkdir(parents=True, exist_ok=True)
    for dest_name, needles in WANTED.items():
        matches = [p for p in onnx if all(n in p.name.lower() for n in needles)]
        if not matches:
            sys.exit(f"Modello '{dest_name}' non trovato in {cache} "
                     f"(presenti: {[p.name for p in onnx]})")
        # a parita' di match si prende il piu' recente: e' quello appena risolto
        src = max(matches, key=lambda p: p.stat().st_mtime)
        shutil.copy2(src, DEST / dest_name)
        print(f"  {dest_name:10s} <- {src.name}  "
              f"({(DEST / dest_name).stat().st_size / 1e6:.1f} MB)")

    print(f"\nModelli pronti in {DEST}")


if __name__ == "__main__":
    main()
