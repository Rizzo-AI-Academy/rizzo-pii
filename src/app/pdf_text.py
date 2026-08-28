# -*- coding: utf-8 -*-
"""
Estrazione testo dai PDF con fallback OCR (Tesseract) e crop verticale.

Il modulo lavora SOLO su bytes + PyMuPDF (+ pytesseract/Pillow, caricati solo se
servono): nessuna dipendenza da torch/transformers, quindi e' verificabile in
isolamento come pdf_export.py e detectors.py.

Due funzionalita' nuove rispetto all'estrazione che usava solo `page.get_text()`:

  * CROP VERTICALE  (crop_top / crop_bottom, in percentuale della pagina):
    esclude dall'alto e dal basso di ogni pagina una fascia. Sui PDF con testo
    nativo il testo i cui blocchi cadono nelle fasce viene scartato dal parsing
    (un blocco e' escluso se il suo rettangolo interseca la fascia: per un
    anonimizzatore tagliare in piu' e' la direzione sicura). Sui PDF
    scannerizzati la stessa fascia viene ritagliata dall'IMMAGINE prima del
    passaggio a Tesseract, cosi' l'OCR non vede affatto quelle zone.

  * OCR DI FALLBACK (Tesseract): quando una pagina ha un layer testuale vuoto o
    quasi vuoto (pagina scannerizzata, nessun testo nativo), la pagina viene
    renderizzata (PyMuPDF, 300 dpi), ritagliata secondo il crop, e il testo lo
    produce Tesseract. I PDF MISTI (alcune pagine native, altre scannerizzate)
    funzionano pagina per pagina: si estrae in modo nativo dove possibile, si fa
    OCR solo sulle pagine che ne hanno bisogno.

Il testo che esce (nativo + OCR) finisce nello stesso flusso di anonimizzazione
gia' usato per i PDF con testo nativo: la pipeline di chunking con overlap non
cambia.

Tesseract e' un binario di SISTEMA, non una libreria Python: pytesseract lo
trova nel PATH. Se non c'e', o se la lingua richiesta manca, il modulo solleva
OcrUnavailableError (sottoclasse di ValueError) con un messaggio chiaro: i
route lo trasformano in un errore JSON visibile nell'UI, mai un crash silenzioso.
Il path del binario si puo' forzare con la variabile d'ambiente TESSERACT_CMD.
"""

import os
import re

import fitz  # PyMuPDF


class OcrUnavailableError(ValueError):
    """L'OCR servirebbe ma Tesseract (o Pillow) non e' disponibile/mal configurato."""


# --------------------------------------------------------------------------- #
# Costanti
# --------------------------------------------------------------------------- #
# Sotto questa lunghezza (spazi esclusi) il layer testuale di una pagina e'
# considerato "quasi vuoto": niente testo nativo di rilievo -> si fa OCR.
# La soglia si valuta sul testo NON filtrato dal crop: una pagina che l'utente
# svuota col crop (es. solo intestazione) non deve riattivare l'OCR.
OCR_MIN_CHARS = 40

# Risoluzione del rendering pagina->immagine per Tesseract. 300 dpi e' il
# minimo consigliato per la qualita' OCR su documenti stampati.
OCR_DPI = 300

# Lingua Tesseract. Richiede il pacchetto di lingua installato a sistema
# (su Windows: eseguibile UB Mannheim con "ita" selezionato all'installazione).
OCR_LANG_DEFAULT = "ita"


def parse_crop(value, default=0.0):
    """Percentuale 0..100 da una stringa/numero; default se manca o non valido."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return float(default)
    return max(0.0, min(100.0, v))


def _frac(percent):
    return parse_crop(percent) / 100.0


def _set_tesseract_cmd(pytesseract):
    """Forza il path del binario se impostato via TESSERACT_CMD (altrimenti
    pytesseract lo cerca nel PATH). Utile col packaging Tauri/PyInstaller."""
    cmd = os.environ.get("TESSERACT_CMD")
    if cmd:
        pytesseract.pytesseract.tesseract_cmd = cmd


# --------------------------------------------------------------------------- #
# Crop su testo nativo
# --------------------------------------------------------------------------- #
def _page_native_text(page, top_frac, bottom_frac):
    """Testo nativo della pagina, esclusi i blocchi che cadono nelle fasce
    tagliate. Con crop 0 equivale a page.get_text() (comportamento invariato)."""
    if top_frac <= 0 and bottom_frac <= 0:
        return page.get_text()
    w, h = page.rect.width, page.rect.height
    cuts = []
    if top_frac > 0:
        cuts.append(fitz.Rect(0, 0, w, h * top_frac))
    if bottom_frac > 0:
        cuts.append(fitz.Rect(0, h * (1 - bottom_frac), w, h))
    parts = []
    for block in page.get_text("blocks"):
        if block[6] != 0:                      # 0 = blocco di testo, 1 = immagine
            continue
        r = fitz.Rect(block[:4])
        # NOTA: `Rect.intersects()` e' un predicato che NON muta `r`; usare
        # `r.intersect(c)` qui svuoterebbe `r` al primo taglio senza sovrapposizione
        # e il secondo confronto non escluderebbe piu' nulla.
        if any(r.intersects(c) for c in cuts):
            continue
        parts.append(block[4])
    return "".join(parts)


# --------------------------------------------------------------------------- #
# OCR (Tesseract) su una singola pagina
# --------------------------------------------------------------------------- #
def _ocr_page(page, top_frac, bottom_frac, lang, dpi=OCR_DPI):
    """Renderizza la pagina, ritaglia le fasce, passa l'immagine a Tesseract.

    Solleva OcrUnavailableError se pytesseract/Pillow mancano, se il binario
    tesseract non e' trovato o se fallisce (es. lingua non installata)."""
    try:
        import pytesseract
        from PIL import Image
    except ImportError as e:
        raise OcrUnavailableError(
            "L'OCR dei PDF scannerizzati non e' disponibile: manca la libreria "
            "Python pytesseract/Pillow. Installa le dipendenze di requirements.txt."
        ) from e
    _set_tesseract_cmd(pytesseract)

    try:
        pix = page.get_pixmap(dpi=dpi, alpha=False)
        mode = "RGBA" if pix.n == 4 else "RGB"
        img = Image.frombytes(mode, (pix.width, pix.height), pix.samples)
    except Exception as e:
        raise OcrUnavailableError(
            f"Impossibile renderizzare la pagina per l'OCR: {e}"
        ) from e

    if top_frac > 0 or bottom_frac > 0:
        w, h = img.size
        top = max(0, min(int(round(h * top_frac)), h))
        bottom = max(top, min(int(round(h * (1 - bottom_frac))), h))
        if bottom - top <= 0:
            return ""                           # pagina completamente tagliata
        img = img.crop((0, top, w, bottom))

    try:
        return pytesseract.image_to_string(img, lang=lang) or ""
    except pytesseract.TesseractNotFoundError as e:
        raise OcrUnavailableError(
            "Tesseract non trovato: l'OCR dei PDF scannerizzati richiede il "
            "binario tesseract-ocr installato a livello di sistema (su Windows: "
            "eseguibile UB Mannheim o equivalente, con il pacchetto di lingua "
            "'ita'; vedi docs/BUILD.md). Se non e' nel PATH, imposta la "
            "variabile TESSERACT_CMD col percorso del binario."
        ) from e
    except pytesseract.TesseractError as e:
        raise OcrUnavailableError(
            f"Tesseract ha restituito un errore (la lingua '{lang}' e' "
            f"installata?): {e}"
        ) from e


# --------------------------------------------------------------------------- #
# API pubblica
# --------------------------------------------------------------------------- #
def extract_pdf_text(data, crop_top=0, crop_bottom=0, ocr_lang=OCR_LANG_DEFAULT):
    """Testo estratto da un PDF (bytes). Ritorna (text, info).

    crop_top / crop_bottom: percentuali (0..100) tagliate da ogni pagina, in
    alto e in basso. 0 = nessun crop (comportamento di default invariato).

    info = {
        "pages":        numero di pagine,
        "ocr_pages":    pagine in cui si e' usato l'OCR,
        "page_methods": ["native" | "ocr", ...] per ogni pagina,
    }

    Solleva ValueError su PDF non valido/protetto e OcrUnavailableError quando
    serve l'OCR ma Tesseract non e' disponibile."""
    top_frac, bottom_frac = _frac(crop_top), _frac(crop_bottom)

    try:
        doc = fitz.open(stream=data, filetype="pdf")
    except Exception:
        raise ValueError("File PDF non valido o danneggiato.")
    if doc.needs_pass:
        doc.close()
        raise ValueError("PDF protetto da password: rimuovi la protezione e riprova.")

    methods, texts = [], []
    for page in doc:
        raw = page.get_text()
        if len(raw.strip()) >= OCR_MIN_CHARS:
            texts.append(_page_native_text(page, top_frac, bottom_frac))
            methods.append("native")
        else:
            texts.append(_ocr_page(page, top_frac, bottom_frac, ocr_lang))
            methods.append("ocr")

    doc.close()
    info = {
        "pages": len(texts),
        "ocr_pages": methods.count("ocr"),
        "page_methods": methods,
    }
    return "\n".join(texts), info