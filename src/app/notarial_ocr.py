# -*- coding: utf-8 -*-
"""
OCR delle scansioni (PDF fatti di immagini, foto, .png/.jpg/.tiff) per Rizzo PII.

Motore: `docmd` (src/app/docmd/), pipeline PaddleOCR su ONNX Runtime, solo CPU:
  PP-DocLayout-M (layout) -> PP-OCRv6 det + rec (CTC) -> SLANet_plus (tabelle).
Niente Paddle e niente PyTorch: bastano onnxruntime + opencv + pyclipper + pyyaml.

Cosa fa questo modulo, e cosa NO:
  - decide quali pagine di un PDF vanno lette con l'OCR (`mode`):
        auto  -> solo le pagine senza layer di testo (scansioni)   [default]
        force -> tutte le pagine (layer di testo assente o inaffidabile)
        off   -> nessuna (comportamento di prima: solo il testo nativo)
  - converte le immagini caricate in un PDF di una pagina per fotogramma, cosi' da
    lì in poi immagini e PDF seguono la stessa strada (anteprima, analisi, redazione);
  - esegue l'OCR e ne tiene il risultato in una piccola cache LRU in memoria, per
    hash del file: /preview, /ocr, /analyze e /pdf sullo stesso documento pagano
    l'OCR UNA volta sola;
  - produce, per ogni pagina letta, il testo (Markdown) e le righe con i box di ogni
    carattere, gia' riportati in PUNTI PDF della pagina originale (anche quando la
    pagina e' stata raddrizzata): `pdf_export.redact_pdf` li usa per coprire i
    pixel delle PII senza sapere nulla del modello.
  - NON tocca il PDF: la redazione resta in pdf_export.py.

Modelli: cartella con una sottocartella per modello (inference.onnx + inference.yml).
Ricerca, in ordine: env PII_OCR_MODELS > `ocr_models` impacchettato (PyInstaller) >
models/ocr/ nella repo. Vedi scripts/fetch_ocr_models.py.
"""

from __future__ import annotations

import hashlib
import os
import sys
import threading
import time
from collections import OrderedDict
from pathlib import Path

import fitz  # PyMuPDF

PRESETS = ("fast", "balanced", "accurate")
MODES = ("auto", "force", "off")
DEFAULT_PRESET = "balanced"
DEFAULT_MODE = "auto"

# Una pagina con meno caratteri alfanumerici di cosi' nel layer di testo e' una
# scansione (o una pagina vuota, che l'OCR attraversa in pochi millisecondi).
MIN_NATIVE_ALNUM = 20

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp", ".gif",
              ".jfif", ".pnm", ".pbm", ".pgm", ".ppm"}
_IMAGE_MAGIC = (b"\x89PNG", b"\xff\xd8\xff", b"II*\x00", b"MM\x00*", b"BM", b"GIF8")

_CACHE_DOCS = 1              # solo il documento attivo, mai una cronologia
_CACHE_MAX_BYTES = 64 * 1024 * 1024
import ocr as _standard_ocr
_lock = _standard_ocr._lock  # one inference lock across both modes
_cache_lock = threading.Lock()
_pipelines = _standard_ocr._pipelines  # share weights, never document results
_cache: "OrderedDict[str, dict]" = OrderedDict()   # chiave doc -> {page: risultato}
_img_pdf: "OrderedDict[str, bytes]" = OrderedDict()  # hash immagine -> PDF convertito
_cache_generation = 0       # impedisce a un lavoro annullato di ripopolare la cache


def _content_bytes(value, seen=None):
    """Budget RAM dei risultati serializzabili, inclusi box e testo OCR."""
    seen = set() if seen is None else seen
    ident = id(value)
    if ident in seen:
        return 0
    seen.add(ident)
    size = sys.getsizeof(value)
    if isinstance(value, dict):
        size += sum(_content_bytes(k, seen) + _content_bytes(v, seen)
                    for k, v in value.items())
    elif isinstance(value, (list, tuple)):
        size += sum(_content_bytes(v, seen) for v in value)
    return size


def _trim_cache():
    """Chiamata con _cache_lock acquisito; un limite in documenti non basta."""
    while len(_cache) > _CACHE_DOCS:
        _cache.popitem(last=False)
    while len(_img_pdf) > _CACHE_DOCS:
        _img_pdf.popitem(last=False)
    while _cache and _content_bytes(_cache) > _CACHE_MAX_BYTES:
        _cache.popitem(last=False)
    while _img_pdf and sum(len(v) for v in _img_pdf.values()) > _CACHE_MAX_BYTES:
        _img_pdf.popitem(last=False)


def clear_cache(pdf_bytes=None):
    """Rilascia tutti i contenuti, o quelli del documento indicato (solo RAM).

    Le sessioni dei modelli restano riusabili: non contengono documenti. L'app
    chiama questa funzione anche su rimozione, cambio documento e timeout.
    """
    global _cache_generation
    with _cache_lock:
        _cache_generation += 1
        if pdf_bytes is None:
            _cache.clear()
            _img_pdf.clear()
            return
        key = sha(pdf_bytes)
        for cached in list(_cache):
            if cached.startswith(key + ":"):
                del _cache[cached]
        for cached, data in list(_img_pdf.items()):
            if cached == key or data == pdf_bytes:
                del _img_pdf[cached]


# --------------------------------------------------------------------------- #
# Modelli
# --------------------------------------------------------------------------- #
_REQUIRED = {
    "fast": ("PP-DocLayout-M", "PP-OCRv6_tiny_det", "PP-OCRv6_tiny_rec"),
    "balanced": ("PP-DocLayout-M", "PP-OCRv6_small_det", "PP-OCRv6_small_rec"),
    "accurate": ("PP-DocLayout_plus-L", "PP-OCRv6_small_det", "PP-OCRv6_small_rec"),
}


def models_dir() -> Path:
    env = os.environ.get("PII_OCR_MODELS")
    if env:
        return Path(env)
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        return Path(meipass) / "ocr_models"
    return Path(__file__).resolve().parents[2] / "models" / "ocr"


def _has(name: str) -> bool:
    return (models_dir() / name / "inference.onnx").is_file()


def available_presets() -> list[str]:
    """Preset utilizzabili con i modelli presenti. Senza PP-DocLayout-M il preset
    "balanced" ripiega sul layout accurato, se c'e' (lo fa _resolve_models)."""
    out = []
    for p in PRESETS:
        try:
            _resolve_models(p)
            out.append(p)
        except FileNotFoundError:
            pass
    return out


def _resolve_models(preset: str) -> dict:
    layout, det, rec = _REQUIRED[preset]
    # Senza il layout veloce (es. build con i soli modelli ufficiali) fast/balanced usano
    # quello accurato. Il contrario no: "accurate" senza il suo layout non esiste.
    if layout == "PP-DocLayout-M" and not _has(layout) and _has("PP-DocLayout_plus-L"):
        layout = "PP-DocLayout_plus-L"
    missing = [m for m in (layout, det, rec) if not _has(m)]
    if missing:
        raise FileNotFoundError(", ".join(missing))
    return {"layout": layout, "det": det, "rec": rec}


def normalize_preset(preset) -> str:
    p = str(preset or "").strip().lower()
    avail = available_presets()
    if p in avail:
        return p
    if DEFAULT_PRESET in avail:
        return DEFAULT_PRESET
    return avail[0] if avail else DEFAULT_PRESET


def normalize_mode(mode) -> str:
    m = str(mode or "").strip().lower()
    if m in ("0", "false", "no", "none"):
        return "off"
    if m in ("1", "true", "yes", "si", "sì", "always", "all"):
        return "force"
    return m if m in MODES else DEFAULT_MODE


def status() -> dict:
    avail = available_presets()
    return {
        "available": bool(avail),
        "presets": avail,
        "loaded": sorted(_pipelines),
        "models_dir": str(models_dir()),
        "engine": "PaddleOCR PP-OCRv6 + PP-DocLayout (ONNX Runtime, CPU)",
    }


def get_pipeline(preset: str = DEFAULT_PRESET):
    """DocPipeline del preset, caricata alla prima richiesta (~1 s) e poi riusata."""
    preset = normalize_preset(preset)
    with _cache_lock:
        pipe = _pipelines.get(preset)
    if pipe is not None:
        return pipe
    from docmd import DocPipeline      # import pigro: cv2/onnxruntime solo se serve l'OCR
    m = _resolve_models(preset)
    with _lock:                        # due richieste insieme non caricano due volte
        with _cache_lock:
            pipe = _pipelines.get(preset)
        if pipe is not None:
            return pipe
        return _load(DocPipeline, preset, m)


def _load(DocPipeline, preset, m):
    t = time.perf_counter()
    pipe = DocPipeline(
        models_dir(), preset=preset,
        layout_model=m["layout"], det_model=m["det"], rec_model=m["rec"],
        # in un anonimizzatore anche intestazioni, pie' di pagina, timbri e loghi
        # contano: una ragione sociale nell'intestazione e' una PII come le altre
        keep_page_furniture=True, read_figure_text=True, figure_markers=False,
        # righe incerte: meglio leggerle (e magari anonimizzarle) che perderle
        min_rec_score=0.4,
    )
    print(f"[ocr] preset {preset} caricato in {time.perf_counter() - t:.2f}s "
          f"({m['layout']}, {m['det']}, {m['rec']})")
    with _cache_lock:
        _pipelines[preset] = pipe
    return pipe


# --------------------------------------------------------------------------- #
# Immagini -> PDF, pagine da leggere, risoluzione
# --------------------------------------------------------------------------- #
def is_image(name: str, data: bytes) -> bool:
    ext = os.path.splitext((name or "").lower())[1]
    if ext in IMAGE_EXTS:
        return True
    return any(data.startswith(m) for m in _IMAGE_MAGIC) or (
        data[:4] == b"RIFF" and data[8:12] == b"WEBP")


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def image_to_pdf(data: bytes, name: str = "") -> bytes:
    """Immagine (anche TIFF multipagina) -> PDF, una pagina per fotogramma, alla
    risoluzione dichiarata dal file. Il risultato e' in cache per hash: la stessa
    immagine produce sempre lo STESSO PDF (stesso hash -> la cache OCR regge)."""
    key = sha(data)
    with _cache_lock:
        generation = _cache_generation
        if key in _img_pdf:
            _img_pdf.move_to_end(key)
            return _img_pdf[key]
    ext = os.path.splitext((name or "").lower())[1].lstrip(".") or None
    try:
        src = fitz.open(stream=data, filetype=ext)
    except Exception:
        src = fitz.open(stream=data)
    try:
        if src.page_count == 0:
            raise ValueError("immagine vuota")
        pdf = src.convert_to_pdf()
    finally:
        src.close()
    with _cache_lock:
        if generation == _cache_generation:
            _img_pdf[key] = pdf
            _trim_cache()
    return pdf


def _alnum(s: str) -> int:
    return sum(ch.isalnum() for ch in s or "")


def page_needs_ocr(page) -> bool:
    """Pagina senza un vero layer di testo: scansione, foto, testo convertito in curve."""
    try:
        return _alnum(page.get_text()) < MIN_NATIVE_ALNUM
    except Exception:
        return True


def pages_to_ocr(doc, mode: str) -> list[int]:
    mode = normalize_mode(mode)
    if mode == "off":
        return []
    if mode == "force":
        return list(range(doc.page_count))
    return [i for i in range(doc.page_count) if page_needs_ocr(doc[i])]


def _page_dpi(page) -> int:
    """Risoluzione di rendering per l'OCR: quella nativa dell'immagine che copre la
    pagina (una scansione a 300 dpi si legge a 250, una foto a 96 dpi resta a 100),
    200 dpi se la pagina non ha un'immagine dominante."""
    try:
        area = abs(page.rect)
        best = None
        for info in page.get_image_info():
            r = fitz.Rect(info["bbox"]) & page.rect
            if r.is_empty or abs(r) < 0.4 * area:
                continue
            native = info["width"] / max(fitz.Rect(info["bbox"]).width, 1) * 72
            if best is None or native > best:
                best = native
        if best:
            return int(max(100, min(250, round(best))))
    except Exception:
        pass
    return 200


def _render(page, dpi: int):
    import numpy as np
    pix = page.get_pixmap(dpi=dpi, alpha=False, colorspace=fitz.csRGB)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
    return img[:, :, ::-1].copy()      # RGB -> BGR (convenzione OpenCV di docmd)


# --------------------------------------------------------------------------- #
# OCR di una pagina -> dict serializzabile, coordinate in punti PDF
# --------------------------------------------------------------------------- #
def _affine_to_page(img_shape, angle: float, dpi: int, rotated_shape):
    """Matrice 2x3: coordinate dell'immagine letta dall'OCR (eventualmente raddrizzata
    su una tela piu' grande) -> punti della pagina PDF NON ruotata."""
    import cv2
    import numpy as np
    h, w = img_shape[:2]
    s = 72.0 / dpi
    if not angle:
        return [s, 0.0, 0.0, 0.0, s, 0.0]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)     # stessa di docmd.rotate
    nh, nw = rotated_shape[:2]
    m[0, 2] += nw / 2 - w / 2
    m[1, 2] += nh / 2 - h / 2
    inv = cv2.invertAffineTransform(m)                           # raddrizzata -> originale
    a = np.vstack([inv, [0, 0, 1]])
    a = np.diag([s, s, 1.0]) @ a
    return [float(x) for x in a[:2].ravel()]


def _apply(aff, box):
    """Box [x1,y1,x2,y2] -> bbox in punti dei 4 angoli trasformati."""
    a, b, c, d, e, f = aff
    xs, ys = [], []
    for x, y in ((box[0], box[1]), (box[2], box[1]), (box[2], box[3]), (box[0], box[3])):
        xs.append(a * x + b * y + c)
        ys.append(d * x + e * y + f)
    return [round(min(xs), 2), round(min(ys), 2), round(max(xs), 2), round(max(ys), 2)]


def ocr_page(page, preset: str, page_no: int = 0) -> dict:
    """Legge una pagina. Ritorna:
        {"page", "dpi", "angle", "text" (Markdown), "timings",
         "lines": [{"text", "score", "box", "chars": [box per carattere] | None}],
         "seals": [box dei timbri], "size": [w, h] in punti}
    Tutti i box sono in PUNTI della pagina (sistema di page.rect, senza rotazione)."""
    pipe = get_pipeline(preset)
    dpi = _page_dpi(page)
    img = _render(page, dpi)
    with _lock:
        res = pipe.process_page(img, page_no)
    aff = _affine_to_page(img.shape, res.angle, dpi, res.image.shape)
    lines = []
    for ln in res.lines:
        lines.append({
            "text": ln.text,
            "score": round(float(ln.score), 3),
            "box": _apply(aff, ln.box),
            "chars": [_apply(aff, b) for b in ln.char_boxes] if ln.char_boxes else None,
        })
    seals = [_apply(aff, b.box) for b in res.blocks if b.label == "seal"]
    return {
        "page": page_no,
        "dpi": dpi,
        "angle": round(float(res.angle), 3),
        "text": res.markdown.strip(),
        "lines": lines,
        "seals": seals,
        "size": [round(page.rect.width, 2), round(page.rect.height, 2)],
        "timings": {k: round(v, 3) for k, v in res.timings.items()},
    }


# --------------------------------------------------------------------------- #
# Documento intero, con cache
# --------------------------------------------------------------------------- #
def _cache_get(key: str, page: int):
    with _cache_lock:
        d = _cache.get(key)
        if d is None:
            return None
        _cache.move_to_end(key)
        return d.get(page)


def _cache_put(key: str, page: int, value: dict, generation=None):
    with _cache_lock:
        if generation is not None and generation != _cache_generation:
            return
        d = _cache.setdefault(key, {})
        d[page] = value
        _cache.move_to_end(key)
        _trim_cache()


def cached_pages(pdf_bytes: bytes, preset=None) -> dict:
    """Pagine gia' lette (nessun OCR nuovo): {pagina: risultato}."""
    key = f"{sha(pdf_bytes)}:{normalize_preset(preset)}"
    with _cache_lock:
        return dict(_cache.get(key) or {})


def ocr_pdf(pdf_bytes: bytes, mode: str = DEFAULT_MODE, preset=None, pages=None,
            progress=None) -> dict:
    """OCR delle pagine che lo richiedono (`mode`), con cache per hash del file.

    pages:    sottoinsieme di pagine 0-based da considerare (None = tutte)
    progress: callback(evento: dict) chiamata dopo ogni pagina letta
    Ritorna {"n_pages", "ocr_pages": [...], "pages": {i: risultato}, "elapsed",
             "preset", "mode", "available"}."""
    mode = normalize_mode(mode)
    preset = normalize_preset(preset)
    with _cache_lock:
        generation = _cache_generation
    key = f"{sha(pdf_bytes)}:{preset}"
    t0 = time.perf_counter()
    out = {"pages": {}, "ocr_pages": [], "mode": mode, "preset": preset,
           "available": bool(available_presets())}
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        if doc.needs_pass:
            raise ValueError("PDF protetto da password: rimuovi la protezione e riprova.")
        out["n_pages"] = doc.page_count
        todo = pages_to_ocr(doc, mode)
        if pages is not None:
            wanted = set(pages)
            todo = [i for i in todo if i in wanted]
        out["ocr_pages"] = todo
        if todo and not out["available"]:
            out["elapsed"] = 0.0
            return out
        for k, i in enumerate(todo):
            res = _cache_get(key, i)
            cached = res is not None
            if res is None:
                res = ocr_page(doc[i], preset, i)
                _cache_put(key, i, res, generation)
            out["pages"][i] = res
            if progress:
                progress({"event": "page", "page": i + 1, "done": k + 1, "total": len(todo),
                          "cached": cached, "seconds": res["timings"].get("total", 0.0)})
    out["elapsed"] = round(time.perf_counter() - t0, 3)
    return out


def page_texts(pdf_bytes: bytes, ocr_result: dict) -> list[dict]:
    """Testo pagina per pagina: OCR dove c'e', altrimenti il layer nativo."""
    pages = []
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        for i, page in enumerate(doc):
            r = ocr_result.get("pages", {}).get(i)
            if r is not None:
                pages.append({"page": i + 1, "source": "ocr", "text": r["text"]})
            else:
                pages.append({"page": i + 1, "source": "text",
                              "text": (page.get_text() or "").strip()})
    return pages
