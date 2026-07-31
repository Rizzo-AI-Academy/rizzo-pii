# -*- coding: utf-8 -*-
"""Estrazione del testo da PDF: testo nativo dove c'e', OCR dove manca.

Le pagine con testo nativo NON passano dall'OCR: sono esatte e istantanee.
Solo le pagine scansionate (immagine pura) vengono rasterizzate e lette.

Le funzioni di decisione e ricomposizione sono pure: si testano senza PDF,
senza modelli e senza fitz.
"""

from ocr import get_backend

MIN_CHARS_PER_PAGE = 60   # sotto questa soglia la pagina si considera scansionata
OCR_DPI = 300             # sotto i 300 DPI i caratteri piccoli (CF, dati catastali) sfumano


def page_needs_ocr(native_text: str, min_chars: int = MIN_CHARS_PER_PAGE) -> bool:
    """True se il testo estratto da fitz e' troppo poco per essere il contenuto."""
    return len(native_text.strip()) < min_chars


def pages_needing_ocr(pages, min_chars: int = MIN_CHARS_PER_PAGE) -> list:
    """Indici delle pagine da passare all'OCR."""
    return [i for i, t in enumerate(pages) if page_needs_ocr(t, min_chars)]


def unreadable_pages(ocr_texts, min_chars: int = MIN_CHARS_PER_PAGE) -> list:
    """Indici delle pagine che, DOPO l'OCR, hanno prodotto quasi nulla.

    Misurato: su una scansione molto degradata l'OCR non sbaglia qualche carattere,
    collassa - restituisce poche righe sparse invece del testo. Il documento
    anonimizzato sembra a posto ma ha perso quasi tutto, e le PII non lette non
    sono mai passate sotto il naso ne' del modello ne' della rete regex.
    Va detto all'utente: qui l'anonimizzazione non e' affidabile.
    """
    return [i for i, t in enumerate(ocr_texts) if len(t.strip()) < min_chars]


def merge_ocr_pages(native, indexes, ocr_texts) -> list:
    """Rimpiazza le pagine `indexes` col testo OCR, preservando l'ordine.

    Tollera un backend che ritorni meno pagine del richiesto: le mancanti
    restano come stavano invece di far saltare tutto il documento.
    """
    out = list(native)
    for i, text in zip(indexes, ocr_texts):
        out[i] = text
    return out


def extract_text(pdf_bytes: bytes, use_ocr: bool = True):
    """Ritorna (testo, info).

    info: {"pages", "ocr_pages", "backend", "unreadable_pages"} - serve all'UI per
    dire quante pagine sono state lette via OCR, quali sono illeggibili, e per
    rilassare i checksum (vedi pii_regex).
    """
    import fitz  # import locale: i test delle funzioni pure non lo richiedono

    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        native = [p.get_text() for p in doc]
        todo = pages_needing_ocr(native)
        info = {"pages": len(native), "ocr_pages": 0, "backend": None,
                "unreadable_pages": []}

        if not todo or not use_ocr:
            return "\n".join(native), info

        backend = get_backend()
        if backend is None or not backend.available():
            # come prima: le pagine scansionate restano vuote. Ma ora lo sappiamo.
            info["backend"] = "unavailable"
            return "\n".join(native), info

        images = [doc[i].get_pixmap(dpi=OCR_DPI).tobytes("png") for i in todo]

    # doc chiuso: l'OCR (lento) non tiene aperto un handle sul PDF
    texts = backend.read_pages(images)
    info.update(ocr_pages=len(todo), backend=backend.name,
                unreadable_pages=[todo[i] for i in unreadable_pages(texts)])
    return "\n".join(merge_ocr_pages(native, todo, texts)), info
