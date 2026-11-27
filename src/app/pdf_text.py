# -*- coding: utf-8 -*-
"""Testo leggibile di un PDF: content stream + annotazioni + campi modulo + TOC.

Come `detectors` e `pdf_export`, questo modulo lavora solo su oggetti duck-typed:
nessun import di torch/transformers/fitz, quindi e' testabile in isolamento.
`app._text_from_bytes` e `pdf_export._readable_text` lo usano sullo stesso
documento aperto da PyMuPDF.

Nei PDF fillable (AcroForm) il valore sta nel widget, non nel content stream:
`page.get_text()` torna vuoto e /analyze rispondeva 400 (issue #85).
"""

# PyMuPDF: fitz.PDF_ANNOT_REDACT. Le annotazioni di redazione non sono testo
# del documento, e in verifica residui andrebbero contate due volte.
REDACT_ANNOT = 12


def _native_boxes(page, text):
    """Allinea in ordine i glifi al testo, senza cercare valori ripetuti.

    Sono ammesse solo differenze di whitespace fra le due viste PyMuPDF.
    Ogni altra differenza rende la localizzazione non verificabile.
    """
    raw_chars, raw_boxes = [], []
    try:
        for block in page.get_text("rawdict").get("blocks", []):
            if block.get("type") != 0:
                continue
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    for char in span.get("chars", []):
                        for c in char.get("c") or "":
                            raw_chars.append(c)
                            raw_boxes.append(list(char["bbox"]))
                raw_chars.append("\n")
                raw_boxes.append(None)
    except Exception:
        return None
    boxes, j = [], 0
    for char in text:
        if char.isspace():
            boxes.append(None)
            if j < len(raw_chars) and raw_chars[j] == char:
                j += 1
            continue
        while j < len(raw_chars) and raw_chars[j].isspace():
            j += 1
        if j == len(raw_chars) or raw_chars[j] != char:
            return None
        boxes.append(raw_boxes[j])
        j += 1
    if any(not c.isspace() for c in raw_chars[j:]):
        return None
    return boxes


def _ocr_text_boxes(result):
    text, boxes, valid = [], [], True
    for line in result.get("lines") or ():
        value = line.get("text") or ""
        if not value:
            continue
        chars = line.get("chars") or []
        if len(chars) != len(value):
            valid = False
            chars = [None] * len(value)
        for i, (char, box) in enumerate(zip(value, chars)):
            rect = list(box) if box is not None else None
            if rect is not None and line.get("box"):
                if i == 0:
                    rect[0] = min(rect[0], line["box"][0])
                if i == len(value) - 1:
                    rect[2] = max(rect[2], line["box"][2])
            text.append(char)
            boxes.append(rect)
        text.append("\n")
        boxes.append(None)
    return "".join(text), boxes if valid else None


def collect_readable_document(doc, page_text=None, ocr_pages=None):
    """Snapshot canonico con locator per OGNI stream leggibile del documento.

    Offset Unicode Python [start,end); separatori fra stream senza geometria.
    L'OCR usa le righe originali, non la loro rappresentazione Markdown.
    Errori di lettura/localizzazione vengono conservati, mai interpretati come
    assenza di dati. Questa struttura vive soltanto nella sessione in memoria.
    """
    parts, sources, errors, unverified = [], [], [], []
    position = 0

    def add(value, **locator):
        nonlocal position
        if parts:
            position += 1
        value = str(value or "")
        sources.append({"start": position, "end": position + len(value), **locator})
        parts.append(value)
        position += len(value)

    page_text, ocr_pages = page_text or {}, ocr_pages or {}
    for i, page in enumerate(doc):
        if i in ocr_pages:
            text, boxes = _ocr_text_boxes(ocr_pages[i])
            origin = "ocr"
            unverified.append({"code": "raster_content", "page": i})
        else:
            text = page_text.get(i, page.get_text() or "")
            boxes = _native_boxes(page, text)
            origin = "native"
            try:
                if page.get_images(full=True):
                    unverified.append({"code": "images", "page": i})
            except Exception:
                unverified.append({"code": "images_unchecked", "page": i})
        add(text, kind="page", page=i, id=i, field="text", origin=origin,
            boxes=boxes, mapping_error=boxes is None)
        try:
            for n, annot in enumerate(page.annots() or ()):
                if annot.type[0] == REDACT_ANNOT:
                    continue
                info = annot.info or {}
                for field in ("content", "subject", "title"):
                    add(info.get(field), kind="annotation", page=i,
                        id=getattr(annot, "xref", n), index=n, field=field,
                        annotation_type=annot.type[0],
                        rect=list(annot.rect) if getattr(annot, "rect", None) is not None else None)
        except Exception:
            errors.append({"code": "annotation_read", "page": i})
        try:
            for n, widget in enumerate(page.widgets() or ()):
                if isinstance(widget.field_value, str) and widget.field_value:
                    add(widget.field_value, kind="widget", page=i,
                        id=getattr(widget, "xref", n), index=n, field="field_value",
                        rect=list(widget.rect) if getattr(widget, "rect", None) is not None else None)
        except Exception:
            errors.append({"code": "widget_read", "page": i})
    try:
        for n, entry in enumerate(doc.get_toc(simple=True) or ()):
            add(entry[1], kind="toc", page=None, id=n, field="title")
    except Exception:
        errors.append({"code": "toc_read"})
    try:
        if doc.embfile_names():
            unverified.append({"code": "attachments"})
    except Exception:
        unverified.append({"code": "attachments_unchecked"})
    return {"text": "\n".join(parts), "sources": sources,
            "errors": errors, "unverified": unverified}


def collect_readable_text(doc, redact_annot_type=REDACT_ANNOT, page_text=None):
    """TUTTO il testo leggibile: pagine + annotazioni + campi modulo + segnalibri.

    `doc` e' un iterabile di pagine con `get_text()`, `annots()`, `widgets()`
    e (sul documento) `get_toc(simple=True)`: il contratto di PyMuPDF, senza
    importarlo. Da solo non vede il testo dentro immagini raster: per le pagine
    scansionate il chiamante passa `page_text` = {indice pagina: testo OCR}, che
    prende il posto del layer di testo di quella pagina.
    """
    parts = []
    page_text = page_text or {}
    for i, page in enumerate(doc):
        override = page_text.get(i)
        parts.append(override if override is not None else (page.get_text() or ""))
        try:
            for a in page.annots() or ():
                if a.type[0] == redact_annot_type:
                    continue
                info = a.info or {}
                parts.extend(str(info.get(k) or "") for k in ("content", "subject", "title"))
        except Exception:
            pass
        try:
            for w in page.widgets() or ():
                val = w.field_value
                if isinstance(val, str) and val:
                    parts.append(val)
        except Exception:
            pass
    try:
        parts.extend(str(e[1] or "") for e in (doc.get_toc(simple=True) or ()))
    except Exception:
        pass
    return "\n".join(parts)
