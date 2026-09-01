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


def collect_readable_text(doc, redact_annot_type=REDACT_ANNOT):
    """TUTTO il testo leggibile: pagine + annotazioni + campi modulo + segnalibri.

    `doc` e' un iterabile di pagine con `get_text()`, `annots()`, `widgets()`
    e (sul documento) `get_toc(simple=True)`: il contratto di PyMuPDF, senza
    importarlo. NB: non vede il testo dentro immagini raster.
    """
    parts = []
    for page in doc:
        parts.append(page.get_text() or "")
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
