# -*- coding: utf-8 -*-
"""
Generazione del PDF anonimizzato (issue #7, punto 1).

Due modalita', entrambe 100% in locale e senza dipendenze dal modello
(il modulo lavora solo su bytes + dizionario {placeholder -> valore}):

  redact_pdf(pdf_bytes, mapping)
      Redazione VERA del PDF originale: per ogni valore del dizionario cerca
      le occorrenze nelle pagine, RIMUOVE il testo dal content stream
      (page.apply_redactions(), non un rettangolo disegnato sopra) e scrive il
      placeholder al suo posto. Layout conservato, metadati/XMP azzerati.

      Il matching e' CHAR-PRECISO con CONFINI DI PAROLA: la pagina viene
      indicizzata carattere per carattere (get_text("rawdict"), ogni glifo con
      il suo bbox) e i valori sono cercati con regex ancorate (?<!\\w)...(?!\\w),
      spazi flessibili e supporto multi-riga. Cosi' "DE" NON viene redatto
      dentro "CORDELLA" o "DENSITA'" e una cifra non cancella i numeri di un
      referto: si redige solo il token/la sequenza esatta, ovunque ma intera.
      I valori "rumore" non localizzabili in modo sicuro (meno di 2 caratteri
      alfanumerici, o 2 sole cifre: frammenti prodotti a volte dal modello su
      documenti tabellari) vengono SALTATI e riportati in report["skipped"].

      Ritorna (pdf_bytes_out, report); report["residual"] = valori ancora
      leggibili nel testo estraibile dell'output (verifica finale: deve essere
      vuota, altrimenti l'UI avvisa).

  text_to_pdf(text)
      PDF "ricostruito" solo testo, impaginato da zero a partire dal testo
      anonimizzato (usato quando l'input era testo incollato).

Limiti noti (coerenti con i punti 2-4 della issue #7, sviluppi futuri):
  - testo dentro immagini raster (scansioni, loghi, blocchi firma): non esiste
    nel layer testuale, quindi non puo' essere trovato ne' redatto (serve OCR);
  - parole sillabate a cavallo di riga ("Fran-\\ncesco"): non riconosciute dal
    matcher; in tal caso il valore finisce in "residual" e l'UI avvisa.
"""

import re
import unicodedata

import fitz  # PyMuPDF


class PdfError(ValueError):
    """Errore d'uso (PDF non valido, protetto, dizionario vuoto...)."""


# --------------------------------------------------------------------------- #
# Utility comuni
# --------------------------------------------------------------------------- #
# caratteri tipografici frequenti nei PDF estratti -> equivalenti Latin-1
# (il font base "helv" copre Latin-1; cosi' la sostituzione e' deterministica)
_TRANSLATE = str.maketrans({
    "\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"',
    "\u2013": "-", "\u2014": "-", "\u2026": "...", "\u20ac": "EUR",
    "\u00a0": " ", "\u2022": "-", "\ufb01": "fi", "\ufb02": "fl",
})


def _norm(s):
    """Spazi compressi + casefold: stessa normalizzazione usata in app.py."""
    return re.sub(r"\s+", " ", (s or "").strip()).casefold()


def _fit_fontsize(text, rect, max_fs=10.0, min_fs=4.0):
    """Corpo del testo per far stare `text` dentro `rect` (0 = non ci sta:
    meglio nessuna etichetta che un'etichetta illeggibile o troncata)."""
    try:
        w10 = fitz.get_text_length(text, fontname="helv", fontsize=10.0)
    except Exception:
        return 0
    if w10 <= 0:
        return 0
    fs = min(max_fs, rect.height * 0.82, 10.0 * max(rect.width - 2.0, 0.0) / w10)
    return round(fs, 1) if fs >= min_fs else 0


def _covered(rect, taken, thr=0.85):
    """True se `rect` e' gia' (quasi) tutto dentro una redazione precedente:
    evita doppioni quando un valore e' contenuto in un altro (es. "Rossi"
    dentro "Mario Rossi", redatto prima perche' piu' lungo)."""
    area = rect.get_area()
    if area <= 0:
        return True
    for t in taken:
        inter = fitz.Rect(rect)
        inter.intersect(t)
        if not inter.is_empty and inter.get_area() / area >= thr:
            return True
    return False


# --------------------------------------------------------------------------- #
# Indice char-preciso della pagina + ricerca con confini di parola
# --------------------------------------------------------------------------- #
def _page_char_index(page):
    """(testo, [bbox per carattere]) dalla pagina: ogni carattere del layer
    testuale con il suo rettangolo (None per i newline di fine riga)."""
    raw = page.get_text("rawdict")
    chars, boxes = [], []
    for block in raw.get("blocks", []):
        if block.get("type") != 0:          # solo blocchi di testo
            continue
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                for ch in span.get("chars", []):
                    c = ch.get("c") or ""
                    for cc in c:            # ligature -> piu' caratteri, stesso bbox
                        chars.append(cc)
                        boxes.append(fitz.Rect(ch["bbox"]))
            chars.append("\n")
            boxes.append(None)
    return "".join(chars), boxes


def _value_pattern(value):
    """Regex del valore: token esatti con CONFINI DI PAROLA agli estremi,
    whitespace flessibile (anche a cavallo di riga) tra i token. Niente match
    di sottostringhe dentro altre parole."""
    toks = [re.escape(t) for t in _norm(value).split(" ") if t]
    if not toks:
        return None
    return re.compile(r"(?<!\w)" + r"\s*".join(toks) + r"(?!\w)",
                      re.IGNORECASE)


def _match_rects(boxes, m):
    """Bbox dei caratteri del match, uniti per riga (overlap verticale)."""
    rects, cur = [], None
    for i in range(m.start(), m.end()):
        b = boxes[i]
        if b is None or b.is_empty:
            continue
        if cur is None:
            cur = fitz.Rect(b)
        elif b.y0 < cur.y1 and b.y1 > cur.y0:      # stessa riga
            cur |= b
        else:                                       # riga nuova
            rects.append(cur)
            cur = fitz.Rect(b)
    if cur is not None and not cur.is_empty:
        rects.append(cur)
    return rects


def _too_noisy(value):
    """Valori non localizzabili in modo sicuro in un PDF: frammenti con meno
    di 2 caratteri alfanumerici, o di 2 sole cifre (es. "1", "C", "05"
    prodotti a volte dal modello su testi tabellari). Meglio saltarli e
    dichiararlo nel report che devastare il documento."""
    alnum = re.sub(r"[\W_]+", "", _norm(value))
    return len(alnum) < 2 or (len(alnum) == 2 and alnum.isdigit())


# --------------------------------------------------------------------------- #
# Metadati + verifica finale
# --------------------------------------------------------------------------- #
def _scrub_metadata(doc):
    """Azzera i metadati classici e l'XMP: possono contenere PII (autore...)."""
    try:
        # set_metadata aggiorna SOLO le chiavi passate: vanno azzerate tutte
        # esplicitamente, altrimenti autore/titolo originali restano nel file
        doc.set_metadata({
            "title": "", "author": "", "subject": "", "keywords": "",
            "creationDate": "", "modDate": "", "trapped": "",
            "creator": "rizzo-pii", "producer": "rizzo-pii",
        })
    except Exception:
        pass
    f = getattr(doc, "del_xml_metadata", None) or getattr(doc, "delXmlMetadata", None)
    if f:
        try:
            f()
        except Exception:
            pass


def _verify_residuals(pdf_bytes, items):
    """Placeholder i cui valori sono ANCORA leggibili nel testo estraibile
    dell'output. Rete di sicurezza finale: deve tornare [].
    NB: non vede il testo dentro immagini raster (loghi, firme, scansioni)."""
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        text = _norm(" ".join(page.get_text() for page in doc))
    residual = []
    for ph, val in items:
        nv = _norm(val)
        if not nv:
            continue
        pat = r"(?<!\w)" + re.escape(nv) + r"(?!\w)"
        if re.search(pat, text):
            residual.append(ph)
    return residual


# --------------------------------------------------------------------------- #
# API pubblica
# --------------------------------------------------------------------------- #
def redact_pdf(pdf_bytes, mapping,
               fill=(0.93, 0.92, 0.97), text_color=(0.39, 0.18, 0.50)):
    """PDF originale -> PDF con redazione vera + placeholder al posto delle PII.

    mapping: {"[FULLNAME_1]": "Mario Rossi", ...} (il dizionario di analyze()).
    Ritorna (bytes, report) con report = {
        "occurrences":  occorrenze redatte in totale,
        "by_placeholder": {placeholder: n_occorrenze},
        "not_found":    placeholder cercati ma senza occorrenze,
        "skipped":      placeholder saltati perche' troppo corti/ambigui,
        "residual":     placeholder ancora leggibili nell'output (deve essere []),
    }
    """
    if not isinstance(mapping, dict) or not mapping:
        raise PdfError("Dizionario vuoto: anonimizza prima il documento.")
    items = [(ph, v) for ph, v in mapping.items()
             if isinstance(ph, str) and isinstance(v, str) and v.strip()]
    if not items:
        raise PdfError("Dizionario non valido.")

    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    except Exception:
        raise PdfError("File PDF non valido o danneggiato.")
    if doc.needs_pass:
        doc.close()
        raise PdfError("PDF protetto da password: rimuovi la protezione e riprova.")

    # valori lunghi per primi: cosi' "Rossi" non spezza la redazione di
    # "Mario Rossi" (i rect gia' coperti vengono saltati da _covered)
    items.sort(key=lambda kv: -len(kv[1]))

    skipped, usable = [], []
    for ph, val in items:
        if _too_noisy(val):
            skipped.append(ph)
            continue
        pat = _value_pattern(val)
        if pat:
            usable.append((ph, val, pat))

    by_ph = {ph: 0 for ph, _, _ in usable}
    total = 0
    for page in doc:
        text, boxes = _page_char_index(page)
        if not text.strip():
            continue
        taken = []
        for ph, val, pat in usable:
            for m in pat.finditer(text):
                rects = _match_rects(boxes, m)
                placed_any, labeled = False, False
                for r in rects:
                    if _covered(r, taken):
                        continue
                    fs = 0 if labeled else _fit_fontsize(ph, r)
                    page.add_redact_annot(
                        r,
                        text=ph if fs else None,
                        fontname="helv",
                        fontsize=fs or 6,
                        align=fitz.TEXT_ALIGN_CENTER,
                        fill=fill,
                        text_color=text_color,
                    )
                    taken.append(fitz.Rect(r))
                    placed_any = True
                    labeled = labeled or bool(fs)
                if placed_any:
                    by_ph[ph] += 1
                    total += 1
        if taken:
            page.apply_redactions()   # rimozione VERA dal content stream

    _scrub_metadata(doc)
    out = doc.tobytes(garbage=3, deflate=True)
    doc.close()

    checked = [(ph, v) for ph, v in items if ph in by_ph]
    report = {
        "occurrences": total,
        "by_placeholder": by_ph,
        "not_found": [ph for ph, n in by_ph.items() if n == 0],
        "skipped": skipped,
        "residual": _verify_residuals(out, checked),
    }
    return out, report


def text_to_pdf(text, margin=56.0, fontsize=10.5, leading=15.5):
    """Testo (gia' anonimizzato) -> PDF A4 solo testo, impaginato da zero.
    Nessun contenuto del documento originale finisce nell'output."""
    text = unicodedata.normalize("NFKC", text or "").translate(_TRANSLATE)
    # helv copre Latin-1: sostituzione esplicita dei caratteri fuori codifica
    text = text.encode("latin-1", "replace").decode("latin-1")
    if not text.strip():
        raise PdfError("Nessun testo da impaginare.")

    a4 = fitz.paper_rect("a4")
    width = a4.width - 2 * margin

    def tl(s):
        return fitz.get_text_length(s, fontname="helv", fontsize=fontsize)

    def hard_split(line):
        """Spezza le 'parole' piu' larghe della riga (IBAN, URL...)."""
        out = []
        while tl(line) > width and len(line) > 1:
            k = max(1, int(len(line) * width / tl(line)))
            while k > 1 and tl(line[:k]) > width:
                k -= 1
            while k < len(line) and tl(line[:k + 1]) <= width:
                k += 1
            out.append(line[:k])
            line = line[k:]
        out.append(line)
        return out

    def wrap(par):
        if not par:
            return [""]
        lines, cur = [], ""
        for w in par.split(" "):
            cand = (cur + " " + w) if cur else w
            if not cur or tl(cand) <= width:
                cur = cand
            else:
                lines.append(cur)
                cur = w
        lines.append(cur)
        out = []
        for ln in lines:
            out.extend(hard_split(ln))
        return out

    doc = fitz.open()
    page = doc.new_page(width=a4.width, height=a4.height)
    y = margin + fontsize
    for par in text.split("\n"):
        for ln in wrap(par.rstrip()):
            if y > a4.height - margin:
                page = doc.new_page(width=a4.width, height=a4.height)
                y = margin + fontsize
            if ln:
                page.insert_text((margin, y), ln, fontname="helv",
                                 fontsize=fontsize)
            y += leading

    _scrub_metadata(doc)
    out = doc.tobytes(deflate=True)
    doc.close()
    return out
