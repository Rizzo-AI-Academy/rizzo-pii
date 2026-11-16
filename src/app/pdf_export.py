# -*- coding: utf-8 -*-
"""
Generazione del PDF anonimizzato (issue #7, punto 1).

Due modalita', entrambe 100% in locale e senza dipendenze dal modello: il modulo
lavora solo su bytes + dizionario {placeholder -> valore}, quindi e' testabile in
isolamento (nessun import di torch/transformers).

  redact_pdf(pdf_bytes, mapping)
      Redazione VERA del PDF originale: per ogni valore del dizionario cerca le
      occorrenze nelle pagine, RIMUOVE il testo dal content stream
      (page.apply_redactions(), non un rettangolo disegnato sopra) e scrive il
      placeholder al suo posto. Layout conservato.

      Il matching e' CHAR-PRECISO con CONFINI DI PAROLA: la pagina viene
      indicizzata carattere per carattere (get_text("rawdict"), ogni glifo con il
      suo bbox) e i valori sono cercati con regex ancorate (?<!\\w)...(?!\\w).
      Cosi' "DE" NON viene redatto dentro "CORDELLA" e una cifra non cancella i
      numeri di un referto: si redige il token/la sequenza esatta, ovunque ma
      intera. Il pattern tollera la SILLABAZIONE a fine riga ("Fran-\\ncesco") e
      gli spazi flessibili tra i token (anche a cavallo di riga).

  text_to_pdf(text)
      PDF "ricostruito" solo testo, impaginato da zero a partire dal testo gia'
      anonimizzato (input incollato oppure file .md/.txt).

Cosa viene ripulito oltre al testo di pagina (tutti posti dove si nasconde PII e
che apply_redactions() da solo NON tocca):
  - metadati classici + XMP;
  - contenuto delle ANNOTAZIONI (commenti, FreeText, note);
  - valore dei CAMPI MODULO (widget AcroForm);
  - titoli dei SEGNALIBRI (outline/TOC);
  - ALLEGATI incorporati (embedded files), rimossi in blocco.

Rete di sicurezza finale: `report["residual"]` elenca i placeholder il cui valore
e' ANCORA leggibile nell'output (testo di pagina + annotazioni + widget + TOC).
Deve essere vuota; l'UI avvisa se non lo e'.

Scansioni: le pagine lette con l'OCR (ocr.py) arrivano in `ocr_pages` con righe e
box per carattere; su quelle la redazione avviene SUI PIXEL (_redact_scanned_page).

Limiti noti:
  - testo dentro immagini raster su pagine che NON passano dall'OCR (es. un logo in
    un PDF testuale con ocr=auto): non esiste nel layer testuale, quindi non puo'
    essere trovato ne' redatto. Se in tutto il PDF non si trova NESSUNA occorrenza,
    il chiamante deve rifiutare: meglio un errore che un PDF "anonimizzato" che non lo e';
  - valori troppo corti/ambigui (< 2 caratteri alfanumerici, o 2 sole cifre) NON
    vengono redatti: cercarli ovunque devasterebbe il documento. Finiscono in
    `report["skipped"]` e l'UI DEVE avvisare, perche' restano in chiaro.
"""

import re
import unicodedata

import fitz  # PyMuPDF
import pdf_text


class PdfError(ValueError):
    """Errore d'uso (PDF non valido, protetto, dizionario vuoto...)."""


# --------------------------------------------------------------------------- #
# Utility comuni
# --------------------------------------------------------------------------- #
# caratteri tipografici frequenti nei PDF estratti -> equivalenti Latin-1
# (il font base "helv" copre Latin-1: cosi' la sostituzione e' deterministica)
_TRANSLATE = str.maketrans({
    "‘": "'", "’": "'", "“": '"', "”": '"',
    "–": "-", "—": "-", "…": "...", "€": "EUR",
    " ": " ", "•": "-", "ﬁ": "fi", "ﬂ": "fl",
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


# Sillabazione a fine riga: "Fran-\ncesco", "Fran­\ncesco" o anche solo un
# a-capo dentro la parola. Ammesso TRA due caratteri qualsiasi del valore, ma solo
# se c'e' davvero un newline: senza, il gruppo non consuma nulla e il matching
# resta char-per-char (niente tolleranza allo spacing, che aprirebbe a
# sovra-redazioni tipo "MI" dentro "M I L A N O").
_HYPHEN_BREAK = r"(?:[­‐-]?[ \t]*\n[ \t]*)?"


# Sotto questa soglia di caratteri alfanumerici un frammento NON puo' rinunciare
# ai confini di parola: ".it" senza ancore matcherebbe "it" dentro ogni parola.
_UNANCHORED_MIN_ALNUM = 4


def _value_pattern(value):
    """Regex del valore: caratteri esatti con CONFINI DI PAROLA agli estremi,
    sillabazione a fine riga tollerata, whitespace flessibile tra i token.
    Niente match di sottostringhe dentro altre parole.

    Eccezione: quando il valore ha punteggiatura ai bordi il modello ha tagliato
    a meta' una parola spezzata a fine riga (tipico: "Fran-\\ncesco Cordella" ->
    il modello etichetta "-\\ncesco Cordella"). Li' il confine di parola su quel
    lato e' proprio cio' che impedisce di trovare il valore, che resterebbe in
    chiaro: si toglie la punteggiatura e con essa l'ancora, ma solo se quel che
    resta e' abbastanza lungo da non trasformarsi in una sottostringa qualunque."""
    v = _norm(value)
    core = re.sub(r"^[^\w]+", "", v)
    core = re.sub(r"[^\w]+$", "", core)
    toks = [t for t in core.split(" ") if t]
    if not toks:
        return None
    if len(re.sub(r"[\W_]+", "", core)) >= _UNANCHORED_MIN_ALNUM:
        left = "" if re.match(r"^[^\w]", v) else r"(?<!\w)"
        right = "" if re.search(r"[^\w]$", v) else r"(?!\w)"
    else:                                  # troppo corto: ancore obbligatorie
        toks = [t for t in v.split(" ") if t]
        left, right = r"(?<!\w)", r"(?!\w)"
    body = r"\s*".join(_HYPHEN_BREAK.join(re.escape(c) for c in tok) for tok in toks)
    return re.compile(left + body + right, re.IGNORECASE)


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
    """Valori non localizzabili in modo sicuro in un PDF: frammenti con meno di 2
    caratteri alfanumerici, o di 2 sole cifre (es. "1", "C", "05", prodotti a
    volte dal modello su testi tabellari). Cercarli ovunque cancellerebbe pezzi di
    documento non-PII: si saltano, MA il chiamante deve avvisare l'utente (restano
    in chiaro)."""
    alnum = re.sub(r"[\W_]+", "", _norm(value))
    return len(alnum) < 2 or (len(alnum) == 2 and alnum.isdigit())


# --------------------------------------------------------------------------- #
# Pulizia di tutto cio' che apply_redactions() non tocca
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


def _sub_all(patterns, s):
    """Applica tutte le (regex, placeholder) alla stringa. Ritorna (nuova, n)."""
    n = 0
    for pat, ph in patterns:
        s, k = pat.subn(ph, s)
        n += k
    return s, n


def _scrub_annots(page, patterns):
    """Contenuto e titolo delle annotazioni (commenti, FreeText, note adesive):
    testo vero e proprio, invisibile a get_text() e non toccato dalle redazioni."""
    done = 0
    try:
        annots = list(page.annots())
    except Exception:
        return 0
    for a in annots:
        try:
            if a.type[0] == fitz.PDF_ANNOT_REDACT:      # le nostre, gia' gestite
                continue
            info = a.info
            new = dict(info)
            hit = 0
            for k in ("content", "subject", "title"):
                if info.get(k):
                    new[k], k_hit = _sub_all(patterns, info[k])
                    hit += k_hit
            if hit:
                a.set_info(new)
                a.update()
                done += hit
        except Exception:
            continue
    return done


def _scrub_widgets(page, patterns):
    """Valore dei campi modulo AcroForm: sopravvive a qualsiasi redazione del
    content stream ed e' leggibile aprendo il PDF."""
    done = 0
    try:
        widgets = list(page.widgets())
    except Exception:
        return 0
    for w in widgets:
        try:
            val = w.field_value
            if not isinstance(val, str) or not val:
                continue
            new, hit = _sub_all(patterns, val)
            if hit:
                w.field_value = new
                w.update()
                done += hit
        except Exception:
            continue
    return done


def _scrub_toc(doc, patterns):
    """Titoli dei segnalibri: spesso ricalcano intestazioni con nomi e numeri."""
    try:
        toc = doc.get_toc(simple=True)
    except Exception:
        return 0
    if not toc:
        return 0
    done, new_toc = 0, []
    for entry in toc:
        lvl, title, pg = entry[0], entry[1], entry[2]
        title, hit = _sub_all(patterns, title or "")
        done += hit
        new_toc.append([lvl, title, pg])
    if done:
        try:
            doc.set_toc(new_toc)
        except Exception:
            return 0
    return done


def _strip_embedded(doc):
    """Allegati incorporati: non ispezionabili in modo affidabile (possono essere
    di qualunque formato), quindi si rimuovono tutti."""
    removed = 0
    try:
        names = list(doc.embfile_names())
    except Exception:
        return 0
    for name in names:
        try:
            doc.embfile_del(name)
            removed += 1
        except Exception:
            continue
    return removed


def _readable_text(doc):
    """TUTTO il testo leggibile del documento: pagine + annotazioni + campi
    modulo + segnalibri. E' la base della verifica dei residui: se un valore
    compare qui, l'anonimizzazione NON e' completa.
    NB: non vede il testo dentro immagini raster (loghi, firme, scansioni)."""
    return pdf_text.collect_readable_text(doc, redact_annot_type=fitz.PDF_ANNOT_REDACT)


def _verify_residuals(pdf_bytes, items):
    """Placeholder il cui valore e' ANCORA leggibile nell'output. Usa lo STESSO
    pattern della redazione (sillabazione inclusa), altrimenti dichiarerebbe
    "0 residui" proprio nei casi che il matcher non sa gestire."""
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        text = _readable_text(doc)
    residual = []
    for ph, val in items:
        pat = _value_pattern(val)
        if pat and pat.search(text):
            residual.append(ph)
    return residual


# --------------------------------------------------------------------------- #
# API pubblica
# --------------------------------------------------------------------------- #
# Colore della casella di redazione: il viola del brand (#7c3a9e) pieno, con il
# placeholder in bianco sopra. Deve SALTARE ALL'OCCHIO scorrendo il documento —
# una casella tenue si confonde con un'evidenziazione qualsiasi, e chi rilegge il
# PDF prima di mandarlo fuori deve vedere subito cosa e' stato coperto.
REDACT_FILL = (0.486, 0.227, 0.620)
REDACT_TEXT = (1.0, 1.0, 1.0)


# --------------------------------------------------------------------------- #
# Pagine scansionate (OCR): redazione sui PIXEL
# --------------------------------------------------------------------------- #
# Per una pagina letta con l'OCR il testo non sta nel content stream ma in
# un'immagine: apply_redactions() da solo non ha niente da togliere. Qui:
#   1. si ricostruisce l'indice char-preciso dalle righe OCR (box per carattere,
#      gia' in punti PDF: li calcola ocr.py), cosi' lo STESSO matcher dei PDF
#      testuali (_value_pattern, confini di parola, sillabazione) trova i valori;
#   2. si renderizza la pagina e si SOVRASCRIVONO i pixel delle PII (e dei timbri);
#   3. si toglie dalla pagina tutto il contenuto originale (immagini comprese) e
#      si rimette solo l'immagine gia' coperta, con i placeholder scritti sopra.
# Il risultato non contiene piu' i pixel originali sotto le caselle: non e' un
# rettangolo disegnato sopra una scansione intatta.
OCR_PAD_X = 0.18     # margine orizzontale della casella, in altezze di riga
OCR_PAD_Y = 0.10     # margine verticale
OCR_JPEG_QUALITY = 82
SEAL_LABEL = "[TIMBRO]"


def _ocr_char_index(ocr_page):
    """(testo, [Rect per carattere]) dalle righe OCR di una pagina, nello stesso
    formato di _page_char_index. Il primo e l'ultimo carattere di ogni riga si
    allargano fino ai bordi della riga: la posizione CTC del primo glifo cade
    spesso un po' dentro, e un bordo di lettera fuori dalla casella si legge."""
    chars, boxes = [], []
    for ln in ocr_page.get("lines") or ():
        text = ln.get("text") or ""
        if not text:
            continue
        lb = fitz.Rect(ln["box"])
        cb = ln.get("chars")
        if not cb or len(cb) != len(text):
            cb = [ln["box"]] * len(text)
        for k, (c, b) in enumerate(zip(text, cb)):
            r = fitz.Rect(b)
            if k == 0:
                r.x0 = min(r.x0, lb.x0)
            if k == len(text) - 1:
                r.x1 = max(r.x1, lb.x1)
            chars.append(c)
            boxes.append(r)
        chars.append("\n")
        boxes.append(None)
    return "".join(chars), boxes


def _pad(rect, page_rect):
    h = rect.height
    r = fitz.Rect(rect.x0 - OCR_PAD_X * h, rect.y0 - OCR_PAD_Y * h,
                  rect.x1 + OCR_PAD_X * h, rect.y1 + OCR_PAD_Y * h)
    return r & page_rect


def _redact_scanned_page(page, ocr_page, usable, fill, text_color, redact_seals):
    """Redazione sui pixel di una pagina letta con l'OCR. Ritorna
    ({placeholder: occorrenze}, n_timbri)."""
    import numpy as np

    page.remove_rotation()      # coordinate OCR = pagina com'e' vista (ocr.py)
    prect = page.rect
    text, boxes = _ocr_char_index(ocr_page)
    hits, taken, labels = {}, [], []
    if text.strip():
        for ph, _val, pat in usable:
            for m in pat.finditer(text):
                placed = False
                for r in _match_rects(boxes, m):
                    r = _pad(r, prect)
                    if r.is_empty or _covered(r, taken):
                        continue
                    taken.append(r)
                    labels.append((r, ph))
                    placed = True
                if placed:
                    hits[ph] = hits.get(ph, 0) + 1
    n_seals = 0
    if redact_seals:
        for b in ocr_page.get("seals") or ():
            r = fitz.Rect(b) & prect
            if not r.is_empty:
                taken.append(r)
                labels.append((r, SEAL_LABEL))
                n_seals += 1

    dpi = int(ocr_page.get("dpi") or 200)
    pix = page.get_pixmap(dpi=dpi, alpha=False, colorspace=fitz.csRGB)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
        pix.height, pix.stride // 3, 3)[:, :pix.width].copy()
    sx, sy = pix.width / prect.width, pix.height / prect.height
    color = np.array([round(c * 255) for c in fill], dtype=np.uint8)
    for r in taken:
        x0, y0 = max(0, int(r.x0 * sx)), max(0, int(r.y0 * sy))
        x1 = min(pix.width, int(np.ceil(r.x1 * sx)))
        y1 = min(pix.height, int(np.ceil(r.y1 * sy)))
        if x1 > x0 and y1 > y0:
            img[y0:y1, x0:x1] = color      # i pixel originali spariscono QUI
    out = fitz.Pixmap(fitz.csRGB, pix.width, pix.height, img.tobytes(), 0)
    jpeg = out.tobytes("jpeg", jpg_quality=OCR_JPEG_QUALITY)

    # via TUTTO il contenuto originale della pagina: testo, immagini, grafica
    page.add_redact_annot(prect)
    kw = dict(images=fitz.PDF_REDACT_IMAGE_REMOVE, text=fitz.PDF_REDACT_TEXT_REMOVE)
    try:
        page.apply_redactions(graphics=fitz.PDF_REDACT_LINE_ART_REMOVE_IF_TOUCHED, **kw)
    except (TypeError, AttributeError):
        page.apply_redactions(**kw)
    page.insert_image(prect, stream=jpeg)

    # placeholder in bianco sopra le caselle (testo vero: si legge e si copia)
    for r, ph in labels:
        fs = _fit_fontsize(ph, r)
        if not fs:
            continue
        w = fitz.get_text_length(ph, fontname="helv", fontsize=fs)
        x = r.x0 + (r.width - w) / 2
        y = r.y0 + r.height / 2 + fs * 0.35
        page.insert_text((x, y), ph, fontname="helv", fontsize=fs, color=text_color)
    return hits, n_seals


def redact_pdf(pdf_bytes, mapping, fill=REDACT_FILL, text_color=REDACT_TEXT,
               ocr_pages=None, redact_seals=True):
    """PDF originale -> PDF con redazione vera + placeholder al posto delle PII.

    mapping: {"[FULLNAME_1]": "Mario Rossi", ...} (il dizionario di analyze()).
    ocr_pages: {indice pagina: risultato di ocr.ocr_page} per le pagine lette con
        l'OCR (scansioni). Su quelle la redazione avviene sui PIXEL e la pagina
        esce come immagine coperta + placeholder (vedi _redact_scanned_page).
    redact_seals: sulle pagine OCR copre anche i timbri trovati dal layout.
    Ritorna (bytes, report) con report = {
        "occurrences":    occorrenze redatte nel testo di pagina,
        "by_placeholder": {placeholder: n_occorrenze},
        "not_found":      placeholder cercati ma senza occorrenze di pagina,
        "skipped":        placeholder NON cercati perche' troppo corti/ambigui
                          (restano in chiaro: va segnalato all'utente),
        "residual":       placeholder ancora leggibili nell'output (deve essere []),
        "annots":         sostituzioni nelle annotazioni,
        "widgets":        sostituzioni nei campi modulo,
        "toc":            sostituzioni nei segnalibri,
        "embedded":       allegati rimossi,
        "ocr_pages":      pagine redatte sui pixel (scansioni),
        "seals":          timbri coperti sulle pagine OCR,
    }
    """
    ocr_pages = ocr_pages or {}
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
        else:
            skipped.append(ph)

    by_ph = {ph: 0 for ph, _, _ in usable}
    patterns = [(pat, ph) for ph, _, pat in usable]   # per annot/widget/TOC
    total = n_annots = n_widgets = n_seals = 0
    raster_pages = []

    for pno, page in enumerate(doc):
        if pno in ocr_pages:
            hits, seals = _redact_scanned_page(page, ocr_pages[pno], usable,
                                               fill, text_color, redact_seals)
            for ph, n in hits.items():
                by_ph[ph] += n
                total += n
            n_seals += seals
            raster_pages.append(pno)
            n_annots += _scrub_annots(page, patterns)
            n_widgets += _scrub_widgets(page, patterns)
            continue
        text, boxes = _page_char_index(page)
        taken = []
        if text.strip():
            for ph, val, pat in usable:
                for m in pat.finditer(text):
                    rects = _match_rects(boxes, m)
                    placed_any, labeled = False, False
                    for r in rects:
                        if _covered(r, taken):
                            continue
                        fs = 0 if labeled else _fit_fontsize(ph, r)
                        _add_redact_annot(page, r, ph if fs else None,
                                          fs or 6, fill, text_color)
                        taken.append(fitz.Rect(r))
                        placed_any = True
                        labeled = labeled or bool(fs)
                    if placed_any:
                        by_ph[ph] += 1
                        total += 1
        if taken:
            page.apply_redactions()   # rimozione VERA dal content stream
        # dopo le redazioni: annotazioni e campi modulo non sono content stream
        n_annots += _scrub_annots(page, patterns)
        n_widgets += _scrub_widgets(page, patterns)

    n_toc = _scrub_toc(doc, patterns)
    n_emb = _strip_embedded(doc)
    _scrub_metadata(doc)
    out = doc.tobytes(garbage=3, deflate=True)
    doc.close()

    checked = [(ph, v) for ph, v in items if ph in by_ph]
    return out, {
        "occurrences": total,
        "by_placeholder": by_ph,
        "not_found": [ph for ph, n in by_ph.items() if n == 0],
        "skipped": skipped,
        "residual": _verify_residuals(out, checked),
        "annots": n_annots,
        "widgets": n_widgets,
        "toc": n_toc,
        "embedded": n_emb,
        "ocr_pages": raster_pages,
        "seals": n_seals,
    }


def _add_redact_annot(page, rect, text, fontsize, fill, text_color):
    """add_redact_annot con cross_out disattivato dove il binding lo supporta
    (le X diagonali sporcano il placeholder); fallback per le versioni vecchie."""
    kw = dict(text=text, fontname="helv", fontsize=fontsize,
              align=fitz.TEXT_ALIGN_CENTER, fill=fill, text_color=text_color)
    try:
        return page.add_redact_annot(rect, cross_out=False, **kw)
    except TypeError:
        return page.add_redact_annot(rect, **kw)


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

    # In una riga larga `width` non entrano piu' di `cap` caratteri, qualunque siano:
    # oltre quel punto non c'e' niente da misurare. Serve perche' hard_split misurava
    # e ricopiava TUTTA la stringa residua a ogni giro -> O(n^2): una "parola" senza
    # spazi (JSON incollato, base64, riga di tabella coi soli tab) da 16.000 caratteri
    # costava 15 s, da 128.000 un quarto d'ora.
    # Il conto va fatto sul glifo PIU' STRETTO, non su "l": il testo e' gia' forzato in
    # latin-1 poche righe sopra, quindi il repertorio e' chiuso e il minimo si misura.
    # In helv e' l'apostrofo (2,0055 pt a corpo 10,5) contro i 2,331 di "l": con "l" cap
    # varrebbe 209 dove in colonna ne entrano 240, la finestra ci starebbe, il ciclo
    # uscirebbe subito e la coda finirebbe tutta su una riga larga oltre il bordo pagina.
    stretto = min(tl(chr(c)) for c in list(range(32, 127)) + list(range(160, 256)))
    cap = max(2, int(width / max(stretto, 0.1)) + 2)   # >= 2: con margini assurdi width <= 0

    def hard_split(line):
        """Spezza le 'parole' piu' larghe della riga (IBAN, URL...)."""
        if tl(line) <= width:          # caso normale: la riga ci sta gia', una sola misura
            return [line]
        out, i = [], 0
        while len(line) - i > 1:
            head = line[i:i + cap]
            if tl(head) <= width:
                break
            k = max(1, int(len(head) * width / tl(head)))
            while k > 1 and tl(head[:k]) > width:
                k -= 1
            while k < len(head) and tl(head[:k + 1]) <= width:
                k += 1
            out.append(head[:k])
            i += k
        out.append(line[i:])
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
