# -*- coding: utf-8 -*-
"""Da box+testo sparsi dell'OCR a prosa in ordine di lettura.

L'OCR ritorna righe con bounding box, non un documento: se si concatenano
nell'ordine di detection il testo esce mescolato e le entita' si spezzano
(un nome a cavallo di due box, un IBAN diviso a meta').

Funzione pura: nessuna dipendenza dall'engine, testabile senza modelli.
"""

import numpy as np


def lines_to_text(boxes, txts, scores=None, row_tol=0.6, para_gap=1.6) -> str:
    """Raggruppa i box in righe, le righe in paragrafi, ritorna testo piatto.

    row_tol  - scarto massimo in y (in altezze mediane) perche' due box siano
               considerati la stessa riga.
    para_gap - salto verticale (in altezze mediane) oltre il quale si apre un
               nuovo paragrafo.
    scores   - accettati ma NON usati per filtrare: una riga a bassa confidenza
               scartata e' una PII che sfugge alla redazione. Meglio testo
               storpiato ma presente.

    Assume colonna singola (atti, contratti, sentenze). Su pagine a due colonne
    mescola le colonne.
    """
    if boxes is None or len(boxes) == 0:
        return ""

    b = np.asarray(boxes, dtype=float)          # (N, 4, 2)
    y_top, y_bot = b[:, :, 1].min(1), b[:, :, 1].max(1)
    x_left = b[:, :, 0].min(1)
    y_mid = (y_top + y_bot) / 2
    h = float(np.median(y_bot - y_top)) or 1.0

    # 1) box -> righe (scorrendo dall'alto, accorpando quelli sulla stessa baseline)
    rows, cur, cur_y = [], [], None
    for i in np.argsort(y_mid, kind="stable"):
        if cur_y is not None and abs(y_mid[i] - cur_y) > row_tol * h:
            rows.append((cur_y, cur))
            cur, cur_y = [], None
        cur.append(int(i))
        cur_y = y_mid[i] if cur_y is None else (cur_y + y_mid[i]) / 2
    if cur:
        rows.append((cur_y, cur))

    # 2) righe -> testo, con interruzione di paragrafo sui salti verticali
    parts, prev_y = [], None
    for ry, idxs in rows:
        idxs.sort(key=lambda i: x_left[i])
        line = " ".join(str(txts[i]).strip() for i in idxs if str(txts[i]).strip())
        if not line:
            continue
        if prev_y is not None:
            parts.append("\n\n" if (ry - prev_y) > para_gap * h else "\n")
        parts.append(line)
        prev_y = ry

    return "".join(parts).strip()
