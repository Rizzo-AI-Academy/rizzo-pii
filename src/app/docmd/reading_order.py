"""Reading order con XY-cut ricorsivo: prima prova a separare le colonne (tagli verticali),
poi le fasce orizzontali. Le fasce consecutive che sono entrambe "a colonne" vengono fuse,
così una pagina con titolo a tutta larghezza e due colonne sotto si legge colonna per colonna.
Se invece le "colonne" sono allineate riga per riga (moduli chiave-valore, fatture) si legge
per righe."""

from __future__ import annotations


def xycut_order(boxes: list[list[float]]) -> list[int]:
    return _cut(list(range(len(boxes))), boxes)


def _cut(idxs: list[int], boxes, depth: int = 0) -> list[int]:
    if len(idxs) <= 1 or depth > 30:
        return idxs
    cols = _split(idxs, boxes, axis=0)
    row_aligned = len(cols) > 1 and _row_aligned(cols, boxes)
    if len(cols) > 1 and not row_aligned:
        return [i for g in cols for i in _cut(g, boxes, depth + 1)]

    bands = _split(idxs, boxes, axis=1)
    merged: list[list[int]] = []
    for band in bands:
        if (not row_aligned and merged and len(_split(band, boxes, 0)) > 1
                and len(_split(merged[-1], boxes, 0)) > 1):
            merged[-1] = merged[-1] + band
        else:
            merged.append(band)
    if len(merged) == 1:
        return _by_rows(idxs, boxes)
    return [i for g in merged for i in _cut(g, boxes, depth + 1)]


def _split(idxs: list[int], boxes, axis: int, min_gap: float = 3.0) -> list[list[int]]:
    """Raggruppa gli indici in base ai vuoti della proiezione sull'asse (0 = x, 1 = y)."""
    lo, hi = axis, axis + 2
    order = sorted(idxs, key=lambda i: boxes[i][lo])
    groups: list[list[int]] = []
    end = None
    for i in order:
        if end is None or boxes[i][lo] > end + min_gap:
            groups.append([i])
            end = boxes[i][hi]
        else:
            groups[-1].append(i)
            end = max(end, boxes[i][hi])
    return groups


def _y_iou(a, b) -> float:
    inter = min(a[3], b[3]) - max(a[1], b[1])
    union = max(a[3], b[3]) - min(a[1], b[1])
    return inter / union if inter > 0 and union > 0 else 0.0


def _row_aligned(cols: list[list[int]], boxes) -> bool:
    """True se quasi ogni blocco ha, in un'altra colonna, un compagno sulla stessa riga."""
    if min(len(c) for c in cols) < 2:
        return False
    paired = 0
    total = sum(len(c) for c in cols)
    for ci, col in enumerate(cols):
        others = [j for cj, c in enumerate(cols) if cj != ci for j in c]
        for i in col:
            if any(_y_iou(boxes[i], boxes[j]) > 0.5 for j in others):
                paired += 1
    return paired >= 0.7 * total


def _by_rows(idxs: list[int], boxes) -> list[int]:
    """Ordina per righe visive (centro verticale), poi da sinistra a destra."""
    rows: list[list[int]] = []
    for i in sorted(idxs, key=lambda i: (boxes[i][1] + boxes[i][3]) / 2):
        cy = (boxes[i][1] + boxes[i][3]) / 2
        if rows and any(boxes[j][1] <= cy <= boxes[j][3] for j in rows[-1]):
            rows[-1].append(i)
        else:
            rows.append([i])
    return [i for r in rows for i in sorted(r, key=lambda i: boxes[i][0])]
