"""Blocco 3: ramo tabelle. SLANet_plus predice la struttura HTML e i box delle celle;
il testo delle celle arriva dalle righe OCR del ramo testo."""

from __future__ import annotations

import html
import re
from pathlib import Path

import cv2
import numpy as np

from .common import intersection, load_model

TD_TOKENS = ("<td>", "<td", "<td></td>")


class TableRecognizer:
    def __init__(self, model_dir: Path, threads: int | None = None):
        self.sess, cfg = load_model(model_dir, threads)
        ops = {next(iter(op)): op[next(iter(op))] for op in cfg["PreProcess"]["transform_ops"]}
        self.max_len = ops["ResizeTableImage"]["max_len"]
        norm = ops["NormalizeImage"]
        self.mean = np.array(norm["mean"], dtype=np.float32)
        self.std = np.array(norm["std"], dtype=np.float32)

        chars = list(cfg["PostProcess"]["character_dict"])
        if cfg["PostProcess"].get("merge_no_span_structure", True):
            if "<td></td>" not in chars:
                chars.append("<td></td>")
            if "<td>" in chars:
                chars.remove("<td>")
        self.chars = ["sos"] + chars + ["eos"]
        self.end_idx = len(self.chars) - 1

    def structure(self, img_bgr: np.ndarray) -> tuple[list[str], list[list[float]]]:
        """Restituisce i token HTML e i box [x1,y1,x2,y2] delle celle (coordinate del crop)."""
        h, w = img_bgr.shape[:2]
        ratio = self.max_len / max(h, w)
        rh, rw = max(1, round(h * ratio)), max(1, round(w * ratio))
        x = cv2.resize(img_bgr, (rw, rh)).astype(np.float32)
        x = (x / 255.0 - self.mean) / self.std
        padded = np.zeros((self.max_len, self.max_len, 3), dtype=np.float32)
        padded[:rh, :rw] = x
        outs = self.sess.run(None, {self.sess.get_inputs()[0].name: padded.transpose(2, 0, 1)[None]})
        bbox_pred = next(o for o in outs if o.shape[-1] in (4, 8))[0]
        struct_prob = next(o for o in outs if o.shape[-1] == len(self.chars))[0]

        scale = self.max_len / ratio
        tokens, cells = [], []
        for i, ci in enumerate(struct_prob.argmax(axis=1)):
            if i > 0 and ci == self.end_idx:
                break
            if ci in (0, self.end_idx):
                continue
            tok = self.chars[ci]
            if tok in TD_TOKENS:
                pts = bbox_pred[i].reshape(-1, 2) * scale
                cells.append([float(pts[:, 0].min()), float(pts[:, 1].min()),
                              float(pts[:, 0].max()), float(pts[:, 1].max())])
            tokens.append(tok)
        return tokens, cells

    def to_html(self, img_bgr: np.ndarray, lines: list[tuple[list[float], str]]) -> str:
        """`lines`: righe OCR (box nel sistema del crop, testo)."""
        tokens, cells = self.structure(img_bgr)
        assigned = _match_lines_to_cells(lines, cells)
        if not cells or any(_has_column_gap(items) for items in assigned.values()):
            # SLANet ha fuso colonne che l'OCR vede separate: griglia ricostruita dalle righe OCR
            return grid_from_lines(lines)
        cell_texts = {i: html.escape(_cell_text(items)) for i, items in assigned.items()}
        out, ci = [], -1
        for tok in tokens:
            if tok == "<td></td>":
                ci += 1
                out.append(f"<td>{cell_texts.get(ci, '')}</td>")
            elif tok == "<td":
                ci += 1
                out.append(tok)
            elif tok == "</td>":
                out.append(f"{cell_texts.get(ci, '')}</td>")
            else:
                out.append(tok)
        return "<table>" + "".join(out) + "</table>"


def _match_lines_to_cells(lines, cells) -> dict[int, list]:
    if not cells:
        return {}
    assigned: dict[int, list] = {}
    for box, text in lines:
        best, best_score = -1, 0.0
        for i, c in enumerate(cells):
            s = intersection(box, c)
            if s > best_score:
                best, best_score = i, s
        if best < 0:  # nessuna sovrapposizione: cella col centro più vicino
            cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
            best = int(np.argmin([abs(cx - (c[0] + c[2]) / 2) + abs(cy - (c[1] + c[3]) / 2) for c in cells]))
        assigned.setdefault(best, []).append((box, text))
    return assigned


def _same_row(a, b) -> bool:
    cy = (a[1] + a[3]) / 2
    return b[1] <= cy <= b[3]


def _cell_text(items) -> str:
    return " ".join(t for _, t in sorted(items, key=lambda it: (round(it[0][1] / 8), it[0][0])))


def _has_column_gap(items) -> bool:
    """Due righe OCR sulla stessa riga visiva ma molto distanti in orizzontale = colonne fuse."""
    for i, (a, _) in enumerate(items):
        for b, _ in items[i + 1:]:
            if _same_row(a, b) or _same_row(b, a):
                gap = max(a[0], b[0]) - min(a[2], b[2])
                if gap > 1.5 * max(a[3] - a[1], b[3] - b[1]):
                    return True
    return False


def grid_from_lines(lines) -> str:
    """Tabella senza celle unite ricostruita dalla geometria: righe = righe visive OCR,
    colonne = intervalli orizzontali delle righe OCR che non si sovrappongono."""
    if not lines:
        return "<table></table>"
    rows: list[list] = []
    for box, text in sorted(lines, key=lambda l: (l[0][1] + l[0][3]) / 2):
        if rows and any(_same_row(box, r[0]) for r in rows[-1]):
            rows[-1].append((box, text))
        else:
            rows.append([(box, text)])

    # le righe larghe (titoli, note a tutta larghezza) non definiscono colonne
    table_w = max(b[2] for b, _ in lines) - min(b[0] for b, _ in lines)
    narrow = [b for b, _ in lines if b[2] - b[0] < 0.5 * table_w] or [b for b, _ in lines]
    cols: list[list[float]] = []
    for x1, x2 in sorted((b[0], b[2]) for b in narrow):
        if cols and x1 <= cols[-1][1]:
            cols[-1][1] = max(cols[-1][1], x2)
        else:
            cols.append([x1, x2])

    out = []
    for row in rows:
        cells: list[list] = [[] for _ in cols]
        for box, text in row:
            cx = (box[0] + box[2]) / 2
            ci = next((i for i, c in enumerate(cols) if c[0] <= cx <= c[1]), None)
            if ci is None:  # riga larga: colonna in cui inizia
                ci = max([i for i, c in enumerate(cols) if c[0] <= box[0] + 1] or [0])
            cells[ci].append((box, text))
        out.append("<tr>" + "".join(f"<td>{html.escape(_cell_text(c))}</td>" for c in cells) + "</tr>")
    return "<table>" + "".join(out) + "</table>"


def html_table_to_markdown(table_html: str) -> str:
    """Tabella semplice -> Markdown; con celle unite (colspan/rowspan) resta HTML."""
    if "colspan" in table_html or "rowspan" in table_html:
        return table_html
    rows = []
    for tr in re.findall(r"<tr>(.*?)</tr>", table_html, flags=re.S):
        cells = re.findall(r"<td[^>]*>(.*?)</td>", tr, flags=re.S)
        rows.append([html.unescape(c).replace("|", "\\|").strip() for c in cells])
    rows = [r for r in rows if r]
    if not rows:
        return ""
    ncol = max(len(r) for r in rows)
    rows = [r + [""] * (ncol - len(r)) for r in rows]
    md = ["| " + " | ".join(rows[0]) + " |", "|" + "---|" * ncol]
    md += ["| " + " | ".join(r) + " |" for r in rows[1:]]
    return "\n".join(md)
