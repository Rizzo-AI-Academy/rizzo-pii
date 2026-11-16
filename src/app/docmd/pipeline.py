"""Pipeline "Detect-and-Route": layout -> routing dei blocchi verso testo / tabelle / formule -> Markdown.

Copia di `docmd` (OCR su ONNX Runtime) adattata a rizzo-pii. Rispetto all'originale:
  - ogni riga OCR porta i box dei SINGOLI caratteri (dai passi CTC): servono a coprire
    sull'immagine solo la PII, non l'intera riga;
  - `read_figure_text`: legge anche il testo dentro immagini, loghi e timbri (in un
    anonimizzatore una ragione sociale nel timbro e' una PII come le altre);
  - il ramo formule e' opzionale: senza il modello (321 MB, inutile per atti e contratti)
    i blocchi "formula" si leggono come testo;
  - `PageResult` espone le righe lette, l'angolo di raddrizzamento e l'immagine usata.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import quote

import cv2
import numpy as np

from .common import box_area, crop, overlap_ratio, quad_to_box
from .layout import Block, LayoutDetector
from .reading_order import xycut_order
from .table import TableRecognizer, html_table_to_markdown
from .text import TextDetector, TextRecognizer, crop_quad

DEFAULT_MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

PRESETS = {
    "fast": {"layout": "PP-DocLayout-M", "det": "PP-OCRv6_tiny_det", "rec": "PP-OCRv6_tiny_rec"},
    "balanced": {"layout": "PP-DocLayout-M", "det": "PP-OCRv6_small_det", "rec": "PP-OCRv6_small_rec"},
    "accurate": {"layout": "PP-DocLayout_plus-L", "det": "PP-OCRv6_small_det", "rec": "PP-OCRv6_small_rec"},
}

TEXT_LABELS = {"text", "abstract", "content", "reference", "reference_content",
               "footnote", "aside_text", "algorithm"}
TITLE_LABELS = {"doc_title", "paragraph_title"}
CAPTION_LABELS = {"figure_title", "table_title", "chart_title"}
FIGURE_LABELS = {"image", "chart", "seal", "header_image", "footer_image"}
FIGURE_NAMES = {"image": "immagine", "chart": "grafico", "seal": "timbro",
                "header_image": "immagine", "footer_image": "immagine"}
PAGE_FURNITURE = {"header", "footer", "number", "header_image", "footer_image"}

BULLET_RE = re.compile(r"^\s*([•●▪■◦·∙○‣⁃–—\-\*])\s*")
NUMBERED_RE = re.compile(r"^\s*(\(?\d{1,3}[\.\)]|\(?[a-zA-Z][\)])\s+")
CJK_RE = re.compile(r"[　-〿一-鿿＀-￯]")


@dataclass
class Line:
    box: list[float]
    text: str
    score: float = 1.0
    # box [x1, y1, x2, y2] di ogni carattere di `text` (coordinate immagine), se noti
    char_boxes: list[list[float]] | None = None


@dataclass
class PageResult:
    markdown: str
    blocks: list[Block]
    timings: dict[str, float] = field(default_factory=dict)
    lines: list[Line] = field(default_factory=list)   # righe lette, in ordine di lettura
    angle: float = 0.0                                 # raddrizzamento applicato (gradi)
    image: np.ndarray | None = None                    # immagine su cui valgono i box


class DocPipeline:
    def __init__(
        self,
        models_dir: Path | str = DEFAULT_MODELS_DIR,
        preset: str = "balanced",
        layout_model: str | None = None,
        det_model: str | None = None,
        rec_model: str | None = None,
        table_model: str = "SLANet_plus",
        formula_model: str = "PP-FormulaNet_plus-S",
        layout_threshold: float | None = None,
        det_limit_side_len: int = 1280,
        min_rec_score: float = 0.5,
        keep_page_furniture: bool = False,
        threads: int | None = None,
        read_figure_text: bool = False,
        figure_markers: bool = True,
    ):
        d = Path(models_dir)
        p = PRESETS[preset]
        self.layout = LayoutDetector(d / (layout_model or p["layout"]), layout_threshold, threads)
        # Layout di riserva per le pagine che il modello veloce non capisce (vedi process_page).
        fallback = d / PRESETS["accurate"]["layout"]
        self._fallback_dir = fallback if fallback != d / (layout_model or p["layout"]) and fallback.exists() else None
        self._fallback: LayoutDetector | None = None
        self.layout_threshold = layout_threshold
        self.det = TextDetector(d / (det_model or p["det"]), det_limit_side_len, threads)
        self.rec = TextRecognizer(d / (rec_model or p["rec"]), batch_size=4, threads=threads)
        self.table_dir, self.formula_dir = d / table_model, d / formula_model
        self.formulas_enabled = (self.formula_dir / "inference.onnx").exists()
        self.read_figure_text = read_figure_text
        self.figure_markers = figure_markers
        self._table: TableRecognizer | None = None
        self._formula: FormulaRecognizer | None = None
        self.threads = threads
        self.min_rec_score = min_rec_score
        self.keep_page_furniture = keep_page_furniture

    # I rami pesanti si caricano solo quando la pagina contiene davvero tabelle o formule.
    @property
    def table(self) -> TableRecognizer:
        if self._table is None:
            self._table = TableRecognizer(self.table_dir, self.threads)
        return self._table

    @property
    def formula(self):
        if self._formula is None:
            from .formula import FormulaRecognizer  # PIL + tokenizers solo se servono
            self._formula = FormulaRecognizer(self.formula_dir, threads=self.threads)
        return self._formula

    # ------------------------------------------------------------------ API

    def convert(self, path: str | Path, pages: list[int] | None = None, dpi: int = 150,
                assets_dir: Path | None = None, verbose: bool = False) -> str:
        parts = []
        for page_no, img in iter_pages(Path(path), pages, dpi):
            res = self.process_page(img, page_no, assets_dir)
            if verbose:
                t = " ".join(f"{k}={v * 1000:.0f}ms" for k, v in res.timings.items())
                print(f"[pagina {page_no + 1}] {len(res.blocks)} blocchi | {t}")
            parts.append(res.markdown)
        return "\n\n---\n\n".join(p for p in parts if p).strip() + "\n"

    def process_page(self, img: np.ndarray, page_no: int = 0, assets_dir: Path | None = None,
                     deskew: bool = True) -> PageResult:
        timings: dict[str, float] = {}
        t0 = time.perf_counter()

        # 1. Router: layout analysis. Se il modello veloce vede una "figura" grande più di mezza
        #    pagina (giornali, scontrini) di solito ha sbagliato: si riprova con il layout accurato
        #    e lo si usa se trova immagini vere (giornale con foto). Se non ne trova (scontrino),
        #    resta il layout veloce e la "figura" piena di testo viene letta riga per riga più sotto.
        blocks = self.layout(img)
        timings["layout"] = time.perf_counter() - t0
        page_area = img.shape[0] * img.shape[1]
        if self._fallback_dir and any(b.label in FIGURE_LABELS and box_area(b.box) > 0.5 * page_area
                                      for b in blocks):
            if self._fallback is None:
                self._fallback = LayoutDetector(self._fallback_dir, self.layout_threshold, self.threads)
            alt = self._fallback(img)
            if any(b.label in FIGURE_LABELS and box_area(b.box) <= 0.5 * page_area for b in alt):
                blocks = alt
            timings["layout_L"] = time.perf_counter() - t0 - timings["layout"]
        if not self.keep_page_furniture:
            blocks = [b for b in blocks if b.label not in PAGE_FURNITURE]
        if not self.formulas_enabled:  # niente modello per il LaTeX: le formule si leggono come testo
            for b in blocks:
                if b.label in ("formula", "formula_number"):
                    b.label = "text"
        t1 = time.perf_counter()

        formulas = [b for b in blocks if b.label == "formula"]
        inline = [f for f in formulas if self._host_block(f, blocks) is not None]

        # 2. Ramo testo: det sull'intera pagina con le formule mascherate, poi rec solo
        #    sulle righe che cadono in blocchi testuali o tabelle.
        ocr_img = img.copy()
        for f in formulas:
            x1, y1, x2, y2 = map(int, f.box)
            ocr_img[y1:y2, x1:x2] = 255
        quads = self.det(ocr_img)
        t2 = time.perf_counter()
        timings["det"] = t2 - t1

        # Pagina storta (scansione/foto): si raddrizza e si rifà layout + det una sola volta.
        angle = estimate_skew(quads)
        if deskew and abs(angle) > 0.3:
            first_pass = time.perf_counter() - t0
            res = self.process_page(rotate(img, angle), page_no, assets_dir, deskew=False)
            res.angle = angle
            res.timings = {f"deskew({angle:.1f}°)": first_pass, **res.timings}
            res.timings["total"] = time.perf_counter() - t0
            return res
        quads = split_around(quads, inline)

        # Una "figura" grande più di mezza pagina e piena di righe di testo è in realtà testo
        # (le immagini vere, anche screenshot con testo, sono più piccole e restano immagini).
        for b in blocks:
            if b.label in FIGURE_LABELS and box_area(b.box) > 0.5 * page_area:
                if sum(overlap_ratio(quad_to_box(q), b.box) > 0.5 for q in quads) >= 8:
                    b.label = "text_region"

        targets = [b for b in blocks if b.label not in FIGURE_LABELS and b.label != "formula"]
        to_read: list[tuple[np.ndarray, Block | None]] = []
        for q in quads:
            box = quad_to_box(q)
            owner = _best_owner(box, targets)
            if (owner is None and not self.read_figure_text
                    and any(overlap_ratio(box, b.box) > 0.5 for b in blocks)):
                continue  # riga dentro una figura/formula: non serve leggerla
            to_read.append((q, owner))
        crops = [crop_quad(ocr_img, q) for q, _ in to_read]
        texts = self.rec(crops, positions=True)
        orphans = []
        for (q, owner), crop_img, (text, score, spans) in zip(to_read, crops, texts):
            if score < self.min_rec_score or not text.strip():
                continue
            line = Line(quad_to_box(q), text.strip(), score,
                        _char_boxes(q, crop_img, text, spans))
            if owner is None:
                orphans.append(Block("text", score, line.box, children=[line]))
            else:
                owner.children.append(line)
        blocks += orphans
        blocks = promote_borderless_tables(blocks, orphans)
        t3 = time.perf_counter()
        timings["rec"] = t3 - t2

        # 3. Ramo formule (solo se presenti)
        if formulas:
            latex = self.formula([crop(img, f.box, pad=2) for f in formulas])
            for f, tex in zip(formulas, latex):
                f.content = tex
            placed = []
            for f in inline:
                host = self._host_block(f, blocks)
                if f.content and host is not None:
                    host.children.append(Line(f.box, f"${f.content}$"))
                    placed.append(f)
            blocks = [b for b in blocks if not any(b is f for f in placed)]
        t4 = time.perf_counter()
        timings["formula"] = t4 - t3

        # 4. Ramo tabelle (solo se presenti)
        for b in blocks:
            if b.label == "table":
                ox, oy = b.box[0], b.box[1]
                lines = [([l.box[0] - ox, l.box[1] - oy, l.box[2] - ox, l.box[3] - oy], l.text)
                         for l in b.children]
                b.content = html_table_to_markdown(self.table.to_html(crop(img, b.box), lines))
        t5 = time.perf_counter()
        timings["table"] = t5 - t4

        # 5. Reading order + Markdown
        order = xycut_order([b.box for b in blocks])
        blocks = [blocks[i] for i in order]
        md = self._render(blocks, img, page_no, assets_dir)
        timings["total"] = time.perf_counter() - t0
        return PageResult(md, blocks, timings, lines=_reading_lines(blocks), image=img)

    # ------------------------------------------------------------ helpers

    @staticmethod
    def _host_block(formula: Block, blocks: list[Block]) -> Block | None:
        """Una formula contenuta in un blocco di testo è inline (va resa come $...$)."""
        for b in blocks:
            if (b.label in TEXT_LABELS | TITLE_LABELS and box_area(b.box) > box_area(formula.box)
                    and overlap_ratio(formula.box, b.box) > 0.8):
                return b
        return None

    def _render(self, blocks: list[Block], img, page_no: int, assets_dir: Path | None) -> str:
        out: list[str] = []
        fig_no: dict[str, int] = {}
        captions = _captions(blocks, img.shape[0])
        open_para: int | None = None  # ultimo paragrafo rimasto a metà frase (fine colonna)
        for b in blocks:
            if b.label not in FIGURE_LABELS | CAPTION_LABELS | TEXT_LABELS:
                open_para = None
            if b.label == "formula":
                if b.content:
                    out.append(f"$$\n{b.content}\n$$")
            elif b.label == "formula_number":
                text = " ".join(_rows(b.children)).strip("() ")
                if text and out and out[-1].startswith("$$"):
                    tag = re.search(r"\\tag\{([^}]*)\}\n\$\$$", out[-1])
                    if tag:  # più numeri sullo stesso blocco: un solo \tag
                        out[-1] = out[-1][:tag.start(1)] + f"{tag.group(1)},{text}" + out[-1][tag.end(1):]
                    else:
                        out[-1] = out[-1][:-3] + f"\\tag{{{text}}}\n$$"
                elif text:
                    out.append(f"({text})")
            elif b.label == "table":
                if b.content:
                    out.append(b.content)
            elif b.label in FIGURE_LABELS:
                kind = FIGURE_NAMES.get(b.label, "immagine")
                fig_no[kind] = fig_no.get(kind, 0) + 1
                alt = captions.get(id(b)) or f"{kind} {fig_no[kind]}"
                alt = re.sub(r"[\[\]\n]", " ", alt)[:150].strip()
                if assets_dir is not None:
                    assets_dir.mkdir(parents=True, exist_ok=True)
                    name = f"pagina{page_no + 1}_{kind}{fig_no[kind]}.png"
                    ok, buf = cv2.imencode(".png", crop(img, b.box))
                    if ok:
                        buf.tofile(str(assets_dir / name))  # funziona anche con percorsi accentati
                    out.append(f"![{alt}]({quote(assets_dir.name)}/{name})")
                elif self.figure_markers:
                    out.append(f"<!-- {kind}: {alt} -->")
            elif b.label in TITLE_LABELS:
                text = " ".join(_rows(b.children))
                if text:
                    out.append(("# " if b.label == "doc_title" else "## ") + text)
            elif b.label in CAPTION_LABELS:
                text = _join_rows(_rows(b.children))
                if text:
                    out.append(f"*{text}*")
            elif b.label == "text_region":  # scontrini, moduli: si conservano le righe
                rows = _rows(b.children)
                if rows:
                    out.append("  \n".join(rows))
            else:
                text = _paragraph(_rows(b.children))
                if not text:
                    continue
                plain = "\n" not in text and not BULLET_RE.match(text) and not NUMBERED_RE.match(text)
                if open_para is not None and plain and text[:1].islower():
                    # la frase continua dalla colonna precedente (anche oltre un'immagine)
                    out[open_para] = _join_rows([out[open_para], text])
                    idx = open_para
                else:
                    out.append(text)
                    idx = len(out) - 1
                unfinished = plain and not re.search(r"[.!?:;»\"”)\]]$", out[idx])
                open_para = idx if unfinished else None
        return "\n\n".join(out)


def _captions(blocks: list[Block], page_h: int) -> dict[int, str]:
    """Abbina ogni didascalia alla figura più vicina sopra o sotto (testo alternativo dell'immagine)."""
    figures = [b for b in blocks if b.label in FIGURE_LABELS]
    result: dict[int, str] = {}
    for c in blocks:
        if c.label not in CAPTION_LABELS or not c.children:
            continue
        best, best_d = None, 0.03 * page_h
        for f in figures:
            x_overlap = min(c.box[2], f.box[2]) - max(c.box[0], f.box[0])
            d = min(abs(c.box[1] - f.box[3]), abs(f.box[1] - c.box[3]))
            if x_overlap >= 0.5 * (c.box[2] - c.box[0]) and d < best_d:
                best, best_d = f, d
        if best is not None and id(best) not in result:
            result[id(best)] = _join_rows(_rows(c.children))
    return result


def _char_boxes(quad: np.ndarray, crop_img: np.ndarray, text: str,
                spans: list[tuple[float, float]]) -> list[list[float]] | None:
    """Box immagine di ogni carattere di `text.strip()`. Gli intervalli `spans` sono lungo
    la larghezza del ritaglio prospettico della riga: si riportano sul quadrilatero
    interpolando fra il lato alto (q0->q1) e il lato basso (q3->q2)."""
    crop_w = crop_img.shape[1]
    if not spans or len(spans) != len(text) or crop_w <= 0:
        return None
    lead = len(text) - len(text.lstrip())
    spans = spans[lead:lead + len(text.strip())]
    q = quad.astype(np.float32)
    out = []
    for x0, x1 in spans:
        a, b = x0 / crop_w, x1 / crop_w
        pts = np.array([q[0] + (q[1] - q[0]) * a, q[0] + (q[1] - q[0]) * b,
                        q[3] + (q[2] - q[3]) * a, q[3] + (q[2] - q[3]) * b])
        out.append([float(pts[:, 0].min()), float(pts[:, 1].min()),
                    float(pts[:, 0].max()), float(pts[:, 1].max())])
    return out


def _reading_lines(blocks: list[Block]) -> list[Line]:
    """Tutte le righe OCR dei blocchi, in ordine di lettura (blocchi gia' ordinati con
    XY-cut, dentro il blocco per riga visiva e da sinistra a destra). Le formule inline
    ($...$) non sono testo letto dall'OCR e restano fuori."""
    out: list[Line] = []
    for b in blocks:
        lines = [ln for ln in b.children if isinstance(ln, Line) and not (
            ln.char_boxes is None and len(ln.text) > 1
            and ln.text.startswith("$") and ln.text.endswith("$"))]
        if lines:
            for row in _visual_rows(lines):
                out.extend(row)
    return out


def _best_owner(box, blocks: list[Block], min_overlap: float = 0.5) -> Block | None:
    best, best_r = None, min_overlap
    for b in blocks:
        r = overlap_ratio(box, b.box)
        if r > best_r:
            best, best_r = b, r
    return best


def _rows(lines: list[Line]) -> list[str]:
    """Raggruppa righe OCR e formule inline in righe visive, ordinate da sinistra a destra."""
    if not lines:
        return []
    lines = sorted(lines, key=lambda l: (l.box[1] + l.box[3]) / 2)
    rows: list[list[Line]] = []
    for l in lines:
        cy = (l.box[1] + l.box[3]) / 2
        if rows:
            ref = rows[-1]
            top = min(r.box[1] for r in ref)
            bottom = max(r.box[3] for r in ref)
            if top <= cy <= bottom:
                ref.append(l)
                continue
        rows.append([l])
    return [" ".join(l.text for l in sorted(r, key=lambda l: l.box[0])) for r in rows]


def promote_borderless_tables(blocks: list[Block], orphans: list[Block]) -> list[Block]:
    """Tabelle senza bordi che il layout ha preso per testo (o non ha visto): dentro un blocco
    di testo, o fra le righe orfane (fuori da ogni blocco), almeno 3 righe visive consecutive
    con 2+ celle corte, staccate da spazi larghi e allineate in colonna diventano un blocco `table`.
    Si guarda un blocco alla volta, così due colonne di testo affiancate non sembrano una tabella."""
    orphan_ids = {id(b) for b in orphans}
    groups: list[tuple[Block | None, list[Line]]] = [
        (b, b.children) for b in blocks
        if b.label in TEXT_LABELS and id(b) not in orphan_ids and len(b.children) >= 6
    ]
    if len(orphans) >= 6:
        groups.append((None, [b.children[0] for b in orphans]))

    for owner, lines in groups:
        run = _table_run(lines)
        if not run:
            continue
        y1 = min(l.box[1] for l in run)
        y2 = max(l.box[3] for l in run)
        in_table = [l for l in lines if y1 <= (l.box[1] + l.box[3]) / 2 <= y2]
        table = Block("table", 1.0, _union([l.box for l in in_table]), children=in_table)
        if owner is None:
            moved = {id(l) for l in in_table}
            blocks = [b for b in blocks if not (id(b) in orphan_ids and id(b.children[0]) in moved)]
            blocks.append(table)
        else:  # il blocco di testo si divide in: sopra / tabella / sotto
            above = [l for l in lines if (l.box[1] + l.box[3]) / 2 < y1]
            below = [l for l in lines if (l.box[1] + l.box[3]) / 2 > y2]
            blocks = [b for b in blocks if b is not owner] + [table] + [
                Block(owner.label, owner.score, _union([l.box for l in part]), children=part)
                for part in (above, below) if part
            ]
    return blocks


def _table_run(lines: list[Line]) -> list[Line]:
    if not lines:
        return []
    h = float(np.median([l.box[3] - l.box[1] for l in lines]))
    width = max(l.box[2] for l in lines) - min(l.box[0] for l in lines)
    rows = _visual_rows(lines)

    def tabular(row: list[Line]) -> bool:
        if len(row) < 2 or any(l.box[2] - l.box[0] > 0.45 * width for l in row):
            return False
        return all(b.box[0] - a.box[2] > 1.5 * h for a, b in zip(row, row[1:]))

    def aligned(r1: list[Line], r2: list[Line]) -> int:
        """Quante celle di r2 stanno sotto una cella di r1 (stessa colonna)."""
        return sum(any(min(a.box[2], b.box[2]) - max(a.box[0], b.box[0]) > -h for a in r1) for b in r2)

    best: list[list[Line]] = []
    cur: list[list[Line]] = []
    for row in rows:
        if tabular(row) and (not cur or (row[0].box[1] - cur[-1][0].box[3] < 2.5 * h
                                         and aligned(cur[-1], row) >= 2)):
            cur.append(row)
        else:
            cur = [row] if tabular(row) else []
        if len(cur) > len(best):
            best = list(cur)
    return [l for r in best for l in r] if len(best) >= 3 else []


def _visual_rows(lines: list[Line]) -> list[list[Line]]:
    rows: list[list[Line]] = []
    for l in sorted(lines, key=lambda l: (l.box[1] + l.box[3]) / 2):
        cy = (l.box[1] + l.box[3]) / 2
        if rows and any(r.box[1] <= cy <= r.box[3] for r in rows[-1]):
            rows[-1].append(l)
        else:
            rows.append([l])
    return [sorted(r, key=lambda l: l.box[0]) for r in rows]


def _union(boxes) -> list[float]:
    return [min(b[0] for b in boxes), min(b[1] for b in boxes),
            max(b[2] for b in boxes), max(b[3] for b in boxes)]


def estimate_skew(quads: list[np.ndarray]) -> float:
    """Inclinazione mediana (gradi) delle righe di testo lunghe; 0 se non affidabile."""
    angles = []
    for q in quads:
        dx, dy = q[1] - q[0]
        h = np.linalg.norm(q[3] - q[0])
        if dx > 0 and np.hypot(dx, dy) > 4 * max(h, 1):
            angles.append(np.degrees(np.arctan2(dy, dx)))
    if len(angles) < 3:
        return 0.0
    a = float(np.median(angles))
    return a if abs(a) < 15 else 0.0


def rotate(img: np.ndarray, angle: float) -> np.ndarray:
    """Ruota di `angle` gradi (raddrizza righe inclinate di `angle`), allargando la tela."""
    h, w = img.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    cos, sin = abs(m[0, 0]), abs(m[0, 1])
    nw, nh = int(h * sin + w * cos), int(h * cos + w * sin)
    m[0, 2] += nw / 2 - w / 2
    m[1, 2] += nh / 2 - h / 2
    return cv2.warpAffine(img, m, (nw, nh), flags=cv2.INTER_CUBIC, borderValue=(255, 255, 255))


def split_around(quads: list[np.ndarray], inline: list[Block]) -> list[np.ndarray]:
    """Il detector tende a unire il testo ai due lati di una formula inline mascherata:
    si spezza la riga attorno alla formula, così l'ordine testo/formula resta corretto."""
    if not inline:
        return quads
    out = []
    for q in quads:
        x1, y1, x2, y2 = quad_to_box(q)
        cuts = sorted(
            (f.box for f in inline
             if f.box[0] > x1 and f.box[2] < x2
             and min(y2, f.box[3]) - max(y1, f.box[1]) > 0.5 * (y2 - y1)),
            key=lambda b: b[0],
        )
        if not cuts:
            out.append(q)
            continue
        start = x1
        for c in cuts + [[x2, 0, x2, 0]]:
            if c[0] - start > 0.5 * (y2 - y1):
                out.append(np.array([[start, y1], [c[0], y1], [c[0], y2], [start, y2]], dtype=np.float32))
            start = max(start, c[2])
    return out


def _join_rows(rows: list[str]) -> str:
    text = ""
    for r in rows:
        if not text:
            text = r
        elif re.search(r"\w-$", text) and r[:1].islower():
            text = text[:-1] + r  # parola spezzata dal trattino di fine riga
        elif CJK_RE.match(text[-1]) and CJK_RE.match(r[0]):
            text += r
        else:
            text += " " + r
    return text


def _paragraph(rows: list[str]) -> str:
    """Testo normale, oppure lista Markdown se le righe iniziano con punti elenco/numeri."""
    if not rows:
        return ""
    # In italiano il trattino d'inciso ("– la verifica…") può finire a inizio riga: i trattini
    # valgono come elenco solo se almeno due righe del blocco iniziano così.
    strong = sum(bool(BULLET_RE.match(r)) and r.lstrip()[0] not in "-–—" for r in rows)
    dashes = sum(r.lstrip()[:1] in ("-", "–", "—") for r in rows)
    numbered = sum(bool(NUMBERED_RE.match(r)) for r in rows)
    if not (strong or dashes >= 2 or numbered):
        return _join_rows(rows)
    items: list[list[str]] = []
    for r in rows:
        if BULLET_RE.match(r):
            items.append([BULLET_RE.sub("- ", r, count=1)])
        elif NUMBERED_RE.match(r):
            items.append([r.strip()])
        elif items:
            items[-1].append(r)
        else:
            items.append([r])
    return "\n".join(_join_rows(it) for it in items)


def iter_pages(path: Path, pages: list[int] | None, dpi: int):
    """Restituisce (indice_pagina, immagine BGR) per PDF o immagini."""
    if path.suffix.lower() == ".pdf":
        import pymupdf as fitz

        with fitz.open(path) as doc:
            for i in pages if pages is not None else range(doc.page_count):
                if not 0 <= i < doc.page_count:
                    continue
                pix = doc[i].get_pixmap(dpi=dpi, alpha=False)
                img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
                yield i, cv2.cvtColor(img, cv2.COLOR_RGB2BGR if pix.n == 3 else cv2.COLOR_GRAY2BGR)
    else:
        img = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Immagine non leggibile: {path}")
        yield 0, img
