"""Blocco 2: ramo veloce per il testo. PP-OCRv6 detection (DB) + recognition (CTC) in ONNX."""

from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np
import pyclipper

from .common import load_model


class TextDetector:
    """Detector DB: restituisce i quadrilateri delle righe di testo in coordinate immagine."""

    def __init__(self, model_dir: Path, limit_side_len: int = 1600, threads: int | None = None):
        self.sess, cfg = load_model(model_dir, threads)
        post = cfg.get("PostProcess", {})
        self.thresh = post.get("thresh", 0.3)
        self.box_thresh = post.get("box_thresh", 0.6)
        self.unclip_ratio = post.get("unclip_ratio", 1.5)
        self.max_candidates = post.get("max_candidates", 1000)
        self.limit_side_len = limit_side_len
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __call__(self, img_bgr: np.ndarray) -> list[np.ndarray]:
        src_h, src_w = img_bgr.shape[:2]
        ratio = min(1.0, self.limit_side_len / max(src_h, src_w))
        rh = max(32, int(round(src_h * ratio / 32) * 32))
        rw = max(32, int(round(src_w * ratio / 32) * 32))
        x = cv2.resize(img_bgr, (rw, rh))
        x = (x.astype(np.float32) / 255.0 - self.mean) / self.std
        x = x.transpose(2, 0, 1)[None]
        pred = self.sess.run(None, {self.sess.get_inputs()[0].name: x})[0][0, 0]
        boxes = self._boxes_from_bitmap(pred, src_w, src_h)
        return sort_quads(boxes)

    def _boxes_from_bitmap(self, pred: np.ndarray, dest_w: int, dest_h: int) -> list[np.ndarray]:
        h, w = pred.shape
        bitmap = (pred > self.thresh).astype(np.uint8) * 255
        contours, _ = cv2.findContours(bitmap, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        boxes = []
        for contour in contours[: self.max_candidates]:
            points, sside = _mini_box(contour)
            if sside < 3:
                continue
            if _box_score(pred, points) < self.box_thresh:
                continue
            expanded = _unclip(points, self.unclip_ratio)
            if expanded is None:
                continue
            box, sside = _mini_box(expanded.reshape(-1, 1, 2))
            if sside < 5:
                continue
            box[:, 0] = np.clip(np.round(box[:, 0] / w * dest_w), 0, dest_w)
            box[:, 1] = np.clip(np.round(box[:, 1] / h * dest_h), 0, dest_h)
            boxes.append(box.astype(np.float32))
        return boxes


def _mini_box(contour) -> tuple[np.ndarray, float]:
    rect = cv2.minAreaRect(contour)
    pts = sorted(cv2.boxPoints(rect).tolist(), key=lambda p: p[0])
    left = sorted(pts[:2], key=lambda p: p[1])
    right = sorted(pts[2:], key=lambda p: p[1])
    box = np.array([left[0], right[0], right[1], left[1]], dtype=np.float32)  # tl, tr, br, bl
    return box, min(rect[1])


def _box_score(bitmap: np.ndarray, box: np.ndarray) -> float:
    h, w = bitmap.shape
    xmin = int(np.clip(np.floor(box[:, 0].min()), 0, w - 1))
    xmax = int(np.clip(np.ceil(box[:, 0].max()), 0, w - 1))
    ymin = int(np.clip(np.floor(box[:, 1].min()), 0, h - 1))
    ymax = int(np.clip(np.ceil(box[:, 1].max()), 0, h - 1))
    mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)
    shifted = box.copy()
    shifted[:, 0] -= xmin
    shifted[:, 1] -= ymin
    cv2.fillPoly(mask, shifted.reshape(1, -1, 2).astype(np.int32), 1)
    return cv2.mean(bitmap[ymin:ymax + 1, xmin:xmax + 1], mask)[0]


def _unclip(box: np.ndarray, ratio: float) -> np.ndarray | None:
    area = cv2.contourArea(box)
    length = cv2.arcLength(box, True)
    if length == 0:
        return None
    offset = pyclipper.PyclipperOffset()
    offset.AddPath(box.astype(np.int64).tolist(), pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
    expanded = offset.Execute(area * ratio / length)
    if not expanded:
        return None
    return np.array(expanded[0], dtype=np.float32)


def sort_quads(quads: list[np.ndarray], y_tol: float = 10) -> list[np.ndarray]:
    """Ordina dall'alto in basso e, sulla stessa riga, da sinistra a destra."""
    quads = sorted(quads, key=lambda q: (q[0, 1], q[0, 0]))
    for i in range(len(quads) - 1):
        for j in range(i, -1, -1):
            if abs(quads[j + 1][0, 1] - quads[j][0, 1]) < y_tol and quads[j + 1][0, 0] < quads[j][0, 0]:
                quads[j], quads[j + 1] = quads[j + 1], quads[j]
            else:
                break
    return quads


def crop_quad(img: np.ndarray, quad: np.ndarray) -> np.ndarray:
    """Ritaglio con correzione prospettica; le righe verticali vengono ruotate."""
    w = int(max(np.linalg.norm(quad[0] - quad[1]), np.linalg.norm(quad[2] - quad[3])))
    h = int(max(np.linalg.norm(quad[0] - quad[3]), np.linalg.norm(quad[1] - quad[2])))
    w, h = max(w, 1), max(h, 1)
    dst = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
    m = cv2.getPerspectiveTransform(quad.astype(np.float32), dst)
    out = cv2.warpPerspective(img, m, (w, h), borderMode=cv2.BORDER_REPLICATE, flags=cv2.INTER_CUBIC)
    if h / w >= 1.5:
        out = np.rot90(out)
    return out


class TextRecognizer:
    """Recognizer SVTR/CTC: decodifica greedy non autoregressiva, deterministica."""

    def __init__(self, model_dir: Path, batch_size: int = 16, threads: int | None = None):
        self.sess, cfg = load_model(model_dir, threads)
        chars = cfg["PostProcess"]["character_dict"]
        self.chars = ["<blank>"] + list(chars) + [" "]
        shape = next(
            op["RecResizeImg"]["image_shape"]
            for op in cfg["PreProcess"]["transform_ops"] if "RecResizeImg" in op
        )
        _, self.img_h, self.img_w = shape
        self.batch_size = batch_size

    def __call__(self, crops: list[np.ndarray], positions: bool = False) -> list[tuple]:
        """(testo, score) per ritaglio. Con `positions=True` anche gli intervalli orizzontali
        [(x0, x1), ...] di ogni carattere, in pixel del ritaglio: vengono dai passi temporali
        CTC in cui il carattere è emesso (rizzo-pii li usa per coprire solo la PII, non la riga)."""
        results: list[tuple] = [("", 0.0, []) if positions else ("", 0.0)] * len(crops)
        order = np.argsort([c.shape[1] / max(c.shape[0], 1) for c in crops])
        for start in range(0, len(crops), self.batch_size):
            idx = order[start:start + self.batch_size]
            max_ratio = max(self.img_w / self.img_h,
                            *(crops[i].shape[1] / max(crops[i].shape[0], 1) for i in idx))
            batch_w = int(math.ceil(self.img_h * max_ratio))
            batch = np.stack([self._norm(crops[i], batch_w) for i in idx])
            probs = self.sess.run(None, {self.sess.get_inputs()[0].name: batch})[0]
            for k, i in enumerate(idx):
                if positions:
                    text, score, steps = self._ctc_decode(probs[k], with_steps=True)
                    results[i] = (text, score, _char_spans(steps, probs.shape[1], batch_w,
                                                           crops[i], self.img_h))
                else:
                    results[i] = self._ctc_decode(probs[k])
        return results

    def _norm(self, img: np.ndarray, batch_w: int) -> np.ndarray:
        h, w = img.shape[:2]
        rw = min(batch_w, int(math.ceil(self.img_h * w / max(h, 1))))
        x = cv2.resize(img, (max(rw, 1), self.img_h)).astype(np.float32)
        x = (x / 255.0 - 0.5) / 0.5
        out = np.zeros((3, self.img_h, batch_w), dtype=np.float32)
        out[:, :, :x.shape[1]] = x.transpose(2, 0, 1)
        return out

    def _ctc_decode(self, prob: np.ndarray, with_steps: bool = False):
        idx = prob.argmax(axis=1)
        conf = prob.max(axis=1)
        keep = np.ones(len(idx), dtype=bool)
        keep[1:] = idx[1:] != idx[:-1]
        keep &= idx != 0
        steps = [int(t) for t in np.nonzero(keep)[0] if idx[t] < len(self.chars)]
        text = "".join(self.chars[idx[t]] for t in steps)
        score = float(conf[keep].mean()) if keep.any() else 0.0
        if with_steps:
            return text, score, steps
        return text, score


def _char_spans(steps: list[int], n_steps: int, batch_w: int, crop: np.ndarray,
                img_h: int) -> list[tuple[float, float]]:
    """Passi temporali CTC -> intervalli [x0, x1] dei caratteri nel ritaglio originale.

    Il ritaglio è stato ridimensionato a larghezza `rw` dentro un tensore largo `batch_w`
    (il resto è padding): il passo t copre la colonna (t + 0.5) * batch_w / n_steps del
    tensore, che si riporta al ritaglio con il fattore w / rw. I confini fra due caratteri
    stanno a metà strada fra i loro centri; il primo e l'ultimo si allargano di mezzo passo
    medio. Per le righe verticali (ritaglio ruotato) non c'è un asse affidabile: lista vuota,
    e chi la usa ricade sul box dell'intera riga."""
    h, w = crop.shape[:2]
    if not steps or h / max(w, 1) >= 1.5:
        return []
    rw = min(batch_w, int(math.ceil(img_h * w / max(h, 1))))
    scale = (batch_w / n_steps) * (w / max(rw, 1))
    centers = [(t + 0.5) * scale for t in steps]
    if len(centers) > 1:
        pitch = (centers[-1] - centers[0]) / (len(centers) - 1)
    else:
        pitch = min(float(w), 2 * scale)
    bounds = [centers[0] - pitch / 2]
    bounds += [(a + b) / 2 for a, b in zip(centers, centers[1:])]
    bounds.append(centers[-1] + pitch / 2)
    bounds = [float(min(max(b, 0.0), w)) for b in bounds]
    return list(zip(bounds[:-1], bounds[1:]))
