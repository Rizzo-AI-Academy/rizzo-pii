"""Blocco 1: il "router" visivo. Layout analysis con PP-DocLayout (ONNX)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .common import box_area, iou, load_model, overlap_ratio


@dataclass
class Block:
    label: str
    score: float
    box: list[float]  # x1, y1, x2, y2 in coordinate pagina
    content: str = ""
    children: list = field(default_factory=list)  # righe OCR / formule inline


# Soglie per classe prese da PP-StructureV3 (PaddleX); le classi non elencate usano 0.5.
# Per le formule PaddleX usa 0.3, ma con PP-DocLayout-M i box a bassa confidenza sono spesso
# testo normale che diventa LaTeX sbagliato: qui resta 0.5.
CLASS_THRESHOLDS = {"paragraph_title": 0.3, "text": 0.4, "seal": 0.45}


class LayoutDetector:
    def __init__(self, model_dir: Path, threshold: float | None = None, threads: int | None = None):
        self.sess, cfg = load_model(model_dir, threads)
        self.labels = cfg["label_list"]
        self.threshold = threshold

        resize = next(op for op in cfg["Preprocess"] if op["type"] == "Resize")
        self.target_h, self.target_w = resize["target_size"]
        norm = next(op for op in cfg["Preprocess"] if op["type"] == "NormalizeImage")
        self.mean = np.array(norm.get("mean", [0, 0, 0]), dtype=np.float32)
        self.std = np.array(norm.get("std", [1, 1, 1]), dtype=np.float32)
        self.input_names = [i.name for i in self.sess.get_inputs()]

    def __call__(self, img_bgr: np.ndarray) -> list[Block]:
        h, w = img_bgr.shape[:2]
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        x = cv2.resize(rgb, (self.target_w, self.target_h), interpolation=cv2.INTER_CUBIC)
        x = (x.astype(np.float32) / 255.0 - self.mean) / self.std
        x = x.transpose(2, 0, 1)[None]

        feeds = {
            "image": x,
            "scale_factor": np.array([[self.target_h / h, self.target_w / w]], dtype=np.float32),
            "im_shape": np.array([[self.target_h, self.target_w]], dtype=np.float32),
        }
        dets = self.sess.run(None, {k: v for k, v in feeds.items() if k in self.input_names})[0]

        blocks = []
        for cls, score, x1, y1, x2, y2 in dets:
            if cls < 0:
                continue
            label = self.labels[int(cls)]
            min_score = self.threshold if self.threshold is not None else CLASS_THRESHOLDS.get(label, 0.5)
            if score < min_score:
                continue
            box = [max(0.0, x1), max(0.0, y1), min(float(w), x2), min(float(h), y2)]
            if box[2] - box[0] < 2 or box[3] - box[1] < 2:
                continue
            blocks.append(Block(label, float(score), box))
        return self._nms(blocks)

    @staticmethod
    def _nms(blocks: list[Block], iou_same: float = 0.6, iou_diff: float = 0.95) -> list[Block]:
        """NMS "da layout": soglia bassa fra stessa classe, alta fra classi diverse
        (come layout_nms di PaddleX), poi rimuove i doppioni annidati della stessa classe."""
        blocks = sorted(blocks, key=lambda b: b.score, reverse=True)
        keep: list[Block] = []
        for b in blocks:
            if all(iou(b.box, k.box) < (iou_same if b.label == k.label else iou_diff) for k in keep):
                keep.append(b)
        result = []
        for b in keep:
            nested = any(
                k is not b and k.label == b.label and box_area(k.box) > box_area(b.box)
                and _contains(k.box, b.box)
                for k in keep
            )
            if not nested:
                result.append(b)
        return result


def _contains(outer, inner, tol: float = 0.9) -> bool:
    return overlap_ratio(inner, outer) >= tol
