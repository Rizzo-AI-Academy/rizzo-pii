"""Utility condivise: sessioni ONNX Runtime, config dei modelli, geometria dei box."""

from __future__ import annotations

import os
from pathlib import Path

# Deve precedere l'import nativo: ORT 1DS puo' emettere un evento iniziale
# prima che disable_telemetry_events() sia invocabile (Privacy.md della wheel).
os.environ["ORT_DISABLE_TELEMETRY"] = "1"

import numpy as np
import onnxruntime as ort
import yaml


def create_session(model_path: Path, threads: int | None = None) -> ort.InferenceSession:
    ort.disable_telemetry_events()
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.log_severity_level = 3
    if threads:
        so.intra_op_num_threads = threads
    return ort.InferenceSession(str(model_path), so, providers=["CPUExecutionProvider"])


def load_model(model_dir: Path, threads: int | None = None) -> tuple[ort.InferenceSession, dict]:
    """Carica `inference.onnx` e la config Paddle `inference.yml` (dizionari, label, preprocess)."""
    model_dir = Path(model_dir)
    onnx_path = model_dir / "inference.onnx"
    if not onnx_path.exists():
        raise FileNotFoundError(
            f"Modello mancante: {onnx_path}. Esegui prima tools/setup_models.py"
        )
    with open(model_dir / "inference.yml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return create_session(onnx_path, threads), cfg


def box_area(b) -> float:
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def intersection(a, b) -> float:
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    return w * h if w > 0 and h > 0 else 0.0


def iou(a, b) -> float:
    inter = intersection(a, b)
    union = box_area(a) + box_area(b) - inter
    return inter / union if union > 0 else 0.0


def overlap_ratio(inner, outer) -> float:
    """Frazione dell'area di `inner` che cade dentro `outer`."""
    a = box_area(inner)
    return intersection(inner, outer) / a if a > 0 else 0.0


def quad_to_box(quad: np.ndarray) -> list[float]:
    return [float(quad[:, 0].min()), float(quad[:, 1].min()),
            float(quad[:, 0].max()), float(quad[:, 1].max())]


def crop(img: np.ndarray, box, pad: int = 0) -> np.ndarray:
    h, w = img.shape[:2]
    x1 = max(0, int(box[0]) - pad)
    y1 = max(0, int(box[1]) - pad)
    x2 = min(w, int(np.ceil(box[2])) + pad)
    y2 = min(h, int(np.ceil(box[3])) + pad)
    return img[y1:y2, x1:x2]
