"""Blocco 4: ramo formule. PP-FormulaNet_plus (encoder-decoder) esportato in ONNX con il loop
di decodifica autoregressivo incluso nel grafo: in uscita ci sono già gli id dei token LaTeX."""

from __future__ import annotations

import json
import re
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps
from tokenizers import Tokenizer

from .common import load_model

EOS_ID = 2


class FormulaRecognizer:
    def __init__(self, model_dir: Path, batch_size: int = 4, threads: int | None = None):
        self.sess, cfg = load_model(model_dir, threads)
        ops = {next(iter(op)): op[next(iter(op))] for op in cfg["PreProcess"]["transform_ops"]}
        self.input_h, self.input_w = ops["UniMERNetImgDecode"]["input_size"]
        tok = cfg["PostProcess"]["character_dict"]["fast_tokenizer_file"]
        self.tokenizer = Tokenizer.from_str(json.dumps(tok))
        self.batch_size = batch_size

    def __call__(self, crops_bgr: list[np.ndarray]) -> list[str]:
        out: list[str] = []
        for start in range(0, len(crops_bgr), self.batch_size):
            batch = np.stack([self._preprocess(c) for c in crops_bgr[start:start + self.batch_size]])
            ids = self.sess.run(None, {self.sess.get_inputs()[0].name: batch})[0]
            out.extend(self._decode(row) for row in ids)
        return out

    def _preprocess(self, img_bgr: np.ndarray) -> np.ndarray:
        img = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        img = _crop_margin(img)
        w, h = img.size
        short = min(self.input_h, self.input_w)
        if w <= h:
            img = img.resize((short, max(1, int(short * h / w))), resample=Image.BILINEAR)
        else:
            img = img.resize((max(1, int(short * w / h)), short), resample=Image.BILINEAR)
        img.thumbnail((self.input_w, self.input_h))
        dw, dh = self.input_w - img.width, self.input_h - img.height
        img = ImageOps.expand(img, (dw // 2, dh // 2, dw - dw // 2, dh - dh // 2))
        x = (np.asarray(img, dtype=np.float32) / 255.0 - 0.7931) / 0.1738
        gray = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)  # identico a PaddleX (input RGB)
        return gray[None]

    def _decode(self, ids: np.ndarray) -> str:
        ids = ids.tolist()
        if EOS_ID in ids:
            ids = ids[: ids.index(EOS_ID) + 1]
        text = self.tokenizer.decode(ids, skip_special_tokens=True)
        text = re.sub(r"\\text\s*{([^{}]*[\u4e00-\u9fff]+[^{}]*)}", r"\1", text).replace('"', "")
        return _normalize_latex(text)


def _crop_margin(img: Image.Image) -> Image.Image:
    data = np.array(img.convert("L")).astype(np.float32)
    lo, hi = data.min(), data.max()
    if hi == lo:
        return img
    data = (data - lo) / (hi - lo) * 255
    coords = cv2.findNonZero((data < 200).astype(np.uint8) * 255)
    if coords is None:
        return img
    x, y, w, h = cv2.boundingRect(coords)
    return img.crop((x, y, x + w, y + h))


def _normalize_latex(s: str) -> str:
    """Toglie gli spazi superflui fra token LaTeX (port di UniMERNetDecode.normalize)."""
    text_reg = r"(\\(operatorname|mathrm|text|mathbf)\s?\*? {.*?})"
    letter, noletter = r"[a-zA-Z]", r"[\W_^\d]"
    names = []
    for x in re.findall(text_reg, s):
        for m in re.findall(r"(\\[a-zA-Z]+)\s(?=\w)|\\[a-zA-Z]+\s(?=})", x[0]):
            if m not in ("\\operatorname", "\\mathrm", "\\text", "\\mathbf") and m.strip():
                s = s.replace(m, m + "XXXXXXX").replace(" ", "")
                names.append(s)
    if names:
        s = re.sub(text_reg, lambda _: str(names.pop(0)), s)
    news = s
    while True:
        s = news
        news = re.sub(r"(?!\\ )(%s)\s+?(%s)" % (noletter, noletter), r"\1\2", s)
        news = re.sub(r"(?!\\ )(%s)\s+?(%s)" % (noletter, letter), r"\1\2", news)
        news = re.sub(r"(%s)\s+?(%s)" % (letter, noletter), r"\1\2", news)
        if news == s:
            break
    s = s.replace("XXXXXXX", " ")
    # \operatorname{t r i l} -> \operatorname{tril}
    s = re.sub(r"\\(operatorname|mathrm|mathit|mathbf|text)\s*\{((?:\w ?)+)\}",
               lambda m: f"\\{m.group(1)}{{{m.group(2).replace(' ', '')}}}"
               if re.fullmatch(r"(\w )+\w", m.group(2)) else m.group(0), s)
    return s.strip()
