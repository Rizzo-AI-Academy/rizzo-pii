#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Backend d'inferenza ONNX per l'app desktop — separa l'APP dal FRAMEWORK DI TRAINING.

Fase 4 del piano di integrazione (docs/PIANO_INTEGRAZIONE.md). L'app non dipende piu' da
PyTorch/Transformers (centinaia di MB): il modello si esporta UNA TANTUM con
src/export/export_onnx.py (PR #12) e qui lo si CONSUMA con onnxruntime + tokenizers, molto
piu' leggeri.

    Training:  PyTorch + Transformers  --export-->  ONNX (fp32 + INT8)
    Desktop:   Tokenizer -> ONNX Runtime -> aggregazione BIO -> entita' con offset
                             ├── INT8  su CPU moderne
                             └── FP32  fallback

`OnnxTagger` e' un drop-in della pipeline HF "token-classification"
(aggregation_strategy="simple"): chiamato con un testo (o una lista) ritorna la stessa
struttura [{entity_group, score, word, start, end}], cosi' app.py lo usa senza altre
modifiche. In Python la libreria `tokenizers` fornisce gia' gli offset carattere
(Encoding.offsets), quindi non serve il trucco del cursore Metaspace del lato Transformers.js.

La funzione `aggregate_entities` e' PURA (solo stdlib): tutta la logica delicata
(raggruppamento BIO, offset, rifilo degli spazi) e' testabile senza un modello.
"""
import json
import os
from pathlib import Path


# --------------------------------------------------------------------------- #
# Aggregazione BIO -> entita' (funzione pura, testabile senza onnxruntime)     #
# --------------------------------------------------------------------------- #
def aggregate_entities(per_token, text):
    """Raggruppa i token etichettati in entita', come aggregation_strategy="simple".

    `per_token`: lista ordinata di dict {label, score, start, end} SENZA i token speciali
    ([CLS]/[SEP]). `label` e' in schema BIO ("B-CF", "I-CF", "O", ...). I token consecutivi
    dello stesso TIPO si fondono (ignorando B/I, come "simple"); "O" chiude il gruppo.

    Ritorna [{entity_group, score, word, start, end}] con offset carattere sul testo
    originale e gli spazi ai bordi rifilati."""
    groups, cur = [], None
    for t in per_token:
        lab = t["label"]
        typ = lab.split("-", 1)[1] if "-" in lab else (None if lab == "O" else lab)
        if typ is None:
            cur = None
            continue
        if cur is not None and typ == cur["type"]:
            cur["end"] = t["end"]
            cur["scores"].append(t["score"])
        else:
            cur = {"type": typ, "start": t["start"], "end": t["end"], "scores": [t["score"]]}
            groups.append(cur)

    # rifila spazi e separatori ai bordi: il modello a volte include la virgola dopo il
    # nome ("Mario Rossi,"), che verrebbe inghiottita nel placeholder. Non si tocca il '.'
    # (fa parte di abbreviazioni: "S.r.l.").
    trim = " \t\n\r\f\v,;:"
    out = []
    for g in groups:
        s, e = g["start"], g["end"]
        while s < e and text[s] in trim:
            s += 1
        while e > s and text[e - 1] in trim:
            e -= 1
        if s >= e:
            continue
        out.append({"entity_group": g["type"], "score": sum(g["scores"]) / len(g["scores"]),
                    "word": text[s:e], "start": s, "end": e})
    return out


# --------------------------------------------------------------------------- #
# Risoluzione del bundle ONNX                                                  #
# --------------------------------------------------------------------------- #
def resolve_bundle(models_dir=None):
    """Cartella `bundle/` da usare, oppure None se non c'e' un export ONNX.

    Precedenza: env PII_ONNX_BUNDLE > ultima models/rizzo-pii-0.3B-v*-onnx/bundle."""
    env = os.environ.get("PII_ONNX_BUNDLE")
    if env:
        p = Path(env)
        return p if p.is_dir() else None
    root = Path(models_dir) if models_dir else Path(__file__).resolve().parents[2] / "models"
    cands = sorted(root.glob("rizzo-pii-0.3B-v*-onnx/bundle")) if root.is_dir() else []
    if not cands:
        return None
    import re
    def _key(p):
        m = re.search(r"-v([0-9][0-9.]*)-onnx", str(p))
        return tuple(int(x) for x in m.group(1).split(".")) if m else ()
    return max(cands, key=_key)


def build_tagger(models_dir=None, prefer_int8=None):
    """OnnxTagger sull'ultimo bundle disponibile, o None se non c'e' (l'app fa fallback
    su PyTorch). prefer_int8=None -> da env PII_ONNX_FP32 (default: INT8 se presente)."""
    bundle = resolve_bundle(models_dir)
    if bundle is None:
        return None
    if prefer_int8 is None:
        prefer_int8 = os.environ.get("PII_ONNX_FP32", "") not in ("1", "true", "yes", "on")
    return OnnxTagger(bundle, prefer_int8=prefer_int8)


# --------------------------------------------------------------------------- #
# OnnxTagger: sessione ORT + tokenizer, drop-in della pipeline HF             #
# --------------------------------------------------------------------------- #
class OnnxTagger:
    """Inferenza token-classification su ONNX Runtime. Dipendenze: onnxruntime, tokenizers,
    numpy (nessun torch/transformers)."""

    def __init__(self, bundle_dir, prefer_int8=True, max_length=8192):
        import numpy as np
        import onnxruntime as ort
        from tokenizers import Tokenizer
        self._np = np
        bundle = Path(bundle_dir)

        self.tokenizer = Tokenizer.from_file(str(bundle / "tokenizer.json"))
        try:
            self.tokenizer.enable_truncation(max_length=max_length)
        except Exception:
            pass

        cfg = json.loads((bundle / "config.json").read_text(encoding="utf-8"))
        # id2label ha chiavi stringa nel config di transformers
        self.id2label = {int(k): v for k, v in cfg["id2label"].items()}

        onnx_dir = bundle / "onnx"
        int8 = onnx_dir / "model_quantized.onnx"
        fp32 = onnx_dir / "model.onnx"
        if prefer_int8 and int8.exists():
            model_path, self.dtype = int8, "INT8"
        elif fp32.exists():
            model_path, self.dtype = fp32, "FP32"
        elif int8.exists():
            model_path, self.dtype = int8, "INT8"
        else:
            raise FileNotFoundError(f"nessun modello ONNX in {onnx_dir}")

        self.model_name = model_path.name
        self.session = ort.InferenceSession(str(model_path),
                                            providers=["CPUExecutionProvider"])
        self._input_names = {i.name for i in self.session.get_inputs()}

    def _softmax_max(self, row):
        e = self._np.exp(row - row.max())
        p = e / e.sum()
        idx = int(p.argmax())
        return idx, float(p[idx])

    def _tag_one(self, text):
        if not text:
            return []
        enc = self.tokenizer.encode(text)
        ids = self._np.asarray([enc.ids], dtype=self._np.int64)
        mask = self._np.asarray([enc.attention_mask], dtype=self._np.int64)
        feeds = {"input_ids": ids}
        if "attention_mask" in self._input_names:
            feeds["attention_mask"] = mask
        if "token_type_ids" in self._input_names:
            feeds["token_type_ids"] = self._np.zeros_like(ids)
        logits = self.session.run(None, {k: v for k, v in feeds.items()
                                         if k in self._input_names})[0][0]

        # aggrega per PAROLA (come aggregation_strategy="simple"): la label del PRIMO
        # sub-token vale per l'intera parola, e i sub-token di continuazione ne estendono
        # solo lo span. Senza questo "RSSMRA80A01H501U" (piu' sub-token, stesso word_id)
        # si spezzerebbe in tanti frammenti.
        word_ids = enc.word_ids
        per_word, seen = [], {}
        for i, (lo, hi) in enumerate(enc.offsets):
            if enc.special_tokens_mask[i] or (lo == 0 and hi == 0):
                continue                       # [CLS]/[SEP]/pad: nessun carattere
            wid = word_ids[i]
            if wid is not None and wid in seen:
                seen[wid]["end"] = hi          # sub-token di continuazione: estendi lo span
                continue
            idx, score = self._softmax_max(logits[i])
            entry = {"label": self.id2label[idx], "score": score, "start": lo, "end": hi}
            per_word.append(entry)
            if wid is not None:
                seen[wid] = entry
        return aggregate_entities(per_word, text)

    def __call__(self, texts):
        """Un testo -> lista di entita'; una lista di testi -> lista di liste (come la
        pipeline HF, cosi' app.py resta invariato)."""
        if isinstance(texts, str):
            return self._tag_one(texts)
        return [self._tag_one(t) for t in texts]
