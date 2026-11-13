# -*- coding: utf-8 -*-
"""
Inferenza del modello PII con ONNX Runtime, SENZA torch e senza transformers.

Serve all'app desktop: il sidecar impacchettato non deve portarsi dietro PyTorch
(centinaia di MB) solo per fare forward di un encoder. Qui bastano onnxruntime,
tokenizers (la stessa libreria Rust che sta sotto i tokenizer "fast" di HF) e numpy.

Il modello NON e' quantizzato: e' lo stesso grafo fp32 di PyTorch esportato con
src/training/export_onnx.py, che verifica anche che logits ed entita' coincidano.

`OnnxTokenClassifier` replica `pipeline("token-classification",
aggregation_strategy="simple")` di transformers 4.57:
  - tokenizzazione con truncation a model_max_length, special token esclusi;
  - softmax sui logits, argmax per token;
  - raggruppamento dei token adiacenti con lo stesso tag (un B- apre un gruppo nuovo),
    score = media degli score dei token, start/end = offset del primo/ultimo token;
  - scarto dei gruppi "O".
Chiamato con una lista di stringhe ritorna una lista di liste di dict
{"entity_group", "score", "word", "start", "end"}, come la pipeline HF: app.py e i test
(che sostituiscono `app.nlp` con un finto) non vedono la differenza.

Layout della cartella del modello (uno dei due):
  <dir>/onnx/model.onnx     (cartella di sviluppo, convenzione Hugging Face)
  <dir>/model.onnx          (dentro l'exe, dove si impacchetta solo il necessario)
piu' config.json (id2label), tokenizer.json, tokenizer_config.json (model_max_length).
"""

import json
import os

import numpy as np

ONNX_CANDIDATES = ("onnx/model.onnx", "model.onnx")


def find_onnx(model_dir):
    """Percorso del model.onnx nella cartella del modello, o None se non c'e'."""
    for rel in ONNX_CANDIDATES:
        p = os.path.join(model_dir, rel)
        if os.path.isfile(p):
            return p
    return None


def _get_tag(name):
    """Come TokenClassificationPipeline.get_tag: senza prefisso vale come I-."""
    if name.startswith("B-"):
        return "B", name[2:]
    if name.startswith("I-"):
        return "I", name[2:]
    return "I", name


class OnnxTokenClassifier:
    def __init__(self, model_dir, onnx_path=None, intra_op_threads=None):
        import onnxruntime as ort
        from tokenizers import Tokenizer

        onnx_path = onnx_path or find_onnx(model_dir)
        if not onnx_path:
            raise FileNotFoundError(f"model.onnx non trovato in {model_dir}")

        with open(os.path.join(model_dir, "config.json"), encoding="utf-8") as f:
            cfg = json.load(f)
        self.id2label = {int(k): v for k, v in cfg["id2label"].items()}

        max_len = 0
        tc_path = os.path.join(model_dir, "tokenizer_config.json")
        if os.path.isfile(tc_path):
            with open(tc_path, encoding="utf-8") as f:
                max_len = int(json.load(f).get("model_max_length") or 0)
        # tokenizer.json porta con se' la truncation del training (768): la pipeline HF la
        # sovrascrive con model_max_length, quindi qui si fa lo stesso.
        self.tokenizer = Tokenizer.from_file(os.path.join(model_dir, "tokenizer.json"))
        self.tokenizer.no_padding()
        pad = self.tokenizer.token_to_id("<pad>")
        self.pad_id = int(pad) if pad is not None else 0   # padding manuale in _predict_batch
        if 0 < max_len < 10**9:
            self.tokenizer.enable_truncation(max_len)
        else:
            self.tokenizer.no_truncation()

        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if intra_op_threads:
            so.intra_op_num_threads = int(intra_op_threads)
        self.session = ort.InferenceSession(onnx_path, sess_options=so,
                                            providers=["CPUExecutionProvider"])
        self.input_names = {i.name for i in self.session.get_inputs()}

    # ------------------------------------------------------------------ #
    def logits(self, text):
        """Logits [seq, n_labels] di un testo + l'Encoding del tokenizer."""
        enc = self.tokenizer.encode(text)
        ids = np.asarray([enc.ids], dtype=np.int64)
        feeds = {"input_ids": ids}
        if "attention_mask" in self.input_names:
            feeds["attention_mask"] = np.asarray([enc.attention_mask], dtype=np.int64)
        out = self.session.run(["logits"], feeds)[0][0]
        return out.astype(np.float32, copy=False), enc

    def _predict_one(self, text):
        logits, enc = self.logits(text)
        return self._entities(text, logits, enc)

    def _predict_batch(self, texts, batch_size):
        """Come _predict_one su ogni testo, ma con UNA session.run per gruppo di
        batch_size testi: si ordinano per lunghezza (poco padding), si allineano con
        <pad> + attention_mask a 0 e di ogni riga si tengono solo i logits dei token
        veri. Su tante stringhe brevi (le celle di un CSV) il costo fisso di ogni
        session.run domina: farne una per gruppo e' molto piu' veloce."""
        encs = self.tokenizer.encode_batch(list(texts))
        out = [None] * len(texts)
        order = sorted(range(len(texts)), key=lambda i: len(encs[i].ids))
        for k in range(0, len(order), batch_size):
            idx = order[k:k + batch_size]
            width = max(len(encs[i].ids) for i in idx)
            ids = np.full((len(idx), width), self.pad_id, dtype=np.int64)
            mask = np.zeros((len(idx), width), dtype=np.int64)
            for row, i in enumerate(idx):
                n = len(encs[i].ids)
                ids[row, :n] = encs[i].ids
                mask[row, :n] = 1
            feeds = {"input_ids": ids}
            if "attention_mask" in self.input_names:
                feeds["attention_mask"] = mask
            logits = self.session.run(["logits"], feeds)[0]
            for row, i in enumerate(idx):
                n = len(encs[i].ids)
                out[i] = self._entities(texts[i], logits[row, :n].astype(np.float32, copy=False),
                                        encs[i])
        return out

    def _entities(self, text, logits, enc):
        maxes = np.max(logits, axis=-1, keepdims=True)
        shifted = np.exp(logits - maxes)
        scores = shifted / shifted.sum(axis=-1, keepdims=True)

        tokens = []
        for idx, (special, (s, e)) in enumerate(zip(enc.special_tokens_mask, enc.offsets)):
            if special:
                continue
            k = int(scores[idx].argmax())
            tokens.append((self.id2label[k], float(scores[idx][k]), s, e))

        groups, cur = [], []
        for tok in tokens:
            if cur:
                bi, tag = _get_tag(tok[0])
                _, last_tag = _get_tag(cur[-1][0])
                if tag == last_tag and bi != "B":
                    cur.append(tok)
                    continue
                groups.append(cur)
            cur = [tok]
        if cur:
            groups.append(cur)

        ents = []
        for g in groups:
            label = g[0][0].split("-", 1)[-1]
            if label == "O":
                continue
            start, end = g[0][2], g[-1][3]
            ents.append({
                "entity_group": label,
                "score": float(np.nanmean([t[1] for t in g])),
                "word": text[start:end],
                "start": start,
                "end": end,
            })
        return ents

    def __call__(self, inputs, batch_size=None):
        """Come la pipeline HF: una stringa -> lista di entita'; una lista -> lista di
        liste. batch_size > 1 raggruppa i testi in una sola inferenza per gruppo."""
        if isinstance(inputs, str):
            return self._predict_one(inputs)
        if batch_size and batch_size > 1 and len(inputs) > 1:
            return self._predict_batch(inputs, int(batch_size))
        return [self._predict_one(t) for t in inputs]
