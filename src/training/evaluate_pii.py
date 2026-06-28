# -*- coding: utf-8 -*-
"""
Valutazione COMPLETA del modello PII salvato, con metriche entity-level PER-TAG
(precision/recall/F1 + support) oltre all'aggregato — quello che train_pii.py non stampa.

Carica di default models/rizzo-pii-0.3B e valuta su dataset/validation/validation_real.jsonl.
Salva il report in experiments/full_run/eval_validation.{json,txt}.

Uso:
  python src/training/evaluate_pii.py
  python src/training/evaluate_pii.py --model models/pii_model_legacy --data dataset/validation/validation_real.jsonl
"""

import argparse
import io
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

ROOT = Path(__file__).resolve().parents[2]
MAX_LEN = 768
BATCH = 32

# --- stessa tassonomia di train_pii.py (i file di validation sono grezzi) ---
TAG_MAP = {
    "GIVENNAME": "FULLNAME", "SURNAME": "FULLNAME",
    "GIUDICE": "FULLNAME", "AVVOCATO": "FULLNAME", "CONVENUTO": "FULLNAME",
    "ATTORE": "FULLNAME", "TESTIMONE": "FULLNAME",
    "SEX": "GENDER", "TAXNUM": "PIVA", "PEC": "EMAIL", "RG": "DOCID",
    "IDCARDNUM": "ID_DOC", "PASSPORTNUM": "ID_DOC",
    "DRIVERLICENSENUM": "ID_DOC", "SOCIALNUM": "ID_DOC", "CONTO": "IBAN",
}
DROP_TYPES = {"TITLE", "TRIBUNAL"}


def normalize_labels(labels):
    out, prev = [], None
    for l in labels:
        typ = TAG_MAP.get(l[2:], l[2:]) if l != "O" else None
        if typ is None or typ in DROP_TYPES:
            out.append("O"); prev = None
            continue
        out.append(("I-" if typ == prev else "B-") + typ)
        prev = typ
    return out


def spans(tags):
    """Insieme di entità come (tipo, start, end) dai tag BIO word-level."""
    out, cur = [], None
    for i, t in enumerate(tags + ["O"]):
        if t == "O" or t.startswith("B-") or (cur and t[2:] != cur[0]):
            if cur:
                out.append((cur[0], cur[1], i)); cur = None
        if t.startswith("B-") or (t.startswith("I-") and cur is None):
            cur = (t[2:], i)
    return set(out)


def load_data(path):
    recs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            toks, labs = r.get("tokens"), r.get("bio_labels")
            if toks and labs and len(toks) == len(labs):
                recs.append((toks, normalize_labels(labs)))
    return recs


def resolve_model_dir(models_dir):
    """Ultima versione models/rizzo-pii-0.3B-v* (storico); fallback al vecchio non versionato, poi legacy."""
    import re
    versioned = [p for p in models_dir.glob("rizzo-pii-0.3B-v*") if p.is_dir()]
    if versioned:
        def _key(p):
            m = re.search(r"-v([0-9][0-9.]*)$", p.name)
            return tuple(int(x) for x in m.group(1).split(".")) if m else ()
        return str(max(versioned, key=_key))
    base = models_dir / "rizzo-pii-0.3B"
    return str(base if base.exists() else models_dir / "pii_model_legacy")


def main():
    ap = argparse.ArgumentParser(description="Valutazione per-tag del modello PII.")
    ap.add_argument("--model", default=resolve_model_dir(ROOT / "models"))
    ap.add_argument("--data", default=str(ROOT / "dataset" / "validation" / "validation_real.jsonl"))
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Modello: {args.model}\nDati:    {args.data}\nDevice:  {device}")
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForTokenClassification.from_pretrained(args.model).to(device).eval()
    id2label = model.config.id2label

    recs = load_data(args.data)
    print(f"Esempi di validation: {len(recs)}")

    # accumulatori entity-level: per-tag e globale
    tp = Counter(); fp = Counter(); fn = Counter(); support = Counter()
    tok_ok = tok_tot = 0
    t0 = time.time()

    for start in range(0, len(recs), BATCH):
        batch = recs[start:start + BATCH]
        toks_list = [t for t, _ in batch]
        enc = tok(toks_list, is_split_into_words=True, truncation=True,
                  max_length=MAX_LEN, padding=True, return_tensors="pt").to(device)
        with torch.no_grad():
            logits = model(**enc).logits
        preds = logits.argmax(-1).cpu().tolist()

        for bi, (toks, gold_words) in enumerate(batch):
            word_ids = enc.word_ids(batch_index=bi)
            seen = set(); g_tags = []; p_tags = []
            for pos, wid in enumerate(word_ids):
                if wid is None or wid in seen:
                    continue
                seen.add(wid)
                g = gold_words[wid]; p = id2label[preds[bi][pos]]
                g_tags.append(g); p_tags.append(p)
                tok_tot += 1; tok_ok += int(g == p)
            gs, ps = spans(g_tags), spans(p_tags)
            for typ, _, _ in gs:
                support[typ] += 1
            inter = gs & ps
            for typ, _, _ in inter:
                tp[typ] += 1
            for typ, _, _ in (ps - gs):
                fp[typ] += 1
            for typ, _, _ in (gs - ps):
                fn[typ] += 1

    def prf(t, f_p, f_n):
        p = t / (t + f_p) if t + f_p else 0.0
        r = t / (t + f_n) if t + f_n else 0.0
        f = 2 * p * r / (p + r) if p + r else 0.0
        return p, r, f

    tags = sorted(support, key=lambda x: -support[x])
    TP, FP, FN = sum(tp.values()), sum(fp.values()), sum(fn.values())
    P, R, F = prf(TP, FP, FN)

    # micro (globale) e macro (media non pesata sui tag)
    per_tag = {t: prf(tp[t], fp[t], fn[t]) for t in tags}
    macro_f = sum(v[2] for v in per_tag.values()) / len(per_tag) if per_tag else 0.0

    lines = []
    lines.append(f"VALUTAZIONE  modello={Path(args.model).name}  dati={Path(args.data).name}")
    lines.append(f"esempi={len(recs)}  token_acc={tok_ok/tok_tot:.4f}  ({time.time()-t0:.0f}s)")
    lines.append("")
    lines.append(f"{'TAG':<18}{'support':>9}{'precision':>11}{'recall':>9}{'F1':>9}")
    lines.append("-" * 56)
    for t in tags:
        p, r, f = per_tag[t]
        lines.append(f"{t:<18}{support[t]:>9}{p:>11.4f}{r:>9.4f}{f:>9.4f}")
    lines.append("-" * 56)
    lines.append(f"{'MICRO (overall)':<18}{TP+FN:>9}{P:>11.4f}{R:>9.4f}{F:>9.4f}")
    lines.append(f"{'MACRO (media tag)':<18}{'':>9}{'':>11}{'':>9}{macro_f:>9.4f}")
    report = "\n".join(lines)
    print("\n" + report)

    out_dir = ROOT / "experiments" / "full_run"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "eval_validation.txt").write_text(report + "\n", encoding="utf-8")
    summary = {
        "model": args.model, "data": args.data, "n": len(recs),
        "token_acc": tok_ok / tok_tot if tok_tot else 0.0,
        "micro": {"precision": P, "recall": R, "f1": F, "support": TP + FN},
        "macro_f1": macro_f,
        "per_tag": {t: {"support": support[t], "precision": per_tag[t][0],
                        "recall": per_tag[t][1], "f1": per_tag[t][2]} for t in tags},
    }
    (out_dir / "eval_validation.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nReport salvato in {out_dir/'eval_validation.txt'} e .json")


if __name__ == "__main__":
    main()
