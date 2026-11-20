# -*- coding: utf-8 -*-
"""
Esporta il modello PII in ONNX (fp32, NESSUNA quantizzazione) e verifica che dia gli
stessi risultati di PyTorch.

Lo scopo non e' cambiare il modello ma il runtime: l'app desktop carica il .onnx con
onnxruntime (src/app/onnx_ner.py) e il sidecar non deve piu' impacchettare torch.
I pesi sono gli stessi, in float32: l'unica differenza ammessa e' il rumore numerico
dell'ordine delle operazioni (1e-4 sui logits), che non deve cambiare nessuna entita'.

Output: <model_dir>/onnx/model.onnx + model.onnx_data (pesi; convenzione Hugging Face,
        app.py lo trova da solo). I due file vanno sempre tenuti insieme.

Verifica (sempre, a meno di --no-verify):
  1. max |logits_torch - logits_onnx| su testi di lunghezza diversa (shape dinamiche);
  2. entita' della pipeline HF (aggregation "simple") == entita' di OnnxTokenClassifier
     su N frasi di dataset/validation/validation_real.jsonl (se c'e') + esempi fissi.
Esce con codice 1 se qualcosa non torna.

Uso:
  python src/training/export_onnx.py                       # modello dell'app (v1.5.0)
  python src/training/export_onnx.py --model-dir models/rizzo-pii-0.3B-v1.5.0 -n 500
"""

import argparse
import io
import json
import os
import sys
import time
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "app"))
DEFAULT_MODEL = ROOT / "models" / "rizzo-pii-0.3B-v1.5.0"
VALIDATION = ROOT / "dataset" / "validation" / "validation_real.jsonl"

SAMPLES = [
    "Mi chiamo Mario Rossi, nato a Napoli il 12/03/1980, CF RSSMRA80C12F839K.",
    "Il sig. Giuseppe Verdi, residente in Via Roma 15, 20121 Milano (MI), versa "
    "l'importo di € 1.250,00 sull'IBAN IT60X0542811101000000123456.",
    "Contattare l'avv. Laura Bianchi all'indirizzo laura.bianchi@studiolegale.it "
    "o al numero +39 333 1234567.",
    "x",
]


def export(model_dir, out_path, opset):
    import torch
    from transformers import AutoModelForTokenClassification, AutoTokenizer

    # eager: niente flash-attention/unpadding (non esportabili) ne' SDPA; stessi pesi.
    model = AutoModelForTokenClassification.from_pretrained(
        model_dir, attn_implementation="eager")
    model.config.reference_compile = False
    model.eval()
    tok = AutoTokenizer.from_pretrained(model_dir)
    enc = tok(SAMPLES[1], return_tensors="pt")

    class Wrapper(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, input_ids, attention_mask):
            return self.m(input_ids=input_ids, attention_mask=attention_mask).logits

    out_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with torch.no_grad():
        torch.onnx.export(
            Wrapper(model),
            (enc["input_ids"], enc["attention_mask"]),
            str(out_path),
            input_names=["input_ids", "attention_mask"],
            output_names=["logits"],
            dynamic_axes={"input_ids": {0: "batch", 1: "seq"},
                          "attention_mask": {0: "batch", 1: "seq"},
                          "logits": {0: "batch", 1: "seq"}},
            opset_version=opset,
            do_constant_folding=True,
            dynamo=False,
        )

    # Pesi in un file a parte (model.onnx_data): onnxruntime lo mappa in memoria invece di
    # copiarlo, e della tabella degli embedding (256k x 768, il 64% del modello) entrano in
    # RAM solo le righe dei token visti. Misurato: 513 MB di RSS contro 1,26 GB del file unico
    # (e 859 MB di PyTorch). I numeri e la velocita' non cambiano.
    import onnx
    data_name = out_path.name + "_data"
    m = onnx.load(str(out_path))
    (out_path.parent / data_name).unlink(missing_ok=True)
    onnx.save_model(m, str(out_path), save_as_external_data=True,
                    all_tensors_to_one_file=True, location=data_name, size_threshold=1024)
    size = out_path.stat().st_size + (out_path.parent / data_name).stat().st_size
    print(f"Esportato {out_path} (+ {data_name}, {size / 2**20:,.0f} MB) "
          f"in {time.time() - t0:.0f}s")


def load_texts(n):
    texts = list(SAMPLES)
    if VALIDATION.is_file() and n > 0:
        with open(VALIDATION, encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                t = row.get("source_text") or " ".join(row.get("tokens", []))
                if t:
                    texts.append(t)
                if len(texts) >= n + len(SAMPLES):
                    break
    # un testo lungo (oltre la finestra locale di 128 token) per le shape dinamiche
    texts.append(" ".join(SAMPLES[:3]) * 6)
    return texts


def verify(model_dir, onnx_path, n):
    import numpy as np
    import torch
    from transformers import (AutoModelForTokenClassification, AutoTokenizer,
                              pipeline)

    from onnx_ner import OnnxTokenClassifier

    tok = AutoTokenizer.from_pretrained(model_dir)
    # riferimento = esattamente cio' che usava l'app (pipeline HF, attention di default)
    ref_model = AutoModelForTokenClassification.from_pretrained(model_dir).eval()
    hf = pipeline("token-classification", model=ref_model, tokenizer=tok,
                  aggregation_strategy="simple", device=-1)
    ox = OnnxTokenClassifier(str(model_dir), onnx_path=str(onnx_path))

    texts = load_texts(n)
    print(f"Verifica su {len(texts)} testi...")
    max_diff, mismatches = 0.0, 0
    t_hf = t_ox = 0.0
    for i, text in enumerate(texts):
        enc = tok(text, return_tensors="pt", truncation=True)
        with torch.no_grad():
            lt = ref_model(**enc).logits[0].numpy()
        lo, _ = ox.logits(text)
        if lt.shape != lo.shape:
            print(f"  [{i}] shape diversa: torch {lt.shape} onnx {lo.shape}")
            mismatches += 1
            continue
        max_diff = max(max_diff, float(np.abs(lt - lo).max()))

        t0 = time.perf_counter(); a = hf(text); t_hf += time.perf_counter() - t0
        t0 = time.perf_counter(); b = ox(text); t_ox += time.perf_counter() - t0
        ka = [(e["entity_group"], int(e["start"]), int(e["end"])) for e in a]
        kb = [(e["entity_group"], e["start"], e["end"]) for e in b]
        sa = [float(e["score"]) for e in a]
        sb = [e["score"] for e in b]
        if ka != kb or any(abs(x - y) > 1e-3 for x, y in zip(sa, sb)):
            mismatches += 1
            print(f"  [{i}] entita' diverse:\n    torch {ka}\n    onnx  {kb}")

    print(f"max |logits torch - onnx| = {max_diff:.2e}")
    print(f"testi con entita' diverse: {mismatches}/{len(texts)}")
    print(f"tempo pipeline: torch {t_hf:.1f}s  onnx {t_ox:.1f}s  "
          f"({t_hf / max(t_ox, 1e-9):.2f}x)")
    ok = mismatches == 0 and max_diff < 1e-2
    print("OK: ONNX equivalente a PyTorch" if ok else "ERRORE: ONNX diverge da PyTorch")
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", default=os.environ.get("PII_MODEL_DIR", str(DEFAULT_MODEL)))
    ap.add_argument("--out", help="default: <model-dir>/onnx/model.onnx")
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("-n", type=int, default=300, help="frasi di validation da confrontare")
    ap.add_argument("--no-verify", action="store_true")
    ap.add_argument("--verify-only", action="store_true", help="non riesporta, verifica soltanto")
    args = ap.parse_args()

    model_dir = Path(args.model_dir)
    if not model_dir.is_absolute():
        model_dir = ROOT / model_dir
    out = Path(args.out) if args.out else model_dir / "onnx" / "model.onnx"

    if not args.verify_only:
        export(model_dir, out, args.opset)
    if not args.no_verify and not verify(model_dir, out, args.n):
        sys.exit(1)


if __name__ == "__main__":
    main()
