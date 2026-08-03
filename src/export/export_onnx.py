# -*- coding: utf-8 -*-
"""
Export del modello PII in ONNX, con quantizzazione INT8 opzionale.

Serve a far girare rizzo-pii dove PyTorch non c'e': runtime ONNX nativi, e
soprattutto **nel browser** via Transformers.js (estensioni, WebAssembly), che
oggi non e' un target raggiungibile. Il modello resta interamente locale: ONNX
non cambia nulla della garanzia di privacy, cambia solo il motore che lo esegue.

La quantizzazione INT8 riduce il file di ~4x. Con --verify si misura quanto
costa in accuratezza, confrontando fp32 e INT8 sugli stessi testi (vedi
docs/ONNX.md per i numeri misurati su v1.2.0 e v1.3.0).

Dipendenze aggiuntive (non nel requirements.txt principale, per non appesantire
chi vuole solo addestrare):

  pip install -r requirements-onnx.txt

Uso:
  python src/export/export_onnx.py                      # fp32 + INT8, ultima versione
  python src/export/export_onnx.py --no-quantize        # solo fp32
  python src/export/export_onnx.py --verify             # export + confronto fp32/INT8
  python src/export/export_onnx.py --arch avx512        # profilo CPU (default: arm64 su Apple silicon)
  PII_MODEL_DIR=... python src/export/export_onnx.py    # modello specifico
"""

import argparse
import io
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

_ROOT = Path(__file__).resolve().parents[2]

# File del tokenizer da affiancare al modello ONNX: Transformers.js li carica
# dalla stessa cartella. Non tutti esistono per ogni tokenizer, si copia cio' che c'e'.
TOKENIZER_FILES = (
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.txt",
)


def resolve_model_dir(models_dir):
    """Ultima versione models/rizzo-pii-0.3B-v*; fallback al non versionato, poi al legacy.

    Stessa logica di src/training/test_pii.py. Override puntuale: env PII_MODEL_DIR.
    """
    if os.environ.get("PII_MODEL_DIR"):
        return Path(os.environ["PII_MODEL_DIR"])
    versioned = [p for p in models_dir.glob("rizzo-pii-0.3B-v*") if p.is_dir()]
    if versioned:
        def _key(p):
            m = re.search(r"-v([0-9][0-9.]*)$", p.name)
            return tuple(int(x) for x in m.group(1).split(".")) if m else ()
        return max(versioned, key=_key)
    base = models_dir / "rizzo-pii-0.3B"
    return base if base.exists() else models_dir / "pii_model_legacy"


def dir_size(path):
    return sum(f.stat().st_size for f in Path(path).rglob("*") if f.is_file())


def mb(n):
    return f"{n / 1024 / 1024:.1f} MB"


def default_arch():
    """Profilo CPU per la quantizzazione: arm64 su Apple silicon, avx2 altrove."""
    import platform
    return "arm64" if platform.machine().lower() in ("arm64", "aarch64") else "avx2"


def run_step(label, cmd):
    """Esegue un comando mostrando il tempo impiegato; esce con l'output se fallisce."""
    print(f"-> {label}...")
    t0 = time.perf_counter()
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print(res.stdout[-4000:])
        print(res.stderr[-4000:], file=sys.stderr)
        sys.exit(f"   FALLITO: {label}")
    dt = time.perf_counter() - t0
    print(f"   fatto in {dt:.0f}s")
    return dt


def export_fp32(model_dir, out_dir):
    run_step(
        "export ONNX fp32",
        [sys.executable, "-m", "optimum.exporters.onnx",
         "--model", str(model_dir), "--task", "token-classification", str(out_dir)],
    )


def quantize_int8(fp32_dir, staging_dir, arch):
    run_step(
        f"quantizzazione INT8 ({arch})",
        ["optimum-cli", "onnxruntime", "quantize",
         "--onnx_model", str(fp32_dir), f"--{arch}", "-o", str(staging_dir)],
    )


def assemble_bundle(model_dir, fp32_dir, staging_dir, bundle_dir):
    """Compone la cartella nel layout atteso da Transformers.js: tokenizer alla
    radice, pesi in onnx/model_quantized.onnx."""
    bundle_dir.mkdir(parents=True, exist_ok=True)
    (bundle_dir / "onnx").mkdir(exist_ok=True)

    for name in TOKENIZER_FILES:
        for candidate in (staging_dir / name, fp32_dir / name, model_dir / name):
            if candidate.exists():
                shutil.copy2(candidate, bundle_dir / name)
                break

    if staging_dir is not None:
        weights = next(staging_dir.glob("*_quantized.onnx"), None)
        target = "model_quantized.onnx"
    else:
        weights = None
    if weights is None:
        weights = next(fp32_dir.glob("model.onnx"), None)
        target = "model.onnx"
    if weights is None:
        sys.exit("Nessun file .onnx prodotto dall'export.")
    shutil.copy2(weights, bundle_dir / "onnx" / target)

    # Pesi esterni: l'export li separa quando il modello supera i 2 GB del formato protobuf.
    source = staging_dir if staging_dir is not None else fp32_dir
    for extra in source.glob("*.onnx_data"):
        shutil.copy2(extra, bundle_dir / "onnx" / extra.name)
    return bundle_dir / "onnx" / target


def load_examples(limit):
    """Testi per il confronto fp32/INT8: la validation reale se c'e', altrimenti
    gli esempi di test_pii.py (lo script deve restare utile senza il dataset)."""
    val = _ROOT / "dataset" / "validation" / "validation_real.jsonl"
    if val.exists():
        texts = []
        with val.open(encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                text = row.get("text") or " ".join(row.get("tokens", []))
                if text.strip():
                    texts.append(text)
                if len(texts) >= limit:
                    break
        if texts:
            return texts, f"validation_real.jsonl ({len(texts)} righe)"

    # Fallback: esempi che toccano tutti i 22 tag, cosi' la verifica resta
    # significativa anche senza il dataset (che e' gitignorato e rigenerabile).
    return [
        "Mi chiamo Mario Rossi e la mia email e' mario.rossi@gmail.com, "
        "telefono +39 333 1234567.",
        "Il sottoscritto, nato a Milano il 12/06/1985, residente in Via Garibaldi 24, "
        "chiede la restituzione della somma.",
        "Per il pagamento usare l'IBAN IT60X0542811101000000123456 intestato "
        "alla societa'.",
        "L'avvocato ha depositato la comparsa presso il Tribunale di Roma in data "
        "3 marzo 2024.",
        "La societa' Edilnord S.r.l., P.IVA 12345678903, e' titolare dell'immobile "
        "al Foglio 12, particella 345, sub. 6.",
        "Il cliente Giulia Bianchi, codice fiscale BNCGLI90S43H501W, ha 35 anni "
        "e risiede in Corso Vittorio Emanuele 118, 00185 Roma (RM).",
        "Fattura n. 2451/2025 del 14 marzo 2025, imponibile € 12.500,00, "
        "scadenza alle ore 15:30.",
        "Carta di credito 4111 1111 1111 1111 intestata a Luca Esposito, "
        "documento di identita' CA12345AB.",
        "Il veicolo targato AB 123 CD e' assegnato alla dipendente Chiara Ferrari, "
        "sesso femminile, assunta il 01/09/2021.",
        "Ordine ricevuto da Meccanica Padana S.p.A. presso la sede di Via Dante 7, "
        "35121 Padova, provincia PD.",
        "Contattare il dott. Andrea Greco al numero 02 87654321 oppure "
        "a.greco@studiolegale.it per la pratica RG 4821/2024.",
        "Il conto corrente IT60X0542811101000000123456 presso la filiale di Torino "
        "e' intestato a Federico Conti, nato il 22.11.1978.",
    ], "esempi predefiniti (dataset/validation assente)"


def predict_labels(model, tokenizer, texts, max_len):
    """Etichetta prevista per ogni token di ogni testo, appiattita in una lista."""
    import numpy as np
    out = []
    for text in texts:
        enc = tokenizer(text, truncation=True, max_length=max_len, return_tensors="np")
        logits = model(**{k: v for k, v in enc.items()}).logits
        arr = logits.numpy() if hasattr(logits, "numpy") else np.asarray(logits)
        out.extend(arr[0].argmax(axis=-1).tolist())
    return out


def verify(model_dir, fp32_dir, bundle_dir, max_len, limit):
    """Confronta le predizioni fp32 e INT8 token per token sugli stessi testi."""
    from optimum.onnxruntime import ORTModelForTokenClassification
    from transformers import AutoTokenizer

    texts, source = load_examples(limit)
    print(f"\n-> verifica fp32 vs INT8 su {source}")

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    ref = ORTModelForTokenClassification.from_pretrained(fp32_dir)
    quant = ORTModelForTokenClassification.from_pretrained(
        bundle_dir, subfolder="onnx", file_name="model_quantized.onnx"
    )

    labels_fp32 = predict_labels(ref, tokenizer, texts, max_len)
    labels_int8 = predict_labels(quant, tokenizer, texts, max_len)

    total = len(labels_fp32)
    agree = sum(1 for a, b in zip(labels_fp32, labels_int8) if a == b)

    # I token 'O' dominano: l'accordo sui soli token etichettati e' la misura severa.
    o_id = next((int(k) for k, v in ref.config.id2label.items() if v == "O"), None)
    ent_idx = [i for i, a in enumerate(labels_fp32) if a != o_id]
    ent_agree = sum(1 for i in ent_idx if labels_fp32[i] == labels_int8[i])

    print(f"   token totali          {total}")
    print(f"   accordo complessivo   {100 * agree / total:.2f}%")
    if ent_idx:
        print(f"   token di entita'      {len(ent_idx)}")
        print(f"   accordo sulle entita' {100 * ent_agree / len(ent_idx):.2f}%")
    return {
        "verify_source": source,
        "tokens": total,
        "agreement_all": round(100 * agree / total, 2),
        "entity_tokens": len(ent_idx),
        "agreement_entities": round(100 * ent_agree / len(ent_idx), 2) if ent_idx else None,
    }


def main():
    ap = argparse.ArgumentParser(description="Export ONNX (+ INT8) del modello rizzo-pii.")
    ap.add_argument("--model", default=None,
                    help="cartella del modello (default: ultima versione in models/)")
    ap.add_argument("--out", default=None,
                    help="cartella di output (default: models/<nome>-onnx/)")
    ap.add_argument("--no-quantize", action="store_true", help="esporta solo in fp32")
    ap.add_argument("--arch", default=None,
                    help="profilo CPU per la quantizzazione: arm64 | avx2 | avx512 | avx512_vnni")
    ap.add_argument("--verify", action="store_true",
                    help="confronta le predizioni fp32 e INT8 dopo l'export")
    ap.add_argument("--verify-n", type=int, default=200,
                    help="quanti testi usare per la verifica (default: 200)")
    ap.add_argument("--max-len", type=int, default=768,
                    help="lunghezza massima in subword, come nel training (default: 768)")
    args = ap.parse_args()

    model_dir = Path(args.model) if args.model else resolve_model_dir(_ROOT / "models")
    if not Path(model_dir).exists():
        sys.exit(f"Modello non trovato: {model_dir}\n"
                 f"Addestralo (src/training/train_pii.py) o indica --model / PII_MODEL_DIR.")

    out_dir = Path(args.out) if args.out else _ROOT / "models" / f"{Path(model_dir).name}-onnx"
    fp32_dir = out_dir / "fp32"
    staging_dir = out_dir / "quant-staging"
    bundle_dir = out_dir / "bundle"
    for path in (fp32_dir, staging_dir, bundle_dir):
        if path.exists():
            shutil.rmtree(path)
    out_dir.mkdir(parents=True, exist_ok=True)

    arch = args.arch or default_arch()
    print(f"Modello:  {model_dir}")
    print(f"Output:   {out_dir}\n")

    export_fp32(model_dir, fp32_dir)
    if args.no_quantize:
        staging = None
    else:
        quantize_int8(fp32_dir, staging_dir, arch)
        staging = staging_dir
    weights_path = assemble_bundle(Path(model_dir), fp32_dir, staging, bundle_dir)

    tokenizer_json = bundle_dir / "tokenizer.json"
    stats = {
        "model": str(model_dir),
        "arch": arch if not args.no_quantize else None,
        "onnx_fp32_bytes": dir_size(fp32_dir),
        "weights_bytes": weights_path.stat().st_size,
        "tokenizer_bytes": tokenizer_json.stat().st_size if tokenizer_json.exists() else 0,
        "bundle_bytes": dir_size(bundle_dir),
    }

    print("\n── dimensioni ──")
    for label, value in (
        ("ONNX fp32", stats["onnx_fp32_bytes"]),
        (f"pesi ({weights_path.name})", stats["weights_bytes"]),
        ("tokenizer.json", stats["tokenizer_bytes"]),
        ("BUNDLE TOTALE", stats["bundle_bytes"]),
    ):
        print(f"  {label:<30}{mb(value):>10}")

    if args.verify and not args.no_quantize:
        stats.update(verify(model_dir, fp32_dir, bundle_dir, args.max_len, args.verify_n))

    (out_dir / "export_stats.json").write_text(
        json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nBundle pronto: {bundle_dir}")
    print(f"Statistiche:   {out_dir / 'export_stats.json'}")
    print("\nCaricamento con Transformers.js: vedi docs/ONNX.md")


if __name__ == "__main__":
    main()
