# -*- coding: utf-8 -*-
"""
Procura i modelli OCR (ONNX) che usa l'app: src/app/ocr.py -> src/app/docmd/.

Cartella di destinazione (default models/ocr/, gitignorata), una sottocartella per
modello con inference.onnx + inference.yml, la stessa struttura di docmd:

    models/ocr/PP-OCRv6_small_det/     text detection        ~10 MB  ONNX ufficiale
    models/ocr/PP-OCRv6_small_rec/     text recognition      ~21 MB  ONNX ufficiale
    models/ocr/PP-OCRv6_tiny_det/      (preset "fast")        ~2 MB  ONNX ufficiale
    models/ocr/PP-OCRv6_tiny_rec/      (preset "fast")        ~5 MB  ONNX ufficiale
    models/ocr/SLANet_plus/            struttura tabelle      ~8 MB  ONNX ufficiale
    models/ocr/PP-DocLayout-M/         layout veloce         ~23 MB  convertito (paddle2onnx)
    models/ocr/PP-DocLayout_plus-L/    layout accurato      ~124 MB  ONNX ufficiale

PaddlePaddle pubblica l'ONNX ufficiale su Hugging Face (`PaddlePaddle/<nome>_onnx`)
per tutti tranne PP-DocLayout-M, che va convertito una volta con paddle2onnx (vedi
la pipeline docmd, tools/setup_models.py). Per questo il layout si prende, in ordine:
  1. da --from-dir, se indicata (cartella models/ di docmd, gia' convertita);
  2. dal repo HF --layout-repo (default rizzoaiacademy/rizzo-ocr), sottocartella
     PP-DocLayout-M/, se esiste;
  3. altrimenti il layout accurato ufficiale PP-DocLayout_plus-L (piu' pesante, ma
     ocr.py lo usa al posto del veloce senza bisogno di configurare niente).

Uso:
    # sviluppo: copia dalla pipeline docmd gia' pronta sul disco
    python scripts/fetch_ocr_models.py --from-dir D:/percorso/ocr_pipeline/models

    # CI / macchina pulita: scarica da Hugging Face
    python scripts/fetch_ocr_models.py

    # anche il preset "accurate" (layout L) oltre a quello veloce
    python scripts/fetch_ocr_models.py --with-accurate

Esce con 1 se alla fine manca qualcosa di indispensabile (det/rec small + un layout).
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FILES = ("inference.onnx", "inference.yml")

OFFICIAL = ["PP-OCRv6_small_det", "PP-OCRv6_small_rec", "PP-OCRv6_tiny_det",
            "PP-OCRv6_tiny_rec", "SLANet_plus"]
LAYOUT_FAST = "PP-DocLayout-M"
LAYOUT_ACCURATE = "PP-DocLayout_plus-L"
REQUIRED = ["PP-OCRv6_small_det", "PP-OCRv6_small_rec"]


def ok(dest: Path, name: str) -> bool:
    return all((dest / name / f).is_file() for f in FILES)


def copy_local(src_root: Path, dest: Path, name: str) -> bool:
    src = src_root / name
    if not all((src / f).is_file() for f in FILES):
        return False
    (dest / name).mkdir(parents=True, exist_ok=True)
    for f in FILES:
        shutil.copy2(src / f, dest / name / f)
    print(f"  copiato   {name}  <- {src}")
    return True


def download(repo: str, dest: Path, name: str, subfolder: str | None = None,
             revision: str | None = None) -> bool:
    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import EntryNotFoundError, RepositoryNotFoundError
    try:
        (dest / name).mkdir(parents=True, exist_ok=True)
        for f in FILES:
            p = hf_hub_download(repo, f, subfolder=subfolder, revision=revision)
            shutil.copy2(p, dest / name / f)
        print(f"  scaricato {name}  <- hf:{repo}{'/' + subfolder if subfolder else ''}")
        return True
    except (RepositoryNotFoundError, EntryNotFoundError):
        shutil.rmtree(dest / name, ignore_errors=True)
        return False
    except Exception as e:          # rete, permessi, repo privato...
        print(f"  ! {name}: {type(e).__name__}: {e}")
        shutil.rmtree(dest / name, ignore_errors=True)
        return False


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dest", type=Path, default=ROOT / "models" / "ocr")
    ap.add_argument("--from-dir", type=Path, help="cartella models/ di docmd da cui copiare")
    ap.add_argument("--layout-repo", default="rizzoaiacademy/rizzo-ocr",
                    help="repo HF con PP-DocLayout-M/ gia' convertito")
    ap.add_argument("--revision", default=None, help="revision dei repo HF (default main)")
    ap.add_argument("--with-accurate", action="store_true",
                    help="scarica anche PP-DocLayout_plus-L (preset accurate, +124 MB)")
    ap.add_argument("--force", action="store_true", help="riscarica anche i modelli presenti")
    args = ap.parse_args()
    dest: Path = args.dest
    dest.mkdir(parents=True, exist_ok=True)
    print(f"Modelli OCR in {dest}")

    def have(name):
        return ok(dest, name) and not args.force

    for name in OFFICIAL:
        if have(name):
            print(f"  presente  {name}")
            continue
        if args.from_dir and copy_local(args.from_dir, dest, name):
            continue
        if not download(f"PaddlePaddle/{name}_onnx", dest, name, revision=args.revision):
            print(f"  ! {name}: non trovato")

    # layout veloce: locale -> repo HF del progetto -> (fallback) layout accurato ufficiale
    if have(LAYOUT_FAST):
        print(f"  presente  {LAYOUT_FAST}")
    elif not ((args.from_dir and copy_local(args.from_dir, dest, LAYOUT_FAST))
              or download(args.layout_repo, dest, LAYOUT_FAST, subfolder=LAYOUT_FAST,
                          revision=args.revision)):
        print(f"  - {LAYOUT_FAST} non disponibile (serve la conversione con paddle2onnx): "
              f"uso {LAYOUT_ACCURATE}")
        args.with_accurate = True

    if args.with_accurate:
        if have(LAYOUT_ACCURATE):
            print(f"  presente  {LAYOUT_ACCURATE}")
        elif not ((args.from_dir and copy_local(args.from_dir, dest, LAYOUT_ACCURATE))
                  or download(f"PaddlePaddle/{LAYOUT_ACCURATE}_onnx", dest, LAYOUT_ACCURATE,
                              revision=args.revision)):
            print(f"  ! {LAYOUT_ACCURATE}: non trovato")

    missing = [m for m in REQUIRED if not ok(dest, m)]
    if not (ok(dest, LAYOUT_FAST) or ok(dest, LAYOUT_ACCURATE)):
        missing.append(f"{LAYOUT_FAST} o {LAYOUT_ACCURATE}")
    total = sum(f.stat().st_size for f in dest.rglob("*") if f.is_file())
    print(f"Totale {total / 1e6:.0f} MB")
    if missing:
        print("MANCANO: " + ", ".join(missing), file=sys.stderr)
        return 1
    print("OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
