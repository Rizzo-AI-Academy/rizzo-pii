#!/usr/bin/env bash
# ============================================================================
# Release macOS di Rizzo PII in UN comando, dal tuo Mac (Apple Silicon).
#
#   bash release_mac.sh
#
# Fa tutto quello che su Windows e Linux fa la CI:
#   1. crea il venv (.venv) con le sole dipendenze del sidecar (niente torch);
#   2. scarica da Hugging Face il modello ONNX (serve alla variante offline);
#   3. costruisce, firma e notarizza (se c'e' .env.notarize) le due varianti con
#      build_mac.sh:
#        Rizzo-PII-<ver>-macOS-arm64.dmg          leggera: modello scaricato al primo avvio
#        Rizzo-PII-<ver>-macOS-arm64-offline.dmg  completa: modello dentro
#   4. le allega alla release GitHub v<ver> (serve `gh auth login`; UPLOAD=0 per saltare).
#
# Perche' sul Mac e non in CI: la firma Developer ID e la notarizzazione Apple usano il
# certificato nel TUO portachiavi + le credenziali in .env.notarize (gitignorato).
#
# Prerequisiti: Xcode CLT, Rust (rustup), Node 18+, Python 3.10+ (PYTHON=... per sceglierlo).
# ============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-python3}"
VENV="${VENV:-.venv}"
MODEL_REPO="rizzoaiacademy/rizzo-pii-0.3B"
MODEL_REV="v1.5.0"
MODEL_DIR="models/rizzo-pii-0.3B-v1.5.0"   # deve combaciare con build_sidecar.spec
VER="$(python3 -c 'import json;print(json.load(open("tauri/src-tauri/tauri.conf.json"))["version"])')"
OUT="$ROOT/build/mac_release"

echo "=== Rizzo PII $VER — release macOS ==="

# ---- 1) venv: solo cio' che serve al sidecar ONNX ---------------------------
if ! "$VENV/bin/python" -c "import onnxruntime, tokenizers, flask, fitz, PyInstaller, huggingface_hub" 2>/dev/null; then
  echo ">>> creo/aggiorno il venv $VENV"
  [ -x "$VENV/bin/python" ] || "$PYTHON" -m venv "$VENV"
  "$VENV/bin/pip" install --upgrade pip
  "$VENV/bin/pip" install tokenizers numpy onnxruntime flask pymupdf pyinstaller "huggingface_hub[cli]" \
    opencv-python-headless pyclipper pyyaml
fi

# ---- 2) modello ONNX da Hugging Face (file elencati: --include ne prende uno solo) ----
if [ ! -f "$MODEL_DIR/onnx/model.onnx_data" ]; then
  echo ">>> scarico il modello ONNX ($MODEL_REPO@$MODEL_REV)"
  "$VENV/bin/hf" download "$MODEL_REPO" onnx/model.onnx onnx/model.onnx_data \
    config.json tokenizer.json tokenizer_config.json special_tokens_map.json \
    --revision "$MODEL_REV" --local-dir "$MODEL_DIR"
fi

# ---- 3) le due varianti -----------------------------------------------------
rm -rf "$OUT"; mkdir -p "$OUT"
for V in lite offline; do
  echo
  echo "=================== variante $V ==================="
  VARIANT="$V" VENV="$VENV" bash build_mac.sh
  SUFFIX=""; [ "$V" = offline ] && SUFFIX="-offline"
  DMG="tauri/src-tauri/target/release/bundle/dmg/Rizzo-PII-$VER-macOS-arm64$SUFFIX.dmg"
  [ -f "$DMG" ] || { echo "ERRORE: $DMG non prodotto"; exit 1; }
  cp "$DMG" "$OUT/"
done

echo
ls -lh "$OUT"

# ---- 4) upload sulla release ------------------------------------------------
if [ "${UPLOAD:-1}" = "1" ]; then
  if command -v gh >/dev/null && gh auth status >/dev/null 2>&1; then
    echo ">>> allego i .dmg alla release v$VER"
    gh release upload "v$VER" "$OUT"/*.dmg --clobber
    echo "FATTO: https://github.com/Rizzo-AI-Academy/rizzo-pii/releases/tag/v$VER"
  else
    echo "gh non disponibile o non autenticato: carica a mano i file in $OUT"
    echo "  (oppure: gh auth login && gh release upload v$VER $OUT/*.dmg --clobber)"
  fi
fi
