#!/usr/bin/env bash
# ============================================================================
# Build dell'app desktop Rizzo PII per LINUX (.deb + .AppImage), CPU/offline.
#
# NON si compila da Windows: PyInstaller e i bundle Tauri Linux (webkit2gtk)
# vanno costruiti SU Linux. Lancia questo script su Ubuntu/Debian o in WSL2.
#
# Speculare a docs/BUILD.md (Windows): stesso build_sidecar.spec, ma qui il
# sidecar esce come 'pii-backend' (senza .exe) e i bundle sono deb/appimage.
# Il Rust (lib.rs) sceglie il nome del binario in base al SO (cfg!(windows)).
#
# Uso:
#   bash build_linux.sh
#
# Prerequisiti di sistema (una volta, Ubuntu 22.04/24.04):
#   sudo apt update && sudo apt install -y \
#     build-essential curl wget file libssl-dev libxdo-dev patchelf \
#     libwebkit2gtk-4.1-dev librsvg2-dev libayatana-appindicator3-dev
#   # + Rust (https://rustup.rs) e Node.js 18+ (con npm)
# ============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

MODEL_DIR="models/rizzo-pii-0.3B-v1.2.0"   # deve combaciare con build_sidecar.spec
VENV="${VENV:-build_env_linux}"            # override: in Docker si usa il venv gia' nell'immagine
BUNDLES="${*:-deb appimage}"               # bundle da produrre (es. "bash build_linux.sh deb")

# ---- 0) controlli ----------------------------------------------------------
[ -d "$MODEL_DIR" ] || { echo "ERRORE: modello mancante: $MODEL_DIR (copialo qui)"; exit 1; }
command -v cargo >/dev/null || { echo "ERRORE: Rust/cargo non trovato (installa da https://rustup.rs)"; exit 1; }
command -v npm   >/dev/null || { echo "ERRORE: npm non trovato (installa Node.js 18+)"; exit 1; }

# ---- 1) venv CPU + dipendenze (come docs/BUILD.md, ma su Linux) ------------
if [ ! -d "$VENV" ]; then
  python3 -m venv "$VENV"
  "$VENV/bin/pip" install --upgrade pip
  "$VENV/bin/pip" install --index-url https://download.pytorch.org/whl/cpu torch
  "$VENV/bin/pip" install "transformers==4.57.3" tokenizers safetensors flask pymupdf pyinstaller
fi

# ---- 2) sidecar PyInstaller -> tauri/src-tauri/backend/pii-backend/pii-backend ----
"$VENV/bin/pyinstaller" build_sidecar.spec --noconfirm \
  --distpath tauri/src-tauri/backend --workpath build/sidecar_work_linux
[ -f tauri/src-tauri/backend/pii-backend/pii-backend ] \
  || { echo "ERRORE: sidecar Linux non prodotto"; exit 1; }

# ---- 3) build Tauri: override bundle a deb + appimage (la conf di default e' nsis) ----
cd tauri
npm install
npx tauri build --bundles $BUNDLES

echo
echo "FATTO. Artefatti Linux in:"
echo "  tauri/src-tauri/target/release/bundle/deb/*.deb"
echo "  tauri/src-tauri/target/release/bundle/appimage/*.AppImage"
