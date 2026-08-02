#!/usr/bin/env bash
# ============================================================================
# Build dell'app desktop Rizzo PII per macOS (.app + .dmg), CPU/offline.
#
# Speculare a build_linux.sh: stesso build_sidecar.spec, sidecar 'pii-backend'
# senza estensione (il Rust in lib.rs sceglie il nome col cfg!(windows)), e
# bundle 'app'/'dmg' invece di deb/appimage. La conf Tauri resta su nsis per
# Windows: qui si sovrascrive da CLI con --bundles.
#
# Uso:
#   bash build_macos.sh            # .app + .dmg
#   bash build_macos.sh app        # solo il .app (piu' veloce, per testare)
#
# Prerequisiti (una volta):
#   - Xcode Command Line Tools:  xcode-select --install
#   - Rust (https://rustup.rs) e Node.js 18+ (con npm)
# Il modello e' gitignorato: dev'essere presente in $MODEL_DIR.
#
# NOTA firma: il bundle non e' firmato ne' notarizzato. In locale si apre
# senza problemi (niente quarantena); se lo copi su un'altra macchina serve
# "tasto destro > Apri" oppure:  xattr -dr com.apple.quarantine "Rizzo PII.app"
# ============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

MODEL_DIR="models/rizzo-pii-0.3B-main"     # deve combaciare con build_sidecar.spec
VENV="${VENV:-.venv}"                      # override: VENV=build_env_macos bash build_macos.sh
BUNDLES="${*:-app dmg}"

# ---- 0) controlli ----------------------------------------------------------
[ "$(uname -s)" = "Darwin" ] || { echo "ERRORE: questo script gira solo su macOS"; exit 1; }
[ -d "$MODEL_DIR" ] || { echo "ERRORE: modello mancante: $MODEL_DIR"; exit 1; }
command -v cargo >/dev/null || { echo "ERRORE: Rust/cargo non trovato (https://rustup.rs)"; exit 1; }
command -v npm   >/dev/null || { echo "ERRORE: npm non trovato (installa Node.js 18+)"; exit 1; }

# ---- 1) venv con le dipendenze (su macOS torch e' gia' CPU/MPS, niente indice extra) ----
if [ ! -d "$VENV" ]; then
  python3 -m venv "$VENV"
  "$VENV/bin/pip" install --upgrade pip
  "$VENV/bin/pip" install torch "transformers==4.57.3" tokenizers safetensors flask pymupdf pyinstaller
fi
"$VENV/bin/python" -c "import PyInstaller" 2>/dev/null \
  || "$VENV/bin/pip" install pyinstaller

# ---- 2) sidecar PyInstaller -> tauri/src-tauri/backend/pii-backend/pii-backend ----
rm -rf tauri/src-tauri/backend
"$VENV/bin/pyinstaller" build_sidecar.spec --noconfirm \
  --distpath tauri/src-tauri/backend --workpath build/sidecar_work_macos
[ -f tauri/src-tauri/backend/pii-backend/pii-backend ] \
  || { echo "ERRORE: sidecar macOS non prodotto"; exit 1; }

# ---- 3) firma + notarizzazione (opzionali, attivate dall'ambiente) ---------
# FIRMA: se nel portachiavi c'e' una "Developer ID Application", Tauri la usa e
# applica hardened runtime + entitlements.plist. Senza, la build resta ad-hoc
# (gira in locale, ma su un altro Mac serve "tasto destro > Apri").
if [ -z "${APPLE_SIGNING_IDENTITY:-}" ]; then
  APPLE_SIGNING_IDENTITY="$(security find-identity -v -p codesigning 2>/dev/null \
    | sed -n 's/.*"\(Developer ID Application: .*\)".*/\1/p' | head -1)"
fi
if [ -n "$APPLE_SIGNING_IDENTITY" ]; then
  export APPLE_SIGNING_IDENTITY
  echo ">> firma con: $APPLE_SIGNING_IDENTITY"
else
  echo ">> ATTENZIONE: nessuna 'Developer ID Application' nel portachiavi -> firma ad-hoc"
fi

# NOTARIZZAZIONE: serve un profilo di credenziali Apple. Creane uno una volta con
#   xcrun notarytool store-credentials rizzo-pii --apple-id <id> --team-id <team> --password <app-specific-password>
# poi esporta le variabili qui sotto (Tauri notarizza e fa lo staple da solo).
# Con l'app-specific password:  APPLE_ID, APPLE_PASSWORD, APPLE_TEAM_ID
# Con la API key App Store Connect:  APPLE_API_KEY, APPLE_API_ISSUER, APPLE_API_KEY_PATH
if [ -n "${APPLE_ID:-}" ] || [ -n "${APPLE_API_KEY:-}" ]; then
  echo ">> notarizzazione attiva"
else
  echo ">> notarizzazione SALTATA (nessuna credenziale Apple nell'ambiente)"
fi

# ---- 3b) smonta i volumi DMG rimasti appesi -------------------------------
# bundle_dmg.sh fa 'hdiutil attach': se una build precedente e' stata interrotta
# il volume resta montato come /Volumes/dmg.XXXXXX e il bundle fallisce con un
# laconico "failed to run bundle_dmg.sh".
for v in /Volumes/dmg.*; do
  [ -d "$v" ] || continue
  echo ">> smonto volume DMG residuo: $v"
  hdiutil detach "$v" -force >/dev/null 2>&1 || true
done

# ---- 4) build Tauri: la conf di default e' nsis, qui si sovrascrive --------
cd tauri
npm install
npx tauri build --bundles $BUNDLES

echo
echo "FATTO. Artefatti macOS in:"
echo "  tauri/src-tauri/target/release/bundle/macos/Rizzo PII.app"
echo "  tauri/src-tauri/target/release/bundle/dmg/*.dmg"
