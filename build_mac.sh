#!/usr/bin/env bash
# ============================================================================
# Build dell'app desktop Rizzo PII per macOS (.app + .dmg), CPU/offline.
#
# Va costruito SU macOS. Speculare a build_linux.sh / docs/BUILD.md (Windows):
# stesso build_sidecar.spec, ma qui il sidecar esce come 'pii-backend' (senza
# .exe). Il Rust (lib.rs) sceglie il nome del binario in base al SO.
#
# Tre modalita' (rilevate dalle credenziali in .env.notarize, gitignorato):
#   - NOTARIZE : Developer ID + APPLE_ID/APPLE_PASSWORD/APPLE_TEAM_ID
#                -> .app/.dmg firmati e NOTARIZZATI (scarica e apri, zero avvisi)
#   - SIGN     : solo Developer ID -> firmati ma non notarizzati
#   - ADHOC    : niente credenziali -> firma ad-hoc (richiede xattr/"Apri comunque")
#
# Perche' la pipeline di notarizzazione e' "manuale": Tauri, copiando il backend
# nelle risorse dell'.app, APPIATTISCE i symlink del Python.framework (PyInstaller)
# -> "bundle ambiguous" -> notarizzazione invalida. Quindi ricopiamo il backend
# preservando i symlink (cp -R) e firmiamo a mano, poi notarizziamo .app e .dmg.
#
# Uso:
#   bash build_mac.sh                # app + dmg
#   bash build_mac.sh app            # solo .app
#
# Prerequisiti: Xcode CLT, Rust, Node18+, un venv CPU con pyinstaller (VENV=...).
# ============================================================================
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

MODEL_DIR="models/rizzo-pii-0.3B-v1.5.0"   # deve combaciare con build_sidecar.spec
VENV="${VENV:-.venv}"
BUNDLES="${*:-app dmg}"
APP="$ROOT/tauri/src-tauri/target/release/bundle/macos/Rizzo PII.app"
ENT="$ROOT/tauri/macos_python_entitlements.plist"
DMG_OUT="$ROOT/tauri/src-tauri/target/release/bundle/dmg/Rizzo-PII-1.0.0-macOS-arm64.dmg"

# ---- 0) controlli ----------------------------------------------------------
[ -d "$MODEL_DIR" ] || { echo "ERRORE: modello mancante: $MODEL_DIR"; exit 1; }
[ -x "$VENV/bin/pyinstaller" ] || { echo "ERRORE: pyinstaller mancante in $VENV"; exit 1; }
command -v cargo >/dev/null || { echo "ERRORE: Rust/cargo non trovato"; exit 1; }
command -v npm   >/dev/null || { echo "ERRORE: npm non trovato"; exit 1; }

# ---- 1) sidecar PyInstaller ------------------------------------------------
"$VENV/bin/pyinstaller" build_sidecar.spec --noconfirm \
  --distpath tauri/src-tauri/backend --workpath build/sidecar_work_mac
[ -f tauri/src-tauri/backend/pii-backend/pii-backend ] \
  || { echo "ERRORE: sidecar macOS non prodotto"; exit 1; }

# ---- 1b) credenziali / modalita' -------------------------------------------
[ -f "$ROOT/.env.notarize" ] && { set -a; . "$ROOT/.env.notarize"; set +a; }
APPLE_SIGNING_IDENTITY="${APPLE_SIGNING_IDENTITY:--}"
MODE="ADHOC"
if [ "$APPLE_SIGNING_IDENTITY" != "-" ]; then
  if [ -n "${APPLE_ID:-}" ] && [ -n "${APPLE_PASSWORD:-}" ] && [ -n "${APPLE_TEAM_ID:-}" ]; then
    MODE="NOTARIZE"
  else
    MODE="SIGN"
  fi
fi
echo ">>> Modalita' macOS: $MODE  (identity: $APPLE_SIGNING_IDENTITY)"

cd "$ROOT/tauri"
npm install
cd "$ROOT"

# ---- 2a) percorso ADHOC/SIGN: build standard Tauri -------------------------
if [ "$MODE" != "NOTARIZE" ]; then
  ( cd "$ROOT/tauri" && APPLE_SIGNING_IDENTITY="$APPLE_SIGNING_IDENTITY" \
      npx tauri build --bundles $BUNDLES )
  echo "FATTO ($MODE). Artefatti in tauri/src-tauri/target/release/bundle/{macos,dmg}/"
  exit 0
fi

# ---- 2b) percorso NOTARIZE -------------------------------------------------
ID="$APPLE_SIGNING_IDENTITY"
sign() { codesign --force --options runtime --timestamp --sign "$ID" "$@"; }
# Notarizza un file e ritorna lo stato (Accepted/Invalid/...) via JSON.
# NB: NON grezzare l'output di --wait con grep: usa le barre di avanzamento con
# ritorni-carrello e l'ultima riga puo' non finire nel file -> falso negativo.
notarize_status() {
  xcrun notarytool submit "$1" --apple-id "$APPLE_ID" --password "$APPLE_PASSWORD" \
    --team-id "$APPLE_TEAM_ID" --wait --output-format json 2>/dev/null \
    | python3 -c 'import sys,json;print(json.load(sys.stdin).get("status","?"))'
}

echo ">>> [1/8] tauri build (firma Developer ID, SENZA notarize di Tauri)"
( cd "$ROOT/tauri" && env -u APPLE_ID -u APPLE_PASSWORD -u APPLE_API_KEY -u APPLE_API_ISSUER \
    APPLE_SIGNING_IDENTITY="$ID" npx tauri build --bundles app >/dev/null )
[ -d "$APP" ] || { echo "ERRORE: .app non prodotto"; exit 1; }

echo ">>> [2/8] sostituisco il backend appiattito preservando i symlink"
rm -rf "$APP/Contents/Resources/backend"
cp -R "$ROOT/tauri/src-tauri/backend" "$APP/Contents/Resources/backend"

echo ">>> [3/8] firmo i Mach-O annidati (dal piu' profondo)"
find "$APP" -type f ! -type l | while IFS= read -r f; do
  file "$f" 2>/dev/null | grep -q "Mach-O" && printf '%s\t%s\n' "$(printf '%s' "$f" | awk -F/ '{print NF}')" "$f"
done | sort -rn | cut -f2- | while IFS= read -r f; do
  sign "$f" >/dev/null 2>&1 || echo "  FAIL $f"
done

echo ">>> [4/8] framework + sidecar(entitlements) + app esterna"
sign "$APP/Contents/Resources/backend/pii-backend/_internal/Python.framework" >/dev/null 2>&1
codesign --force --options runtime --timestamp --entitlements "$ENT" --sign "$ID" \
  "$APP/Contents/Resources/backend/pii-backend/pii-backend" >/dev/null 2>&1
sign "$APP" >/dev/null 2>&1
codesign --verify --deep --strict "$APP" || { echo "ERRORE: verify app fallita"; exit 1; }

echo ">>> [5/8] notarizzo la .app (attendo Apple)"
ZIP="$ROOT/build/RizzoPII_app.zip"; mkdir -p "$ROOT/build"; rm -f "$ZIP"
/usr/bin/ditto -c -k --keepParent "$APP" "$ZIP"
ST=$(notarize_status "$ZIP"); echo "    .app notarization: $ST"
[ "$ST" = "Accepted" ] || { echo "ERRORE: notarizzazione .app = $ST"; exit 1; }

echo ">>> [6/8] staple .app"
xcrun stapler staple "$APP"

echo ">>> [7/8] costruisco il .dmg (hdiutil, preserva i symlink)"
DMGROOT="$ROOT/build/dmgroot"; rm -rf "$DMGROOT"; mkdir -p "$DMGROOT"
cp -R "$APP" "$DMGROOT/"; ln -s /Applications "$DMGROOT/Applications"
mkdir -p "$(dirname "$DMG_OUT")"; rm -f "$DMG_OUT"
hdiutil create -volname "Rizzo PII" -srcfolder "$DMGROOT" -fs HFS+ -format UDZO -ov "$DMG_OUT" >/dev/null

echo ">>> [8/8] firmo + notarizzo + staple il .dmg"
codesign --force --timestamp --sign "$ID" "$DMG_OUT" >/dev/null 2>&1
ST=$(notarize_status "$DMG_OUT"); echo "    .dmg notarization: $ST"
[ "$ST" = "Accepted" ] || { echo "ERRORE: notarizzazione .dmg = $ST"; exit 1; }
xcrun stapler staple "$DMG_OUT"

echo
echo "FATTO (NOTARIZE). DMG notarizzato:"
echo "  $DMG_OUT"
spctl -a -t open --context context:primary-signature -vv "$DMG_OUT" 2>&1 | head -2
