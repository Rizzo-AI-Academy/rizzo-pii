# Build dell'app desktop (CPU — Windows e Linux)

Crea un eseguibile/installer **standalone e offline** dell'anonimizzatore PII.
Usa una build **CPU di PyTorch** (gira su qualsiasi PC, niente CUDA richiesta).
Windows: vedi sotto. **Linux** (.deb/.AppImage): vedi **[§ Build Linux](#build-linux-debappimage)**.

Esistono due modi di impacchettare, entrambi CPU/offline:

- **Tauri (consigliato per distribuire)** — vera **finestra nativa** (WebView2) + installer NSIS
  per-utente. Vedi **[§ App Tauri](#app-tauri-finestra-nativa--installer-nsis)**.
- **PyInstaller + Inno Setup (legacy)** — apre l'UI nel **browser di sistema** su localhost.
  Vedi **[§ PyInstaller](#pyinstaller--inno-setup-legacy-browser)**. È anche la base del sidecar Tauri.

## Architettura (perché un "sidecar")

Il motore dell'app è **Python + PyTorch + il modello mmBERT**: non gira dentro Rust/WebView.
Per questo, sia con Tauri sia col build legacy, il backend è il **server Flask** (`src/app/app.py`)
impacchettato con PyInstaller. Con Tauri la finestra nativa lo lancia come **processo figlio
(sidecar)**, attende che risponda su `127.0.0.1:5005`, poi mostra l'UI; alla chiusura lo termina.
La porta è la **5005** (non la 5000, occupata di default da AirPlay Receiver su macOS); override con `PII_PORT`.

---

## App Tauri (finestra nativa + installer NSIS)

### Prerequisiti (una volta)
- **Rust** (`rustup`), **Node.js** + **npm**, **WebView2 Runtime** (già presente su Win10/11 aggiornati).
- Il **venv CPU** `build_env/` (vedi § PyInstaller, passo 1) per costruire il sidecar.

### 1. Costruire il sidecar (backend headless)
Entry dedicato `src/app/serve.py` (solo Flask, niente browser); spec `build_sidecar.spec`.
Si costruisce **dentro le risorse Tauri**:
```powershell
build_env\Scripts\pyinstaller.exe build_sidecar.spec --noconfirm `
  --distpath tauri\src-tauri\backend --workpath build\sidecar_work
```
Output: `tauri\src-tauri\backend\pii-backend\pii-backend.exe` (+ `_internal\`, modello, asset) ≈ 1,8 GB.

### 2. Build dell'app + installer
```powershell
cd tauri
npm install                  # prima volta: scarica la CLI di Tauri
npx tauri icon ..\src\app\assets\mascot_shield.png   # (ri)genera le icone (già fatto)
npx tauri build              # compila Rust + bundle + installer NSIS
```
Output installer: `tauri\src-tauri\target\release\bundle\nsis\Anonimizzatore PII_1.0.0_x64-setup.exe`.
Installer **per-utente** (niente admin), in italiano, con shortcut e disinstallazione.

### Sviluppo / debug
- `npx tauri dev` avvia l'app collegata ai sorgenti (ricompila Rust al volo).
- Log del backend: `%LOCALAPPDATA%\rizzo-pii\backend.log` (il sidecar è windowed, niente console).
- Per rigenerare con un **nuovo modello**: riaddestra (crea `models\rizzo-pii-0.3B-v{VERSION}\`),
  aggiorna il path in **`build_sidecar.spec`** (riga `datas += [("models/rizzo-pii-0.3B-v...", "pii_model")]`),
  poi rifai il passo 1 e il passo 2. Build attuale: **v1.2.0**.

---

## Build Linux (.deb/.AppImage)

**Non si compila da Windows** (PyInstaller e i bundle Tauri Linux/`webkit2gtk` vanno fatti su
Linux). Usa una macchina Ubuntu/Debian o **WSL2**. Tutto è automatizzato in **`build_linux.sh`**
(root): stesso `build_sidecar.spec`, ma il sidecar esce come `pii-backend` (senza `.exe`) e i
bundle sono `deb`/`appimage`. Il Rust ([`lib.rs`](../tauri/src-tauri/src/lib.rs)) sceglie già il
nome del binario in base al SO (`cfg!(windows)`).

```bash
# prerequisiti di sistema (una volta)
sudo apt update && sudo apt install -y \
  build-essential curl wget file libssl-dev libxdo-dev patchelf \
  libwebkit2gtk-4.1-dev librsvg2-dev libayatana-appindicator3-dev
# + Rust (https://rustup.rs) e Node.js 18+

# copia il modello addestrato sulla macchina Linux in models/rizzo-pii-0.3B-v1.2.0/
bash build_linux.sh
```
Output: `tauri/src-tauri/target/release/bundle/{deb/*.deb, appimage/*.AppImage}`. Il modello è
gitignorato (~1,23 GB): va copiato a mano sulla macchina Linux, non è nel repo.

### Con Docker (consigliato: riproducibile, non sporca il sistema)

`Dockerfile.linux` (root) crea un'immagine con tutta la toolchain + le dipendenze Python già
installate (torch CPU, transformers, pyinstaller). Sorgenti e modello si **montano** a runtime
(`-v`), così l'immagine resta riutilizzabile e l'output finisce sull'host. Serve Docker (su
Windows: Docker Desktop con backend WSL2).

```bash
cd /mnt/d/documenti/rizzo_pii     # o una copia in ~/ (più veloce: vedi nota)
docker build -t rizzo-pii-builder -f Dockerfile.linux .
docker run --rm -e VENV=/opt/venv -e APPIMAGE_EXTRACT_AND_RUN=1 \
  -v "$PWD":/work -w /work rizzo-pii-builder
# artefatti -> tauri/src-tauri/target/release/bundle/{deb,appimage}/  (visibili anche da Windows)
```
- L'immagine si ricostruisce solo se cambiano le dipendenze; le build successive sono veloci.
- Se l'**AppImage** fallisce nel container (FUSE): `... rizzo-pii-builder bash build_linux.sh deb`
  produce solo il `.deb`.
- **Velocità**: buildare sul mount `/mnt/d` (filesystem Windows) è lento. Per build ripetute,
  `rsync` i sorgenti + il modello in `~/` dentro WSL e monta quella copia.

---

## PyInstaller + Inno Setup (legacy, browser)

## Componenti
- `src/app/desktop_app.py` — entry point: avvia il server locale e apre il browser.
- `src/app/app.py` — logica (modello + chunking); `MODEL_DIR` si risolve anche dentro l'exe.
- `build.spec` (root) — configurazione PyInstaller. Impacchetta `models/rizzo-pii-0.3B-v1.2.0` come
  `pii_model` dentro l'exe; esclude TF/CUDA/ecc. Entry: `src/app/desktop_app.py`, `pathex=src/app`.
- `installer.iss` — script Inno Setup per l'installer `.exe`.
- `build_env\` — virtualenv CPU dedicato (NON il Python di sistema, che ha torch CUDA).

## Passi

### 1. Ambiente CPU (una volta)
```powershell
python -m venv build_env
build_env\Scripts\python.exe -m pip install --upgrade pip
build_env\Scripts\python.exe -m pip install --index-url https://download.pytorch.org/whl/cpu torch
build_env\Scripts\python.exe -m pip install "transformers==4.57.3" tokenizers safetensors flask pymupdf pyinstaller
```

### 2. Build dell'eseguibile
```powershell
build_env\Scripts\pyinstaller.exe build.spec --noconfirm
```
Output: `dist\AnonimizzatorePII\AnonimizzatorePII.exe` (cartella autocontenuta, include il modello).
Avviabile direttamente con doppio clic — apre il browser su http://127.0.0.1:5005/.

### 3. Installer (opzionale, per distribuirlo)
Installa Inno Setup (https://jrsoftware.org/isdl.php), poi:
```powershell
iscc installer.iss
```
Output: `installer_out\AnonimizzatorePII-Setup.exe` — installer per-utente (niente admin),
con shortcut nel menu Start e disinstallazione.

## Note
- **Dimensione**: la cartella/installer è di alcuni GB (PyTorch + modello incluso). Normale.
- **Rigenerare col modello definitivo**: il modello è impacchettato dalla cartella versionata
  `models\rizzo-pii-0.3B-v{VERSION}\` (vedi `datas` in `build.spec` / `build_sidecar.spec`; build
  attuale **v1.2.0**). Quando riaddestri una nuova versione, aggiorna quel path negli spec e rifai
  il passo 2 (e 3). Per provare senza modello si può puntare a `models\pii_model_legacy`.
- **console=True** in `build.spec` mostra una finestra con i log; mettila `False` per nasconderla
  (consigliato solo dopo che tutto funziona).
- **SmartScreen**: un exe non firmato mostra l'avviso "editore sconosciuto". Per la distribuzione
  serve un certificato di code signing.
