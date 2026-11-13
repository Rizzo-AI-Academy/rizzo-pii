# Build dell'app desktop (CPU — Windows, Linux e macOS)

Crea un eseguibile/installer **standalone e offline** dell'anonimizzatore PII.
Il modello gira con **ONNX Runtime su CPU** (qualsiasi PC, niente CUDA, **niente PyTorch** nel
pacchetto): il `model.onnx` è lo stesso modello fp32, non quantizzato — vedi [§ Modello ONNX](#modello-onnx-niente-pytorch-nel-pacchetto).
Windows: vedi sotto. **Linux** (.deb/.AppImage): vedi **[§ Build Linux](#build-linux-debappimage)**.
**macOS** (.app/.dmg): vedi **[§ Build macOS](#build-macos-appdmg)**.

Esistono due modi di impacchettare, entrambi CPU/offline:

- **Tauri (consigliato per distribuire)** — vera **finestra nativa** (WebView2) + installer NSIS
  per-utente. Vedi **[§ App Tauri](#app-tauri-finestra-nativa--installer-nsis)**.
- **PyInstaller + Inno Setup (legacy)** — apre l'UI nel **browser di sistema** su localhost.
  Vedi **[§ PyInstaller](#pyinstaller--inno-setup-legacy-browser)**. È anche la base del sidecar Tauri.

## Architettura (perché un "sidecar")

Il motore dell'app è **Python + ONNX Runtime + il modello mmBERT**: non gira dentro Rust/WebView.
Per questo, sia con Tauri sia col build legacy, il backend è il **server Flask** (`src/app/app.py`)
impacchettato con PyInstaller. Con Tauri la finestra nativa lo lancia come **processo figlio
(sidecar)**, attende che risponda su `127.0.0.1:5005`, poi mostra l'UI; alla chiusura lo termina.

---

## App Tauri (finestra nativa + installer NSIS)

### Prerequisiti (una volta)
- **Rust** (`rustup`), **Node.js** + **npm**, **WebView2 Runtime** (già presente su Win10/11 aggiornati).
- Il **venv CPU** `build_env/` (vedi § PyInstaller, passo 1) per costruire il sidecar.

### 0. Esportare il modello in ONNX (una volta per versione)
```powershell
python src/training/export_onnx.py --model-dir models/rizzo-pii-0.3B-v1.5.0
```
Crea `onnx/model.onnx` + `onnx/model.onnx_data` nella cartella del modello e verifica che le
entità coincidano con PyTorch (esce con 1 se no). Senza questi file lo spec si ferma subito.

### 1. Costruire il sidecar (backend headless)
Entry dedicato `src/app/serve.py` (solo Flask, niente browser); spec `build_sidecar.spec`.
Si costruisce **dentro le risorse Tauri**:
```powershell
build_env\Scripts\pyinstaller.exe build_sidecar.spec --noconfirm `
  --distpath tauri\src-tauri\backend --workpath build\sidecar_work
```
Output: `tauri\src-tauri\backend\pii-backend\pii-backend.exe` (+ `_internal\`, modello, asset) ≈ 1,4 GB
(1,2 GB sono il modello; con torch erano 1,8 GB).

Con **`$env:PII_BUNDLE_MODEL = "0"`** il sidecar esce **senza modello** (~155 MB) e l'installer
pesa ~60 MB: l'app scarica il modello da Hugging Face al primo avvio — vedi
[§ Modello al primo avvio](#modello-al-primo-avvio-installer-leggero-e-offline).
Nota: `target\release\backend\` (usata lanciando `app.exe` direttamente) si aggiorna solo quando
ricompila `build.rs`; dopo aver cambiato il sidecar, `rm -r target\release\backend` + tocca
`src-tauri\build.rs`. L'installer NSIS invece prende sempre `src-tauri\backend\`.

### 2. Build dell'app + installer
```powershell
cd tauri
npm install                  # prima volta: scarica la CLI di Tauri
npx tauri icon ..\src\app\assets\mascot_shield.png   # (ri)genera le icone (già fatto)
npx tauri build              # compila Rust + bundle + installer NSIS
```
Output installer: `tauri\src-tauri\target\release\bundle\nsis\Rizzo PII_2.0.0_x64-setup.exe`
(il nome viene da `productName` in `tauri.conf.json`).
Installer **per-utente** (niente admin), in italiano, con shortcut e disinstallazione.

### Sviluppo / debug
- `npx tauri dev` avvia l'app collegata ai sorgenti (ricompila Rust al volo).
- Log del backend: `%LOCALAPPDATA%\rizzo-pii\backend.log` (il sidecar è windowed, niente console).
- Per rigenerare con un **nuovo modello**: riaddestra (crea `models\rizzo-pii-0.3B-v{VERSION}\`),
  aggiorna il path in **`build_sidecar.spec`** (riga `datas += [("models/rizzo-pii-0.3B-v...", "pii_model")]`),
  poi rifai il passo 1 e il passo 2. Build attuale: **v1.5.0**.

### Build Windows da macOS/Linux → GitHub Actions

**Non si compila da un altro SO**, nemmeno con Docker: su macOS i container sono Linux, e
PyInstaller non cross-compila (il sidecar è Python + onnxruntime, va costruito sul SO di destinazione).
La strada è un runner Windows di GitHub Actions: **[`.github/workflows/build-windows.yml`](../.github/workflows/build-windows.yml)**
fa gli stessi passi 1 e 2 su `windows-latest` e allega l'installer alla release del tag.

Il modello è gitignorato: il workflow lo scarica da Hugging Face, repo **pubblico**
[`rizzoaiacademy/rizzo-pii-0.3B`](https://huggingface.co/rizzoaiacademy/rizzo-pii-0.3B)
branch `v1.5.0` → nessun token, nessun secret. Scarica **solo l'ONNX** (`onnx/model.onnx` +
`model.onnx_data`) e il tokenizer. Per cambiare modello si aggiornano insieme
`MODEL_REPO`/`MODEL_REV`/`MODEL_DIR` nel workflow e il path in `build_sidecar.spec`, e **prima**
si pubblica l'ONNX sul nuovo ramo HF (export + verifica in locale, poi upload in `onnx/`):
senza, il workflow si ferma con "modello non scaricato".

Due modi di lanciarlo:
```bash
# a) build di prova: l'installer resta come artifact del run (5 giorni, serve accesso al repo)
gh workflow run build-windows.yml        # oppure Actions > build-windows > Run workflow

# b) release vera: il tag fa partire anche release.yml, che crea la release;
#    build-windows la attende (max 5 min) e ci carica l'.exe
git tag -a v2.0.1 -m "Rizzo PII 2.0.1" && git push origin v2.0.1

# c) riempire una release GIA' pubblicata (es. l'.exe mancante sulla 2.0.0):
#    l'asset lo carica il runner, non passa da casa
gh workflow run build-windows.yml -f release_tag=v2.0.0
```
L'installer viene rinominato **`Rizzo-PII-<versione>-Windows-Setup.exe`** (la versione arriva da
`tauri.conf.json`), la stessa convenzione degli artefatti macOS/Linux a cui puntano i pulsanti di
download della landing in [`docs/index.html`](index.html). Attenzione: gli **artifact dei run non
sono pubblici** (servono login e accesso al repo, e scadono) — per la landing serve sempre un
**asset di release**, cioè la via (b) o la (c).
Ri-pushare lo stesso tag sostituisce l'asset (viene cancellato e ricaricato: l'API di GitHub
non sovrascrive un asset con lo stesso nome).

Accortezze:
- **Limite di 2 GiB per asset di release.** L'NSIS sta sui ~1,5-1,8 GB: passa, ma con poco
  margine. Se cresce, l'upload risponde 422.
- **Disco del runner** ~25 GB liberi: il workflow non installa torch e scarica da HF solo
  `onnx/` + tokenizer/config (niente `model.safetensors`); si cancella `build/sidecar_work` dopo PyInstaller.
- **Tempi**: ~35 min il primo run, ~15 i successivi (cache di cargo su `src-tauri/target`).
- L'installer **non è firmato** → SmartScreen avvisa al primo avvio, come per la build locale.

---

## Modello ONNX (niente PyTorch nel pacchetto)

L'app impacchettata non usa PyTorch: carica `model.onnx` con **onnxruntime**
([`src/app/onnx_ner.py`](../src/app/onnx_ner.py), dipendenze: onnxruntime + tokenizers + numpy).
È lo **stesso modello fp32**, senza quantizzazione: cambia il runtime, non i pesi.
[`src/training/export_onnx.py`](../src/training/export_onnx.py) lo esporta e lo confronta con
PyTorch (pipeline HF `aggregation_strategy="simple"`) su 300 frasi della validation:
**0 entità diverse**, scarto massimo sui logits ~3e-4 (rumore numerico).

- **Due file, sempre insieme**: `model.onnx` (grafo, 0,5 MB) + `model.onnx_data` (pesi).
  I pesi separati vengono mappati in memoria: della tabella degli embedding (256k token, il 64%
  del modello) entra in RAM solo ciò che serve.
- **Selezione in `app.py`**: se nella cartella del modello c'è `onnx/model.onnx` (o `model.onnx`)
  usa ONNX, altrimenti la pipeline transformers (CPU/GPU). `PII_BACKEND=onnx|torch` forza la scelta.
  `/health` riporta `backend`.
- **Nello spec** entrano solo `model.onnx`, `model.onnx_data`, `config.json` e i file del tokenizer:
  il `model.safetensors` resta fuori, e così non si impacchetta il modello due volte.
- **Misure sul sidecar Windows** (v1.5.0, stessa macchina):

  | | torch (prima) | ONNX (ora) |
  |---|---|---|
  | cartella del sidecar | 1,8 GB | 1,4 GB |
  | avvio fino a `/health` 200 | 7,6–9,5 s | 3,2–4,0 s |
  | prima `/analyze` (frase breve) | 3,6 s (warm-up) | 0,09 s |
  | `/analyze` ~800 parole | ~4,1 s | ~3,8–4,6 s (pari) |
  | RAM dopo l'uso | ~870 MB | ~670 MB |

  Sul testo lungo il throughput è pari: il guadagno è in installazione, avvio e memoria.

---

## Modello al primo avvio (installer leggero e offline)

La CI produce **due installer Windows** dalla stessa build:

| installer | dimensione | modello |
|---|---|---|
| `Rizzo-PII-<ver>-Windows-Setup.exe` | ~60 MB | scaricato da Hugging Face al primo avvio |
| `Rizzo-PII-<ver>-Windows-Setup-offline.exe` | ~1,15 GB | dentro l'installer, nessuna rete richiesta |

Il leggero è quello linkato dalla landing. L'offline serve a PC senza internet o a reti
aziendali che bloccano `huggingface.co`.

**Come funziona** ([`tauri/src-tauri/src/model.rs`](../tauri/src-tauri/src/model.rs)):

1. All'avvio Tauri guarda se il sidecar ha il modello dentro (`_internal/pii_model/`): se sì,
   è l'offline e non scarica niente.
2. Altrimenti cerca il modello in `%LOCALAPPDATA%\rizzo-pii\models\rizzo-pii-0.3B-v<ver>\`
   (Linux `~/.local/share/rizzo-pii/…`, macOS `~/Library/Application Support/rizzo-pii/…`).
   Il marcatore `.verified` dice che è già stato scaricato e verificato.
3. Se manca, lo scarica da `huggingface.co/rizzoaiacademy/rizzo-pii-0.3B` a un **commit
   fissato** (`MODEL_COMMIT`), solo i 6 file ONNX + tokenizer (1,18 GB). Ogni file è verificato
   con **dimensione + sha256** (`MODEL_FILES`) prima di essere usato. Il download va in
   `*.part` e **riprende** da dove si era fermato (chiusura dell'app, rete caduta), con 4
   tentativi per file; il proxy si legge da `HTTPS_PROXY`/`HTTP_PROXY`.
4. Lancia il sidecar con **`PII_MODEL_DIR`** puntato a quella cartella (in `app.py` vince sul
   modello impacchettato).

Lo splash mostra MB scaricati, percentuale, velocità e tempo stimato. In caso di errore mostra
un motivo breve (il dettaglio è in `~/rizzo-pii/tauri.log`) con **Riprova** e **Apri cartella**,
per copiare i file a mano.

**Misure** (Windows, rete domestica): download completo ~55 s a ~22 MB/s; ripresa dopo
chiusura verificata (560 MB già scaricati non vengono riscaricati); avvii successivi senza
download, backend pronto in ~4 s.

**Nuova versione del modello**: esporta e verifica l'ONNX
(`src/training/export_onnx.py`), caricalo in `onnx/` sul nuovo ramo HF, poi aggiorna in
`model.rs` `MODEL_VERSION`, `MODEL_COMMIT` (commit del ramo) e `MODEL_FILES` (dimensioni e
sha256), e in CI `MODEL_REV`/`MODEL_DIR`. Se sha256 e file pubblicati non coincidono, il
download fallisce di proposito.

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

## Build macOS (.app/.dmg)

Come per Linux, **non si compila da un altro SO**: PyInstaller e il bundle Tauri macOS vanno
fatti su un Mac. Tutto automatizzato in **`build_macos.sh`** (root), speculare a `build_linux.sh`:
stesso `build_sidecar.spec`, sidecar `pii-backend` senza estensione (il Rust in
[`lib.rs`](../tauri/src-tauri/src/lib.rs) sceglie il nome col `cfg!(windows)`), bundle `app`/`dmg`
al posto di `nsis` — la conf Tauri resta su NSIS per Windows e qui si sovrascrive da CLI con
`--bundles`.

```bash
# prerequisiti (una volta)
xcode-select --install                 # Command Line Tools
# + Rust (https://rustup.rs) e Node.js 18+

bash build_macos.sh                    # .app + .dmg
bash build_macos.sh app                # solo il .app (piu' veloce, per provare)
VENV=build_env_macos bash build_macos.sh   # venv dedicato invece di .venv
```
Output: `tauri/src-tauri/target/release/bundle/macos/Rizzo PII.app` e `.../dmg/*.dmg`.

Note specifiche di macOS:
- **Niente indice PyTorch CPU**: su macOS la ruota di default è già CPU/MPS, quindi lo script
  installa `torch` senza `--index-url` (a differenza di Windows/Linux).
- **Firma**: il bundle **non è firmato né notarizzato**. In locale si apre senza problemi (un file
  che non arriva da internet non ha l'attributo di quarantena). Copiandolo su un altro Mac serve
  "tasto destro → Apri", oppure `xattr -dr com.apple.quarantine "Rizzo PII.app"`. Per distribuirlo
  davvero servono un Developer ID e la notarizzazione Apple.
- **Architettura**: il `.app` esce per l'arch della macchina che compila (arm64 su Apple Silicon).
  Per un binario universale servirebbero due build di PyInstaller + `lipo`.
- **Porta 5005**: se stai già facendo girare `python src/app/app.py` in sviluppo, fermalo prima di
  lanciare il `.app`, altrimenti il sidecar esce con codice 76 e lo splash chiede una porta nuova.

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
build_env\Scripts\python.exe -m pip install "transformers==4.57.3" tokenizers safetensors flask pymupdf pyinstaller onnx onnxruntime
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
- **Dimensione**: ~1,4 GB, quasi tutto modello (1,2 GB). torch serve solo nel venv per l'export
  ONNX: gli spec lo escludono dal pacchetto.
- **Rigenerare col modello definitivo**: il modello è impacchettato dalla cartella versionata
  `models\rizzo-pii-0.3B-v{VERSION}\` (vedi `datas` in `build.spec` / `build_sidecar.spec`; build
  attuale **v1.5.0**). Quando riaddestri una nuova versione, aggiorna quel path negli spec e rifai
  il passo 2 (e 3). Per provare senza modello si può puntare a `models\pii_model_legacy`.
- **console=True** in `build.spec` mostra una finestra con i log; mettila `False` per nasconderla
  (consigliato solo dopo che tutto funziona).
- **SmartScreen**: un exe non firmato mostra l'avviso "editore sconosciuto". Per la distribuzione
  serve un certificato di code signing.
