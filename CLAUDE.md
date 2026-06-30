# CLAUDE.md

Guida per Claude Code (claude.ai/code) quando lavora in questa repo. Panoramica e struttura
delle cartelle in **[README.md](README.md)**. Documenti di dettaglio:
**[docs/TASSONOMIA_TAG.md](docs/TASSONOMIA_TAG.md)** (i 22 tag) e
**[docs/DATASET.md](docs/DATASET.md)** (composizione completa di train/validation).

## Cos'è questo progetto

Pipeline per addestrare un modello **mmBERT** (`jhu-clsp/mmBERT-base`) a fare **token
classification di PII**, con focus su **testi legali italiani** (atti, contratti, sentenze)
ma con training **multilingue**. Obiettivo finale: anonimizzare documenti **in locale**
prima di mandarli a LLM closed (anonymize → placeholder + dizionario reversibile locale →
API → ricostruzione), per studi legali / compliance GDPR.

Scelta di mmBERT (encoder multilingue, architettura ModernBERT, context nativa 8192) e non
ModernBERT vanilla perché quest'ultimo è quasi solo inglese.

## Ambiente — vincoli critici e non ovvi

GPU **RTX 5060 Ti (Blackwell, sm_120)** su Windows; Python in `D:\programmi\python`.
Questi punti fanno fallire tutto se ignorati:

- **torch DEVE essere build cu128**: `torch 2.11.0+cu128` (da `https://download.pytorch.org/whl/cu128`).
  Le build cpu/cu121 non supportano sm_120 → `torch.cuda.is_available()` False o crash.
- **torchvision/torchaudio vanno disinstallati** se a versioni vecchie: rompono l'import di
  transformers con `operator torchvision::nms does not exist`. Non servono qui.
- **accelerate ≥ 1.14** (transformers 4.57 chiama `unwrap_model(keep_torch_compile=...)`).
- Windows: nel Trainer **`dataloader_num_workers=0`** (altrimenti `RuntimeError ... bootstrapping`).
- **seqeval non si compila** (bug setuptools_scm) → metriche entity-level (P/R/F1) calcolate
  a mano dentro gli script, nessuna dipendenza.
- Dipendenze extra installate: **`wandb`** (tracking) e **`python-dotenv`** (legge `.env`).
- **Log da PowerShell**: redirezioni `>` e `Tee-Object` scrivono **UTF-16** → leggerli con
  `Get-Content`/`python`, NON con lo strumento Read. Gli script forzano `sys.stdout` UTF-8.

## Mappa della repo

Struttura completa in [README.md](README.md). In breve: codice in `src/`, dati in `dataset/`,
modelli in `models/<versione>/`, artefatti dei run in `experiments/<run>/`, doc in `docs/`.
**I path negli script sono assoluti (risolti da `__file__`): girano da qualsiasi CWD.**

**Pipeline dati — `src/data_pipeline/` (ordine di esecuzione):**
- `llm_template_bank.py` — Gemini scrive documenti legali con soli segnaposto `{SLOT}` →
  `dataset/synthetic/legal_templates.json` (72 template). Guard scarta i template con nomi inline.
- `generate_synthetic_pii.py` — inietta nei template dati con **checksum validi** (CF/PIVA/IBAN)
  → `dataset/synthetic/synthetic_pii_it_200k.jsonl` (`tokens` + `bio_labels`).
- `augment_real_pii.py` — inietta entità sintetiche in **frasi reali** Ai4Privacy (it) in posizioni
  variabili → `dataset/synthetic/synthetic_pii_it_realaug.jsonl`. Spezza il legame template/posizione.
- `prepare_deepmount.py` — rimappa `DeepMount00/pii-masking-ita` (56 tipi Faker IT) sui 22 tag →
  `dataset/processed/deepmount_pii_it_{train,test}.jsonl`.
- `build_validation.py` — **unica validation reale** → `dataset/validation/validation_real.jsonl`.
- `build_subset.py` — subset stratificati (multilingua + tag) per smoke test →
  `dataset/subsets/{train_subset_10k,val_subset_5k}.jsonl`.

**Training — `src/training/`:**
- `train_pii.py` — fonde tutte le fonti, addestra, salva il modello, plotta la loss, stampa
  metriche train/val, logga su **W&B** (run `rizzo-pii:0.3B-v{VERSION}`). **Modello versionato**:
  ogni run grande salva in `models/rizzo-pii-0.3B-v{VERSION}/` (storico, niente sovrascrittura;
  `MODEL_VERSION` in cima al file o `--version`; storia in `models/registry.json`). Vedi "Parametri".
- `test_pii.py` — inferenza CLI sul modello salvato (entità + testo anonimizzato). Risolve in automatico
  l'**ultima** versione `rizzo-pii-0.3B-v*` (fallback al vecchio non versionato → legacy; override `PII_MODEL_DIR`).

**Utility / ispezione (read-only) — `src/inspect/`:**
- `validate_checksums.py` — ricalcola i checksum CF/IBAN/PIVA; **blueprint della rete regex+checksum**
  da affiancare al modello in produzione.
- `inspect_ai4privacy.py` (conteggi lingue/tag), `inspect_lengths.py` (lunghezze), `inspect_no_iban.py`.

**App di anonimizzazione locale — `src/app/` (+ packaging in `docs/BUILD.md`):**
- `server_config.py` — **configurazione host/porta** condivisa tra tutti gli entry point e con Tauri.
  Catena di precedenza: **CLI `--host`/`--port` > env `PII_HOST`/`PII_PORT` > `config.json` > default
  `127.0.0.1:5005`**. Il config.json è in `%LOCALAPPDATA%\rizzo-pii\` (Windows) /
  `~/.local/share/rizzo-pii/` (Linux) / `~/Library/Application Support/rizzo-pii/` (macOS), lo
  stesso file letto/scritto da Tauri (lib.rs). Include `port_available()` (pre-bind check) e il
  codice di uscita `EXIT_PORT_CONFLICT = 76`: i 3 entry point escono con 76 se la porta è occupata
  **prima di caricare il modello** (evita secondi sprecati). Tauri riconosce il codice 76 e mostra
  il form di configurazione nello splash screen.
- `app.py` — server Flask + **UI**: testo o PDF, chunking con overlap, offset globali + dedup.
  Anonimizzazione **reversibile** (ogni PII → `[FULLNAME_1]`/`[IBAN_1]`… + dizionario locale; tab
  "Ripristina"). Affianca al modello una **rete regex/checksum** (EMAIL/TELEFONO/IBAN/CF/PIVA/carta/
  importo/targa; IBAN/CF/PIVA/carta validati con checksum, che ha priorità sul modello). `APP_VERSION`.
  Endpoint `GET/POST /config` e `GET /port-check` per la configurazione host/porta dall'UI (⚙️
  gear icon nell'header); `--host`/`--port` come argomenti CLI.
- `serve.py` — entry **headless** (solo Flask, niente browser): è il backend dell'app Tauri; log su
  `%LOCALAPPDATA%\rizzo-pii\backend.log`. Pre-check porta + `sys.exit(76)` se occupata.
  `desktop_app.py` — entry PyInstaller legacy (apre il browser); stesso pre-check.
  `assets/` — mascotte (il riccio) + icone. `smoke_app.py`, `make_test_pdf.py`.

**App desktop Tauri — `tauri/`:** finestra nativa **Rizzo PII** (WebView2) che lancia il backend
`serve.py` impacchettato come **sidecar** (`build_sidecar.spec` → `tauri/src-tauri/backend/`).
All'avvio legge `config.json` (host/porta), passa i valori al sidecar via env `PII_HOST`/`PII_PORT`,
attende il server sulla porta configurata e mostra l'UI. Se il sidecar esce con codice **76** (porta
occupata), lo splash mostra un form di configurazione (host + porta) con "Salva e riprova": Tauri
scrive `config.json`, rilancia il sidecar e riprova. Splash con badge UE/GDPR + versione. `npx tauri build`
→ installer NSIS per-utente. Dettagli e comandi in `docs/BUILD.md`. Comandi Tauri esposti allo splash:
`save_config(host, port)` e `retry_backend`.

**Config:** `.env` (segreti W&B, **gitignorato**), `.gitignore`, `build.spec`/`build_sidecar.spec`/
`installer.iss` + `tauri/` (packaging).
**Artefatti non versionati** (gitignored): `dataset/`, `models/`, `experiments/`, `wandb/`, `build_env/`,
`dist/`, `build/`, `tauri/node_modules/`, `tauri/src-tauri/{target,gen,backend}/`.

## Idea architetturale chiave: "LLM autore, codice etichettatore"

Il cuore della generazione sintetica: **l'LLM scrive solo la prosa con segnaposto, il codice
inietta i dati**. Risolve insieme tre problemi: label BIO esatte (sappiamo dove iniettiamo),
checksum matematicamente validi, nessuna PII reale rigurgitata dall'LLM. **Non far mai scrivere
all'LLM i dati sensibili veri.**

## Tassonomia: 22 tag (dettaglio in TASSONOMIA_TAG.md)

`TAG_MAP` + `DROP_TYPES` + `normalize_labels()` in `train_pii.py` rimappano **al caricamento**
(file grezzi intatti). Per cambiare la tassonomia si edita **solo** `TAG_MAP`/`DROP_TYPES`.

Fusioni: nomi+ruoli legali → `FULLNAME`; `SEX`→`GENDER`; `TAXNUM`→`PIVA`; `PEC`→`EMAIL`;
`RG`→`DOCID`; `IDCARDNUM`/`PASSPORTNUM`/`DRIVERLICENSENUM`/`SOCIALNUM`→`ID_DOC`; `CONTO`→`IBAN`.
Rimossi (→`O`): `TITLE` (appellativo), `TRIBUNAL` (ente pubblico, non PII).
Tag aggiunti via sintetico: `ORG`, `DOCID`, `CATASTO`, `CONTO`(→IBAN), `PROVINCE`.

I 5 tag legali IT-specifici (`CF`, `PIVA`, `CATASTO`, `DOCID`, `PROVINCE`) non esistono come
dato reale da nessuna parte → vengono solo dai sintetici.

## Fonti dati (dettaglio in DATASET.md)

Quattro fonti, tutte ricondotte ai 22 tag al caricamento:
1. **Ai4Privacy** `open-pii-masking-500k` — reale, **8 lingue**, ~464k righe train (it ~55k).
   `hf download ai4privacy/open-pii-masking-500k-ai4privacy`. Ha `mbert_tokens`+`mbert_token_classes`.
2. **Sintetico da template** — `generate_synthetic_pii.py`, 200k righe, copre i tag legali IT.
3. **Augment** — `augment_real_pii.py`, 40k, entità sintetiche in frasi reali it.
4. **DeepMount** `DeepMount00/pii-masking-ita` — Faker IT, 41k; dà contesto reale a IBAN/ORG/AMOUNT/TARGA.

Train pool ≈ **745k righe** (multilingue; italiano rinforzato al ~45%). ~38% sintetico.

## Validation: UNA validation reale unificata

`validation_real.jsonl` (7k righe, **solo italiano**, da `build_validation.py`):
- base reale held-out = Ai4Privacy val (it) + DeepMount test;
- i 5 tag senza dato reale sono **iniettati in frasi reali held-out** (frasi della validation
  Ai4, non nel training) → contesto reale, niente leakage.

Tutti i sintetici e il DeepMount **train** stanno nel pool di training; il DeepMount **test**
è consumato solo nella validation. Scelta: validation italiana perché l'uso reale è il dominio
legale IT (il training resta multilingue; le altre lingue non sono validate).

## Parametri di training importanti (`src/training/train_pii.py`)

- `LANG = None` → **multilingue** (8 lingue Ai4Privacy); `"it"` = solo italiano. Synth/DeepMount
  sempre inclusi (già italiani, rinforzo).
- `MAX_LEN = 768` → copre i sintetici (max 771 subword) e DeepMount (660); con 512 si troncava il
  **33% dei sintetici**. Padding dinamico.
- `BATCH = 16` + `GRAD_ACCUM = 2` (batch **effettivo 32**) → a MAX_LEN 768 su una 16 GB **condivisa
  col desktop**, batch più grandi saturano la VRAM e causano **thrashing** (allocatore CUDA che
  libera/ri-alloca ad ogni batch lungo → 24-37 s/step). Con 16 il picco di attivazioni resta ~8-9 GB.
  Altre mitigazioni: `group_by_length=True` (`LengthGroupedTrainer` usa lunghezze precalcolate per
  non ri-tokenizzare il dataset lazy) e `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
  Chiudere le app che usano la GPU (Chrome/WhatsApp/…) libera 1-2 GB.
- `EPOCHS = 1`. L'eval **non** gira durante il training: train loss a ogni step, **metriche P/R/F1
  ALLA FINE** su validation e su un campione di train (`TRAIN_EVAL_N`).
- **Modalità del run** (`--type {full,subset}`, default `full`; anche `PII_SUBSET=1` forza subset):
  `subset` = smoke test / tuning sui subset `dataset/subsets/` (10k/5k), MAX_LEN 256, BATCH 32,
  ~3 min → `experiments/subset_smoke/`. `full` = run grande.
- Output run grande: `models/rizzo-pii-0.3B-v{VERSION}/` (+ tokenizer) e `experiments/full_run_v{VERSION}/`
  (loss + out/) + append a `models/registry.json`. La versione viene da `MODEL_VERSION` o `--version`.

## Weights & Biases

`train_pii.py` carica `.env` con `load_dotenv()`; se c'è `WANDB_API_KEY` attiva
`report_to=["wandb"]` (progetto da `WANDB_PROJECT`, default impostato a `pii-mmbert-it`).
Logga la **train loss live a ogni step** e le **metriche finali** (`final/train_*`, `final/val_*`).
Il `.env` è **gitignorato**: non committarlo. Senza chiave, W&B si disattiva da solo.

## Comandi

Tutti gli script forzano UTF-8 e risolvono i path da soli (girano da qualsiasi CWD; i comandi
sotto si lanciano dalla root per comodità).

```powershell
# 1) (opz.) template legali via Gemini   (richiede GEMINI_API_KEY)
python src/data_pipeline/llm_template_bank.py --per-type 5 --append

# 2) sintetico da template (200k)
python src/data_pipeline/generate_synthetic_pii.py -n 200000 --out dataset/synthetic/synthetic_pii_it_200k.jsonl

# 3) augment in testo reale (40k)
python src/data_pipeline/augment_real_pii.py -n 40000 --out dataset/synthetic/synthetic_pii_it_realaug.jsonl

# 4) DeepMount rimappato 56->22 tag  (richiede login HF: hf auth login)
python src/data_pipeline/prepare_deepmount.py

# 5) validation reale unificata (7k)
python src/data_pipeline/build_validation.py

# 6) subset stratificati per smoke test (10k/5k)
python src/data_pipeline/build_subset.py

# 7a) smoke test / tuning sul subset (~3 min)  -> experiments/subset_smoke/
python src/training/train_pii.py --type subset
# 7b) run grande su tutto  -> models/rizzo-pii-0.3B-v{VERSION}/ + experiments/full_run_v{VERSION}/ + W&B
python src/training/train_pii.py --type full                  # usa MODEL_VERSION nel file
python src/training/train_pii.py --type full --version 1.2.0  # oppure versione esplicita

# inferenza / app
python src/training/test_pii.py "Mi chiamo Mario Rossi, IBAN ..."
python src/app/app.py            # http://127.0.0.1:5005  (PII_PORT per override; 5000 = AirPlay su macOS)

# ispezione read-only
python src/inspect/inspect_ai4privacy.py
python src/inspect/inspect_lengths.py
```

## Limiti noti / aspettative oneste

- **Overfit strutturale**: i tag che vengono solo dai template rischiano di imparare la
  *struttura* invece dell'entità. Mitigazioni in atto: 72 template (non più ~29), augment in
  testo reale, e il contesto vario di DeepMount per IBAN/ORG/AMOUNT/TARGA.
- **Validation solo italiana**: non misura le 7 lingue non-it (scelta voluta).
- **Tag legali IT-only** (`CF`/`PIVA`/`CATASTO`/`DOCID`/`PROVINCE`): in validation sono entità
  generate in frasi reali → buon proxy, non eval completamente cieco. `PROVINCE` nel train ha
  poca diversità di contesto (quasi solo dai template).
- **Sbilanciamento di classe**: FULLNAME ≫ CREDITCARDNUMBER (~66×) → i tag rari sono più rumorosi.
- **Valori off-domain di DeepMount**: nomi/indirizzi USA; utili per forma/contesto, non come
  valori italiani.
- In produzione affiancare **sempre** la rete regex+checksum (`src/inspect/validate_checksums.py`) al modello.
