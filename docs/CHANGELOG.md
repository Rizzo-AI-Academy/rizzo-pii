# Changelog / note di modifica

Registro delle modifiche significative alla pipeline di training, con motivazione.
Le voci più recenti in alto. (Codice: `src/training/train_pii.py` salvo diverso.)

---

## 2026-06-28 — App di anonimizzazione: revisione completa + app desktop Tauri

Riscrittura dell'app locale (`src/app/`) e nuovo packaging desktop. Nessun impatto su training.

**1. Anonimizzazione reversibile** (`app.py`). Ogni PII riceve un **ID univoco** (`[FULLNAME_1]`,
`[IBAN_1]`…); valori identici condividono lo stesso ID → l'LLM resta coerente e il **reverse è 1:1**.
Si genera un **dizionario locale** `{placeholder → valore}` scaricabile in `.json`; nuovo tab
**"Ripristina"** che rimette i valori veri nella risposta dell'LLM (matching tollerante a parentesi
alterate / grassetto markdown). Tutto in locale.

**2. Rete regex + checksum** a supporto del modello (`detect_regex` + `_merge`). Detector per
EMAIL/TELEFONO/IBAN/CF/PIVA/carta/importo/targa. IBAN/PIVA/carta richiedono il **checksum valido**
(mod-97 / Luhn) per non avere falsi positivi; il CF si redige sulla sola forma (molto specifica) e
prende il ✓ solo se il checksum passa. Priorità in caso di sovrapposizione: **checksum-valido ›
regex › modello** (risolve la frammentazione di CF/IBAN del modello).

**3. UI rifatta** (tema chiaro, flusso a 2 step, highlight a colori per tag, hover col valore
originale, legenda cliccabile, drag&drop PDF). **Fix layout**: altezza fissa a finestra → lo scroll
avviene **dentro la textarea e l'anteprima**, non sulla pagina.

**4. Mascotte** (il riccio): logo header + favicon (`mascot_shield`) ed empty state (`mascot_doc`),
serviti da `/assets/` con fallback emoji. Asset in `src/app/assets/`.

**5. App desktop Tauri** (`tauri/`). Architettura **sidecar**: il backend Python/Flask
(`serve.py`, headless) è impacchettato con PyInstaller (`build_sidecar.spec`, CPU, ~1,8 GB col
modello) in `tauri/src-tauri/backend/`; la finestra nativa **Rizzo PII** (WebView2) lo lancia come
processo figlio, attende il server su `127.0.0.1:5000`, mostra l'UI e lo termina alla chiusura.
Splash con badge **UE / GDPR compliant**, **versione** (iniettata da Rust dalla config) e crediti
nell'app (Simone Rizzo · Rizzo AI Academy). `npx tauri build` → **installer NSIS per-utente**
(`Rizzo PII_1.0.0_x64-setup.exe`, ~1,3 GB, non firmato → avviso SmartScreen atteso).

**6. Pulizia repo.** Rimossi output/cache rigenerabili: vecchia build PyInstaller `dist/`, intermedi
`build/`, `__pycache__/`, `_archive/`, e log W&B stray in `src/training/`. `.gitignore` esteso agli
artefatti Tauri. README/CLAUDE/BUILD aggiornati.

---

## 2026-06-28 — Primo run grande `rizzo-pii:0.3B`: risultati e fix PROVINCE

Primo training completo (1 epoca, ~1h40, BATCH 16 ×2). Modello salvato in
`models/rizzo-pii-0.3B/`. Valutazione per-tag su `validation_real.jsonl` (nuovo
`src/training/evaluate_pii.py`, report in `experiments/full_run/eval_validation.*`):

- **F1 micro (overall) = 0,977** · precision 0,989 · recall 0,965 · token-acc 0,997. Forte.
- Quasi tutti i tag 0,95-1,00; `CATASTO`/`CF`/`PIVA`/`ID_DOC`/`DOCID`/`GENDER`/`TARGA` = 1,000.
- **`PROVINCE` = 0,000** (support 400): unico fallimento totale.

Causa-radice (diagnosticata): nei sintetici `PROVINCE` appare **quasi solo** come `Citta' (XX)`
(sigla tra parentesi dopo una città), e nell'**augment era del tutto assente** (0 occorrenze) —
l'unico tag IT-only mai mostrato in testo reale con connettori vari. La validation la testa come
`in provincia di XX` / `prov. XX` → contesto mai visto → il modello predice `O`. Overfit
strutturale puro (gli altri 8 tag iniettati stanno nell'augment → fanno ~1,0).

Fix applicato: aggiunti snippet `PROVINCE` a `INJECTION_SNIPPETS` in `augment_real_pii.py`
(`in provincia di {PROVINCE}`, `prov. {PROVINCE}`, `Prov. di {PROVINCE}`, `in provincia ({PROVINCE})`).
**Per avere effetto serve rigenerare l'augment + riaddestrare** (vedi comandi sotto).
Nota pratica: in produzione `PROVINCE` è un set chiuso (~110 sigle valide) → catturabile con
gazetteer/regex nella rete di sicurezza, quindi è il tag meno critico da sbagliare.

Inoltre: il salvataggio di `metrics.{json,txt}` ora avviene anche per il run **full** (prima
solo subset) → i prossimi run grandi scrivono le metriche in `experiments/full_run/`.

Rigenerare + riaddestrare per la fix PROVINCE:
```powershell
python src/data_pipeline/augment_real_pii.py -n 40000 --out dataset/synthetic/synthetic_pii_it_realaug.jsonl
python src/data_pipeline/build_subset.py        # opz., aggiorna i subset
python src/training/train_pii.py --type full
```

---

## 2026-06-28 — VRAM al limite durante il run grande: BATCH 16 + accumulo gradiente

Sintomo (run grande, osservato): ETA ~1h che di colpo schizza a ~23h in concomitanza col
caricamento dei batch lunghi. Diagnosi con `nvidia-smi` durante il training: **memoria GPU a
16006/16311 MiB (98%, satura)**. Causa: a `BATCH=24`/`MAX_LEN=768` i batch di documenti lunghi
(raggruppati da `group_by_length`) chiedono ~12 GB di attivazioni; con la VRAM già piena — anche
per le app desktop che usano la GPU (Chrome, WhatsApp, Claude, ecc., ~1-2 GB) — l'allocatore CUDA
va in thrashing sui cambi di batch. (RAM di sistema 51 GB: non è quello il limite.)

Fix:
- `BATCH` 24 → **16**: picco attivazioni ~8-9 GB, margine ampio anche col desktop sulla GPU.
- `gradient_accumulation_steps = 2` (`GRAD_ACCUM`): **batch effettivo 32**, a costo VRAM di 16
  (qualità del gradiente invariata/migliore, niente thrashing).
- `EVAL_EVERY` ora calcolato sugli **step ottimizzatore** (`microbatches // GRAD_ACCUM`), così
  l'eval intermedia resta a ~4 valutazioni reali.

Consiglio operativo: chiudere le app che usano la GPU (Chrome/WhatsApp/Video) per liberare 1-2 GB.

---

## 2026-06-28 — Flag `--type {full,subset}` per scegliere il run

La modalità si seleziona da riga di comando invece che con la variabile d'ambiente:
`python src/training/train_pii.py --type full` (default) o `--type subset`. Per compatibilità
`PII_SUBSET=1` continua a forzare la modalità subset. Implementato con `argparse` (parse_known_args)
in cima allo script.

---

## 2026-06-28 — Iperparametri di training: warmup, weight decay, eval intermedia

Tre fix standard al `TrainingArguments` del run grande, decisi prima del primo run completo
di `rizzo-pii:0.3B`. Applicati anche al subset (modalità `PII_SUBSET=1`).

| Modifica | Prima | Dopo | Perché |
|---|---|---|---|
| `warmup_ratio` | assente (0) | **0.05** | ~5% di step di riscaldamento (~1.300 su ~27k). La testa di classificazione è inizializzata da zero: partire a LR pieno destabilizza i primi step. È la mancanza più importante. |
| `weight_decay` | 0.0 | **0.01** | Regolarizzazione AdamW canonica per il fine-tuning di transformer. Piccolo guadagno atteso, rischio nullo. |
| `eval_strategy` | `"no"` | `"steps"`, `eval_steps = steps_per_epoch // 4` | ~4 valutazioni (solo `eval_loss`) durante l'epoca → su W&B si vede la curva **train-vs-val** e si colgono overfit/anomalie senza aspettare la fine del run (4-5h). Le metriche **P/R/F1 entity-level restano calcolate ALLA FINE** (sezione 6 dello script), come prima. |

**Costo dell'eval intermedia**: trascurabile. ~4 pass forward sulla validation (7k righe, run
grande) a `EVAL_BATCH=64` → ~minuto a valutazione, contro 4-5h di training.

**Note di implementazione**:
- `EVAL_EVERY = max(1, steps_per_epoch // 4)`: cadenza relativa, così vale sia per il run grande
  (~27k step → eval ogni ~6.700) sia per il subset (313 step → eval ogni ~78).
- `save_strategy="no"` invariato: niente checkpoint su disco, niente selezione del best model.
  L'eval intermedia serve solo a **osservare** la curva, non a fare early-stopping.
- Il plot di fine run continua a usare solo i punti di *train* loss (le voci di eval nel
  `log_history` hanno chiave `eval_loss`, non `loss`, quindi non inquinano il plot).
- `EVAL_BATCH` 64 → **32**: con l'eval ora *durante* il training (optimizer residente in VRAM),
  un batch di eval 64×768 rischiava OOM. 32 lo evita; l'eval è infrequente, costo trascurabile.

**LR lasciato a 5e-5**: scelta sicura per mmBERT/ModernBERT. Da rivedere solo guardando W&B.

### Da valutare DOPO il primo baseline (non ancora applicati)
Decisioni rimandate, da prendere guardando le metriche **per-tag** su W&B:
- `EPOCHS=2` se a fine epoca 1 la val F1 sta ancora salendo (occhio all'overfit sui sintetici).
- `gradient_accumulation_steps=2` (batch effettivo 48) per gradienti più lisci, costo ~nullo.
- `LR=8e-5` se converge lento; `3e-5` se instabile.
- Pesi di classe / focal loss se i tag rari (`TARGA`, `CREDITCARDNUMBER`) hanno recall basso
  (sbilanciamento FULLNAME ≫ CREDITCARDNUMBER ~66×).

> ⚠️ Il subset 10k **non** è il banco per tunare LR/epoche: le dinamiche (numero di step ~30×
> minore, regime di scheduler diverso) non rispecchiano 645k×1epoca. Serve solo a validare la
> pipeline. Il tuning vero si fa sul run grande osservando W&B.

---

## 2026-06-28 — Performance VRAM: fix del thrashing a MAX_LEN 768

Diagnosi (misurata con micro-benchmark): a `MAX_LEN=768` un batch denso da 32 usa **15,5 GB su
17,1** → l'allocatore CUDA, vicino al tetto, libera/ri-alloca blocchi grossi a ogni batch lungo
(padding dinamico) causando **thrashing**: primi 2 step veloci, poi 24-37 s/step (~3h per 10k).
L'attenzione era già `sdpa` (non era quello il problema).

Fix applicati:
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (impostato in cima allo script, prima di
  importare torch) — riduce la frammentazione. Vale per tutti i run.
- `BATCH` 32 → **24** per il run grande (margine VRAM).
- `group_by_length=True` + `LengthGroupedTrainer`: raggruppa sequenze di lunghezza simile (meno
  padding sprecato, batch a memoria uniforme). Usa **lunghezze precalcolate** (conteggio parole)
  per non ri-tokenizzare il dataset lazy — il difetto del `group_by_length` standard.
- Modalità SUBSET: `MAX_LEN=256`, `BATCH=32` → smoke test da ~3 min (era ~3h).

Risultato subset: training in ~108 s, VRAM picco 9,2 GB.

---

## 2026-06-28 — Subset rappresentativi per smoke test/tuning

Nuovo `src/data_pipeline/build_subset.py`: genera `dataset/subsets/train_subset_10k.jsonl` e
`val_subset_5k.jsonl`, stratificati per `(fonte × lingua)` + floor sui tag rari (proporzionale +
floor). Attivati nel training con `PII_SUBSET=1` → artefatti in `experiments/subset_smoke/`.
Servono a validare la pipeline e fare cicli rapidi prima del run grande.

---

## 2026-06-28 — Riorganizzazione della repo

Struttura professionale: codice in `src/{data_pipeline,training,inspect,app}/`, dati in
`dataset/{raw,synthetic,processed,validation,subsets}/`, modelli in `models/<versione>/`,
artefatti dei run in `experiments/<run>/`, documentazione in `docs/`. Tutti i path negli script
sono assoluti (risolti da `__file__`): girano da qualsiasi CWD. Modello di produzione rinominato
`models/rizzo-pii-0.3B` (precedente conservato in `models/pii_model_legacy`). `build.spec`,
`app.py`, `.gitignore` e i doc aggiornati di conseguenza. Dettaglio struttura in
[../README.md](../README.md).
