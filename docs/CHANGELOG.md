# Changelog / note di modifica

Registro delle modifiche significative alla pipeline di training, con motivazione.
Le voci più recenti in alto. (Codice: `src/training/train_pii.py` salvo diverso.)

---

## 2026-08-03 — `_merge()` era quadratica: 100 s su un documento lungo (`app.py`)

Il controllo delle sovrapposizioni confrontava **ogni** candidato con **tutta** la lista già
tenuta, e `analyze()` chiama `_merge()` una volta sull'**intero documento**, non per chunk.
Il costo esplode dopo l'inferenza, quando l'utente crede di aver finito (issue #61):

| entità nel documento | prima | dopo |
|---:|---:|---:|
| 5.000 | 1,14 s | **0,025 s** |
| 20.000 | 21,25 s | **0,181 s** |
| 40.000 | 111,35 s | **0,398 s** |
| 100.000 | non misurata (ore) | **2,08 s** |

Le entità tenute sono per costruzione ordinate e non sovrapposte, quindi un candidato può
accavallarsi solo con i due vicini: una ricerca binaria li trova subito.

Comportamento invariato, verificato confrontando l'output della funzione intera fra vecchia e
nuova implementazione: **60.000 casi casuali** (span sovrapposti, annidati, densi, punteggi
tutti uguali per far contare la stabilità del `sorted`, 64.822 candidati di lunghezza zero) e
**48 casi costruiti** (lista vuota, lunghezza zero dentro e fuori un altro span, candidati
identici, stesso `start`, annidamento nei due ordini, adiacenti, sovrapposti di un carattere)
→ **0 output diversi**.

Non risolve da solo la issue #61: lì il grosso è l'inferenza su CPU. Questo è il pezzo che
resta lento anche dopo.
## 2026-08-02 — "Pulisci" non cancellava il dizionario PII (`src/app/app.py`)

Il bottone **Pulisci** nascondeva la card del dizionario ma lasciava i valori in `MAP` e in
`localStorage['pii_map']`: nomi, CF e IBAN in chiaro restavano sul disco, e dall'interfaccia
non c'era nessun modo di toglierli (`removeItem` non compariva da nessuna parte nel repo).

Non è solo igiene. All'avvio il dizionario della sessione precedente viene ricaricato in `MAP`,
quindi la guardia di `reverse()` — «Nessun dizionario: caricane uno .json» — non scatta mai:
chi riapre l'app e incolla la risposta dell'LLM di oggi **senza** caricare il `.json` si vede
sostituire i valori del caso precedente, con l'esito «Valori ripristinati».

    dizionario rimasto sul disco: {"[FULLNAME_1]": "Mario Rossi", "[IBAN_1]": "IT60X054…"}
    "Il ricorrente [FULLNAME_1] chiede il bonifico su [IBAN_1]."
      ->  "Il ricorrente Mario Rossi chiede il bonifico su IT60X054…"

Ora «Pulisci» azzera `MAP`, rimuove `pii_map` da `localStorage` e ripulisce l'etichetta: è
quello che l'UI già dichiara in italiano e in inglese («il dizionario **di questa sessione**…
se hai chiuso e riaperto l'app, **carica il dizionario .json**»). Nient'altro cambia.
## 2026-08-03 — La porta poteva risultare libera mentre era occupata (`src/app/server_config.py`)

`port_available()` provava a legare solo `host:porta`. Se un altro programma ascolta su
`0.0.0.0:5005`, il bind su `127.0.0.1:5005` riesce lo stesso: il controllo diceva **libera**,
l'app partiva, e da quel momento non è definito chi serve le connessioni — l'utente può
ritrovarsi sul server dell'altro programma. È il sintomo riportato nella #17: «You're speaking
plain HTTP to an SSL-enabled server port».

Ora si prova a legare anche l'indirizzo jolly, **senza** `SO_REUSEADDR`: quando ce l'ha anche
il socket dell'altro programma — Werkzeug la imposta di default, quindi vale per qualsiasi
Flask — su Windows il bind riesce lo stesso, ed è proprio il caso da rilevare. Se il jolly
risulta occupato si guarda, con una connessione, se qualcuno risponde davvero su `host:porta`:
potrebbe ascoltare su un'altra interfaccia, e allora la nostra è libera.

| chi occupa la porta | prima | dopo |
|---|---|---|
| nessuno | libera | libera |
| `127.0.0.1:5005` | occupata | occupata |
| **`0.0.0.0:5005`** | **libera** | **occupata** |
| un'altra interfaccia | libera | libera |

Costo: 0,12 ms a chiamata quando la porta è libera. Vale per i tre entry point (`app.py`,
`serve.py`, `desktop_app.py`), che escono con `EXIT_PORT_CONFLICT`, e per `/port-check`.
## 2026-08-03 — Con host `localhost` l'app desktop non partiva mai (`tauri/src-tauri/src/lib.rs`)

`is_our_backend()` costruiva `"{host}:{port}"` e lo passava a `parse::<SocketAddr>()`, che
accetta **solo IP letterali**: con `host = localhost` (valore che si può scrivere nel form
dello splash o passare in `PII_HOST`) il parse fallisce e la funzione restituisce sempre
`false`. Il backend parte e scrive nel log che è pronto, ma `poll_backend` non lo riconosce:
900 iterazioni × 200 ms → **~180 secondi** di splash, poi «il backend non si è avviato».

Ora l'indirizzo si risolve con `to_socket_addrs()` e si provano **tutti** gli indirizzi
restituiti, non il primo: su Windows `localhost` risolve prima `::1` e poi `127.0.0.1`, e il
backend può ascoltare solo sul secondo. I 500 ms restano il budget dell'**intera chiamata** e
vengono divisi fra gli indirizzi: `poll_backend` la ripete 900 volte, e senza questo l'attesa
prima dell'errore sarebbe passata da ~10 a ~18 minuti.

| host | prima | dopo |
|---|---|---|
| `127.0.0.1` | riconosciuto | riconosciuto |
| **`localhost`** | **mai riconosciuto** | riconosciuto |
| servizio estraneo sulla porta | `false` | `false` |
| nessuno in ascolto | `false` | `false` |

Il controllo che distingue il nostro Flask da un servizio estraneo (`GET /config` → il corpo
contiene `config_path`) resta invariato.
## 2026-07-31 — Span delle entità allineate ai confini di parola (`app.py`)

Il modello etichetta i **sotto-token** e a volte ne copre solo una parte: `' No'` di
`' No' + 'vara'`. Sostituendo la span così com'è, l'`anonymized_text` conservava frammenti
leggibili — `Direzione Provinciale di [CITY_2]vara` — da cui il valore originale è banalmente
ricostruibile. In un anonimizzatore è **peggio di un falso negativo pulito**: l'utente vede il
segnaposto e conclude che il dato sia stato rimosso.

`_merge` ora, dopo aver rifilato gli spazi, **estende ogni span fino a coprire per intero le
parole che tocca**, e fonde le span che l'estensione rende sovrapposte o adiacenti con la stessa
etichetta (prima `foglio 21` → `[CATASTO_1][CATASTO_2]`). L'estensione è sempre nella direzione
sicura: nel dubbio si maschera un carattere in più. Nessun impatto sul training.

Riscontrato su 5 documenti sintetici (perizia CTU, ricorso tributario, relazione dell'organo di
controllo, istanza di composizione negoziata, verbale assembleare): spariscono tutti i frammenti
osservati su `CITY`/`DATE`/`DOCID`/`CATASTO`/`ORG`/`TARGA`, le PII dirette restano mascherate
23/23. Resta fuori portata il caso del richiamo parziale su entità multi-parola
(`Guardia di Finanza` → `Guardia di [ORG_2]`), che è un problema del modello, non della
sostituzione. Rif. issue #11.

---

## 2026-06-30 — Porta del backend 5000 → 5005 (conflitto AirPlay su macOS)

Gli utenti macOS vedevano una **pagina bianca**: la porta **5000** è occupata di default
dall'**AirPlay Receiver** (ControlCenter), quindi il WebView Tauri si collegava al servizio
sbagliato. Backend spostato su **5005** (`app.py`, `serve.py`, `desktop_app.py` con default
`PII_PORT=5005`; `lib.rs` `ADDR`/`URL` aggiornati). Override sempre possibile con la env
`PII_PORT`. Nessun impatto sul training. Richiede rebuild/ri-notarizzazione del bundle macOS.

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
