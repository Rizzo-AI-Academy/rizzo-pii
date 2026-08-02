# Training — dati, tag, iperparametri

Procedura del run grande di `src/training/train_pii.py`. Composizione dei dataset in
[DATASET.md](DATASET.md), tassonomia in [TASSONOMIA_TAG.md](TASSONOMIA_TAG.md).

## 1. Dati: unione dei due dataset Hugging Face

Dalla v1.3.0 il training non parte più dai soli file generati in locale: si scaricano
**entrambi** i dataset pubblicati e si usa **l'unione dei train per il training e
l'unione delle validation per la validation**.

| Dataset | Train | Validation | Cosa aggiunge |
|---|--:|--:|---|
| [`rizzoaiacademy/rizzo-pii-it-dataset`](https://huggingface.co/datasets/rizzoaiacademy/rizzo-pii-it-dataset) | ~745k | 7.000 | pool storico: Ai4Privacy multilingue, sintetico da template, augment, DeepMount. È il dataset con cui è stata fatta la prima versione |
| [`rizzoaiacademy/anonimizzazione-testi-italiano-clean`](https://huggingface.co/datasets/rizzoaiacademy/anonimizzazione-testi-italiano-clean) | 1.431.762 | 29.297 | corpus community deduplicato e bilanciato, italiano, prosa legale sintetica lunga (media ~1.100 caratteri). Validation **template-disjoint** dal suo train |
| **Unione** | **~2,18M** | **36.297** | |

Perché unirli e non sostituire: il primo porta il **multilinguismo** (8 lingue di
Ai4Privacy) e il testo reale di DeepMount, il secondo porta **volume e densità di
entità** sull'italiano legale. Tenere solo il secondo renderebbe il modello monolingue;
tenere solo il primo lascerebbe fuori il grosso dei dati.

Sulle validation vale lo stesso: unirle misura sia il dominio storico sia quello nuovo.
Sono **complementari, non sovrapposte** — se si valuta solo su una delle due si prendono
decisioni sbagliate. Caso reale: confrontando due modelli sulla sola `validation_real`
uno sembrava migliore su `FULLNAME` e `CATASTO`; sulla validation `-clean` lo stesso
confronto si ribaltava su **tutti** i tag.

> ⚠️ Le due validation hanno formati diversi: la prima è JSONL con `tokens` +
> `bio_labels` (label **grezze**: `GIVENNAME`, `SURNAME`, `TITLE`, `SEX`…), la seconda è
> parquet con in più `source_text` ed `entities` con offset di carattere e label già sui
> 22 tag. Entrambe passano comunque da `normalize_labels()` al caricamento (§2), che è
> idempotente sui tag già normalizzati.

### Scaricare e comporre l'unione

```powershell
# 1) dataset storico -> dataset/  (mantiene la struttura processed/ synthetic/ validation/)
hf download rizzoaiacademy/rizzo-pii-it-dataset --repo-type dataset --local-dir dataset

# 2) dataset clean (parquet) -> dataset/clean/
hf download rizzoaiacademy/anonimizzazione-testi-italiano-clean --repo-type dataset --local-dir dataset/clean

# 3) parquet -> jsonl nel formato atteso dal loader (tokens + bio_labels)
python - <<'PY'
import json, glob
from pathlib import Path
import pyarrow.parquet as pq

D = Path("dataset")
for split, out in (("train", D/"synthetic"/"clean_train.jsonl"),
                   ("validation", D/"clean"/"validation_clean.jsonl")):
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for shard in sorted(glob.glob(f"dataset/clean/data/{split}-*.parquet")):
            for r in pq.read_table(shard, columns=["tokens", "bio_labels"]).to_pylist():
                f.write(json.dumps({"tokens": r["tokens"],
                                    "bio_labels": r["bio_labels"]}, ensure_ascii=False) + "\n")
    print(out)

# validation unificata = validation_real + validation clean
with (D/"validation"/"validation_union.jsonl").open("w", encoding="utf-8") as f:
    for src in (D/"validation"/"validation_real.jsonl", D/"clean"/"validation_clean.jsonl"):
        f.write(src.read_text(encoding="utf-8"))
PY
```

Poi in `train_pii.py`: aggiungere `dataset/synthetic/clean_train.jsonl` a `SYNTH_PATHS` e
puntare `VALIDATION_PATH` a `dataset/validation/validation_union.jsonl`.

## 2. Accortezza sui tag (obbligatoria, altrimenti il modello si rompe sui codici)

Due passaggi distinti, entrambi in `train_pii.py`. **Nessuno dei due tocca i dataset**,
che restano quelli pubblicati.

### 2.1 Normalizzazione della tassonomia — `TAG_MAP` / `DROP_TYPES`

I file grezzi hanno più tipi dei 22 finali. `normalize_labels()` rimappa al
**caricamento** (nomi e ruoli legali → `FULLNAME`, `PEC` → `EMAIL`, `CONTO` → `IBAN`,
`IDCARDNUM`/`PASSPORTNUM`/… → `ID_DOC`) e manda a `O` ciò che non è PII (`TITLE`,
`TRIBUNAL`). Per cambiare tassonomia si edita **solo** `TAG_MAP`/`DROP_TYPES`.

### 2.2 Propagazione della label sui subword — `LABEL_ALL_SUBWORDS`

Il tokenizer spezza ogni parola in subword. La label del dataset è **una per parola** e
va distribuita su quei pezzi:

```
'Mario'             -> 1 subword     nessun pezzo interno
'MNTCRL58D07H163B'  -> 13 subword    12 pezzi interni
'IT60X05428111…'    -> 26 subword    25 pezzi interni
```

Fino alla v1.2.0 la label andava **solo al primo subword**, gli altri prendevano `-100`
= esclusi dalla loss. Quelle posizioni non venivano **mai** addestrate, ma in inferenza
la pipeline (`aggregation_strategy="simple"`) le legge comunque: il modello rispondeva a
caso con score ~1.0 e lo span si frantumava.

```
C.F. RCCMRT60T58H703I  ->  CF:'RCC' + CF:'M' + ID_DOC:'RT60T58H7' + TARGA:'I'
```

Misurato sulla validation `-clean` (3.000 righe, span esatto): `CF`, `PIVA`, `IBAN` a
**F1 0,000**, mentre `FULLNAME` — una parola, un subword, nessun pezzo interno — stava a
**0,987**. Stessi dati, stessa qualità di etichettatura: la differenza era solo in
quanti subword fa quella parola.

Ora il subword interno **eredita la label come continuazione** (`B-X → I-X`; `O` e `I-X`
restano invariati), così ogni posizione letta in inferenza è supervisionata.
`LABEL_ALL_SUBWORDS = False` ripristina il comportamento vecchio per un confronto A/B.

**Corollario non ovvio**: per ogni `B-X` deve esistere la classe `I-X` anche se nei dati
non compare mai. Otto tag hanno solo `B-` perché l'entità non supera mai il token —
`CF`, `PIVA`, `CONTO`, `ZIPCODE`, `PROVINCE`, `GENDER`, `IDCARDNUM`, `DRIVERLICENSENUM`
(verificato sulla validation `-clean`: `B-CF` 57.995, `I-CF` **0**). Senza estendere il
set, `label2id.get("I-CF")` non esisterebbe e i subword interni cadrebbero in silenzio
su `O`, **insegnando** che dentro un codice fiscale non c'è niente — peggio del bug. Da
qui la riga:

```python
label_set |= {"I-" + l[2:] for l in list(label_set) if l.startswith("B-")}
```

Non sono tag nuovi: la tassonomia resta **22**. È la metà `I-` di tag che esistevano già.

Conseguenze pratiche sul run:
- il numero di classi cresce di ~8 → la testa cambia dimensione, **non si riprende un
  checkpoint precedente**, si riparte da `mmBERT-base`;
- `token_acc` e i valori di loss **non sono confrontabili** con i run ≤ v1.2.0: ora sono
  supervisionate molte più posizioni, e sono le più difficili. Loss più alta a parità di
  qualità è attesa, non una regressione. Precision/recall/F1 entity-level restano
  confrontabili;
- dopo un training con questa modifica, `aggregation_strategy="simple"` in `app.py`
  torna la lettura corretta e non serve altro.

## 3. Iperparametri

Valori nel file dopo l'analisi delle curve del run v1.2.0 (W&B, 1 epoca, ~26k step).
✅ = già applicato in `train_pii.py`; ⚙️ = da decidere sulla macchina di training.

| Parametro | Valore | Perché |
|---|---|---|
| ✅ `EPOCHS` | **2** (era 1) | `eval/loss` scendeva monotona 0,0135 → 0,0060 ed era **ancora in discesa** all'ultimo punto; `test/loss` 0,0058 ≈ eval finale, nessun divario train/eval. Non c'era overfitting da prevenire, c'era capacità inutilizzata. La 2ª epoca però **ripassa** su dati in gran parte sintetici: il freno vero è l'early stopping, non questo numero |
| ✅ `EVAL_EVERY` | **min(2000, …)** (era `steps_per_epoch // 4`) | con 4 punti in tutto il run l'overfitting non è osservabile: quando lo vedi il training è finito. A 2.000 step si hanno ~13 punti per epoca |
| ✅ checkpoint | `save_strategy="steps"`, `save_steps=EVAL_EVERY`, `save_total_limit=2`, `load_best_model_at_end=True` | prima era `"no"`: **nessun checkpoint**, un run lungo che degenera a metà era da buttare intero. `save_steps` **deve** coincidere con `eval_steps`, altrimenti `load_best_model_at_end` solleva `ValueError` |
| ✅ `metric_for_best_model` | **`f1_macro`** (non la loss) | vedi §3.1: la loss è dominata dagli `O` |
| ✅ early stopping | `EarlyStoppingCallback(patience=3)` | con eval frequente costa nulla e su 2 epoche serve davvero |
| ⚙️ `BATCH` / `GRAD_ACCUM` | 14 / 2 (eff. 28) → **effettivo ≥ 64** | il 14 è tarato su **16 GB condivisi col desktop** (vedi CLAUDE.md): su RTX Pro 6000 Blackwell quel vincolo non esiste |
| ⚙️ `LR` | 5e-5 → **1e-4** se il batch effettivo sale a ~96-128 | scaling lineare col batch. A batch invariato lasciare 5e-5 |
| `warmup_ratio` | 0.05, invariato | `grad_norm` spara a ~275 nei primi step e poi si appiattisce: il warmup sta già facendo il suo |
| `MAX_LEN` | 768, invariato | il corpus `-clean` ha documenti lunghi (media ~1.100 caratteri); scendere troncherebbe |
| ⚙️ `dataloader_num_workers` | 0 | è un workaround Windows, non una scelta di performance: su Linux alzare a 4-8 |

### 3.1 Quali metriche guardare (e perché non la sola loss)

`compute_metrics` gira a **ogni** eval e logga su W&B: `precision`, `recall`, `f1_micro`,
`f1_macro`, `token_acc` e `f1_<TAG>` per i cinque tag critici (`WATCH_TAGS`: `CF`, `PIVA`,
`IBAN`, `ZIPCODE`, `ID_DOC`). Prima veniva loggata **solo `eval_loss`** — ecco perché i
run precedenti hanno su W&B quei soli quattro grafici — e P/R/F1 si calcolavano una volta
sola alla fine, quando non si può più decidere niente.

- **`eval_loss`** — guardrail, serve a vedere *quando* peggiora. Da sola mente per
  omissione: oltre il 90% dei token è `O`, quindi la loss è una media dominata dalla
  classe facile e può scendere mentre i tag rari stanno fermi. È letteralmente successo:
  il bug dei subword (§2.2) ha attraversato un'epoca intera con la loss che scendeva
  liscia da 0,0135 a 0,0060 e tre tag inchiodati a zero.
- **`f1_macro`** — la metrica di selezione. La micro è dominata da `FULLNAME` e `CITY`
  (già a 0,95+) e resta alta anche con metà dei tag rari a pezzi; la macro crolla appena
  un tag si rompe. Se se ne guarda **una** oltre alla loss, è questa.
- **`f1_micro`** — qualità complessiva, per confrontare run diversi.
- **`f1_CF`, `f1_PIVA`, `f1_IBAN`, `f1_ZIPCODE`, `f1_ID_DOC`** — verifica diretta che la
  propagazione dei subword sia attiva. Se restano a zero, non lo è.
- **F1 spezzata per fonte** (`validation_real` vs `-clean`) — il rilevatore di
  overfitting sui template, che la loss aggregata non può dare. Non è ancora automatica:
  oggi si ottiene valutando le due validation separatamente.

Nota su `preprocess_logits_for_metrics`: fa l'argmax **sulla GPU** prima che il Trainer
accumuli. Senza, il Trainer terrebbe in RAM i logit interi della validation unita (36k
righe × 768 token × ~50 classi = svariati GB) → OOM. Conseguenza: `trainer.predict()`
restituisce già gli id delle classi, non i logit — `evaluate_metrics()` non deve rifare
`argmax` (c'è una difesa `if preds.ndim == 3`).

Altre due letture delle curve:

**`train/loss` collassa sotto 0,5 entro ~1.000 step e poi resta piatta.** Non vuol dire
che il compito è risolto: la stragrande maggioranza dei token è `O`, quindi la loss è
dominata dalla classe facile. È il motivo per cui il bug dei subword è sopravvissuto a
un'intera epoca senza dare segnale. **Non usare la train loss per decidere quando
fermarsi**: alla eval affiancare la **F1 entity-level**, che è l'unica che si muove sui
tag rari.

**Sbilanciamento di classe.** `FULLNAME` ≫ `CREDITCARDNUMBER` (~66×): i tag rari restano
più rumorosi a prescindere dagli iperparametri. Con la propagazione dei subword la quota
di posizioni non-`O` sale, e sale soprattutto sugli identificatori lunghi — cioè proprio
i tag che erano a zero.

## 4. Cosa guardare nel prossimo run

1. `CF`, `PIVA`, `IBAN`, `ZIPCODE`, `ID_DOC` in F1 entity-level: devono passare da ~0 a
   valori confrontabili con gli altri tag. Se restano a zero, la propagazione non è
   attiva o `I-X` non è nel set di label.
2. `PROVINCE` deve **restare** alta: è il tag che si rompe se si prova a rimediare al
   problema lato inferenza con `aggregation_strategy="first"` invece che qui.
3. `eval/loss` con ~13 punti per epoca: se dopo l'epoca 1 risale, l'early stopping tiene
   il best checkpoint e la seconda epoca costa solo tempo macchina.
4. Bumpare `MODEL_VERSION` (nel file è ancora `1.1.0` mentre su disco esistono già dei
   `v1.2.0`), o passare `--version` esplicito, altrimenti il run si autonumera indietro.
