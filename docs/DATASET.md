# Dataset di training e validation — documentazione completa

Questo documento descrive **tutti i dati** usati per addestrare il modello PII:
da dove vengono, come sono stati creati, quanto è sintetico, quante lingue, e la
distribuzione per lingua e per tag. È la fonte di verità sui dati quando si riprende
il progetto. Per la **tassonomia dei 22 tag** vedi [TASSONOMIA_TAG.md](TASSONOMIA_TAG.md).

Tutti i numeri sono calcolati sui file reali, **dopo** la normalizzazione `TAG_MAP`
(cioè sui 22 tag finali, come li vede il modello). `LANG = None` → training **multilingue**.

---

## 1. Sintesi numerica

| | Righe | Quota |
|---|---:|---:|
| **TRAIN totale** | **744.912** | 100% |
| **VALIDATION (reale, solo it)** | **7.000** | — |
| Lingue nel train | **8** | — |
| Parole nel train | ~67,8 M | — |
| **Token (subword mmBERT)** | **~105,9 M** | — |
| Sequenza più lunga | **962 token** | — |
| Entità taggate nel train | ~5,70 M | — |

> Dettaglio testo/lunghezze e lunghezza max per fonte in **§3.4**.

Composizione del **train** per fonte:

| Fonte | Righe | Quota | Tipo |
|---|---:|---:|---|
| Ai4Privacy (multilingue) | 464.124 | 62,3% | riferimento, testo umano + PII mascherata |
| Sintetico da template (nostro) | 200.000 | 26,8% | generato: prosa con segnaposto + iniezione |
| Augment (nostro) | 40.000 | 5,4% | ibrido: entità sintetiche in **frasi reali** it |
| DeepMount (pii-masking-ita) | 40.788 | 5,5% | Faker tradotto in italiano |

**Quanto è sintetico:** generato interamente da noi (template + augment) = **240.000 righe
(32,2%)**; includendo DeepMount (Faker) il materiale sintetico sale a **280.788 righe (37,7%)**.
Il restante **62,3%** è Ai4Privacy (il dataset di riferimento, multilingue).

---

## 2. Le quattro fonti dati

### 2.1 Ai4Privacy (`open-pii-masking-500k`)
Dataset di riferimento multilingue (CC-BY-4.0), testo realistico con PII mascherata e
label BIO word-level (`mbert_tokens` / `mbert_token_classes`). **8 lingue.** È la spina
dorsale multilingue del training. Caricato con filtro lingua opzionale (`LANG`).

### 2.2 Sintetico da template (`generate_synthetic_pii.py`)
Principio **"LLM autore, codice etichettatore"**: un LLM (Gemini) scrive la prosa legale
italiana con soli segnaposto `{SLOT}` (72 template in `legal_templates.json`), poi il
**codice inietta** i dati con **checksum validi** (CF/P.IVA/IBAN con algoritmi reali) e
produce le label BIO esatte (le posizioni sono note → zero annotazione manuale, nessuna
PII reale). Copre i tag legali italiani assenti altrove: `CF`, `PIVA`, `CATASTO`, `DOCID`,
`PROVINCE`, e rinforza `AMOUNT`/`ORG`/`IBAN`/`TARGA`.

### 2.3 Augment (`augment_real_pii.py`)
Inietta entità sintetiche (con checksum validi) dentro **frasi reali** italiane di
Ai4Privacy, in **posizioni variabili** (non sempre a fine frase). Serve a mostrare i tag
legali in prosa reale e varia, **non** dentro i template → mitiga l'overfit strutturale.

### 2.4 DeepMount (`DeepMount00/pii-masking-ita` → `prepare_deepmount.py`)
Tassonomia ai4privacy-200k (56 tipi, Faker) **tradotta in italiano**, rimappata sui nostri
22 tag. Testo italiano **vario ma non-legale** (email, chat, business). Valore principale:
dà contesto naturale a `IBAN`/`ORG`/`AMOUNT`/`TARGA`. ⚠️ Il **contesto** è italiano ma molti
**valori** sono Faker inglesi/USA (nomi inglesi, ZIP/civici americani).

---

## 3. Training set

### 3.1 Lingue — Ai4Privacy train (la parte multilingue)

| Lingua | Codice | Righe | Quota Ai4 |
|---|---|---:|---:|
| Inglese | en | 120.526 | 26,0% |
| Francese | fr | 89.668 | 19,3% |
| Tedesco | de | 65.897 | 14,2% |
| Spagnolo | es | 62.585 | 13,5% |
| Italiano | it | 55.002 | 11,9% |
| Hindi | hi | 27.021 | 5,8% |
| Telugu | te | 22.144 | 4,8% |
| Olandese | nl | 21.281 | 4,6% |
| **Totale** | | **464.124** | 100% |

### 3.2 Distribuzione per lingua del POOL completo (italiano rinforzato)

I dati sintetici/DeepMount sono **tutti italiani** e si sommano all'italiano di Ai4Privacy.
Risultato: l'italiano diventa la lingua **più rappresentata** (~45%), pur restando un
training multilingue.

| Lingua | Righe nel pool | Quota | Note |
|---|---:|---:|---|
| Italiano | 335.790 | 45,1% | 55.002 Ai4 + 200k synth + 40k augment + 40.788 DeepMount |
| Inglese | 120.526 | 16,2% | solo Ai4 |
| Francese | 89.668 | 12,0% | solo Ai4 |
| Tedesco | 65.897 | 8,8% | solo Ai4 |
| Spagnolo | 62.585 | 8,4% | solo Ai4 |
| Hindi | 27.021 | 3,6% | solo Ai4 |
| Telugu | 22.144 | 3,0% | solo Ai4 |
| Olandese | 21.281 | 2,9% | solo Ai4 |
| **Totale** | **744.912** | 100% | |

> Italiano = 45,1% del pool (di cui 55k reale Ai4 + 280,8k materiale italiano dedicato).
> Multilingue (non-it) = 409.122 righe (54,9%).

### 3.3 Distribuzione per tag (22 tag, pool multilingue completo)

| Tag | Entità | Quota | | Tag | Entità | Quota |
|---|---:|---:|---|---|---:|---:|
| FULLNAME | 1.339.633 | 23,5% | | EMAIL | 158.243 | 2,8% |
| CITY | 870.604 | 15,3% | | DOCID | 156.984 | 2,8% |
| DATE | 411.336 | 7,2% | | TELEPHONENUM | 104.803 | 1,8% |
| STREET | 363.594 | 6,4% | | ID_DOC | 98.643 | 1,7% |
| BUILDINGNUM | 356.834 | 6,3% | | IBAN | 93.057 | 1,6% |
| ZIPCODE | 329.149 | 5,8% | | TIME | 71.018 | 1,2% |
| PROVINCE | 315.179 | 5,5% | | PIVA | 60.487 | 1,1% |
| AMOUNT | 268.349 | 4,7% | | AGE | 30.608 | 0,5% |
| CF | 247.126 | 4,3% | | GENDER | 27.702 | 0,5% |
| ORG | 194.863 | 3,4% | | TARGA | 19.716 | 0,3% |
| CATASTO | 173.285 | 3,0% | | CREDITCARDNUMBER | 13.759 | 0,2% |
| | | | | **TOTALE** | **5.704.972** | 100% |

**Sbilanciamento di classe:** FULLNAME (1,34 M) vs CREDITCARDNUMBER (13,7k) ≈ **97×**. I tag in
coda (`CREDITCARDNUMBER`, `TARGA`, `GENDER`, `AGE`) saranno più deboli — da tenere a mente
leggendo le metriche. Rispetto alla versione precedente l'iniezione 200k + i template-elenco
hanno rinforzato soprattutto `ORG` (81k → 195k), `PROVINCE`, `CATASTO`, `CF` e `AMOUNT`.

### 3.4 Dimensione del testo e lunghezza delle sequenze

Numeri **esatti**, ottenuti tokenizzando l'intero training set col tokenizer di mmBERT
(`is_split_into_words=True`, senza troncamento). Il "token" rilevante per la context window
è il **subword** di mmBERT.

| | Totale |
|---|---:|
| Righe | 744.912 |
| Parole | 67.830.590 (~67,8 M) |
| **Token (subword mmBERT)** | **105.915.543 (~105,9 M)** |
| Caratteri (circa) | ~334.000.000 (~334 M) |
| Media token/riga | 142,2 |
| Espansione token/parola | 1,56× |

**Lunghezza per fonte** (parole e token):

| Fonte | Righe | Parole | Token (subword) | Max parole | **Max token** | Righe > 512 token |
|---|---:|---:|---:|---:|---:|---:|
| Ai4 (tutte le lingue) | 464.124 | 15.365.883 | 23.521.541 | 510 | **962** | 11 |
| Synth 200k | 200.000 | 49.221.243 | 77.050.194 | 519 | 787 | 59.516 |
| Augment | 40.000 | 1.734.315 | 2.834.181 | 297 | 431 | 0 |
| DeepMount | 40.788 | 1.509.149 | 2.509.627 | 480 | 660 | 5 |
| **TOTALE** | **744.912** | **67.830.590** | **105.915.543** | **519** | **962** | **59.532** |

- **Sequenza più lunga in assoluto: 962 token** (un esempio Ai4 multilingue — telugu + codici,
  che si espandono molto in subword).
- **Implicazione context window**: con `MAX_LEN = 768` (in `train_pii.py`) Augment (max 431) e
  DeepMount (max 660) entrano per intero; del **Synth 200k** si troncano **1.912 righe** (0,96%,
  max 787 — code di atti lunghi con elenchi) più poche centinaia di esempi Ai4 estremi (max 962).
  Con il vecchio `MAX_LEN = 512` si troncavano **59.532 sequenze**, di cui **59.516 sintetiche**
  (~30% del Synth 200k) → si perdevano entità etichettate nella coda dei documenti. mmBERT
  (ModernBERT) ha context nativa 8192, quindi 768 è ampiamente nei limiti del modello.

---

## 4. Validation set

File: **`validation_real.jsonl` — 7.000 righe, SOLO ITALIANO** (`build_validation.py`,
seed 2024 indipendente dal training). Scelta deliberata: l'obiettivo finale è anonimizzare
**atti italiani**, quindi la validation misura il caso d'uso reale. (Il training resta
multilingue; le altre lingue non sono validate — vedi §5.)

### 4.1 Come è stata costruita
- **Base reale held-out**: 2.500 frasi Ai4Privacy validation (it) + 2.500 DeepMount test.
- **Supplementi** per i 5 tag che non hanno alcun esempio reale (`CF`, `PIVA`, `CATASTO`,
  `DOCID`, `PROVINCE`): l'entità generata è **iniettata in frasi reali held-out** della
  validation Ai4 (400 frasi per tag = 2.000 righe). Contesto reale, niente leakage col train
  (che usa frasi di *train*, non di *validation*).

Tutti i sintetici e il DeepMount **train** stanno interamente nel pool di training; il
DeepMount **test** è consumato solo qui. La validation è quindi un benchmark "pulito"
(frasi mai viste in training) e copre **tutti i 22 tag**.

### 4.2 Distribuzione per tag (validation)

| Tag | Entità | | Tag | Entità |
|---|---:|---|---|---:|
| FULLNAME | 4.390 | | PROVINCE | 400 |
| CATASTO | 1.200 | | DOCID | 400 |
| CITY | 953 | | CF | 400 |
| DATE | 922 | | AGE | 385 |
| TELEPHONENUM | 874 | | ZIPCODE | 299 |
| ID_DOC | 800 | | IBAN | 278 |
| EMAIL | 748 | | CREDITCARDNUMBER | 257 |
| TIME | 637 | | AMOUNT | 146 |
| STREET | 617 | | ORG | 145 |
| BUILDINGNUM | 594 | | TARGA | 43 |
| PIVA | 514 | | | |
| GENDER | 472 | | | |

> `TARGA`/`ORG`/`AMOUNT` sono sottili (vengono solo dalla base DeepMount): metriche più
> rumorose per questi. I 5 tag iniettati usano connettori fissi (`C.F. …`, `prot. n. …`)
> come nell'augment di training: eval in **contesto reale** ma non perfettamente cieco.

---

## 5. Limiti noti (onestà sulle metriche)

- **Validation solo italiana**: il training è multilingue ma non misuriamo le 7 lingue non
  italiane. Scelta voluta (dominio = atti italiani). Per certificare il multilingue servirebbe
  una fetta di validation Ai4 nelle altre lingue.
- **Tag legali IT-only** (`CF`/`PIVA`/`CATASTO`/`DOCID`/`PROVINCE`): non esistono come dato
  reale da nessuna parte; in validation sono entità generate in frasi reali. Buon proxy, non
  un eval completamente indipendente dalla struttura di iniezione.
- **`PROVINCE` nel train** viene quasi solo dai template (poca diversità di contesto).
- **Valori off-domain di DeepMount** (nomi/indirizzi USA): utili per forma/contesto, non come
  valori italiani.

---

## 6. Come rigenerare tutto

```powershell
# 1) (opz.) template legali via Gemini  -> dataset/synthetic/legal_templates.json
python src/data_pipeline/llm_template_bank.py --per-type 5 --append

# 2) sintetico da template (200k)       -> dataset/synthetic/synthetic_pii_it_200k.jsonl
python src/data_pipeline/generate_synthetic_pii.py -n 200000 --out dataset/synthetic/synthetic_pii_it_200k.jsonl

# 3) augment in testo reale (40k)        -> dataset/synthetic/synthetic_pii_it_realaug.jsonl
python src/data_pipeline/augment_real_pii.py -n 40000 --out dataset/synthetic/synthetic_pii_it_realaug.jsonl

# 4) DeepMount rimappato 56->22 tag      -> dataset/processed/deepmount_pii_it_{train,test}.jsonl
python src/data_pipeline/prepare_deepmount.py

# 5) validation reale unificata (7k)     -> dataset/validation/validation_real.jsonl
python src/data_pipeline/build_validation.py

# 6) (opz.) subset per smoke test 10k/5k -> dataset/subsets/
python src/data_pipeline/build_subset.py

# 7) training multilingue (LANG=None)    -> models/rizzo-pii-0.3B-v{VERSION}/ + registry.json
python src/training/train_pii.py --type full
```
