# Tassonomia dei tag PII — versione definitiva

Documento di riferimento sulla tassonomia dei tag che il modello mmBERT impara a
riconoscere. È la **fonte di verità** quando si riprende il progetto: descrive i
**22 tipi finali**, come si ottengono dai dati grezzi, e le decisioni di fusione.

> I file dei dataset (reale Ai4Privacy + sintetico) restano **grezzi e intatti**.
> La tassonomia qui descritta si ottiene al **caricamento** in `train_pii.py` tramite
> `TAG_MAP` + `DROP_TYPES` + `normalize_labels()`. Per cambiare la mappatura si edita
> solo `TAG_MAP` — niente va riannotato a mano.

---

## I 22 tag finali

| Tag | Cos'è | Esempio | Fonte |
|---|---|---|---|
| `FULLNAME` | Nome di persona (anche ruoli legali: giudice, avvocato, parti, teste) | `Mario Rossi` | reale + sintetico |
| `AGE` | Età | `45 anni` | reale |
| `GENDER` | Sesso/genere | `M`, `Femmina`, `Non-binario` | reale |
| `DATE` | Data di calendario | `12/06/1985` | reale + sintetico |
| `TIME` | Ora | `ore 15:30` | reale |
| `STREET` | Via / piazza / corso | `Via Garibaldi` | reale + sintetico |
| `BUILDINGNUM` | Numero civico | `24` | reale + sintetico |
| `ZIPCODE` | CAP | `00185` | reale + sintetico |
| `CITY` | Città | `Milano` | reale + sintetico |
| `PROVINCE` | Sigla provincia | `MI` | sintetico |
| `EMAIL` | Email (inclusa la PEC) | `m.rossi@studio.it` | reale + sintetico |
| `TELEPHONENUM` | Numero di telefono | `+39 333 1234567` | reale + sintetico |
| `CF` | Codice fiscale (persona fisica) | `RSSMRA85H12F205Z` | sintetico |
| `PIVA` | Partita IVA (impresa / VAT) | `12345678901` | reale + sintetico |
| `ID_DOC` | Numero di un documento d'identità personale (carta d'identità, passaporto, patente, n. previdenziale) | `CA12345AB` | reale + sintetico |
| `IBAN` | IBAN / numero di conto corrente | `IT60X0542811101000000123456` | sintetico |
| `CREDITCARDNUMBER` | Numero di carta di credito | `4111 1111 1111 1111` | reale |
| `AMOUNT` | Importo in denaro | `€ 12.500,00` | sintetico |
| `TARGA` | Targa di veicolo | `AB 123 CD` | sintetico |
| `ORG` | Ragione sociale: società, studio legale, banca (parti **private**) | `Edilnord S.r.l.`, `Studio Legale Gallo` | sintetico |
| `DOCID` | Codice identificativo di un atto: n. Ruolo Generale, protocollo, repertorio/raccolta, sentenza | `1234/2024` | sintetico |
| `CATASTO` | Dati catastali di un immobile (foglio, particella, subalterno) | `Foglio 12, particella 345, sub. 6` | sintetico |

Formato BIO: ogni tag esiste come `B-<TAG>` (inizio entità) e `I-<TAG>` (continuazione),
più `O` (token non sensibile). I token adiacenti dello stesso tipo vengono fusi in una
sola entità da `normalize_labels()`.

---

## Decisioni di fusione (in `TAG_MAP`)

Tag grezzi diversi che descrivono **lo stesso concetto** sono stati uniti, per togliere
ambiguità al modello ed evitare confini decisi solo da una parola-chiave di contesto:

| Tag grezzi (nei file) | → | Tag finale | Motivo |
|---|---|---|---|
| `GIVENNAME`, `SURNAME` | → | `FULLNAME` | parti del nome di persona |
| `GIUDICE`, `AVVOCATO`, `ATTORE`, `CONVENUTO`, `TESTIMONE` | → | `FULLNAME` | i ruoli legali **sono nomi di persona** da mascherare (vedi sotto) |
| `SEX`, `GENDER` | → | `GENDER` | stesso contenuto (`M/F/Maschio/Femmina/...`) |
| `TAXNUM`, `PIVA` | → | `PIVA` | `TAXNUM` nei dati reali = partita IVA; la fusione dà a `PIVA` copertura reale **+** sintetica |
| `PEC`, `EMAIL` | → | `EMAIL` | la PEC è un'email |
| `RG`, `DOCID` | → | `DOCID` | il n. di Ruolo Generale è un codice di documento; stesso formato `NNNN/AAAA` |
| `IDCARDNUM`, `PASSPORTNUM`, `DRIVERLICENSENUM`, `SOCIALNUM` | → | `ID_DOC` | tutti "numero di documento d'identità personale" |
| `CONTO`, `IBAN` | → | `IBAN` | il numero di conto è un sottoinsieme dell'IBAN |

### Tag rimossi (→ `O`, restano nel testo ma non sono PII)

| Tag grezzo | → | Motivo |
|---|---|---|
| `TITLE` (`Dott.`, `Avv.`, `Sig.`) | → `O` | è un **appellativo**, non un identificatore |
| `TRIBUNAL` (`Tribunale di Roma`) | → `O` | **ente pubblico**: non è PII da mascherare; il nome resta come contesto |

---

## Perché i ruoli legali → `FULLNAME` (e non tag a sé)

Il ruolo (giudice / avvocato / attore / convenuto / teste) **non è una proprietà della
stringa "Mario Rossi"**: lo stesso nome può essere giudice in un atto e avvocato in un
altro. Il modello potrebbe deciderlo solo dal contesto attorno, e nei dati sintetici il
ruolo è determinato dalla **posizione nel template** → imparerebbe keyword-matching
(«nome dopo la parola *avvocato* = AVVOCATO»), non comprensione reale.

Decisione: il modello tagga **cosa** è (`FULLNAME`, ciò che si maschera). Se a valle
serve il ruolo, lo si deriva come **metadato** (es. dalle parole-chiave intorno),
non chiedendolo alla token-classification.

---

## Note di anonimizzazione (politiche diverse per tag)

- Solo le **parti private** sono PII: `ORG` (società, studi, banche) va mascherato;
  il **tribunale** (ente pubblico) no → non è un tag, resta `O` nel testo.
- `AGE`, `GENDER`, `DATE`, `TIME` sono dati personali GDPR ma **non identificatori
  diretti**; in un atto spesso servono. Politica di mascheramento da decidere a valle,
  non necessariamente "sempre".

---

## Fonti dei dati e validation

Il training fonde **tre fonti**, tutte ricondotte ai 22 tag:

| Fonte | Cosa porta | Preparazione |
|---|---|---|
| **Ai4Privacy** (reale, multilingue, ~reali it) | 14 tag in contesto reale vario | `train_pii.py` via `TAG_MAP` |
| **Sintetici nostri** (template + iniezione in testo reale) | i tag legali italiani assenti altrove: `CF`, `PIVA`, `CATASTO`, `DOCID`, `PROVINCE` (+ checksum validi) | `generate_synthetic_pii.py`, `augment_real_pii.py` |
| **DeepMount00/pii-masking-ita** (Faker tradotto, prosa varia non-legale) | contesto naturale e vario per `IBAN`/`ORG`/`AMOUNT`/`TARGA` (prima solo da template) → mitiga l'overfit strutturale | `prepare_deepmount.py` (rimappa 56→22 tipi) |

> Nota su DeepMount: il **contesto** è italiano ma molti **valori** sono Faker inglesi/USA
> (nomi inglesi, ZIP/civici americani). Utile soprattutto per i tag forma-dipendenti e
> per la varietà di contesto; i valori italiani veri arrivano da Ai4Privacy + sintetici.

**Una validation reale unificata** (`validation_real.jsonl`, da `build_validation.py`):
- **base reale held-out** = Ai4Privacy validation (it) + DeepMount test;
- per i 5 tag senza alcun esempio reale (`CF`, `PIVA`, `CATASTO`, `DOCID`, `PROVINCE`)
  le entità generate sono **iniettate in frasi reali held-out** (frasi della validation Ai4,
  non nel training) → il contesto resta reale, niente leakage col train.

Copre tutti i 22 tag. Tutti i sintetici (template + augment) e il train DeepMount stanno
**interamente nel pool di train**; la validation resta un benchmark "pulito" (frasi mai viste
in training).

## Catena di derivazione

```
file grezzi (reale Ai4Privacy + sintetico)   ← tag grezzi: GIVENNAME, TAXNUM, PEC, RG, IDCARDNUM, CONTO, ...
        │
        ▼  caricamento in train_pii.py
normalize_labels()  =  TAG_MAP  +  DROP_TYPES
        │
        ▼
23 tag finali  →  BIO (B-/I-/O)  →  training mmBERT
```

Per modificare la tassonomia: edita **solo** `TAG_MAP` / `DROP_TYPES` in `train_pii.py`.
Per aggiungere un tag *nuovo* (non presente nei dati) serve invece generare dati
sintetici: un generatore in `generate_synthetic_pii.py` + lo slot nei template
(`TEMPLATES` built-in e/o `ALLOWED_SLOTS` in `llm_template_bank.py`).
