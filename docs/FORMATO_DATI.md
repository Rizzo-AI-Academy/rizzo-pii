# Formato dei dati del dataset community

Questo documento spiega **esattamente** il formato di ogni esempio del dataset
[`rizzoaiacademy/anonimizzazione-testi-italiano`](https://huggingface.co/datasets/rizzoaiacademy/anonimizzazione-testi-italiano),
così da poter generare contributi **a mano, in modo semi-automatico, o con uno script**.

> ⚠️ **Solo dati SINTETICI.** Non inserire MAI dati personali reali. Nomi, codici fiscali,
> IBAN, indirizzi ecc. devono essere inventati (con checksum validi dove richiesto).
> Vedi il principio *"LLM autore, codice etichettatore"* in [CLAUDE.md](../CLAUDE.md).

Il modo consigliato per contribuire è lo script
[`src/data_pipeline/contribute_dataset.py`](../src/data_pipeline/contribute_dataset.py),
che genera e valida tutto da solo e apre una Pull Request. Questo documento serve a chi
vuole capire/produrre il formato direttamente.

---

## 1. File: JSON Lines (`.jsonl`)

- **Un esempio per riga**, ogni riga è un oggetto JSON valido (UTF-8, niente BOM).
- Niente virgole tra le righe, niente array che racchiude il file.
- Nome file dei contributi: `contributions/<handle>-<timestamp>-seed<seed>-n<conteggio>.jsonl`.

## 2. Schema di una riga

| Campo | Tipo | Obbligatorio | Descrizione |
|---|---|:---:|---|
| `source_text` | `string` | ✅ | Il testo completo dell'esempio (prosa legale italiana sintetica). |
| `language` | `string` | ✅ | Codice lingua ISO. Per il dataset community: `"it"`. |
| `tokens` | `string[]` | ✅ | Il testo tokenizzato (vedi §3). |
| `bio_labels` | `string[]` | ✅ | Una label BIO per ogni token. **`len(bio_labels) == len(tokens)`**. |
| `entities` | `object[]` | ✅ | Le entità PII con offset di carattere (vedi §4). |
| `template_id` | `int` | ⛔️ facolt. | Id del template usato (tracciamento; può mancare). |
| `meta` | `object` | ⛔️ facolt. | Provenienza: `{contributor, seed, generator_version, synthetic, new_template}`. Ignorato dal training. |

I campi `tokens`/`bio_labels` sono ciò che il training consuma; `entities` è la stessa
informazione in forma di span ed è usata per la validazione (checksum, offset).

## 3. Tokenizzazione (regola esatta)

I token si ottengono con questa regex (Unicode), in ordine di apparizione nel testo:

```python
import re
TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)
tokens = TOKEN_RE.findall(source_text)
```

In pratica: **ogni sequenza alfanumerica è un token**; **ogni carattere di punteggiatura
è un token a sé**; gli spazi non producono token. Esempio: `C.F.` → `["C", ".", "F", "."]`.

## 4. Le entità (`entities`)

Ogni entità è:

```json
{ "value": "RSSMRA85H12F205Y", "label": "CF", "start": 34, "end": 50 }
```

- `start`/`end` sono **offset di carattere** in `source_text` (slice Python: `source_text[start:end] == value`).
- Le entità non si sovrappongono.
- `label` è uno dei tag della tassonomia (§5).

## 5. Schema BIO e tag

Per ogni token:
- `B-<TAG>` se il token **inizia** un'entità (offset iniziale del token == `start` dell'entità);
- `I-<TAG>` se il token è **dentro** un'entità ma non è il primo;
- `O` se il token non appartiene ad alcuna entità.

I 22 tag finali del modello (dettaglio in [docs/TASSONOMIA_TAG.md](TASSONOMIA_TAG.md)):

```
FULLNAME  AGE  GENDER  DATE  TIME  STREET  BUILDINGNUM  ZIPCODE  CITY  PROVINCE
EMAIL  TELEPHONENUM  CF  PIVA  ID_DOC  IBAN  CREDITCARDNUMBER  AMOUNT  TARGA
ORG  DOCID  CATASTO
```

> **Label "grezze" accettate.** Il training rimappa le label al caricamento tramite
> `TAG_MAP` (vedi `src/training/train_pii.py`), quindi puoi usare anche le label di
> dettaglio che poi confluiscono in un tag finale. Le principali fusioni:
> `GIVENNAME` + `SURNAME` + ruoli legali (`GIUDICE`, `AVVOCATO`, `ATTORE`, `CONVENUTO`,
> `TESTIMONE`) → **FULLNAME**; `PEC` → **EMAIL**; `CONTO` → **IBAN**; `RG` → **DOCID**;
> `IDCARDNUM` / `DRIVERLICENSENUM` → **ID_DOC**. In dubbio, usa direttamente il tag finale.

### Entità a più parole

Una `STREET` come `"Via Roma"` copre due token: il primo è `B-STREET`, il secondo `I-STREET`.

### Entità consecutive (elenchi)

Due entità adiacenti **devono** essere separate da almeno un token `O` (es. una virgola),
altrimenti si fonderebbero in un'unica entità. Per gli elenchi usa separatori che producono
un token (`, ` `; ` ` - ` ` e `), **mai** un semplice `\n` (lo spazio bianco non è un token).

## 6. Validità obbligatoria dei checksum

Per questi tag il `value` **deve** superare il proprio checksum (lo stesso che il modello
affianca in produzione, vedi [src/inspect/validate_checksums.py](../src/inspect/validate_checksums.py)):

- **CF** — algoritmo ufficiale del codice di controllo (16° carattere).
- **PIVA** — 11 cifre, checksum Luhn-like dell'Agenzia delle Entrate.
- **IBAN** — IT + 2 cifre di controllo mod-97.

Lo script di contribuzione genera questi valori già validi e li ricontrolla; se li scrivi a
mano, validali prima di inviarli.

## 7. Esempio completo (verificato)

`source_text`:

```
Il sottoscritto Mario Rossi, C.F. RSSMRA85H12F205Y, residente in Via Roma 10.
```

Riga JSONL corrispondente (formattata su più righe per leggibilità — nel file è **una sola riga**):

```json
{
  "source_text": "Il sottoscritto Mario Rossi, C.F. RSSMRA85H12F205Y, residente in Via Roma 10.",
  "language": "it",
  "tokens": ["Il","sottoscritto","Mario","Rossi",",","C",".","F",".","RSSMRA85H12F205Y",",","residente","in","Via","Roma","10","."],
  "bio_labels": ["O","O","B-GIVENNAME","B-SURNAME","O","O","O","O","O","B-CF","O","O","O","B-STREET","I-STREET","B-BUILDINGNUM","O"],
  "entities": [
    {"value":"Mario","label":"GIVENNAME","start":16,"end":21},
    {"value":"Rossi","label":"SURNAME","start":22,"end":27},
    {"value":"RSSMRA85H12F205Y","label":"CF","start":34,"end":50},
    {"value":"Via Roma","label":"STREET","start":65,"end":73},
    {"value":"10","label":"BUILDINGNUM","start":74,"end":76}
  ],
  "meta": {"contributor":"esempio","seed":123,"generator_version":"1.0.0","synthetic":true}
}
```

Nota: `Mario` e `Rossi` sono due entità distinte (`B-GIVENNAME`, `B-SURNAME`) che il
training fonde entrambe in `FULLNAME`; sono separate dal contesto e dalla virgola che segue.

## 8. Checklist prima di inviare

- [ ] Ogni riga è un JSON valido, una per esempio, UTF-8.
- [ ] `len(tokens) == len(bio_labels)`.
- [ ] `source_text[start:end] == value` per ogni entità.
- [ ] Le label BIO sono coerenti (`I-` solo dopo un `B-`/`I-` dello stesso tag).
- [ ] CF / PIVA / IBAN passano il checksum.
- [ ] **Nessun dato personale reale.** Tutto sintetico.
