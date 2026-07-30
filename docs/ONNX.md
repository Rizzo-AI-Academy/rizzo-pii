# Export ONNX (+ INT8)

Come esportare il modello in **ONNX** per farlo girare dove PyTorch non arriva:
runtime ONNX nativi e, soprattutto, **nel browser** via
[Transformers.js](https://github.com/huggingface/transformers.js) — estensioni,
WebAssembly, applicazioni web. Il modello resta interamente locale: ONNX non
cambia nulla della garanzia di privacy, cambia solo il motore che lo esegue.

Script: [`src/export/export_onnx.py`](../src/export/export_onnx.py).

---

## Installazione

Le dipendenze ONNX sono tenute **fuori** da `requirements.txt`, per non
appesantire chi vuole solo addestrare o usare l'app desktop:

```bash
pip install -r requirements-onnx.txt
```

## Uso

```bash
python src/export/export_onnx.py                   # fp32 + INT8, ultima versione in models/
python src/export/export_onnx.py --verify          # + confronto fp32 vs INT8
python src/export/export_onnx.py --no-quantize     # solo fp32
python src/export/export_onnx.py --arch avx512     # profilo CPU
PII_MODEL_DIR=/path/al/modello python src/export/export_onnx.py
```

Il modello si risolve con la stessa logica di `src/training/test_pii.py`: ultima
versione `models/rizzo-pii-0.3B-v*`, con override `PII_MODEL_DIR`.

Il profilo CPU per la quantizzazione si sceglie da solo (`arm64` su Apple
silicon, `avx2` altrove) e si può forzare con `--arch`
(`arm64` | `avx2` | `avx512` | `avx512_vnni`).

## Risultato

```
models/rizzo-pii-0.3B-v1.2.0-onnx/
├─ fp32/                        export intermedio (non serve spedirlo)
├─ quant-staging/               staging della quantizzazione
├─ export_stats.json            dimensioni + esito della verifica
└─ bundle/                      ← questo è ciò che si distribuisce
   ├─ config.json
   ├─ tokenizer.json
   ├─ tokenizer_config.json
   ├─ special_tokens_map.json
   └─ onnx/
      └─ model_quantized.onnx
```

Il layout di `bundle/` è quello che Transformers.js si aspetta: file del
tokenizer alla radice, pesi sotto `onnx/`.

---

## Dimensioni misurate (v1.2.0, profilo arm64)

| Artefatto | Dimensione |
|---|---:|
| ONNX fp32 | 1206,6 MB |
| `model_quantized.onnx` (INT8) | **294,3 MB** |
| `tokenizer.json` | 32,8 MB |
| **Bundle totale** | **327,1 MB** |

Export in ~10 s, quantizzazione in ~10 s (Apple M-series).

Due note utili a chi deve distribuire il modello:

- Il **`tokenizer.json` pesa 32,8 MB** ed è una conseguenza diretta del
  vocabolario mmBERT da 256k voci. Va spedito insieme ai pesi: Transformers.js
  lo carica dalla stessa cartella.
- Degli ~307 M parametri, **~197 M (il 64%) sono la tabella di embedding**
  (256.000 × 768). L'encoder vero e proprio, 22 layer, ne pesa ~110 M. È il
  motivo per cui il file INT8 resta sui ~294 MB nonostante il modello sia
  descritto come "0.3B".

## Fedeltà della quantizzazione INT8

`--verify` confronta le predizioni fp32 e INT8 **token per token** sugli stessi
testi, usando `dataset/validation/validation_real.jsonl` se presente e altrimenti
un insieme di esempi che copre tutti i 22 tag.

Sulla v1.2.0, con gli esempi predefiniti (537 token, 242 dei quali di entità):

| Misura | Valore |
|---|---:|
| Accordo su tutti i token | 98,51% |
| Accordo sui soli token di entità | 98,76% |

La quantizzazione è quindi praticamente gratuita in accuratezza, a fronte di un
file **4× più piccolo**. Per una misura più solida conviene rigenerare la
validation (`python src/data_pipeline/build_validation.py`) ed eseguire
`--verify --verify-n 2000`.

---

## Uso con Transformers.js

```js
import { pipeline, env } from '@huggingface/transformers';

// Nessuna chiamata di rete: i pesi vengono dalla cartella locale.
env.allowRemoteModels = false;
env.localModelPath = '/percorso/a/models/rizzo-pii-0.3B-v1.2.0-onnx/';

const nlp = await pipeline('token-classification', 'bundle', { dtype: 'q8' });
const out = await nlp("Mario Rossi, C.F. RSSMRA85H12F205Y.", { ignore_labels: [] });
```

### Attenzione: gli offset carattere non arrivano dalla pipeline

Transformers.js restituisce per ogni token `entity`, `score`, `index` e `word`,
**ma non `start`/`end`**. Chi deve sostituire il testo (l'anonimizzazione
reversibile lo richiede) deve ricostruirseli.

Con il tokenizer mmBERT (Metaspace) la ricostruzione è **esatta**: la
concatenazione dei `word` in ordine riproduce il testo di partenza, preceduto da
**un solo spazio**. Bastano quindi le lunghezze cumulative, sfalsate di uno.

Due dettagli da cui dipende la correttezza:

1. **`ignore_labels: []` è obbligatorio.** Il valore predefinito è `['O']`: senza
   passarlo esplicitamente i token `O` vengono scartati, la sequenza si
   interrompe e le lunghezze cumulative vanno fuori fase.
2. I `word` includono lo spazio iniziale (`" Mario"`), quindi conviene rifilare i
   bordi degli span alla fine.

```js
/** Ricostruisce gli offset carattere e raggruppa i token BIO in entità. */
function spansFromTokens(text, tokens) {
  let cursor = -1;                       // compensa lo spazio iniziale di Metaspace
  const placed = tokens.map((t) => {
    const start = cursor;
    cursor += t.word.length;
    return { ...t, start: Math.max(0, start), end: Math.min(text.length, cursor) };
  });

  const groups = [];
  for (const t of placed) {
    const m = /^([BI])-(.+)$/.exec(t.entity);
    if (!m) continue;
    const [, bio, type] = m;
    const last = groups[groups.length - 1];
    if (last && last.type === type && (bio === 'I' || last.lastIdx + 1 === t.index)) {
      last.end = t.end;
      last.lastIdx = t.index;
    } else {
      groups.push({ type, start: t.start, end: t.end, lastIdx: t.index });
    }
  }

  return groups.map(({ type, start, end }) => {          // rifila gli spazi ai bordi
    while (start < end && /\s/.test(text[start])) start++;
    while (end > start && /\s/.test(text[end - 1])) end--;
    return { type, start, end };
  });
}

const spans = spansFromTokens(text, await nlp(text, { ignore_labels: [] }));
```

### Documenti lunghi

Il modello è addestrato a `MAX_LEN = 768` subword. Per testi più lunghi vale la
stessa strategia dell'app Flask (`src/app/app.py`): chunk con sovrapposizione,
offset riportati in coordinate globali e deduplica finale.

### Da affiancare sempre alla rete regex + checksum

Vale anche qui quanto detto nel README: in produzione il modello **non va usato
da solo**. La rete regex + checksum
([`src/inspect/validate_checksums.py`](../src/inspect/validate_checksums.py))
resta necessaria, e un checksum valido ha la precedenza sul modello. È la
mitigazione della frammentazione dei codici lunghi, che si osserva anche via
Transformers.js: su un codice fiscale il modello può alternare `CF` e `ID_DOC`
fra i subword dello stesso codice.

---

## Limiti di questa documentazione

- Le dimensioni e i numeri di fedeltà sono misurati sulla **v1.2.0** con profilo
  `arm64`. Altri profili CPU producono file di dimensione paragonabile ma non
  identica.
- La verifica `--verify` misura l'**accordo fra fp32 e INT8**, non l'accuratezza
  assoluta: dice quanto costa la quantizzazione, non quanto è buono il modello.
  Per quello c'è `src/training/evaluate_pii.py`.
- Il bundle è stato caricato ed eseguito con `@huggingface/transformers` 3.8.1
  **in Node**; il percorso browser usa lo stesso codice della pipeline ma non è
  stato eseguito in un browser reale come parte di questo lavoro.
