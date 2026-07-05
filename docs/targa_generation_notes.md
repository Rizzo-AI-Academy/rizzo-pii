# Generazione sintetica TARGA

`src/data_pipeline/generate_targa_it.py` genera un corpus italiano mirato alla
disambiguazione contestuale delle targhe.

Il corpus contiene:

- **40% positivi**: targhe in contesti automobilistici, assicurativi e legali;
- **30% hard negative**: codici con forma `AA999AA` usati come ordini, ticket,
  password o identificativi documentali;
- **30% mixed PII**: targa insieme a nomi, organizzazioni, indirizzi, e-mail,
  telefoni e partite IVA valide.

Esecuzione riproducibile:

```powershell
python src/data_pipeline/generate_targa_it.py `
  -n 3000 `
  --seed 20260704 `
  --out dataset/synthetic/synthetic_targa_it_3k.jsonl
```

Ogni record viene validato prima della scrittura:

- tokenizzazione conforme a `docs/FORMATO_DATI.md`;
- stessa lunghezza per `tokens` e `bio_labels`;
- offset delle entità coerenti con `source_text`;
- label BIO ricalcolate dagli offset;
- nessun `source_text` duplicato.

Con seed `20260704` il generatore riproduce byte per byte il file proposto nella
[PR dataset #2](https://huggingface.co/datasets/rizzoaiacademy/rizzo-pii-it-dataset/discussions/2):
SHA-256 `F84E1771749F8D452C600E96990E269D1B701C06F198AC7C93D3B7A17F717098`.

Nota di produzione: gli hard negative migliorano il modello, ma l'applicazione
desktop assegna attualmente priorità alla regex TARGA. Per trasferire il beneficio
all'app, la regex dovrà diventare un candidato contestuale invece di un override.

## Stress test separato dal training

Lo stress test usa famiglie di template diverse da quelle dei 3.000 record e non
deve essere aggiunto al training:

```powershell
python src/data_pipeline/generate_targa_stress.py
python src/training/evaluate_targa_stress.py
```

Produce 300 righe bilanciate:

- 100 targhe positive in contesti non presenti nel training;
- 100 codici `AA999AA` semanticamente non riferiti a veicoli;
- 100 targhe insieme ad altre PII.

La valutazione confronta il modello neurale puro con la pipeline usata
dall'applicazione (`modello + regex`) usando match esatto degli offset di `TARGA`.
Il report include precision, recall, F1, quota di documenti completamente corretti
e falsi positivi sugli hard negative.

### Baseline v1.2.0

Sul modello `rizzo-pii-0.3B-v1.2.0`:

| Pipeline | Precision | Recall | F1 | Documenti esatti | Hard negative con FP |
|---|---:|---:|---:|---:|---:|
| Modello puro | 0,295 | 0,600 | 0,395 | 39,3% | 100/100 |
| Modello + regex | 0,575 | 0,900 | 0,702 | 59,7% | 100/100 |

La regex migliora nettamente la copertura dei positivi, ma marca come `TARGA`
tutti i 100 codici formalmente validi usati come password, ordini, ticket o altri
identificativi. Lo stress test rende quindi misurabile il compromesso che i nuovi
hard negative intendono correggere.
