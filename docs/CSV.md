# Anonimizzazione di file CSV

Rizzo PII anonimizza in locale anche **CSV grandi** (export applicativi, log, anagrafiche, elenchi di transazioni): scegli quali
colonne analizzare, il file viene elaborato **in streaming** e ottieni un CSV con lo stesso
formato in cui le PII sono sostituite da pseudonimi `[TAG_N]`, più un dizionario per il ripristino.
Come per il resto dell'app, **niente lascia la macchina**: è tutto su `127.0.0.1`.

Codice: `src/app/csv_engine.py` (motore),
`src/app/csv_jobs.py` (upload, coda, job, download — Blueprint Flask sotto `/csv`),
`src/app/registry.py` (registro degli pseudonimi). I tre moduli sono **solo stdlib**: non importano
torch né `app.py` e si testano in CI (`tests/test_csv_engine.py`, `tests/test_csv_jobs.py`,
`tests/test_registry.py`).

## Perché

Gli export tabellari grandi (log applicativi o di audit, export di CRM e ticketing, anagrafiche,
elenchi di transazioni) pesano centinaia di MB e hanno milioni di righe. Mandarli "a pezzi" a un LLM
closed è proibito dal GDPR. Il CSV è quindi trattato **colonna per colonna**, con una scelta sola:
analizzarla o lasciarla intatta.

Le colonne analizzate passano dallo **stesso motore del resto dell'app** (`analyze()`: modello + rete
regex/checksum), con lo stesso comportamento, **date comprese**: una data in una colonna analizzata
si anonimizza come in qualunque testo, salvo escludere i tag `DATE`/`TIME` dai toggle 🏷️ oppure
disattivare la colonna. Niente euristiche sugli header né modalità speciali: un solo comportamento da
conoscere, quello dell'app.

Il costo è il **modello** (un forward pass per cella di testo). Per questo, sui file oltre il limite
di upload dell'app l'UI avvisa di tenere l'analisi solo sulle colonne rilevanti (vedi "Prestazioni").

Risultato: un file che un LLM può analizzare (chi ha fatto cosa, quando, con quale esito, quante volte) senza
vedere un solo dato personale, e un dizionario locale per tornare ai valori veri.

## Le due modalità di colonna

Il piano è una lista allineata all'header, un oggetto per colonna: `{"mode": ...}`.

| Modalità | Cosa fa |
|---|---|
| `anonymize` | la cella passa dal motore dell'app (modello + rete regex/checksum); ogni PII trovata diventa `[TAG_N]` (via registro). Default per **tutte** le colonne (`default_plan(header)`) |
| `preserve` | la cella resta identica, byte per byte |

Nella UI è il checkbox **Anonimizza** di ogni colonna, con i comandi "tutte · nessuna".
Le **celle vuote restano vuote**. I **tag esclusi** (🏷️ nella UI, `exclude_tags`) valgono anche qui:
vengono rilevati ma non sostituiti (es. escludi `DATE`/`TIME` per tenere intatte le date).
Per tenere intatta un'intera colonna (timestamp, ID tecnici, esiti) si disattiva "Anonimizza" e resta in chiaro.


## Coerenza degli pseudonimi

Ogni **job** ha **un solo `Registry`** (`registry.py`):

- la **chiave** è il valore normalizzato (spazi compressi, `casefold`), **senza il tag**: se lo
  stesso valore è già stato visto, il suo pseudonimo vince anche se altrove viene etichettato con un
  tag diverso (**il primo tag vince**);
- il formato è `[TAG_N]`, la stessa numerazione di `analyze()`: il dizionario scaricato
  (`{"[TAG_N]": valore}`) è **compatibile con la tab "Ripristina"** e si usa come quello del testo;
- la coerenza vale **tra righe, colonne e file dello stesso job**: `mario.rossi@…` è `[EMAIL_3]`
  ovunque compaia, in qualunque colonna e in ogni file del job;
- **non** vale tra esecuzioni separate: il registro vive in memoria e muore col job. Un secondo job
  ricomincia da `_1`. È una scelta di progetto (niente stato persistente delle chiavi di
  ripristino), non una dimenticanza; per anonimizzare più file in modo coerente, mettili **nello
  stesso job**.

Con il **dizionario disattivato** (switch nella card di input, `include_mapping: false`) il registro
non conserva i valori originali, solo le chiavi normalizzate necessarie alla coerenza:
l'anonimizzazione è definitiva e `/dictionary` risponde `404`.

## Celle JSON e multilinea

- **Celle JSON** (oggetto o array valido) in una colonna `anonymize`: si analizzano solo le
  **foglie stringa** (non le chiavi) e le sostituzioni avvengono **dentro la cella originale**, che
  resta identica byte per byte salvo i valori sostituiti (escape, spazi e ordine delle chiavi
  compresi). Una cella che non è JSON valido è trattata come testo.
- **Celle multilinea** (campi tra virgolette con a capo): sono normali record CSV, si leggono e si
  riscrivono correttamente.

## Fedeltà del formato

Il file di output mantiene **delimitatore** (`,` `;` tab, rilevato dall'inizio del file),
**codifica** (UTF-8 o UTF-8 con BOM, il BOM resta), **fine riga** (`\n` o `\r\n`) e **ordine** di
righe e colonne. Nessuna colonna viene aggiunta o tolta.

Caveat: la scrittura è **`QUOTE_MINIMAL`**: un campo è tra virgolette solo se contiene il
delimitatore, un doppio apice o un a capo. Un file di partenza che quotava *tutti* i campi, o
quotava campi che non ne avevano bisogno, esce con le virgolette normalizzate: il **contenuto** è
identico, i byte no. (Non si usa `csv.writer`: su Python < 3.12 non quota un `\r` isolato e il file
si spezzerebbe.) Le righe completamente vuote non sono record e vengono scartate.

## Affidabilità

- **`.partial` + rename**: l'output viene scritto in `<out>.partial` e rinominato solo a fine run
  riuscita. In caso di errore o annullamento il `.partial` viene cancellato: non esistono output
  parziali che sembrino completi.
- **Righe malformate** (numero di campi diverso dall'header, errore di parsing): **saltate e
  contate**. Il report dà `rows_skipped` e `skipped_lines`, i **soli numeri di riga** (al massimo
  1000 nel report del motore, 50 mostrati dall'API), mai il contenuto. Con **`strict: true`** il job
  si ferma alla prima riga malformata (errore con il numero di riga). Se il file diventa
  illeggibile, il job fallisce invece di ciclare.
- **Annullamento**: `POST /csv/jobs/<jid>/cancel` ferma il job al prossimo batch; un job ancora in
  coda viene tolto subito.
- Formati non gestiti: UTF-16 e file non UTF-8 vengono rifiutati con un messaggio chiaro
  (`422` alla finalizzazione), non letti male in silenzio.
- Limite per cella: 16 MB. Estensioni accettate: `.csv`, `.tsv`, `.txt`.

## Privacy

L'app tiene i PDF **solo in memoria**; i CSV possono pesare GB e questo non è praticabile. È
quindi un'**eccezione esplicita** alla regola "mai su disco":

- i file di upload e gli output stanno in **file temporanei** nella cartella dati dell'app,
  `<config_dir>/csv_jobs/<istanza>/` (`%LOCALAPPDATA%\rizzo-pii\` su Windows,
  `~/Library/Application Support/rizzo-pii/` su macOS, `~/.local/share/rizzo-pii/` su Linux);
- vengono **cancellati a fine job** (`DELETE /csv/jobs/<jid>`, fatto dalla UI quando chiudi o
  ricominci) e **spazzati all'avvio successivo** se il processo è morto prima: ogni istanza tiene un
  lock sulla propria cartella, e all'avvio si eliminano quelle il cui lock è libero;
- upload e job **abbandonati** (tab chiusa senza "Pulisci") vengono cancellati, registro compreso,
  dopo **2 ore senza richieste** (`IDLE_TTL` in `csv_jobs.py`). Un job in coda o in corso non scade mai;
- il **dizionario non va mai su disco**: sta in memoria nel job e muore con `DELETE`/arresto. Si
  scarica solo se lo chiedi (`/dictionary`);
- **nessun valore di cella in log, eccezioni o messaggi d'errore**: gli errori espongono solo
  messaggi del motore privi di contenuto o il *tipo* dell'eccezione; nel log del backend finisce al
  più `job <id> fallito: <TipoEccezione>`;
- le anteprime (header e prime 20 righe) tornano solo alla UI che ha caricato il file.

Chi ha accesso al profilo utente della macchina può quindi leggere i temporanei **mentre un job è in
corso**: valgono le stesse cautele degli altri file in chiaro sul disco (disco cifrato, sessione
bloccata).

## Memoria e prestazioni

**Memoria.** Il motore è in streaming: il file non si carica in RAM. Cresce solo il registro, circa
**150-250 byte per valore unico** (~**200 MB per milione** di valori unici) con il dizionario
attivo; senza dizionario si conserva solo la chiave normalizzata. Lo stato del job espone
`unique_values` e `dictionary_size`. La cache dei risultati è un **LRU limitato** (100 000 voci,
32 M di caratteri in tutto; i testi oltre 4096 caratteri non entrano in cache). Un dizionario molto
grande si scarica a parte: la UI lo copia in `localStorage` per la tab "Ripristina" solo se ha
**≤ 20 000 ID**.

**Prestazioni.**
- Il costo è il **modello**: su CPU ogni cella di testo è un forward pass. Le colonne `preserve` non
  costano nulla: **disattiva "Anonimizza"** sulle colonne che non contengono dati personali (timestamp,
  esiti, ID tecnici). Le celle identiche si risolvono una volta (cache).
- **Avviso file grande**: se il file supera la soglia, `finalize` risponde `large: true` e
  `large_threshold` (byte), e la UI mostra un avviso che invita a disattivare le colonne senza dati
  personali per ridurre i tempi. La soglia **non è fissa nella feature**: è il `MAX_CONTENT_LENGTH`
  dell'app (oggi 50 MB), letto a runtime da `app.py` (`large_threshold=lambda: app.config[...]`) e
  passato al Blueprint, quindi segue la configurazione dell'app. È solo un avviso: il job parte comunque.
- Le celle da analizzare vengono raccolte in **batch** (64 righe per flush) e il modello vede tutti
  i chunk del batch in una sola chiamata (`detect_spans_batch`, `MODEL_BATCH_SIZE = 32` chunk per
  forward pass).

**Misura indicativa** (CPU, percorso con il modello):

| Scenario | Righe | Tempo | Throughput |
|---|---|---|---|
| una colonna di testo libero analizzata | 2 000 | ~15 s | ~130 righe/s |

Nei file veri i testi si ripetono e la cache abbassa molto il conto.

- Un job alla volta: i job successivi sono in **coda FIFO** (`position` nello stato).
- Le inferenze sono **serializzate** (un lock sulla pipeline). Un job CSV con colonne analizzate e
  un'anonimizzazione di testo/PDF lanciata nel frattempo si alternano: la seconda attende il batch in
  corso.

## API HTTP

Tutti gli endpoint sono sotto `/csv`, JSON salvo i download. Il flusso è: **upload a pezzi →
finalize → job → polling → download → delete**.

| Metodo e path | Corpo / risposta |
|---|---|
| `POST /csv/uploads` | `{"name","size"}` → `{"upload_id","chunk_size"(8 MB),"name","size"}`. `400` se vuoto o estensione non valida, `507` se manca spazio su disco (servono ~`2×size + 256 MB`) |
| `PUT /csv/uploads/<uid>?offset=N` | corpo grezzo (≤ 8 MB + 64 KB), pezzi **in ordine** → `{"received","size","complete"}`. `409` se l'offset non è in sequenza, `413` se troppo grande |
| `POST /csv/uploads/<uid>/finalize` | → `{"upload_id","name","size","header","preview","delimiter","encoding","newline","plan","modes","large","large_threshold"}` (`preview` = prime 20 righe, `plan` = piano di default: tutte `anonymize`, `modes` = `["anonymize","preserve"]`; `large` = `true` se `size` supera `large_threshold`, il `MAX_CONTENT_LENGTH` dell'app in byte). `422` se non è un CSV UTF-8 |
| `DELETE /csv/uploads/<uid>` | scarta un upload non ancora usato da un job |
| `POST /csv/jobs` | `{"files":[{"upload_id","plan"?}],"strict"?,"exclude_tags"?,"include_mapping"?}` → `202 {"job_id","state":"queued","position"}`. `plan` = `[{"mode": "anonymize"|"preserve"}, …]`, una voce per colonna; senza `plan` si usa il default (tutte `anonymize`); senza `exclude_tags`/`include_mapping` valgono le preferenze del server |
| `GET /csv/jobs/<jid>` | stato: `state` (`queued`/`running`/`done`/`error`/`cancelled`), `files[]` (`name`, `out_name`, `state`, `rows_done`, `rows_out`, `rows_skipped`, `skipped_lines`, `by_label`, `bytes_done`, `bytes_total`, `error`), `rows_done`, `rows_skipped`, `bytes_done`, `bytes_total`, `mapping_enabled`, `dictionary_size`, `unique_values`, `excluded_tags`, `strict`, `seconds`, `error`, `position` (se in coda) |
| `POST /csv/jobs/<jid>/cancel` | interrompe il job (o lo toglie dalla coda); ritorna lo stato |
| `GET /csv/jobs/<jid>/files/<i>` | output i-esimo, `<nome>.anon.csv`, in streaming. `409` se non ancora pronto |
| `GET /csv/jobs/<jid>/files/<i>/preview` | `{header, preview, rows_out}`: intestazione e prime 20 righe dell'output (l'anteprima nella card Risultato). `409` se non ancora pronto |
| `GET /csv/jobs/<jid>/dictionary` | `dizionario_anonimizzazione.json`, `{"[TAG_N]": valore}`. `404` se il dizionario è disattivato, `409` se il job non è terminato |
| `DELETE /csv/jobs/<jid>` | cancella input, output e registro (se il job è in corso lo annulla e pulisce a fine file) |

Esempio completo (stdlib, nessuna dipendenza):

```python
import json, time, urllib.request

BASE = "http://127.0.0.1:5005"
PATH = "audit.csv"

def call(method, url, body=None, headers=None):
    data = json.dumps(body).encode() if isinstance(body, (dict, list)) else body
    h = {"Content-Type": "application/json"} if isinstance(body, (dict, list)) else {}
    h.update(headers or {})
    with urllib.request.urlopen(urllib.request.Request(BASE + url, data, h, method=method)) as r:
        return r.read()

raw = open(PATH, "rb").read()          # per file enormi: leggi a pezzi invece
up = json.loads(call("POST", "/csv/uploads", {"name": "audit.csv", "size": len(raw)}))
uid, chunk = up["upload_id"], up["chunk_size"]
for off in range(0, len(raw), chunk):
    call("PUT", f"/csv/uploads/{uid}?offset={off}", raw[off:off + chunk],
         {"Content-Type": "application/octet-stream"})

fin = json.loads(call("POST", f"/csv/uploads/{uid}/finalize"))
print(fin["header"], fin["plan"], fin["large"])
plan = fin["plan"]                     # default: tutte "anonymize"
plan[0] = {"mode": "preserve"}         # es. lascia intatta la prima colonna (timestamp)

job = json.loads(call("POST", "/csv/jobs", {
    "files": [{"upload_id": uid, "plan": plan}],
    "strict": False,
}))["job_id"]

while True:
    st = json.loads(call("GET", f"/csv/jobs/{job}"))
    if st["state"] not in ("queued", "running"):
        break
    time.sleep(1)

if st["state"] == "done":
    open("audit.anon.csv", "wb").write(call("GET", f"/csv/jobs/{job}/files/0"))
    open("dizionario_anonimizzazione.json", "wb").write(call("GET", f"/csv/jobs/{job}/dictionary"))
call("DELETE", f"/csv/jobs/{job}")     # via i temporanei e il registro
```

Con `curl`, lo stesso flusso per un file piccolo (un solo pezzo):

```bash
SIZE=$(wc -c < audit.csv)
UID_=$(curl -s -X POST localhost:5005/csv/uploads -H 'Content-Type: application/json' \
       -d "{\"name\":\"audit.csv\",\"size\":$SIZE}" | python -c 'import sys,json;print(json.load(sys.stdin)["upload_id"])')
curl -s -X PUT "localhost:5005/csv/uploads/$UID_?offset=0" --data-binary @audit.csv
curl -s -X POST localhost:5005/csv/uploads/$UID_/finalize
curl -s -X POST localhost:5005/csv/jobs -H 'Content-Type: application/json' \
     -d "{\"files\":[{\"upload_id\":\"$UID_\"}]}"
# poi: GET /csv/jobs/<job_id>, GET /csv/jobs/<job_id>/files/0 -o audit.anon.csv, DELETE /csv/jobs/<job_id>
```

## Uso dalla UI, passo per passo

1. Apri l'app e **trascina uno o più `.csv`/`.tsv`** nella card di input (o clicca per sceglierli).
   La UI passa in **modalità CSV**: ogni file si carica a pezzi da 8 MB con una barra di
   avanzamento, poi se ne legge l'intestazione.
2. Per ogni file compare l'**anteprima tabellare** (prime 20 righe, separatore e codifica rilevati)
   con un checkbox **Anonimizza** per colonna (tutte attive di default) e i comandi **tutte · nessuna**.
   Disattiva le colonne che vuoi lasciare intatte (timestamp, ID tecnici) o che non contengono dati
   personali: meno colonne = meno tempo. Se il file supera il limite di upload dell'app compare un
   avviso che lo suggerisce.
3. Opzioni: **"Fermati alla prima riga malformata"** (strict); lo switch del **dizionario
   reversibile** e i **tag esclusi** (🏷️) valgono anche per i CSV.
4. Premi **Anonimizza**. A destra compaiono avanzamento per file, righe elaborate/saltate (con i
   numeri di riga), conteggio per tag e, volendo, **Annulla**.
5. A job finito: **⬇️ Scarica CSV** per ogni file (`<nome>.anon.csv`) e **⬇️ Scarica dizionario**.
   Se il dizionario ha ≤ 20 000 ID viene anche caricato nella tab **Ripristina**, pronto per
   ricostruire i valori nella risposta dell'LLM.
6. Chiudendo la modalità CSV (✕ o cambiando input) la UI chiama `DELETE` e i temporanei spariscono.

La UI esegue un job alla volta; chi usa l'API può accodarne altri (FIFO).

## Limiti noti

- **Nessuna persistenza tra esecuzioni**: la coerenza degli pseudonimi vale dentro un job, non tra
  job diversi (fuori scopo per scelta, vedi sopra).
- **Nessuna CLI** per i CSV: si usano la UI o l'API HTTP.
- **Niente dialoghi file nativi di Tauri**: l'upload passa dall'HTTP a pezzi, anche nell'app
  desktop.
- **Solo UTF-8 e UTF-8 con BOM**: UTF-16, Latin-1 ecc. vengono rifiutati (converti prima il file).
- **Virgolette normalizzate** (`QUOTE_MINIMAL`), righe vuote scartate: il contenuto è fedele, i byte
  non sempre.
- L'analisi col modello è **lenta** su CPU e vede una cella (o una foglia JSON) alla volta, senza il
  contesto delle altre colonne.
- **Date e numeri non hanno trattamento speciale**: in una colonna analizzata si anonimizzano come nel resto dell'app (escludi `DATE`/`TIME` o disattiva la colonna per tenerli).
- **Rilanciare un job concluso ricarica i file**: per cambiare il piano dopo un'elaborazione, la UI
  cancella il job e ricarica gli input dal browser (su file da GB, il tempo dell'upload torna).
- Il motore non ha un rilevatore proprio: con colonne `anonymize` richiede `detect_fn` (l'app passa `detect_spans_batch`).
- Il modello ha i limiti di sempre (vedi README, "Limitations"): affianca la rete regex/checksum,
  ma una PII in una colonna `preserve` **non viene vista**.
