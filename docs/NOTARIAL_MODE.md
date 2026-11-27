# Modalità notarile

La modalità notarile estende Rizzo PII con una revisione selettiva delle singole
occorrenze. Dalla schermata standard scegliere **Modalità notarile**, oppure
aprire `/notarial/` sullo stesso indirizzo locale dell'applicazione.
Il collegamento **Rizzo PII standard** torna alla schermata originale.

## Classificazione e revisione

Il modello NER esistente riconosce le entità; un classificatore separato applica
regole contestuali senza riaddestrare il modello. La tassonomia comprende nascita,
residenza, domicilio, immobile e dati catastali, sedi societarie, luoghi notarili
e degli uffici, date documentali, recapiti, dati economici e riferimenti degli atti.

La stessa città può essere luogo di nascita, residenza e ubicazione dell'immobile.
La selezione usa identificativi e intervalli di ciascuna occorrenza, invece di
sostituire globalmente tutti i valori uguali. L'utente può selezionare categorie,
sottocategorie e occorrenze, correggere la classificazione e revisionare i casi
ambigui. Testo e geometria originali non vengono riscritti durante la revisione.

La modalità irreversibile è predefinita per questo profilo. Prima di copiare o
esportare, il backend controlla la revisione corrente e i residui riconosciuti.
Le ambiguità richiedono una decisione esplicita; immagini o allegati non verificati
richiedono una conferma e non attestano anonimizzazione completa.

Foto, volti, firme e timbri non vengono rimossi automaticamente dal profilo
notarile. Le funzioni esistenti della modalità standard restano nel loro percorso.

## Integrazione

Il profilo è montato sotto `/notarial` nello stesso processo Flask, con lo stesso
modello NER già caricato dall'app standard. Le richieste e le risposte del profilo
hanno un proprio controllo di sessione; la revisione e i documenti temporanei sono
tenuti in memoria. Il profilo non legge né cancella il dizionario `localStorage`
della schermata standard. L'OCR riusa i motori locali e separa la cache dei risultati.

La modalità notarile richiede accesso da loopback e non introduce API esterne,
telemetria o download del modello durante l'elaborazione. Per una distribuzione
senza download al primo avvio usare la variante **offline** già prevista upstream;
la variante leggera conserva il provisioning del modello dell'app originale.

Nome del prodotto, identificativo del bundle, versione, porte e procedura di
rilascio restano quelli upstream. L'aggiunta dei sorgenti non rende notarizzato
un nuovo binario: il manutentore deve compilare, firmare e notarizzare la release
con i propri strumenti e credenziali. La PR non include certificati, credenziali,
pesi del modello o pacchetti DMG/EXE.

## Verifiche

I test impiegano esclusivamente documenti sintetici e distinguono la classificazione
contestuale dal riconoscimento. I risultati delle fixture di regressione non
stimano l'accuratezza su atti o perizie inediti.

```sh
python -m unittest discover -v tests
python tests/test_ripristino_placeholder.py
python scripts/evaluate_notarial.py
```

La suite di integrazione usa Flask 3.1.3, PyMuPDF 1.26.7 e ONNX Runtime 1.32.0.
Il workflow aggiunge verifiche su Linux, macOS e Windows senza scaricare pesi o
usare dati personali reali. Le prove sui binari firmati e sull'hardware da 8 GB
restano parte della verifica di rilascio del manutentore.

Il progetto originale è di Simone Rizzo, sponsorizzato da Rizzo AI Academy.
Adattamento per l'ambito notarile: **Nader Deghaili**. Il sorgente conserva la
licenza MIT; l'inclusione di PyMuPDF nei binari mantiene gli obblighi
AGPL/commerciali documentati in `THIRD_PARTY_LICENSES.md`.
