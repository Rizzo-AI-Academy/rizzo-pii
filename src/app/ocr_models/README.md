# Modelli OCR (ONNX)

Qui vanno i 3 file che il backend OCR carica per leggere i **PDF scansionati**.
Non sono versionati (vedi `.gitignore`): si generano prima della build.

```
det.onnx    PP-OCRv6 det small   rilevamento righe di testo    ~9,9 MB
cls.onnx    ch_ppocr_mobile_v2   orientamento delle righe      ~0,6 MB
rec.onnx    multi_PP-OCRv6_rec   riconoscimento caratteri     ~21,2 MB
```

Totale ~32 MB.

## Come popolarli

```bash
pip install rapidocr onnxruntime
cd src/app && python -m ocr.fetch_models
```

Lo script istanzia RapidOCR una volta, lascia che scarichi i modelli dalla sua
fonte ufficiale, e li copia qui con nomi stabili (così `ocr/rapid_ocr.py` non
dipende dai nomi di versione di PaddleOCR).

## Una sola lingua? No: il modello di riconoscimento è multilingua

PP-OCRv6 usa **un unico modello di riconoscimento** (`multi_PP-OCRv6_rec_small`)
per tutte le 52 lingue supportate, italiano incluso. `Rec.lang_type: "it"` serve
solo a validare che la lingua sia supportata: non cambia il file caricato e non
esiste un dizionario di caratteri separato da impacchettare.

(Nelle versioni PP-OCRv4/v5 esistevano modelli per famiglia — `latin`, `cyrillic`
ecc. — e un `rec_keys.txt`. Con la v6 non serve più.)

## Perché impacchettati e non scaricati al primo uso

RapidOCR di suo scarica i modelli alla prima chiamata. In un'app che promette
**100% locale, nessuna telemetria**, una richiesta di rete all'apertura del primo
PDF scansionato è un bug di prodotto, non un dettaglio. Per questo
`ocr/rapid_ocr.py` passa i path espliciti e le spec di PyInstaller li includono.

Se i file mancano, `RapidOcr.models_present()` ritorna `False`, l'app degrada al
comportamento precedente (pagine scansionate = nessun testo) e lo dice all'utente.
