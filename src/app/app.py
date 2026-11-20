# -*- coding: utf-8 -*-
"""
App locale per l'anonimizzazione reversibile di documenti con il modello PII.

Flusso d'uso:
  1) ANONIMIZZA  - incolli testo o carichi un PDF; il modello + una rete regex/checksum
     trovano le PII. Ogni entita' riceve un ID univoco e reversibile: [FULLNAME_1],
     [IBAN_1], ... (valori uguali condividono lo stesso ID).
  2) COPIA       - copi il testo anonimizzato e lo incolli in ChatGPT/altro LLM.
  3) RIPRISTINA  - incolli la risposta dell'LLM (che contiene i placeholder) e l'app
     rimette i valori veri usando il dizionario locale.

Tutto in locale: il testo e il dizionario {placeholder -> valore} non lasciano la macchina.

Il modello e' affiancato da una rete REGEX + CHECKSUM (EMAIL, TELEFONO, IBAN, CF, PIVA,
carta di credito, importi, targhe, URL, IP). Le entita' validate matematicamente (IBAN/CF/
PIVA/carta) hanno priorita' sul modello in caso di sovrapposizione.

Il dizionario di ripristino si puo' DISATTIVARE (switch nell'UI, --no-mapping, PII_MAPPING=0):
l'anonimizzazione diventa definitiva, nessuna chiave placeholder->valore viene costruita.

Scansioni e immagini: le pagine senza layer di testo (PDF scansionati, foto, .png/
.jpg/.tiff) passano dall'OCR locale (ocr.py -> docmd: PaddleOCR su ONNX Runtime, CPU).
Il testo letto entra nell'analisi come quello nativo; nel PDF anonimizzato quelle
pagine vengono coperte SUI PIXEL (pdf_export._redact_scanned_page).

Endpoint HTTP:
  GET  /health    liveness/readiness senza inference (200 = modello caricato, 503 = no)
  POST /analyze   {"text": ...} oppure multipart con un file (.pdf/.md/.txt/immagine) o
                  con "doc_id" (documento gia' caricato con /preview); campi opzionali
                  "exclude_tags", "include_mapping", "ocr" (auto|force|off), "ocr_preset"
  POST /ocr       OCR di un PDF o di un'immagine (file o doc_id): testo Markdown per
                  pagina, opz. righe con box ("lines=1") e avanzamento NDJSON ("stream=1")
  POST /pdf       stesso input di /analyze -> scarica il PDF ANONIMIZZATO (redazione
                  vera del PDF caricato, oppure PDF ricostruito dal testo)
  POST /preview   multipart con un .pdf -> lo tiene in memoria per l'ANTEPRIMA a video e
                  ritorna {doc_id, n_pages, text}
  POST /pdf/preview  come /pdf, ma invece del binario ritorna {doc_id, n_pages, ...}:
                  il PDF anonimizzato resta in memoria e si guarda pagina per pagina
  GET  /doc/<id>/page/<n>.png   pagina renderizzata (anteprima a video)
  GET  /doc/<id>/file.pdf       download del documento tenuto in memoria
  GET  /settings  legenda dei tag, tag esclusi, dizionario reversibile, opzioni OCR
  POST /settings  {"excluded_tags": [...], "mapping_enabled": bool, "ocr_mode": ...,
                  "ocr_preset": ..., "ocr_redact_seals": bool} -> salva in prefs.json
                  (/tags e' un alias storico degli stessi due endpoint)
  GET  /config, POST /config, GET /port-check   host/porta del server

Avvio:  python app.py   ->   http://127.0.0.1:5005
Configurazione host/porta (precedenza): CLI --host/--port > env PII_HOST/PII_PORT >
  config.json (vedi server_config.py) > default 127.0.0.1:5005
Preferenze di anonimizzazione (precedenza): campo nella richiesta > CLI --exclude-tags/
  --no-mapping > env PII_EXCLUDE_TAGS/PII_MAPPING > prefs.json > default
"""

import bisect
import json
import os
import queue
import re
import secrets
import sys
import threading
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import fitz  # PyMuPDF
from flask import Flask, Response, jsonify, request, send_from_directory

import ocr
import pdf_export
import pdf_text
import server_config
# Rete REGEX + CHECKSUM: modulo a parte, senza dipendenze dal modello. I nomi
# restano importabili da qui (`app.detect_regex`) per non rompere chi li usa.
from detectors import (DETECTORS, SOFT_REGEX_LABELS, cf_ok,  # noqa: F401
                       complete_time, detect_iban, detect_regex, iban_ok, luhn_ok,
                       piva_ok)
from onnx_ner import OnnxTokenClassifier, find_onnx


def _resource_path(rel):
    """Percorso risorsa valido sia in sviluppo sia dentro l'exe PyInstaller."""
    base = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base, rel)


# MODELLO usato dall'app. Accetta tre forme, provate in quest'ordine:
#   "1.2.0"  -> models/rizzo-pii-0.3B-v1.2.0/   (versione di un run di training)
#   "main"   -> models/rizzo-pii-0.3B-main/     (revision scaricata da HF, senza numero)
#   percorso -> usato tal quale (assoluto o relativo alla root della repo)
# Metti None per usare AUTOMATICAMENTE l'ultima versione `-v*` disponibile.
APP_MODEL_VERSION = "1.5.0"

# Override a runtime: env PII_MODEL_DIR, che vince su tutto. E' anche il modo in cui
# l'app desktop passa il modello SCARICATO al primo avvio (installer leggero, vedi
# tauri/src-tauri/src/model.rs). Dentro l'exe, altrimenti, il modello e' quello
# impacchettato come "pii_model" (installer offline, vedi build_sidecar.spec).
# In sviluppo: pin sopra -> auto-ultima versione -> vecchio non versionato -> legacy.
if os.environ.get("PII_MODEL_DIR"):
    MODEL_DIR = os.environ["PII_MODEL_DIR"]
elif getattr(sys, "_MEIPASS", None):
    MODEL_DIR = _resource_path("pii_model")
else:
    import re
    _models = Path(__file__).resolve().parents[2] / "models"
    _cands = []
    if APP_MODEL_VERSION:
        _cands = [_models / f"rizzo-pii-0.3B-v{APP_MODEL_VERSION}",
                  _models / f"rizzo-pii-0.3B-{APP_MODEL_VERSION}"]
        if os.sep in APP_MODEL_VERSION or "/" in APP_MODEL_VERSION:   # e' un percorso
            _cands += [Path(APP_MODEL_VERSION), _models.parent / APP_MODEL_VERSION]
    _pinned = next((p for p in _cands if p.is_dir()), None)
    _versioned = [p for p in _models.glob("rizzo-pii-0.3B-v*") if p.is_dir()]
    if _pinned:
        MODEL_DIR = str(_pinned)
    elif _versioned:
        MODEL_DIR = str(max(_versioned, key=lambda p: tuple(
            int(x) for x in re.search(r"-v([0-9][0-9.]*)$", p.name).group(1).split("."))))
    else:
        _prod = _models / "rizzo-pii-0.3B"
        MODEL_DIR = str(_prod if _prod.exists() else _models / "pii_model_legacy")

# Se la cartella non c'e', dirlo QUI e con il comando giusto: senza questo controllo si
# arriva a from_pretrained() con un path inventato (pii_model_legacy, ultimo fallback) e
# l'errore parla di una cartella che l'utente non ha mai sentito nominare.
if not getattr(sys, "_MEIPASS", None) and not os.path.isdir(MODEL_DIR):
    _v = APP_MODEL_VERSION or "1.5.0"
    print(f"ERRORE: modello non trovato in {MODEL_DIR}\n"
          f"Scaricalo con (la revision e la cartella devono combaciare):\n"
          f"  hf download rizzoaiacademy/rizzo-pii-0.3B --revision v{_v} "
          f"--local-dir models/rizzo-pii-0.3B-v{_v}\n"
          f"Oppure indica una cartella tua con la variabile PII_MODEL_DIR.",
          file=sys.stderr)
    sys.exit(2)

ASSETS_DIR = _resource_path("assets")   # mascotte / icone (servite su /assets/<file>)
APP_VERSION = "2.2.0"                    # versione mostrata nell'UI (allineata a tauri.conf.json)
MAX_WORDS = 120      # parole per chunk (~180 subword, sotto i 512 del training)
OVERLAP = 20         # parole di sovrapposizione tra chunk consecutivi

# --------------------------------------------------------------------------- #
# Caricamento modello (una sola volta all'avvio)
# --------------------------------------------------------------------------- #
# Due backend con lo stesso risultato (verificato da src/training/export_onnx.py):
#   onnx  -> onnxruntime + tokenizers, niente torch: e' quello dell'app impacchettata.
#   torch -> pipeline di transformers (CPU o GPU): serve se il .onnx non c'e'.
# PII_BACKEND=onnx|torch forza la scelta; di default si usa ONNX quando c'e' il model.onnx.
_backend = os.environ.get("PII_BACKEND", "auto").strip().lower()
_onnx_path = find_onnx(MODEL_DIR)
if _backend == "onnx" or (_backend != "torch" and _onnx_path):
    print(f"Carico il modello ONNX da {_onnx_path or MODEL_DIR} (CPU)...")
    nlp = OnnxTokenClassifier(MODEL_DIR, onnx_path=_onnx_path)
    MODEL_BACKEND, MODEL_DEVICE = "onnx", "cpu"
else:
    import torch
    from transformers import pipeline
    device = 0 if torch.cuda.is_available() else -1
    print(f"Carico il modello da {MODEL_DIR} su {'GPU' if device == 0 else 'CPU'}...")
    nlp = pipeline(
        "token-classification",
        model=MODEL_DIR,
        tokenizer=MODEL_DIR,
        aggregation_strategy="simple",
        device=device,
    )
    MODEL_BACKEND, MODEL_DEVICE = "torch", ("cuda" if device == 0 else "cpu")
print(f"Modello pronto ({MODEL_BACKEND}).")

# L'inferenza gira sempre sullo STESSO thread, qualunque thread serva la richiesta
# (issue #139). torch crea il suo pool OpenMP (torch.get_num_threads() - 1 thread) per
# ogni thread che lo chiama, e su Windows non lo libera quando quel thread finisce: con
# app.run(threaded=True), che usa un thread nuovo per richiesta, ogni /analyze lasciava
# un pool intero, senza limite. Con un worker fisso il pool nasce una volta sola.
# Le richieste in parallelo fanno la fila qui, ma sulla CPU torch usa gia' tutti i core.
_INFERENZA = ThreadPoolExecutor(max_workers=1, thread_name_prefix="inferenza")

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  # 50 MB

# Estensioni accettate dall'upload (il PDF passa da PyMuPDF, il resto e' testo puro).
TEXT_EXTS = {".md", ".markdown", ".txt", ".text"}

# --------------------------------------------------------------------------- #
# Anteprima a video dei PDF (documento caricato a sinistra, anonimizzato a destra)
#
# I PDF vengono renderizzati QUI, con PyMuPDF, e serviti come PNG per pagina: non
# serve un viewer PDF nel browser (WebView2/WKWebView dentro Tauri non lo hanno in
# modo affidabile) ne' una libreria JS esterna (l'app e' offline, niente CDN).
#
# I documenti restano in memoria (mai su disco: valgono quanto il documento stesso)
# in una LRU piccola, e muoiono con il processo. Il render di una pagina e' pigro e
# poi tenuto in cache: aprire un PDF di 200 pagine non costa 200 render.
# --------------------------------------------------------------------------- #
PREVIEW_DPI = 110          # ~1150 px di larghezza su un A4: leggibile senza pesare
MAX_DOCS = 6               # documenti tenuti in memoria (LRU, i piu' vecchi cadono)

_DOCS = OrderedDict()      # doc_id -> {"pdf", "pages": {n: png}, "n_pages", "name", "kind", "role"}
_DOCS_LOCK = threading.Lock()


def _safe_name(name, default="documento.pdf"):
    """Nome file pulito per Content-Disposition (ASCII, niente separatori)."""
    base = os.path.basename(name or "").strip()
    base = re.sub(r"[^A-Za-z0-9._\- ]+", "_", base).strip(" ._")
    return base or default


def _store_doc(data, name="documento.pdf", kind="pdf", role="output"):
    """Mette un PDF nello store dell'anteprima. Ritorna (doc_id, n_pages).
    role="source": documento caricato dall'utente (riusabile con doc_id da /analyze,
    /pdf, /ocr senza un secondo upload); "output": PDF anonimizzato da scaricare.
    Le immagini caricate arrivano qui gia' convertite in PDF (kind="image")."""
    with fitz.open(stream=data, filetype="pdf") as doc:
        n_pages = doc.page_count
        # proporzioni delle pagine: l'UI riserva lo spazio prima che il PNG arrivi
        sizes = [[round(p.rect.width, 1), round(p.rect.height, 1)] for p in doc]
    doc_id = secrets.token_urlsafe(12)
    safe = _safe_name(name)
    if kind == "image":                        # il download e' comunque un PDF
        safe = os.path.splitext(safe)[0] + ".pdf"
    with _DOCS_LOCK:
        _DOCS[doc_id] = {"pdf": data, "pages": {}, "n_pages": n_pages,
                         "name": safe, "orig_name": name or safe, "kind": kind,
                         "role": role, "sizes": sizes}
        while len(_DOCS) > MAX_DOCS:
            _DOCS.popitem(last=False)
    return doc_id, n_pages


def _get_doc(doc_id):
    with _DOCS_LOCK:
        d = _DOCS.get(doc_id)
        if d is not None:
            _DOCS.move_to_end(doc_id)          # LRU: l'uso lo tiene in vita
    return d


def _page_png(d, n, dpi=PREVIEW_DPI):
    """PNG della pagina n (0-based) del documento, con cache. None se fuori range."""
    if not (0 <= n < d["n_pages"]):
        return None
    key = (n, dpi)
    with _DOCS_LOCK:
        png = d["pages"].get(key)
    if png is None:
        with fitz.open(stream=d["pdf"], filetype="pdf") as doc:
            png = doc.load_page(n).get_pixmap(dpi=dpi).tobytes("png")
        with _DOCS_LOCK:
            d["pages"][key] = png
    return png


# --------------------------------------------------------------------------- #
# Legenda dei tag + tag esclusi dall'anonimizzazione
#
# I 22 tag del modello (docs/TASSONOMIA_TAG.md) + URL e IPADDR, che sono solo-regex:
# il modello non e' stato addestrato su di essi, li trova la rete regex.
# L'utente puo' DESELEZIONARE un tag: le entita' di quel tipo vengono rilevate ma non
# sostituite (restano in chiaro). Serve a chi deve confrontare gli importi (AMOUNT) o
# tenere eta'/sesso in un caso clinico.
# --------------------------------------------------------------------------- #
TAGS = [
    ("FULLNAME", "Nome di persona (anche ruoli legali: giudice, avvocato, parti, teste)",
     "Person name (legal roles included: judge, lawyer, parties, witness)", "Mario Rossi"),
    ("AGE", "Età", "Age", "45 anni"),
    ("GENDER", "Sesso / genere", "Sex / gender", "Femmina"),
    ("DATE", "Data di calendario", "Calendar date", "12/06/1985"),
    ("TIME", "Ora", "Time of day", "ore 15:30"),
    ("STREET", "Via / piazza / corso", "Street / square", "Via Garibaldi"),
    ("BUILDINGNUM", "Numero civico", "Building number", "24"),
    ("ZIPCODE", "CAP", "ZIP / postal code", "00185"),
    ("CITY", "Città", "City", "Milano"),
    ("PROVINCE", "Sigla della provincia", "Province code", "MI"),
    ("EMAIL", "Email, PEC inclusa", "Email, certified mail included", "m.rossi@studio.it"),
    ("TELEPHONENUM", "Numero di telefono", "Phone number", "+39 333 1234567"),
    ("CF", "Codice fiscale (checksum verificato)", "Italian tax code (checksum verified)",
     "RSSMRA85H12F205Y"),
    ("PIVA", "Partita IVA (checksum verificato)", "VAT number (checksum verified)", "12345678903"),
    ("ID_DOC", "Numero di documento d'identità (carta, passaporto, patente)",
     "Identity document number (ID card, passport, driving licence)", "CA12345AB"),
    ("IBAN", "IBAN / numero di conto (checksum verificato)",
     "IBAN / account number (checksum verified)", "IT60X0542811101000000123456"),
    ("CREDITCARDNUMBER", "Numero di carta di credito (Luhn verificato)",
     "Credit card number (Luhn verified)", "4111 1111 1111 1111"),
    ("AMOUNT", "Importo in denaro", "Money amount", "€ 12.500,00"),
    ("TARGA", "Targa di veicolo", "Vehicle plate", "AB 123 CD"),
    ("ORG", "Ragione sociale privata: società, studio legale, banca",
     "Private organization: company, law firm, bank", "Edilnord S.r.l."),
    ("DOCID", "Codice di un atto: ruolo generale, protocollo, repertorio, sentenza",
     "Document code: case number, protocol, repertoire, judgment", "1234/2024"),
    ("CATASTO", "Dati catastali: foglio, particella, subalterno",
     "Land registry data: sheet, parcel, subordinate", "Foglio 12, part. 345, sub. 6"),
    ("URL", "Indirizzo web (rilevato solo dalla rete regex, non dal modello)",
     "Web address (regex net only, not from the model)", "https://www.studiorossi.it"),
    ("IPADDR", "Indirizzo IP, anche subnet CIDR o range (rilevato solo dalla rete regex)",
     "IP address, including CIDR subnet or range (regex net only)", "192.168.1.1"),
]
TAG_NAMES = [t[0] for t in TAGS]

# Preferenze di default del server (env > prefs.json). Gli argomenti CLI le sovrascrivono
# all'avvio; ogni singola richiesta puo' comunque passare le proprie.
#
# MAPPING_ENABLED = False -> l'anonimizzazione e' DEFINITIVA: nessun dizionario
# placeholder->valore viene costruito, restituito o salvato, e la risposta non contiene
# piu' il testo originale delle entita' (nemmeno dentro `segments`). Serve a chi non
# vuole che una chiave di ripristino esista, da nessuna parte.
_prefs = server_config.load_prefs()
EXCLUDED_TAGS = _prefs["excluded_tags"]
MAPPING_ENABLED = _prefs["mapping_enabled"]
# OCR delle scansioni: modalita' (auto|force|off), preset dei modelli, copertura timbri
OCR_MODE = _prefs["ocr_mode"]
OCR_PRESET = _prefs["ocr_preset"]
OCR_REDACT_SEALS = _prefs["ocr_redact_seals"]
_PREFS_LOCK = threading.Lock()   # protegge la lettura/scrittura ATOMICA della coppia sopra


def _get_prefs():
    """Snapshot coerente di (excluded_tags, mapping_enabled): mai una POST /settings a meta'."""
    with _PREFS_LOCK:
        return EXCLUDED_TAGS, MAPPING_ENABLED


def _get_ocr_prefs():
    with _PREFS_LOCK:
        return OCR_MODE, OCR_PRESET, OCR_REDACT_SEALS


# --------------------------------------------------------------------------- #
# Chunking word-safe + inferenza del modello su tutto il documento
# --------------------------------------------------------------------------- #
def chunk_text(text, max_words=MAX_WORDS, overlap=OVERLAP):
    """Ritorna [(sottostringa, offset_char_globale), ...] senza tagliare parole."""
    words = list(re.finditer(r"\S+", text))
    if not words:
        return []
    chunks, i = [], 0
    step = max(1, max_words - overlap)
    while i < len(words):
        block = words[i:i + max_words]
        start, end = block[0].start(), block[-1].end()
        chunks.append((text[start:end], start))      # slice esatto -> offset diretti
        if i + max_words >= len(words):
            break
        i += step
    return chunks


def detect_model(text):
    """Entita' trovate dal modello mmBERT su tutti i chunk, su offset globali."""
    chunks = chunk_text(text)
    ents = []
    if chunks:
        results = _INFERENZA.submit(nlp, [c for c, _ in chunks]).result()
        if isinstance(results, dict):                 # singolo chunk -> normalizza
            results = [results]
        for (_, off), res in zip(chunks, results):
            for e in res:
                ents.append({
                    "label": e["entity_group"],
                    "start": int(e["start"]) + off,
                    "end": int(e["end"]) + off,
                    "score": float(e["score"]),
                    "validated": False,
                    "source": "modello",
                })
    return ents, len(chunks)


# --------------------------------------------------------------------------- #
# Fusione modello + regex, ID reversibili, testo anonimizzato
# --------------------------------------------------------------------------- #
def _is_word(ch):
    """Carattere interno a una parola (lettere accentate e cifre incluse)."""
    return ch.isalnum() or ch == "_"


def _merge(cands, text):
    """Greedy senza overlap. Priorita': checksum-valido > regex (non soft) > score
    > lunghezza.
    La rete regex copre campi a forma molto specifica: per quegli span e' piu' affidabile
    del modello (evita la frammentazione di CF/IBAN/carta in piu' pezzi). Fanno eccezione
    i tag in SOFT_REGEX_LABELS, dove la forma non ha un checksum a confermarla."""
    order = sorted(
        cands,
        key=lambda e: (1 if e["validated"] else 0,
                       1 if (e["source"] == "regex"
                             and e["label"] not in SOFT_REGEX_LABELS) else 0,
                       e["score"], e["end"] - e["start"]),
        reverse=True,
    )
    kept = []
    for e in order:
        # kept resta ordinata per start e senza sovrapposizioni: allora un candidato puo'
        # accavallarsi solo con i due vicini, che la ricerca binaria trova subito. Il
        # confronto con TUTTA kept era O(n^2): su un documento con 40.000 entita' (una chat
        # esportata di qualche MB) erano 111 s spesi qui, dopo l'inferenza.
        i = bisect.bisect_right(kept, e["start"], key=lambda k: k["start"])
        if (i and kept[i - 1]["end"] > e["start"]) or \
           (i < len(kept) and kept[i]["start"] < e["end"]):
            continue
        kept.insert(i, e)
    # niente spazi inglobati nei placeholder (il modello a volte include lo spazio iniziale)
    for e in kept:
        while e["start"] < e["end"] and text[e["start"]].isspace():
            e["start"] += 1
        while e["end"] > e["start"] and text[e["end"] - 1].isspace():
            e["end"] -= 1
    kept = [e for e in kept if e["end"] > e["start"]]

    # Allineamento ai confini di parola. Il modello etichetta i sotto-token e a volte ne
    # copre solo una parte ("No" di "Novara"): sostituendo la span cosi' com'e' resterebbe
    # "[CITY_1]vara", cioe' un valore ancora ricostruibile. Se una span taglia una parola
    # a meta', la si estende fino a coprirla. Nel dubbio si maschera un carattere in piu':
    # per un anonimizzatore l'errore per eccesso e' l'unico accettabile.
    for e in kept:
        while (e["start"] > 0
               and _is_word(text[e["start"] - 1]) and _is_word(text[e["start"]])):
            e["start"] -= 1
        while (e["end"] < len(text)
               and _is_word(text[e["end"]]) and _is_word(text[e["end"] - 1])):
            e["end"] += 1

    # L'estensione puo' rendere due span sovrapposte o adiacenti: si fondono, altrimenti
    # una stessa parola verrebbe sostituita due volte ("[CATASTO_1][CATASTO_2]").
    kept.sort(key=lambda e: (e["start"], -(e["end"] - e["start"])))
    merged = []
    for e in kept:
        if merged and e["start"] < merged[-1]["end"]:
            merged[-1]["end"] = max(merged[-1]["end"], e["end"])
            continue
        if merged and e["start"] == merged[-1]["end"] and e["label"] == merged[-1]["label"]:
            merged[-1]["end"] = e["end"]
            continue
        merged.append(e)
    return merged


def _norm(s):
    return re.sub(r"\s+", " ", s.strip()).casefold()


def analyze(text, excluded=None, mapping_enabled=True):
    """excluded = tag da NON anonimizzare: le entita' di quel tipo vengono scartate
    prima della fusione, quindi il valore resta in chiaro nel testo di output.

    mapping_enabled=False -> anonimizzazione DEFINITIVA: nessun dizionario
    placeholder->valore, e i segmenti-entita' non riportano il testo originale
    (che altrimenti permetterebbe di ricostruire il dizionario dalla risposta).
    La numerazione dei placeholder resta: dice che due occorrenze sono lo stesso
    soggetto, ma da sola non fa risalire al valore."""
    excluded = set(excluded or ())
    model_ents, n_chunks = detect_model(text)
    # il modello taglia i minuti delle ore: la regex dell'ora completa SOLO le sue span TIME
    complete_time(model_ents, text)
    cands = model_ents + detect_regex(text)
    if excluded:
        cands = [e for e in cands if e["label"] not in excluded]
    kept = _merge(cands, text)

    # ID reversibili: stesso (label, valore-normalizzato) -> stesso placeholder.
    counters, seen, mapping = {}, {}, {}
    for e in kept:
        val = text[e["start"]:e["end"]]
        key = (e["label"], _norm(val))
        if key in seen:
            e["ph"] = seen[key]
        else:
            counters[e["label"]] = counters.get(e["label"], 0) + 1
            ph = f"[{e['label']}_{counters[e['label']]}]"
            seen[key] = ph
            if mapping_enabled:
                mapping[ph] = val
            e["ph"] = ph

    # segmenti per la preview + testo anonimizzato + statistiche
    segments, anon, by_label, by_source, pos = [], [], {}, {}, 0
    for e in kept:
        if e["start"] > pos:
            segments.append({"t": text[pos:e["start"]]})
            anon.append(text[pos:e["start"]])
        seg = {
            "label": e["label"],
            "ph": e["ph"],
            "src": e["source"],
            "validated": e["validated"],
        }
        if mapping_enabled:                       # senza dizionario niente valore originale
            seg["t"] = text[e["start"]:e["end"]]
        segments.append(seg)
        anon.append(e["ph"])
        by_label[e["label"]] = by_label.get(e["label"], 0) + 1
        by_source[e["source"]] = by_source.get(e["source"], 0) + 1
        pos = e["end"]
    if pos < len(text):
        segments.append({"t": text[pos:]})
        anon.append(text[pos:])

    return {
        "segments": segments,
        "anonymized_text": "".join(anon),
        "mapping": mapping,
        "mapping_enabled": mapping_enabled,
        "n_chunks": n_chunks,
        "n_chars": len(text),
        "n_entities": len(kept),
        "n_unique": len(seen),
        "by_label": dict(sorted(by_label.items(), key=lambda x: -x[1])),
        "by_source": by_source,
        "excluded_tags": sorted(excluded),
    }


# --------------------------------------------------------------------------- #
# Endpoints
# --------------------------------------------------------------------------- #
def _page():
    return _load_ui().replace("__VERSION__", APP_VERSION)


@app.route("/")
def index():
    return _page()


@app.route("/assets/<path:fn>")
def assets(fn):
    if os.path.isfile(os.path.join(ASSETS_DIR, fn)):
        return send_from_directory(ASSETS_DIR, fn)
    return ("", 404)


@app.route("/favicon.ico")
def favicon():
    if os.path.isfile(os.path.join(ASSETS_DIR, "mascot_shield.png")):
        return send_from_directory(ASSETS_DIR, "mascot_shield.png")
    return ("", 204)


@app.errorhandler(404)
def not_found(_e):
    return _page()


@app.route("/health")
@app.route("/healthz")
def health():
    """Liveness/readiness SENZA inference: sonda economica per orchestratori e sidecar.
    200 = modello caricato e pronto; 503 = server su ma modello non disponibile."""
    ready = nlp is not None
    excl, mapping = _get_prefs()
    body = {
        "status": "ok" if ready else "loading",
        "model_loaded": ready,
        "model": os.path.basename(str(MODEL_DIR).rstrip("/\\")),
        "model_version": APP_MODEL_VERSION,
        "app_version": APP_VERSION,
        "device": MODEL_DEVICE,
        "backend": MODEL_BACKEND,
        "tags": len(TAG_NAMES),
        "excluded_tags": excl,
        "mapping_enabled": mapping,
        "ocr": {**ocr.status(), "mode": _get_ocr_prefs()[0]},
    }
    return jsonify(body), (200 if ready else 503)


def _is_pdf(name, data):
    return os.path.splitext((name or "").lower())[1] == ".pdf" or data[:5] == b"%PDF-"


def _text_from_bytes(name, data):
    """Testo dai bytes di un upload: PDF via PyMuPDF, .md/.txt come testo puro.

    Per i PDF non basta `page.get_text()`: nei moduli AcroForm il valore sta
    nei widget (issue #85). `pdf_text.collect_readable_text` allinea
    l'estrazione a quanto `pdf_export` gia' ispezionava per i residui.
    """
    name = (name or "").lower()
    ext = os.path.splitext(name)[1]
    if _is_pdf(name, data):
        with fitz.open(stream=data, filetype="pdf") as doc:
            # senza, l'utente leggeva l'errore grezzo di PyMuPDF: "document closed or
            # encrypted". Un PDF con la sola password del proprietario si apre: needs_pass
            # e' falso e si prosegue.
            if doc.needs_pass:
                raise PdfLocked("PDF protetto da password: rimuovi la protezione e riprova.")
            # Non solo page.get_text(): nei PDF fillable il valore sta nei
            # widget AcroForm (issue #85). Stesso contratto di pdf_export.
            return pdf_text.collect_readable_text(doc)
    if ext in TEXT_EXTS or not ext:
        # utf-16 solo col BOM: senza, il codec accetta qualunque sequenza di lunghezza pari
        # e si mangia i .txt a 8 bit. cp1252 prima di latin-1: e' quello che scrivono Word
        # e il Blocco note italiani, e i due differiscono su euro e virgolette tipografiche.
        primo = ("utf-16",) if data[:2] in (b"\xff\xfe", b"\xfe\xff") else ()
        for enc in primo + ("utf-8-sig", "cp1252", "latin-1"):
            try:
                # Via i byte nulli. Un utf-16 SENZA BOM finisce per forza su una codifica a
                # 8 bit e il testo esce "M\0a\0r\0i\0o": a schermo non si vede niente (in
                # HTML il NUL e' invisibile), quindi l'utente legge il documento intero,
                # non trova segnaposti e conclude che non ci fosse nulla da anonimizzare -
                # mentre i rilevatori, che lavorano sul testo vero, non trovano niente.
                # In un utf-16 latino il byte alto e' proprio quello nullo: tolto, il testo
                # torna esatto e le PII tornano trovabili. Riconoscere l'utf-16 contando i
                # byte nulli, invece, sbaglia in entrambe le direzioni.
                return data.decode(enc).replace("\x00", "")
            except UnicodeDecodeError:
                continue
    raise ValueError(
        f"Formato non supportato: {ext or name or 'sconosciuto'}. "
        "Accetto .pdf, .md, .txt oppure testo incollato."
    )


class PdfLocked(ValueError):
    """PDF con password di apertura: /preview lo segnala con `code`, cosi' l'UI lo
    dice nella sua lingua invece di "Anteprima del PDF non disponibile"."""


def _uploaded_file():
    """Il file dell'upload, se c'e'. "pdf" e' il nome storico del campo (l'UI lo
    usa ancora), "file" e' l'alias nuovo."""
    return next((request.files[k] for k in ("pdf", "file")
                 if k in request.files and request.files[k].filename), None)


class _ReqError(Exception):
    """Errore da rendere al client come {"error": ...} con uno status preciso."""

    def __init__(self, msg, status=400, code=None):
        super().__init__(msg)
        self.msg, self.status, self.code = msg, status, code

    def response(self):
        body = {"error": self.msg}
        if self.code:
            body["code"] = self.code
        return jsonify(body), self.status


def _param(name, default=None):
    """Parametro della richiesta: campo multipart, query string o chiave del JSON."""
    if name in request.form:
        return request.form.get(name)
    if request.args.get(name) is not None:
        return request.args.get(name)
    payload = request.get_json(silent=True)
    if isinstance(payload, dict) and name in payload:
        return payload[name]
    return default


# --------------------------------------------------------------------------- #
# Sorgente della richiesta: file caricato, documento gia' in memoria, testo
#
# Le immagini (.png/.jpg/.tiff...) diventano subito un PDF di una pagina per
# fotogramma: da li' in poi seguono la strada dei PDF scansionati (anteprima,
# OCR, redazione sui pixel). Le pagine senza layer di testo passano dall'OCR.
# --------------------------------------------------------------------------- #
class _Source:
    def __init__(self, name, kind, pdf=None, text=None):
        self.name, self.kind, self.pdf, self.text = name, kind, pdf, text


def _source_from_bytes(name, data):
    if _is_pdf(name, data):
        return _Source(name, "pdf", pdf=data)
    if ocr.is_image(name, data):
        try:
            return _Source(name, "image", pdf=ocr.image_to_pdf(data, name))
        except Exception:
            raise _ReqError("Immagine non leggibile o formato non supportato "
                            "(accetto PNG, JPEG, TIFF, BMP, WebP).")
    try:
        return _Source(name, "text", text=_text_from_bytes(name, data))
    except ValueError as e:
        raise _ReqError(str(e) + " Per le scansioni: PDF o immagini (PNG, JPEG, TIFF).")


def _read_source():
    """Il documento della richiesta, oppure None se non c'e' un file (testo in JSON).
    Con "doc_id" si riusa il documento caricato con /preview: niente secondo upload."""
    up = _uploaded_file()
    if up is not None:
        return _source_from_bytes(up.filename, up.read())
    doc_id = _param("doc_id")
    if doc_id:
        d = _get_doc(str(doc_id))
        if d is None or d.get("role") != "source":
            raise _ReqError("Documento non piu' disponibile: ricaricalo.", 410, "doc_gone")
        return _Source(d["orig_name"], d["kind"], pdf=d["pdf"])
    return None


def _ocr_options(src=None):
    mode, preset, _ = _get_ocr_prefs()
    mode = server_config.parse_ocr_mode(_param("ocr"), mode)
    preset = server_config.parse_ocr_preset(_param("ocr_preset"), preset)
    if src is not None and src.kind == "image" and mode == "off":
        mode = "auto"          # un'immagine senza OCR non ha testo: "off" non ha senso
    return mode, preset


def _check_pdf(pdf):
    try:
        with fitz.open(stream=pdf, filetype="pdf") as doc:
            locked = doc.needs_pass
    except Exception as e:
        raise _ReqError(f"Impossibile leggere il file: {e}")
    if locked:
        raise _ReqError("Impossibile leggere il file: PDF protetto da password: "
                        "rimuovi la protezione e riprova.", 400, "pdf_locked")


def _extract(src, mode, preset):
    """(testo, risultato OCR) di una sorgente PDF/immagine. Il testo e' quello di
    sempre (pagine + annotazioni + campi modulo + segnalibri) con il testo OCR al
    posto del layer vuoto delle pagine scansionate."""
    _check_pdf(src.pdf)
    try:
        res = ocr.ocr_pdf(src.pdf, mode, preset)
    except FileNotFoundError:
        res = {"pages": {}, "ocr_pages": [], "available": False, "mode": mode,
               "preset": preset, "elapsed": 0}
    page_text = {i: r["text"] for i, r in res["pages"].items()}
    with fitz.open(stream=src.pdf, filetype="pdf") as doc:
        text = pdf_text.collect_readable_text(doc, page_text=page_text)
    return text, res


def _load_text():
    """(testo, sorgente | None, risultato OCR | None) della richiesta."""
    try:
        src = _read_source()
        if src is None:
            payload = request.get_json(silent=True) or {}
            return payload.get("text", "") or "", None, None
        if src.kind == "text":
            return src.text, src, None
        text, res = _extract(src, *_ocr_options(src))
        return text, src, res
    except _ReqError:
        raise
    except PdfLocked as e:
        raise _ReqError(f"Impossibile leggere il file: {e}", 400, "pdf_locked")
    except ValueError as e:
        raise _ReqError(str(e))
    except Exception as e:                           # PDF corrotto
        raise _ReqError(f"Impossibile leggere il file: {e}")


def _no_text_error(res, what="analizzare"):
    if res and res.get("ocr_pages") and not res.get("available"):
        return _ReqError("Il documento e' una scansione, ma i modelli OCR non sono "
                         "installati: senza OCR non c'e' testo da " + what + ".",
                         422, "ocr_missing")
    if res and res.get("ocr_pages"):
        return _ReqError("L'OCR non ha trovato testo leggibile nel documento.", 422, "ocr_empty")
    return _ReqError(f"Nessun testo da {what}.")


def _ocr_info(res):
    if not res:
        return None
    return {"pages": [i + 1 for i in res.get("ocr_pages", [])],
            "preset": res.get("preset"), "mode": res.get("mode"),
            "elapsed": res.get("elapsed"), "available": res.get("available")}


@app.route("/analyze", methods=["POST"])
def analyze_route():
    try:
        text, src, ocr_res = _load_text()
        text = (text or "").strip()
        if not text:
            raise _no_text_error(ocr_res)
    except _ReqError as e:
        return e.response()
    raw_excl = _param("exclude_tags")
    raw_map = _param("include_mapping")

    # override per-richiesta; senza override vale la configurazione del server
    # (snapshot atomico: una POST /settings concorrente non deve far leggere
    # un tag escluso aggiornato insieme a un mapping_enabled non ancora aggiornato, o viceversa)
    default_excl, default_map = _get_prefs()
    excl = server_config.parse_tag_list(raw_excl) if raw_excl is not None else default_excl
    keep_map = (server_config.parse_bool(raw_map, default_map)
                if raw_map is not None else default_map)
    out = analyze(text, excl, keep_map)
    out["ocr"] = _ocr_info(ocr_res)
    out["source_kind"] = src.kind if src is not None else "text"
    # Il testo originale torna solo col dizionario attivo (l'UI lo rimette nella casella
    # dopo un upload). In modalita' definitiva la risposta non deve contenere nessun
    # valore in chiaro: la chiave manca del tutto, come il valore nei segmenti (#118).
    if keep_map:
        out["source_text"] = text
    return jsonify(out)


# --------------------------------------------------------------------------- #
# PDF anonimizzato da scaricare (issue #7, punto 1)
# --------------------------------------------------------------------------- #
def _pdf_response(data, report, redactions, filename="documento_anonimizzato.pdf"):
    """Risposta binaria + intestazioni diagnostiche. X-PII-Residual e
    X-PII-Skipped sono le due che l'UI trasforma in AVVISO: valori del documento
    che, per motivi diversi, sono rimasti in chiaro."""
    resp = app.response_class(data, mimetype="application/pdf")
    resp.headers["Content-Disposition"] = f"attachment; filename={filename}"
    resp.headers["X-PII-Redactions"] = str(redactions)
    resp.headers["X-PII-Residual"] = str(len(report.get("residual", [])))
    resp.headers["X-PII-Skipped"] = str(len(report.get("skipped", [])))
    resp.headers["X-PII-Notfound"] = str(len(report.get("not_found", [])))
    resp.headers["X-PII-OCR-Pages"] = ",".join(str(i + 1) for i in report.get("ocr_pages", []))
    return resp


def _build_anonymized_pdf():
    """Costruisce il PDF ANONIMIZZATO dall'input della richiesta. Stesso input di
    /analyze:

    - multipart con un file .pdf  -> REDAZIONE VERA del documento originale: le
      PII sono rimosse dal content stream e sostituite dai placeholder, il layout
      resta quello di partenza (piu' metadati/annotazioni/campi modulo/segnalibri
      ripuliti e allegati rimossi);
    - multipart con .md/.txt oppure {"text": ...} -> PDF ricostruito impaginando
      da zero il solo testo anonimizzato.

    Il dizionario placeholder->valore NON viaggia sulla rete in nessuna delle due
    direzioni: viene ricostruito qui, serve solo a localizzare le PII nel PDF e
    muore con la richiesta. Per questo funziona identico anche con il dizionario
    reversibile disattivato (MAPPING_ENABLED=False), dove il client non ne ha
    nessuno.

    Le pagine scansionate (lette con l'OCR) vengono coperte sui pixel: escono
    come immagine con le PII gia' cancellate e i placeholder scritti sopra.

    Ritorna (bytes, report, n_redazioni, nome_file). Solleva _ReqError sugli errori.
    """
    text, src, ocr_res = _load_text()
    raw_excl = _param("exclude_tags")
    name = src.name if src is not None else ""

    text = (text or "").strip()
    if not text:
        raise _no_text_error(ocr_res, "anonimizzare")

    stem = os.path.splitext(_safe_name(name, "documento.pdf"))[0] or "documento"
    out_name = f"{stem}_anonimizzato.pdf"

    excl = (server_config.parse_tag_list(raw_excl) if raw_excl is not None
            else _get_prefs()[0])
    # mapping_enabled=True e' interno: il risultato non esce da questa funzione.
    res = analyze(text, excl, mapping_enabled=True)
    if not res["mapping"]:
        raise _ReqError("Nessuna PII trovata: non c'e' niente da "
                        "anonimizzare in questo documento.", 422)

    if src is not None and src.pdf is not None:
        seals = _param("ocr_redact_seals")
        seals = (server_config.parse_bool(seals, True) if seals is not None
                 else _get_ocr_prefs()[2])
        try:
            out, report = pdf_export.redact_pdf(
                src.pdf, res["mapping"], ocr_pages=(ocr_res or {}).get("pages"),
                redact_seals=seals)
        except pdf_export.PdfError as e:
            raise _ReqError(str(e))
        if report["occurrences"] == 0:
            hint = ("" if (ocr_res or {}).get("ocr_pages") else
                    " Se il documento e' una scansione, attiva l'OCR (modalita' "
                    "\"tutte le pagine\") e riprova.")
            raise _ReqError("Nessuna occorrenza trovata nel PDF: le PII trovate nel "
                            "testo non si localizzano nelle pagine." + hint, 422)
        return out, report, report["occurrences"], out_name

    try:
        out = pdf_export.text_to_pdf(res["anonymized_text"])
    except pdf_export.PdfError as e:
        raise _ReqError(str(e))
    # testo reimpaginato da zero: nell'output c'e' solo il testo anonimizzato,
    # quindi niente residui e niente valori saltati da segnalare.
    return out, {}, res["n_entities"], out_name


@app.route("/pdf", methods=["POST"])
def pdf_route():
    """Scarica il documento ANONIMIZZATO in PDF (vedi _build_anonymized_pdf)."""
    try:
        out, report, redactions, name = _build_anonymized_pdf()
    except _ReqError as e:
        return e.response()
    return _pdf_response(out, report, redactions, name)


@app.route("/pdf/preview", methods=["POST"])
def pdf_preview_route():
    """Come /pdf, ma il binario resta in memoria e si guarda a video: la risposta
    e' il descrittore del documento ({doc_id, n_pages, ...}), le pagine si prendono
    poi da /doc/<id>/page/<n>.png e il file da /doc/<id>/file.pdf.

    Serve a non anonimizzare due volte: l'anteprima a destra e il download del PDF
    usano lo STESSO documento, generato una volta sola."""
    try:
        out, report, redactions, name = _build_anonymized_pdf()
    except _ReqError as e:
        return e.response()
    doc_id, n_pages = _store_doc(out, name, role="output")
    return jsonify({
        "doc_id": doc_id,
        "n_pages": n_pages,
        "page_sizes": (_get_doc(doc_id) or {}).get("sizes"),
        "filename": name,
        "redactions": redactions,
        "residual": len(report.get("residual", [])),
        "skipped": len(report.get("skipped", [])),
        "not_found": len(report.get("not_found", [])),
        "ocr_pages": [i + 1 for i in report.get("ocr_pages", [])],
        "seals": report.get("seals", 0),
    })


# --------------------------------------------------------------------------- #
# Anteprima a video del documento CARICATO + pagine renderizzate
# --------------------------------------------------------------------------- #
@app.route("/preview", methods=["POST"])
def preview_route():
    """Tiene in memoria il PDF (o l'immagine, convertita in PDF) appena caricato per
    mostrarlo a sinistra e ne ritorna il testo nativo. NON esegue l'OCR, che costa
    secondi: dice quali pagine ne hanno bisogno ("ocr.needed") e l'UI lo lancia con
    /ocr?doc_id=...&stream=1 mostrando l'avanzamento. Se l'OCR di quel file e' gia'
    in cache, il testo lo comprende gia'."""
    up = _uploaded_file()
    if up is None:
        return jsonify({"error": "Nessun file caricato."}), 400
    name, data = up.filename, up.read()
    try:
        src = _source_from_bytes(name, data)
        if src.kind == "text":
            raise _ReqError("L'anteprima renderizzata vale solo per PDF e immagini.")
        _check_pdf(src.pdf)
        mode, preset = _ocr_options(src)
        with fitz.open(stream=src.pdf, filetype="pdf") as doc:
            needed = ocr.pages_to_ocr(doc, mode)
            done = {i: r for i, r in ocr.cached_pages(src.pdf, preset).items() if i in needed}
            text = pdf_text.collect_readable_text(
                doc, page_text={i: r["text"] for i, r in done.items()})
        doc_id, n_pages = _store_doc(src.pdf, name, kind=src.kind, role="source")
    except _ReqError as e:
        return e.response()
    except Exception as e:                           # PDF corrotto
        return jsonify({"error": f"Impossibile leggere il file: {e}"}), 400
    return jsonify({"doc_id": doc_id, "n_pages": n_pages, "kind": src.kind,
                    "page_sizes": (_get_doc(doc_id) or {}).get("sizes"),
                    "filename": _safe_name(name), "text": text,
                    "ocr": {"needed": [i + 1 for i in needed],
                            "done": sorted(i + 1 for i in done),
                            "available": bool(ocr.available_presets()),
                            "mode": mode, "preset": preset}})


def _parse_pages(spec):
    """"1-3,5" -> [0, 1, 2, 4] (0-based). None/"" -> None (tutte)."""
    if spec is None or str(spec).strip() == "":
        return None
    out = []
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            out.extend(range(int(a) - 1, int(b)))
        else:
            out.append(int(part) - 1)
    return sorted({p for p in out if p >= 0})


@app.route("/ocr", methods=["POST"])
def ocr_route():
    """OCR locale di un PDF o di un'immagine.

    Input: multipart con "file" (PDF, PNG, JPEG, TIFF, BMP, WebP) oppure "doc_id".
    Parametri (form, query o JSON):
      mode    auto (default: solo le pagine senza testo) | force (tutte) | off
      preset  fast | balanced (default) | accurate
      pages   es. "1-3,5" (default tutte)
      lines   1 -> per ogni pagina OCR anche le righe con box e box per carattere
              (in punti PDF, origine in alto a sinistra); "boxes" -> solo le righe
      stream  1 -> risposta NDJSON: {"event":"start"}, un {"event":"page"} per
              pagina letta, {"event":"done", ...risultato} (o {"event":"error"})
    Risultato: {"text" (Markdown di tutto il documento), "pages": [{"page",
    "source": "ocr"|"text", "text", ...}], "ocr_pages", "n_pages", "elapsed"}.
    Il risultato resta in cache: /analyze e /pdf sullo stesso file non rileggono."""
    try:
        src = _read_source()
        if src is None:
            raise _ReqError("Carica un PDF o un'immagine (campo \"file\") oppure passa doc_id.")
        if src.kind == "text":
            raise _ReqError("L'OCR vale per PDF e immagini, non per file di testo.")
        _check_pdf(src.pdf)
        mode = server_config.parse_ocr_mode(_param("mode", _param("ocr")), "auto")
        if src.kind == "image" and mode == "off":
            mode = "auto"
        preset = server_config.parse_ocr_preset(_param("preset", _param("ocr_preset")),
                                                _get_ocr_prefs()[1])
        try:
            pages = _parse_pages(_param("pages"))
        except ValueError:
            raise _ReqError("Parametro pages non valido: usa ad esempio 1-3,5.")
        lines_opt = str(_param("lines") or "").strip().lower()
        want_boxes = lines_opt == "boxes"      # solo righe e box, senza i box per carattere
        want_lines = not want_boxes and server_config.parse_bool(lines_opt or None, False)
        stream = server_config.parse_bool(_param("stream"), False)
        if not ocr.available_presets():
            raise _ReqError("Modelli OCR non installati (vedi scripts/fetch_ocr_models.py).",
                            503, "ocr_missing")
    except _ReqError as e:
        return e.response()

    pdf, name = src.pdf, src.name

    def result(res):
        pages_out = []
        for p in ocr.page_texts(pdf, res):
            if pages is not None and p["page"] - 1 not in pages:
                continue
            r = res["pages"].get(p["page"] - 1)
            if r is not None:
                p.update(dpi=r["dpi"], angle=r["angle"], timings=r["timings"],
                         size=r["size"])
                if want_lines:
                    p.update(lines=r["lines"], seals=r["seals"])
                elif want_boxes:
                    p.update(lines=[{"text": ln["text"], "box": ln["box"]} for ln in r["lines"]],
                             seals=r["seals"])
            pages_out.append(p)
        return {"filename": _safe_name(name), "n_pages": res["n_pages"],
                "ocr_pages": [i + 1 for i in res["ocr_pages"]],
                "preset": res["preset"], "mode": res["mode"], "elapsed": res["elapsed"],
                "text": "\n\n".join(p["text"] for p in pages_out if p["text"]),
                "pages": pages_out}

    if not stream:
        try:
            return jsonify(result(ocr.ocr_pdf(pdf, mode, preset, pages)))
        except Exception as e:
            return jsonify({"error": f"OCR non riuscito: {e}"}), 500

    with fitz.open(stream=pdf, filetype="pdf") as doc:
        todo = ocr.pages_to_ocr(doc, mode)
        n_pages = doc.page_count
    if pages is not None:
        todo = [i for i in todo if i in set(pages)]
    events = queue.Queue()

    def work():
        try:
            res = ocr.ocr_pdf(pdf, mode, preset, pages, progress=events.put)
            events.put({"event": "done", **result(res)})
        except Exception as e:                       # noqa: BLE001 - va al client
            events.put({"event": "error", "error": f"OCR non riuscito: {e}"})
        finally:
            events.put(None)

    threading.Thread(target=work, daemon=True).start()

    def gen():
        yield json.dumps({"event": "start", "n_pages": n_pages, "preset": preset,
                          "ocr_pages": [i + 1 for i in todo]}) + "\n"
        while True:
            ev = events.get()
            if ev is None:
                break
            yield json.dumps(ev, ensure_ascii=False) + "\n"

    return Response(gen(), mimetype="application/x-ndjson",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.route("/doc/<doc_id>/page/<int:n>.png")
def doc_page(doc_id, n):
    """Pagina n (0-based) renderizzata in PNG. 404 se il documento e' scaduto
    dalla LRU: l'UI in quel caso rifa' l'upload."""
    d = _get_doc(doc_id)
    if d is None:
        return ("", 404)
    png = _page_png(d, n)
    if png is None:
        return ("", 404)
    resp = app.response_class(png, mimetype="image/png")
    resp.headers["Cache-Control"] = "private, max-age=600"
    return resp


@app.route("/doc/<doc_id>/file.pdf")
def doc_file(doc_id):
    """Download del PDF tenuto in memoria (quello gia' anonimizzato da
    /pdf/preview): non ricalcola niente."""
    d = _get_doc(doc_id)
    if d is None:
        return jsonify({"error": "Documento non piu' disponibile."}), 404
    resp = app.response_class(d["pdf"], mimetype="application/pdf")
    resp.headers["Content-Disposition"] = f"attachment; filename={d['name']}"
    return resp


# --------------------------------------------------------------------------- #
# Impostazioni: legenda dei tag, tag esclusi, dizionario reversibile on/off
# (/tags resta come alias storico dello stesso endpoint)
# --------------------------------------------------------------------------- #
@app.route("/settings", methods=["GET"])
@app.route("/tags", methods=["GET"])
def settings_get():
    excl, mapping = _get_prefs()
    o_mode, o_preset, o_seals = _get_ocr_prefs()
    return jsonify({
        "tags": [{"tag": t, "it": it, "en": en, "example": ex} for t, it, en, ex in TAGS],
        "excluded_tags": excl,
        "mapping_enabled": mapping,
        "ocr_mode": o_mode,
        "ocr_preset": o_preset,
        "ocr_redact_seals": o_seals,
        "ocr": ocr.status(),
        "config_path": str(server_config.prefs_path()),
        "env_override": "PII_EXCLUDE_TAGS" in os.environ or "PII_MAPPING" in os.environ,
    })


@app.route("/settings", methods=["POST"])
@app.route("/tags", methods=["POST"])
def settings_post():
    global EXCLUDED_TAGS, MAPPING_ENABLED, OCR_MODE, OCR_PRESET, OCR_REDACT_SEALS
    data = request.get_json(silent=True) or {}
    tags = None
    if "excluded_tags" in data:
        tags = server_config.parse_tag_list(data["excluded_tags"])
        unknown = [t for t in tags if t not in TAG_NAMES]
        if unknown:
            return jsonify({"error": f"Tag sconosciuti: {', '.join(unknown)}"}), 400
    mapping = data.get("mapping_enabled")
    o_mode, o_preset, o_seals = (data.get("ocr_mode"), data.get("ocr_preset"),
                                 data.get("ocr_redact_seals"))
    if o_mode is not None and str(o_mode).lower() not in server_config.OCR_MODES:
        return jsonify({"error": "ocr_mode: usa auto, force oppure off."}), 400
    if o_preset is not None and str(o_preset).lower() not in server_config.OCR_PRESETS:
        return jsonify({"error": "ocr_preset: usa fast, balanced oppure accurate."}), 400
    if all(v is None for v in (tags, mapping, o_mode, o_preset, o_seals)):
        return jsonify({"error": "Niente da salvare: passa excluded_tags, mapping_enabled "
                                 "o un'opzione ocr_*."}), 400
    saved = server_config.save_prefs(excluded_tags=tags, mapping_enabled=mapping,
                                     ocr_mode=o_mode, ocr_preset=o_preset,
                                     ocr_redact_seals=o_seals)
    with _PREFS_LOCK:
        EXCLUDED_TAGS = saved["excluded_tags"]
        MAPPING_ENABLED = saved["mapping_enabled"]
        OCR_MODE = saved["ocr_mode"]
        OCR_PRESET = saved["ocr_preset"]
        OCR_REDACT_SEALS = saved["ocr_redact_seals"]
    return jsonify({"ok": True, "excluded_tags": EXCLUDED_TAGS,
                    "mapping_enabled": MAPPING_ENABLED, "ocr_mode": OCR_MODE,
                    "ocr_preset": OCR_PRESET, "ocr_redact_seals": OCR_REDACT_SEALS})


# --------------------------------------------------------------------------- #
# Config host/porta (GET = leggi, POST = salva per il prossimo avvio)
# --------------------------------------------------------------------------- #
@app.route("/config", methods=["GET"])
def config_get():
    cfg = server_config.load_config()
    return jsonify({
        "host": cfg.get("host", server_config.DEFAULT_HOST),
        "port": cfg.get("port", server_config.DEFAULT_PORT),
        "config_path": str(server_config.config_path()),
    })


@app.route("/config", methods=["POST"])
def config_post():
    data = request.get_json(silent=True) or {}
    host = str(data.get("host", server_config.DEFAULT_HOST)).strip()
    try:
        port = int(data.get("port", server_config.DEFAULT_PORT))
    except (ValueError, TypeError):
        return jsonify({"error": "Porta non valida."}), 400
    if not (1024 <= port <= 65535):
        return jsonify({"error": "La porta deve essere tra 1024 e 65535."}), 400
    server_config.save_config(host, port)
    return jsonify({"ok": True, "host": host, "port": port})


@app.route("/port-check")
def port_check():
    host = request.args.get("host", server_config.DEFAULT_HOST)
    try:
        port = int(request.args.get("port", server_config.DEFAULT_PORT))
    except (ValueError, TypeError):
        return jsonify({"available": False})
    return jsonify({"available": server_config.port_available(host, port)})


# --------------------------------------------------------------------------- #
# UI (single page): src/app/ui/index.html, servita cosi' com'e' (niente CDN, l'app
# e' offline). In sviluppo il file si rilegge a ogni richiesta: si modifica la UI
# senza riavviare il server (e senza ricaricare il modello).
# --------------------------------------------------------------------------- #
UI_FILE = _resource_path(os.path.join("ui", "index.html"))
_UI_CACHE = None


def _load_ui():
    global _UI_CACHE
    if _UI_CACHE is not None:
        return _UI_CACHE
    with open(UI_FILE, encoding="utf-8") as f:
        html = f.read()
    if getattr(sys, "_MEIPASS", None):          # impacchettata: il file non cambia piu'
        _UI_CACHE = html
    return html


PAGE = _load_ui()   # compatibilita': chi importava app.PAGE trova ancora la pagina


if __name__ == "__main__":
    import argparse as _ap
    _p = _ap.ArgumentParser(description="Rizzo PII — server locale di anonimizzazione.")
    _p.add_argument("--host", default=None, help="indirizzo su cui ascoltare (default da config/env/127.0.0.1)")
    _p.add_argument("--port", type=int, default=None, help="porta su cui ascoltare (default da config/env/5005)")
    _p.add_argument("--exclude-tags", default=None, metavar="TAG,TAG",
                    help="tag PII da NON anonimizzare, es. AGE,GENDER (default da env/prefs.json)")
    _p.add_argument("--no-mapping", action="store_true",
                    help="anonimizzazione definitiva: non costruire il dizionario di ripristino")
    _args = _p.parse_args()

    if _args.exclude_tags is not None:
        EXCLUDED_TAGS = server_config.parse_tag_list(_args.exclude_tags)
        _unknown = [t for t in EXCLUDED_TAGS if t not in TAG_NAMES]
        if _unknown:
            print(f"ERRORE: tag sconosciuti in --exclude-tags: {', '.join(_unknown)}")
            print(f"Tag validi: {', '.join(TAG_NAMES)}")
            sys.exit(2)
    if _args.no_mapping:
        MAPPING_ENABLED = False
    if EXCLUDED_TAGS:
        print(f"Tag esclusi dall'anonimizzazione: {', '.join(EXCLUDED_TAGS)}")
    print("Dizionario reversibile: " +
          ("ATTIVO (si puo' ripristinare)" if MAPPING_ENABLED
           else "DISATTIVO (anonimizzazione definitiva)"))

    _host, _port = server_config.resolve(cli_host=_args.host, cli_port=_args.port)

    if not server_config.port_available(_host, _port):
        print(f"ERRORE: porta {_port} occupata su {_host}")
        sys.exit(server_config.EXIT_PORT_CONFLICT)

    print(f"Server su http://{_host}:{_port}")
    try:
        app.run(host=_host, port=_port, threaded=True)
    except OSError as e:
        print(f"ERRORE bind: {e}")
        sys.exit(server_config.EXIT_PORT_CONFLICT)
