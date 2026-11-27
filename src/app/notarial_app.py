# -*- coding: utf-8 -*-
"""Optional local-only notarile workflow mounted at /notarial.

Shares the standard application's already-loaded recognizer and OCR weights.
Classifies individual occurrences by context, keeps revisions and document data
in isolated in-memory stores, and exports only explicitly reviewed selections.
Preferences and session cookies are scoped to this mode; legacy routes remain
on the standard Flask application.
"""

import bisect
import io
import logging
import json
import os
# ORT 1DS puo' inizializzarsi gia' all'import: l'API di disattivazione da sola
# arriverebbe troppo tardi. Impostare il flag prima di qualunque libreria nativa.
os.environ["ORT_DISABLE_TELEMETRY"] = "1"
import queue
import re
import secrets
import sys
import threading
import time
from collections import OrderedDict
from functools import wraps
from pathlib import Path
from urllib.parse import urlsplit

import fitz  # PyMuPDF
from flask import Flask, Request, Response, g, jsonify, request, send_from_directory

import notarial_ocr as ocr
import pdf_export
import pdf_text
import notarial_server_config as server_config
import contextual
import review
import span_repairs
# Rete REGEX + CHECKSUM: modulo a parte, senza dipendenze dal modello. I nomi
# restano importabili da qui (`app.detect_regex`) per non rompere chi li usa.
from detectors import (DETECTORS, SOFT_REGEX_LABELS, cf_ok,  # noqa: F401
                       complete_time, detect_iban, detect_regex, iban_ok, luhn_ok,
                       piva_ok)
from onnx_ner import OnnxTokenClassifier, find_onnx

# Anche il fallback da sorgente deve restare offline. Nessun download runtime.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["DO_NOT_TRACK"] = "1"


def _resource_path(rel):
    """Percorso risorsa valido sia in sviluppo sia dentro l'exe PyInstaller."""
    base = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base, rel)


# Reuse the recognizer already loaded by the standard app. Importing this optional
# mode never creates another ONNX/PyTorch session or downloads model weights.
import app as standard_app
nlp = standard_app.nlp
MODEL_DIR = standard_app.MODEL_DIR
MODEL_BACKEND = standard_app.MODEL_BACKEND
MODEL_DEVICE = standard_app.MODEL_DEVICE
APP_MODEL_VERSION = standard_app.APP_MODEL_VERSION
APP_VERSION = standard_app.APP_VERSION
ASSETS_DIR = standard_app.ASSETS_DIR
MAX_WORDS = standard_app.MAX_WORDS
OVERLAP = standard_app.OVERLAP


class _MemoryRequest(Request):
    def _get_file_stream(self, total_content_length, content_type, filename=None,
                         content_length=None):
        # Evita lo spooling implicito Werkzeug; MAX_CONTENT_LENGTH limita la RAM.
        return io.BytesIO()


app = Flask(__name__)
app.request_class = _MemoryRequest
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  # 50 MB
SESSION_TOKEN = os.environ.get("PII_SESSION_TOKEN") or secrets.token_urlsafe(32)
class _NotarialAccessFilter(logging.Filter):
    def filter(self, record):
        # Only suppress this mode's document identifiers; keep standard diagnostics.
        return "/notarial" not in record.getMessage()

logging.getLogger("werkzeug").addFilter(_NotarialAccessFilter())
app.logger.disabled = True


@app.before_request
def _local_only():
    host = urlsplit(request.host_url).hostname
    try:
        server_config.loopback_host(host)
        server_config.loopback_host(request.remote_addr or "127.0.0.1")
    except ValueError:
        return jsonify(error="Accesso consentito soltanto in locale."), 403
    origin = request.headers.get("Origin")
    top_level_navigation = (request.path == "/" and request.method == "GET"
                            and request.headers.get("Sec-Fetch-Mode") == "navigate"
                            and request.headers.get("Sec-Fetch-Dest") == "document")
    if ((origin and origin != request.host_url.rstrip("/"))
            or (request.headers.get("Sec-Fetch-Site") == "cross-site" and not top_level_navigation)):
        return jsonify(error="Origine non consentita."), 403
    public = request.path in ("/", "/favicon.ico", "/health", "/healthz") or request.path.startswith("/assets/")
    token = request.headers.get("X-PII-Token", "")
    if request.method == "GET" and not token:
        token = request.cookies.get("pii_session", "")
    if not public and not secrets.compare_digest(token, SESSION_TOKEN):
        return jsonify(error="Sessione locale non autorizzata."), 403
    _expire_documents()
    g.document_generation = _DOCUMENT_GENERATION


@app.after_request
def _private_response(response):
    response.headers["Cache-Control"] = "no-store, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; script-src 'self' 'unsafe-inline'; "
        "style-src 'self' 'unsafe-inline'; img-src 'self' data: blob:; "
        "connect-src 'self'; object-src 'none'; frame-ancestors 'none'; base-uri 'none'")
    return response

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
MAX_DOCS = 2               # sorgente e risultato del documento attivo
SESSION_TTL = 30 * 60
MAX_PREVIEW_BYTES = 32 * 1024 * 1024

_DOCS = OrderedDict()      # doc_id -> {"pdf", "pages": {n: png}, "n_pages", "name", "kind", "role"}
_DOCS_LOCK = threading.Lock()
_ANALYSES = {}
_ANALYSIS_LOCK = threading.RLock()
_CLEANUP_TIMER = None
_DOCUMENT_GENERATION = 0


def _clear_documents():
    global _CLEANUP_TIMER, _DOCUMENT_GENERATION
    with _ANALYSIS_LOCK:
        _DOCUMENT_GENERATION += 1
        _ANALYSES.clear()
        if _CLEANUP_TIMER:
            _CLEANUP_TIMER.cancel()
            _CLEANUP_TIMER = None
    with _DOCS_LOCK:
        _DOCS.clear()
    ocr.clear_cache()


def _expire_documents():
    now = time.monotonic()
    with _ANALYSIS_LOCK:
        for key in list(_ANALYSES):
            if now - _ANALYSES[key]["created"] >= SESSION_TTL:
                _ANALYSES.pop(key)
    with _DOCS_LOCK:
        for key in list(_DOCS):
            if now - _DOCS[key]["created"] >= SESSION_TTL:
                _DOCS.pop(key)


def _arm_cleanup():
    global _CLEANUP_TIMER
    with _ANALYSIS_LOCK:
        if _CLEANUP_TIMER:
            _CLEANUP_TIMER.cancel()
        _CLEANUP_TIMER = threading.Timer(SESSION_TTL, _clear_documents)
        _CLEANUP_TIMER.daemon = True
        _CLEANUP_TIMER.start()


def _serialized_document(function):
    @wraps(function)
    def call(*args, **kwargs):
        with _ANALYSIS_LOCK:
            if g.document_generation != _DOCUMENT_GENERATION:
                return jsonify(error="Documento annullato.", code="doc_gone"), 410
            return function(*args, **kwargs)
    return call


@app.route("/session", methods=["DELETE"])
def close_session():
    _clear_documents()
    return jsonify(ok=True)


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
                         "role": role, "sizes": sizes, "created": time.monotonic()}
        while len(_DOCS) > MAX_DOCS:
            _DOCS.popitem(last=False)
    _arm_cleanup()
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
            while sum(len(p) for p in d["pages"].values()) > MAX_PREVIEW_BYTES and len(d["pages"]) > 1:
                d["pages"].pop(next(iter(d["pages"])))
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
        results = nlp([c for c, _ in chunks])
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
    page = _load_ui().replace("__VERSION__", APP_VERSION)
    return page.replace("</head>", "<script>window.PII_TOKEN=" +
                        json.dumps(SESSION_TOKEN) + ";window.PII_BASE=" +
                        json.dumps(request.script_root) + ";</script></head>")


@app.route("/")
def index():
    response = app.make_response(_page())
    response.set_cookie("pii_session", SESSION_TOKEN, httponly=True,
                        samesite="Strict", path=request.script_root + "/")
    return response


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
        "profile": "notarial-selective",
        "taxonomy_version": contextual.TAXONOMY_VERSION,
        "device": MODEL_DEVICE,
        "backend": MODEL_BACKEND,
        "tags": len(TAG_NAMES),
        "excluded_tags": excl,
        "mapping_enabled": mapping,
        "ocr": {"available": bool(ocr.available_presets()), "mode": _get_ocr_prefs()[0]},
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
        raise _ReqError("Impossibile leggere il file PDF.")
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
    with fitz.open(stream=src.pdf, filetype="pdf") as doc:
        src.document = pdf_text.collect_readable_document(doc, ocr_pages=res["pages"])
    return src.document["text"], res


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
        raise _ReqError("Impossibile leggere il documento.")


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
        if not _param("doc_id"):
            _clear_documents()
            g.document_generation = _DOCUMENT_GENERATION
        _arm_cleanup()
        text, src, ocr_res = _load_text()
        if not isinstance(text, str) or not text.strip():
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
    if server_config.parse_bool(_param("review"), False):
        try:
            state = _new_analysis(text, src, ocr_res, excl, keep_map)
            return jsonify(_analysis_result(state, local_review=True))
        except ValueError:
            return jsonify(error="Analisi del documento non valida."), 422
    out = analyze(text, excl, keep_map)
    if g.document_generation != _DOCUMENT_GENERATION:
        return jsonify(error="Documento annullato.", code="doc_gone"), 410
    out["verification"] = {"status": "analysis_only", "complete_anonymization": False}
    out["ocr"] = _ocr_info(ocr_res)
    out["source_kind"] = src.kind if src is not None else "text"
    # Il testo originale torna solo col dizionario attivo (l'UI lo rimette nella casella
    # dopo un upload). In modalita' definitiva la risposta non deve contenere nessun
    # valore in chiaro: la chiave manca del tutto, come il valore nei segmenti (#118).
    if keep_map:
        out["source_text"] = text
    return jsonify(out)


def _new_analysis(text, src, ocr_res, excluded, mapping_enabled):
    generation = g.document_generation
    model, chunks = detect_model(text)
    complete_time(model, text)
    recognized = _merge(model + detect_regex(text), text)
    entities = contextual.classify(text, span_repairs.normalize(text, recognized))
    entities = review.assign_placeholders(entities)
    if getattr(src, "document", None):
        entities = pdf_export.align_occurrence_aliases(entities, src.document)
    review.validate_spans(text, entities)
    state = {"id": secrets.token_urlsafe(18), "revision": 1,
             "created": time.monotonic(), "text": text, "entities": entities,
             "source": src, "ocr": ocr_res, "excluded": excluded,
             "mapping_enabled": mapping_enabled, "n_chunks": chunks,
             "categories": None, "overrides": {}, "classifications": {}}
    with _ANALYSIS_LOCK:
        if generation != _DOCUMENT_GENERATION:
            raise ValueError("Documento annullato durante l'analisi.")
        _ANALYSES.clear()  # un documento attivo; nessuno storico applicativo
        _ANALYSES[state["id"]] = state
    with _DOCS_LOCK:
        for key in list(_DOCS):
            if (_DOCS[key]["role"] == "output" or src is None or src.pdf is None
                    or _DOCS[key]["pdf"] != src.pdf):
                _DOCS.pop(key)
    if src is None or src.pdf is None:
        ocr.clear_cache()
    _arm_cleanup()
    return state


def _analysis_result(state, local_review=False):
    out, chosen = review.render(
        state["text"], state["entities"], categories=state["categories"],
        overrides=state["overrides"], classifications=state["classifications"],
        excluded=state["excluded"], mapping_enabled=state["mapping_enabled"],
        local_review=local_review, n_chunks=state["n_chunks"])
    src = state["source"]
    document = getattr(src, "document", {})
    warning_labels = {
        "images": "Immagini, fotografie, firme e timbri presenti non sono verificati integralmente.",
        "raster_content": "Contenuti grafici delle scansioni non verificati integralmente dall'OCR.",
        "images_unchecked": "Non è stato possibile verificare la presenza di immagini.",
        "attachments": "Gli allegati incorporati non vengono analizzati e saranno rimossi dal PDF esportato.",
        "attachments_unchecked": "Non è stato possibile verificare gli allegati incorporati.",
    }
    warnings = [warning_labels.get(w.get("code"), "Contenuti non verificati presenti.")
                if isinstance(w, dict) else str(w) for w in document.get("unverified", [])]
    if src is not None and src.pdf is not None:
        if (state["ocr"] or {}).get("ocr_pages") and not (state["ocr"] or {}).get("available"):
            warnings.append("Pagine scansionate non lette: modelli OCR assenti.")
    if not any(e["selected"] for e in chosen):
        warnings.append("Nessuna occorrenza selezionata: il testo resta in chiaro.")
    out.update(analysis_id=state["id"], revision=state["revision"],
               taxonomy=contextual.TAXONOMY, taxonomy_version=contextual.TAXONOMY_VERSION,
               warnings=list(dict.fromkeys(warnings)),
               ocr=_ocr_info(state["ocr"]), source_kind=src.kind if src else "text")
    return out


def _requested_analysis(analysis_id=None):
    key = analysis_id or _param("analysis_id")
    state = _ANALYSES.get(str(key))
    if state is None:
        raise _ReqError("Analisi scaduta: analizza nuovamente il documento.", 410, "analysis_gone")
    revision = _param("revision")
    if type(revision) is not int or revision != state["revision"]:
        raise _ReqError("Il documento o le scelte sono cambiati: aggiorna l'anteprima.",
                        409, "stale_revision")
    return state


def _export_gate(state):
    result = _analysis_result(state)
    aliases = {}
    for entity in result["occurrences"]:
        if entity.get("alias_group"):
            aliases.setdefault(entity["alias_group"], set()).add(entity["selected"])
    if any(len(choices) > 1 for choices in aliases.values()):
        raise _ReqError("Campo e apparenza sono lo stesso dato: uniforma le occorrenze collegate.",
                        422, "linked_occurrences")
    if getattr(state["source"], "document", {}).get("errors"):
        raise _ReqError("Esportazione bloccata: alcune parti del documento non sono leggibili.",
                        422, "verification_failed")
    if result["unresolved"]:
        raise _ReqError("Rivedi le classificazioni ambigue prima di esportare.",
                        422, "review_required")
    if result["warnings"] and not server_config.parse_bool(_param("ack_unverified"), False):
        raise _ReqError("Conferma la presa visione dei contenuti non verificati prima di esportare.",
                        422, "unverified_content")
    return result


@app.route("/review/<analysis_id>", methods=["POST"])
def review_route(analysis_id):
    try:
        with _ANALYSIS_LOCK:
            state = _requested_analysis(analysis_id)
            data = request.get_json(silent=True) or {}
            candidate = dict(state)
            for field in ("categories", "overrides", "classifications"):
                if field in data:
                    candidate[field] = data[field]
            result = _analysis_result(candidate, local_review=True)  # valida prima di mutare
            candidate["revision"] += 1
            result["revision"] = candidate["revision"]
            _ANALYSES[analysis_id] = candidate
            with _DOCS_LOCK:
                for key in list(_DOCS):
                    if _DOCS[key]["role"] == "output":
                        _DOCS.pop(key)
            return jsonify(result)
    except _ReqError as e:
        return e.response()
    except (ValueError, TypeError, KeyError):
        return jsonify(error="Selezione o classificazione non valida."), 400


@app.route("/review/<analysis_id>/text", methods=["POST"])
def reviewed_text(analysis_id):
    try:
        with _ANALYSIS_LOCK:
            result = _export_gate(_requested_analysis(analysis_id))
            return jsonify(anonymized_text=result["anonymized_text"],
                           verification={"status": "selective", "complete_anonymization": False},
                           warnings=result["warnings"])
    except _ReqError as e:
        return e.response()


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
    """Esporta lo stesso snapshot e le stesse decisioni della revisione locale."""
    with _ANALYSIS_LOCK:
        if _param("analysis_id"):
            state = _requested_analysis()
        else:
            text, src, ocr_res = _load_text()
            if not isinstance(text, str) or not text.strip():
                raise _no_text_error(ocr_res, "anonimizzare")
            raw = _param("exclude_tags")
            excluded = server_config.parse_tag_list(raw) if raw is not None else _get_prefs()[0]
            state = _new_analysis(text, src, ocr_res, excluded, False)
        result = _export_gate(state)
        _, chosen = review.render(
            state["text"], state["entities"], categories=state["categories"],
            overrides=state["overrides"], classifications=state["classifications"],
            excluded=state["excluded"])
        selected = [e for e in chosen if e["selected"]]
        src = state["source"]
        try:
            if src is not None and src.pdf is not None:
                out, report = pdf_export.redact_occurrences(
                    src.pdf, selected, src.document,
                    ocr_pages=(state["ocr"] or {}).get("pages"),
                    redact_seals=False)
                if any(report.get(k) for k in ("errors", "residual", "skipped", "not_found")):
                    raise _ReqError("Esportazione bloccata: alcune rimozioni non sono verificabili.",
                                    422, "verification_failed")
                if report.get("unverified") and not server_config.parse_bool(_param("ack_unverified"), False):
                    raise _ReqError("Contenuti non verificati: è necessaria la presa visione.",
                                    422, "unverified_content")
                return out, report, report["occurrences"], "documento_selettivo.pdf"
            out = pdf_export.text_to_pdf(result["anonymized_text"])
            # La conversione legacy NFKC/Latin-1 può perdere caratteri: non esportare
            # silenziosamente un testo diverso da quello revisionato.
            import unicodedata
            normalized = unicodedata.normalize("NFKC", result["anonymized_text"]).translate(pdf_export._TRANSLATE)
            if normalized.encode("latin-1", "replace").decode("latin-1") != normalized:
                raise _ReqError("Il PDF di testo non rappresenta tutti i caratteri: usa il testo copiato.",
                                422, "unsupported_glyphs")
            return out, {}, len(selected), "documento_selettivo.pdf"
        except pdf_export.PdfError:
            raise _ReqError("Impossibile verificare il PDF esportato.", 422, "verification_failed")


@app.route("/pdf", methods=["POST"])
@_serialized_document
def pdf_route():
    """Scarica il documento ANONIMIZZATO in PDF (vedi _build_anonymized_pdf)."""
    try:
        out, report, redactions, name = _build_anonymized_pdf()
    except _ReqError as e:
        return e.response()
    return _pdf_response(out, report, redactions, name)


@app.route("/pdf/preview", methods=["POST"])
@_serialized_document
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
        "verification": "selective",
        "unverified": report.get("unverified", []),
    })


# --------------------------------------------------------------------------- #
# Anteprima a video del documento CARICATO + pagine renderizzate
# --------------------------------------------------------------------------- #
@app.route("/preview", methods=["POST"])
@_serialized_document
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
    _clear_documents()
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
        return jsonify({"error": "Impossibile leggere il file."}), 400
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
        _arm_cleanup()
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
    generation = g.document_generation

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
            res = ocr.ocr_pdf(pdf, mode, preset, pages)
            if generation != _DOCUMENT_GENERATION:
                return jsonify(error="Documento annullato."), 410
            return jsonify(result(res))
        except Exception as e:
            return jsonify({"error": "OCR non riuscito."}), 500

    with fitz.open(stream=pdf, filetype="pdf") as doc:
        todo = ocr.pages_to_ocr(doc, mode)
        n_pages = doc.page_count
    if pages is not None:
        todo = [i for i in todo if i in set(pages)]
    events = queue.Queue()

    def work():
        try:
            res = ocr.ocr_pdf(pdf, mode, preset, pages, progress=events.put)
            if generation == _DOCUMENT_GENERATION:
                events.put({"event": "done", **result(res)})
            else:
                events.put({"event": "error", "error": "Documento annullato."})
        except Exception as e:                       # noqa: BLE001 - va al client
            events.put({"event": "error", "error": "OCR non riuscito."})
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
    if d.get("role") != "output":
        return jsonify(error="Il documento originale non è un risultato esportabile."), 403
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
        "taxonomy": contextual.TAXONOMY,
        "taxonomy_version": contextual.TAXONOMY_VERSION,
        "excluded_tags": excl,
        "mapping_enabled": mapping,
        "ocr_mode": o_mode,
        "ocr_preset": o_preset,
        "ocr_redact_seals": o_seals,
        "ocr": {**ocr.status(), "models_dir": "modelli locali"},
        "config_path": "prefs.json",
        "env_override": "PII_NOTARIAL_EXCLUDE_TAGS" in os.environ or "PII_NOTARIAL_MAPPING" in os.environ,
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
    saved = server_config.save_prefs(excluded_tags=tags,
                                     mapping_enabled=_get_prefs()[1] if mapping is None else mapping,
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
        "config_path": "config.json",
    })


@app.route("/config", methods=["POST"])
def config_post():
    data = request.get_json(silent=True) or {}
    try:
        host = server_config.loopback_host(data.get("host", server_config.DEFAULT_HOST))
        port = int(data.get("port", server_config.DEFAULT_PORT))
    except (ValueError, TypeError):
        return jsonify({"error": "Porta non valida."}), 400
    if not (1024 <= port <= 65535):
        return jsonify({"error": "La porta deve essere tra 1024 e 65535."}), 400
    server_config.save_config(host, port)
    return jsonify({"ok": True, "host": host, "port": port})


@app.route("/port-check")
def port_check():
    try:
        host = server_config.loopback_host(request.args.get("host", server_config.DEFAULT_HOST))
        port = int(request.args.get("port", server_config.DEFAULT_PORT))
    except (ValueError, TypeError):
        return jsonify({"available": False})
    return jsonify({"available": server_config.port_available(host, port)})


# --------------------------------------------------------------------------- #
# UI (single page): src/app/ui/index.html, servita cosi' com'e' (niente CDN, l'app
# e' offline). In sviluppo il file si rilegge a ogni richiesta: si modifica la UI
# senza riavviare il server (e senza ricaricare il modello).
# --------------------------------------------------------------------------- #
UI_FILE = _resource_path(os.path.join("ui", "notarial.html"))
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



def install_notarial_mode(application):
    """Mount an isolated, local-only selective workflow without replacing legacy routes."""
    from werkzeug.middleware.dispatcher import DispatcherMiddleware
    if not getattr(application, "_notarial_mode_installed", False):
        application.wsgi_app = DispatcherMiddleware(application.wsgi_app, {"/notarial": app})
        application._notarial_mode_installed = True
    return application
