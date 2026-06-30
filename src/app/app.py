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
carta di credito, importi, targhe). Le entita' validate matematicamente (IBAN/CF/PIVA/
carta) hanno priorita' sul modello in caso di sovrapposizione.

Avvio:  python app.py   ->   http://127.0.0.1:5005
Configurazione host/porta (precedenza): CLI --host/--port > env PII_HOST/PII_PORT >
  config.json (vedi server_config.py) > default 127.0.0.1:5005
"""

import os
import re
import sys
from pathlib import Path

import fitz  # PyMuPDF
import torch
from flask import (Flask, jsonify, render_template_string, request,
                   send_from_directory)

import server_config
from transformers import pipeline


def _resource_path(rel):
    """Percorso risorsa valido sia in sviluppo sia dentro l'exe PyInstaller."""
    base = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base, rel)


# VERSIONE del modello usata dall'app -> models/rizzo-pii-0.3B-v{APP_MODEL_VERSION}/.
# Metti None per usare AUTOMATICAMENTE l'ultima versione disponibile.
APP_MODEL_VERSION = "1.2.0"

# Dentro l'exe il modello e' impacchettato come "pii_model" (vedi build.spec).
# In sviluppo: pin sopra -> auto-ultima versione -> vecchio non versionato -> legacy.
# Override puntuale a runtime: env PII_MODEL_DIR.
if getattr(sys, "_MEIPASS", None):
    MODEL_DIR = _resource_path("pii_model")
elif os.environ.get("PII_MODEL_DIR"):
    MODEL_DIR = os.environ["PII_MODEL_DIR"]
else:
    import re
    _models = Path(__file__).resolve().parents[2] / "models"
    _pinned = _models / f"rizzo-pii-0.3B-v{APP_MODEL_VERSION}" if APP_MODEL_VERSION else None
    _versioned = [p for p in _models.glob("rizzo-pii-0.3B-v*") if p.is_dir()]
    if _pinned and _pinned.is_dir():
        MODEL_DIR = str(_pinned)
    elif _versioned:
        MODEL_DIR = str(max(_versioned, key=lambda p: tuple(
            int(x) for x in re.search(r"-v([0-9][0-9.]*)$", p.name).group(1).split("."))))
    else:
        _prod = _models / "rizzo-pii-0.3B"
        MODEL_DIR = str(_prod if _prod.exists() else _models / "pii_model_legacy")

ASSETS_DIR = _resource_path("assets")   # mascotte / icone (servite su /assets/<file>)
APP_VERSION = "1.0.0"                    # versione mostrata nell'UI (allineata a tauri.conf.json)
MAX_WORDS = 120      # parole per chunk (~180 subword, sotto i 512 del training)
OVERLAP = 20         # parole di sovrapposizione tra chunk consecutivi

# --------------------------------------------------------------------------- #
# Caricamento modello (una sola volta all'avvio)
# --------------------------------------------------------------------------- #
device = 0 if torch.cuda.is_available() else -1
print(f"Carico il modello da {MODEL_DIR} su {'GPU' if device == 0 else 'CPU'}...")
nlp = pipeline(
    "token-classification",
    model=MODEL_DIR,
    tokenizer=MODEL_DIR,
    aggregation_strategy="simple",
    device=device,
)
print("Modello pronto.")

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  # 50 MB


# --------------------------------------------------------------------------- #
# Rete REGEX + CHECKSUM (affianca il modello)
# --------------------------------------------------------------------------- #
def iban_ok(s):
    s = re.sub(r"\s", "", s).upper()
    if not (15 <= len(s) <= 34):
        return False
    r = s[4:] + s[:4]
    try:
        n = int("".join(str(ord(c) - 55) if c.isalpha() else c for c in r))
    except ValueError:
        return False
    return n % 97 == 1


def piva_ok(p):
    p = re.sub(r"\D", "", p)
    if len(p) != 11:
        return False
    t = 0
    for i, c in enumerate(map(int, p[:10])):
        if i % 2 == 0:
            t += c
        else:
            x = c * 2
            t += x - 9 if x > 9 else x
    return (10 - t % 10) % 10 == int(p[10])


_CF_ODD = {"0": 1, "1": 0, "2": 5, "3": 7, "4": 9, "5": 13, "6": 15, "7": 17, "8": 19,
           "9": 21, "A": 1, "B": 0, "C": 5, "D": 7, "E": 9, "F": 13, "G": 15, "H": 17,
           "I": 19, "J": 21, "K": 2, "L": 4, "M": 18, "N": 20, "O": 11, "P": 3, "Q": 6,
           "R": 8, "S": 12, "T": 14, "U": 16, "V": 10, "W": 22, "X": 25, "Y": 24, "Z": 23}


def cf_ok(c):
    c = c.strip().upper()
    if len(c) != 16 or not c.isalnum():
        return False
    b = c[:15]
    try:
        t = sum((_CF_ODD[ch] if i % 2 == 0
                 else (int(ch) if ch.isdigit() else ord(ch) - 65))
                for i, ch in enumerate(b))
    except KeyError:
        return False
    return chr(65 + t % 26) == c[15]


def luhn_ok(s):
    d = re.sub(r"\D", "", s)
    if not (13 <= len(d) <= 19):
        return False
    tot, alt = 0, False
    for ch in reversed(d):
        n = int(ch)
        if alt:
            n *= 2
            if n > 9:
                n -= 9
        tot += n
        alt = not alt
    return tot % 10 == 0


# Ogni detector: (label, regex, validatore-o-None, strict).
#   validatore None  -> match accettato sulla sola forma (validated=False).
#   strict=True      -> il match si scarta se il checksum FALLISCE (forma troppo generica:
#                       IBAN/PIVA/carta -> servono i numeri giusti per non avere falsi positivi).
#   strict=False     -> si redige comunque (forma molto specifica, es. CF: meglio nascondere);
#                       validated=True solo se il checksum passa (mette il ✓).
DETECTORS = [
    ("EMAIL",
     re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}"),
     None, True),
    ("CF",
     re.compile(r"\b[A-Za-z]{6}\d{2}[A-Za-z]\d{2}[A-Za-z]\d{3}[A-Za-z]\b"),
     cf_ok, False),
    ("IBAN",
     re.compile(r"\b[A-Za-z]{2}\d{2}[A-Za-z0-9]{11,30}\b"),
     iban_ok, True),
    ("CREDITCARDNUMBER",
     re.compile(r"(?<!\d)(?:\d[ \-]?){13,19}(?!\d)"),
     luhn_ok, True),
    ("PIVA",
     re.compile(r"(?<!\d)\d{11}(?!\d)"),
     piva_ok, True),
    ("TELEPHONENUM",
     re.compile(r"(?<![\w.])(?:\+39[\s.]?)?(?:3\d{2}[\s.]?\d{3}[\s.]?\d{3,4}"
                r"|0\d{1,3}[\s.]?\d{5,8})(?![\w])"),
     None, True),
    ("AMOUNT",
     re.compile(r"(?:€|EUR|euro)\s?\d{1,3}(?:[.\s]\d{3})*(?:,\d{2})?"
                r"|\d{1,3}(?:\.\d{3})*,\d{2}\s?(?:€|EUR|euro)", re.IGNORECASE),
     None, True),
    ("TARGA",
     re.compile(r"\b[A-Za-z]{2}\s?\d{3}\s?[A-Za-z]{2}\b"),
     None, True),
]


def detect_regex(text):
    """Entita' della rete regex. validated=True solo quando il checksum passa."""
    ents = []
    for label, rx, validator, strict in DETECTORS:
        for m in rx.finditer(text):
            ok = validator(m.group(0)) if validator else False
            if validator and strict and not ok:
                continue
            ents.append({
                "label": label,
                "start": m.start(),
                "end": m.end(),
                "score": 1.0 if ok else 0.9,
                "validated": ok,
                "source": "regex",
            })
    return ents


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
def _merge(cands, text):
    """Greedy senza overlap. Priorita': checksum-valido > fonte regex > score > lunghezza.
    La rete regex copre campi a forma molto specifica: per quegli span e' piu' affidabile
    del modello (evita la frammentazione di CF/IBAN/carta in piu' pezzi)."""
    order = sorted(
        cands,
        key=lambda e: (1 if e["validated"] else 0,
                       1 if e["source"] == "regex" else 0,
                       e["score"], e["end"] - e["start"]),
        reverse=True,
    )
    kept = []
    for e in order:
        if all(e["end"] <= k["start"] or e["start"] >= k["end"] for k in kept):
            kept.append(e)
    # niente spazi inglobati nei placeholder (il modello a volte include lo spazio iniziale)
    for e in kept:
        while e["start"] < e["end"] and text[e["start"]].isspace():
            e["start"] += 1
        while e["end"] > e["start"] and text[e["end"] - 1].isspace():
            e["end"] -= 1
    kept = [e for e in kept if e["end"] > e["start"]]
    kept.sort(key=lambda e: e["start"])
    return kept


def _norm(s):
    return re.sub(r"\s+", " ", s.strip()).casefold()


def analyze(text):
    model_ents, n_chunks = detect_model(text)
    cands = model_ents + detect_regex(text)
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
            mapping[ph] = val
            e["ph"] = ph

    # segmenti per la preview + testo anonimizzato + statistiche
    segments, anon, by_label, by_source, pos = [], [], {}, {}, 0
    for e in kept:
        if e["start"] > pos:
            segments.append({"t": text[pos:e["start"]]})
            anon.append(text[pos:e["start"]])
        segments.append({
            "t": text[e["start"]:e["end"]],
            "label": e["label"],
            "ph": e["ph"],
            "src": e["source"],
            "validated": e["validated"],
        })
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
        "n_chunks": n_chunks,
        "n_chars": len(text),
        "n_entities": len(kept),
        "n_unique": len(mapping),
        "by_label": dict(sorted(by_label.items(), key=lambda x: -x[1])),
        "by_source": by_source,
    }


# --------------------------------------------------------------------------- #
# Endpoints
# --------------------------------------------------------------------------- #
def _page():
    return PAGE.replace("__VERSION__", APP_VERSION)


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


@app.route("/analyze", methods=["POST"])
def analyze_route():
    text = ""
    if "pdf" in request.files and request.files["pdf"].filename:
        data = request.files["pdf"].read()
        with fitz.open(stream=data, filetype="pdf") as doc:
            text = "\n".join(page.get_text() for page in doc)
    else:
        text = (request.get_json(silent=True) or {}).get("text", "")
    text = text.strip()
    if not text:
        return jsonify({"error": "Nessun testo da analizzare."}), 400
    out = analyze(text)
    out["source_text"] = text
    return jsonify(out)


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
# UI (single page)
# --------------------------------------------------------------------------- #
PAGE = r"""
<!doctype html>
<html lang="it">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<link rel="icon" href="/assets/mascot_shield.png">
<title>Rizzo PII · locale</title>
<style>
  :root{
    --bg:#f5f4f8; --card:#ffffff; --ink:#211f29; --muted:#6c6677; --soft:#9d97a9;
    --line:#e9e6f1; --line2:#f2f0f7; --brand:#7c3a9e; --brand-dk:#643183;
    --ok:#1d8a4e; --shadow:0 1px 2px rgba(33,26,48,.05),0 10px 28px rgba(33,26,48,.05);
    --r:14px;
  }
  *{box-sizing:border-box}
  html,body{height:100%}
  body{margin:0;background:var(--bg);color:var(--ink);overflow-x:hidden;overflow-y:auto;
       font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
       font-size:15px;line-height:1.55;-webkit-font-smoothing:antialiased}
  /* desktop: altezza = viewport, lo scroll avviene DENTRO i componenti (no effetto zoom-out);
     su schermi stretti (media query sotto) si sblocca e la pagina scrolla, colonne impilate */
  .app{max-width:1240px;margin:0 auto;padding:14px 22px 10px;height:100vh;
       display:flex;flex-direction:column;overflow:hidden}

  /* header */
  header{display:flex;align-items:center;gap:14px;margin-bottom:14px;flex-wrap:wrap;flex:none}
  .logo{width:46px;height:46px;display:grid;place-items:center;font-size:24px}
  .logo img{width:100%;height:100%;object-fit:contain;
            filter:drop-shadow(0 2px 4px rgba(20,28,46,.18))}
  .empty img{width:128px;height:auto;opacity:.96;margin-bottom:2px}
  header h1{font-size:19px;margin:0;font-weight:700;letter-spacing:-.01em}
  header h1 .ver{font-size:11.5px;font-weight:600;color:var(--soft);vertical-align:middle;margin-left:4px}
  header .tag{font-size:12.5px;color:var(--muted)}
  .badge{margin-left:auto;display:inline-flex;align-items:center;gap:7px;background:#eaf7ef;
         color:var(--ok);border:1px solid #cde8d8;border-radius:999px;padding:6px 13px;
         font-size:12.5px;font-weight:600}
  .badge .dot{width:7px;height:7px;border-radius:50%;background:var(--ok)}
  .lang{margin-left:8px;background:#fff;border:1px solid var(--line);border-radius:10px;
        padding:6px 10px;font:inherit;font-size:12.5px;font-weight:600;color:var(--ink);cursor:pointer}
  .lang:hover{border-color:#cfd5e0}

  /* stepper / tabs */
  .tabs{display:flex;gap:8px;margin-bottom:14px;flex:none}
  .tab{flex:0 0 auto;display:flex;align-items:center;gap:10px;background:var(--card);
       border:1px solid var(--line);border-radius:12px;padding:11px 16px;cursor:pointer;
       color:var(--muted);font-weight:600;font-size:14px;transition:.15s;user-select:none}
  .tab:hover{border-color:#d4d9e4}
  .tab.on{color:var(--ink);border-color:var(--brand);box-shadow:0 0 0 3px rgba(124,58,158,.13)}
  .tab .num{width:22px;height:22px;border-radius:50%;display:grid;place-items:center;
            background:var(--line);color:var(--muted);font-size:12.5px;font-weight:700}
  .tab.on .num{background:var(--brand);color:#fff}
  .tab .arrow{color:var(--soft)}

  /* grid (pane "Ripristina") + workspace (pane "Anonimizza") */
  .grid{display:grid;grid-template-columns:1fr 1fr;grid-template-rows:1fr;gap:16px;flex:1;min-height:0}
  .workspace{display:grid;grid-template-columns:1fr 1fr;grid-template-rows:1fr;gap:16px;
             align-items:stretch;flex:1;min-height:0}
  .card{background:var(--card);border:1px solid var(--line);border-radius:var(--r);
        box-shadow:var(--shadow);display:flex;flex-direction:column;overflow:hidden;min-height:0}
  .card .hd{padding:13px 16px;border-bottom:1px solid var(--line2);display:flex;
            align-items:center;gap:10px}
  .card .hd h2{font-size:13px;margin:0;text-transform:uppercase;letter-spacing:.04em;
               color:var(--muted);font-weight:700}
  .card .hd .right{margin-left:auto;display:flex;gap:8px;align-items:center}
  .card .bd{padding:14px 16px;flex:1;min-height:0;display:flex;flex-direction:column}

  textarea{width:100%;flex:1;min-height:0;resize:none;border:1px solid var(--line);
           border-radius:10px;padding:13px 14px;font-size:14.5px;line-height:1.6;color:var(--ink);
           background:#fcfcfe;font-family:inherit}
  textarea:focus{outline:none;border-color:var(--brand);box-shadow:0 0 0 3px rgba(124,58,158,.13)}
  textarea.mono{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-size:13.5px}

  /* dropzone */
  .drop{border:1.5px dashed var(--line);border-radius:10px;padding:11px 14px;margin-top:11px;
        display:flex;align-items:center;gap:11px;color:var(--muted);font-size:13.5px;cursor:pointer;
        transition:.15s;background:#fcfcfe}
  .drop:hover,.drop.hot{border-color:var(--brand);background:#f8f4fc;color:var(--ink)}
  .drop .ic{font-size:18px}
  .drop b{color:var(--ink)}

  /* buttons */
  .row{display:flex;gap:9px;align-items:center;flex-wrap:wrap;margin-top:12px}
  button{font:inherit;border:0;border-radius:10px;padding:10px 17px;font-weight:600;font-size:14px;
         cursor:pointer;display:inline-flex;align-items:center;gap:8px;transition:.15s}
  .btn{background:var(--brand);color:#fff;box-shadow:0 1px 2px rgba(124,58,158,.22)}
  .btn:hover{background:var(--brand-dk)}
  .btn.lg{padding:12px 22px;font-size:15px}
  .ghost{background:#fff;color:var(--ink);border:1px solid var(--line)}
  .ghost:hover{border-color:#cfd5e0;background:#fafbfd}
  button:disabled{opacity:.55;cursor:default}
  .spin{width:15px;height:15px;border:2px solid rgba(255,255,255,.45);border-top-color:#fff;
        border-radius:50%;animation:sp .7s linear infinite}
  @keyframes sp{to{transform:rotate(360deg)}}
  .hint{color:var(--soft);font-size:12.5px;margin-left:auto}

  /* preview */
  .seg-tabs{display:inline-flex;background:var(--line2);border-radius:9px;padding:3px}
  .seg-tabs button{background:transparent;color:var(--muted);padding:5px 12px;font-size:13px;
                   border-radius:7px;box-shadow:none}
  .seg-tabs button.on{background:#fff;color:var(--ink);box-shadow:var(--shadow)}
  .view{flex:1;overflow:auto;border:1px solid var(--line);border-radius:10px;
        padding:14px;background:#fcfcfe;min-height:0}
  /* editor (sx) e anteprima/testo (dx): riempiono la card -> stessa altezza, scroll sincronizzato */
  #src,#anon{flex:1;min-height:0}
  #pane2 textarea{flex:1;min-height:0}

  /* CON RISULTATO: la pagina scrolla e i componenti hanno altezza fissa generosa (no schiacciamento).
     Il dizionario va sotto e si raggiunge scrollando la pagina; editor/anteprima scrollano internamente. */
  .app.has-result{height:auto;overflow:visible}
  .app.has-result .workspace{flex:none}
  .app.has-result #src,.app.has-result #anon,.app.has-result .view{flex:none;height:60vh}
  /* pane "Ripristina": non deve collassare quando l'app e' in modalita' scroll-pagina */
  #pane2 textarea,#pane2 .view{min-height:60vh}

  /* schermi stretti: la pagina scrolla, una colonna, card impilate con altezze fisse leggibili.
     DEVE stare dopo le regole flex:1 qui sopra per vincere a parita' di specificita'. */
  @media(max-width:920px){
    .grid{grid-template-columns:1fr}
    .app{height:auto;overflow:visible}
    .workspace{grid-template-columns:1fr;grid-template-rows:none;flex:none;min-height:auto}
    #src,#anon,.view,#pane2 textarea{flex:none;height:60vh}
  }
  .preview{white-space:pre-wrap;word-wrap:break-word;font-size:14.5px;line-height:1.7}
  .ph{border-radius:6px;padding:1px 7px 2px;font-weight:600;font-size:12.5px;cursor:help;
      border:1px solid;white-space:nowrap;display:inline-block;line-height:1.4;
      transition:.12s}
  .ph .ck{font-size:10px;opacity:.8;margin-left:3px}
  .ph.dim{opacity:.25;filter:grayscale(.6)}
  .empty{color:var(--soft);display:flex;flex-direction:column;align-items:center;justify-content:center;
         height:100%;gap:9px;text-align:center;font-size:14px}
  .empty .big{font-size:34px;opacity:.6}

  /* legend / stats */
  .legend{display:flex;gap:7px;flex-wrap:wrap;padding:12px 16px;border-top:1px solid var(--line2)}
  .chip{display:inline-flex;align-items:center;gap:7px;border:1px solid var(--line);border-radius:999px;
        padding:4px 11px;font-size:12.5px;font-weight:600;color:var(--ink);cursor:pointer;
        background:#fff;user-select:none;transition:.12s}
  .chip:hover{border-color:#cfd5e0}
  .chip.off{opacity:.4;text-decoration:line-through}
  .chip .sw{width:10px;height:10px;border-radius:3px}
  .chip .n{color:var(--muted);font-weight:700}
  .meta{display:flex;gap:8px;flex-wrap:wrap;padding:0 16px 12px}
  .stat{background:#f7f8fb;border:1px solid var(--line2);border-radius:9px;padding:6px 11px;
        font-size:12.5px;color:var(--muted)}
  .stat b{color:var(--ink);font-weight:700}

  /* mapping table */
  .tablewrap{max-height:240px;overflow:auto;padding:0 16px 16px}
  table{width:100%;border-collapse:collapse;font-size:13.5px}
  th{position:sticky;top:0;background:#fff;text-align:left;color:var(--soft);font-weight:600;
     font-size:11.5px;text-transform:uppercase;letter-spacing:.04em;padding:8px 8px;border-bottom:1px solid var(--line)}
  td{padding:8px 8px;border-bottom:1px solid var(--line2);vertical-align:top}
  td.k{font-family:ui-monospace,Consolas,monospace;font-weight:600;white-space:nowrap}
  td.v{word-break:break-word}
  tr:hover td{background:#fafbfd}

  /* card "Dizionario": a tutta larghezza sotto le due colonne, scrollabile */
  .dict{margin-top:16px;flex:none}
  .dict .bd{padding:0;display:block}
  .dict .meta{padding:13px 16px 6px}
  .dict .legend{padding:0 16px 12px;border-top:none}
  .dict .tablewrap{max-height:300px;overflow:auto;padding:0 16px 16px}

  /* reverse panel */
  .pane{display:none}
  .pane.on{display:flex;flex-direction:column;flex:1;min-height:0}
  .callout{display:flex;gap:11px;background:#fff7ed;border:1px solid #fde6c8;border-radius:11px;
           padding:12px 14px;font-size:13.5px;color:#7a4d12;margin-bottom:14px;flex:none}
  .callout b{color:#5c3a0d}
  .callout .ic{font-size:17px}

  /* angolo alto a destra: la lingua resta in linea col badge; l'icona "galleggia" sotto (no shift) */
  .topright{position:relative;margin-left:8px;display:flex;align-items:center}
  .info{position:absolute;top:calc(100% + 6px);right:0;display:grid;place-items:center;
        width:25px;height:25px;border-radius:8px;background:#fff7ed;border:1px solid #fde6c8;
        font-size:13px;line-height:1;cursor:pointer;user-select:none;transition:.12s}
  .info:hover{border-color:#f3c98b}
  .info.open{border-color:#f3c98b;background:#ffedd5}
  .info .tip{position:absolute;top:calc(100% + 8px);right:0;width:300px;
             background:#fff;border:1px solid var(--line);border-radius:11px;box-shadow:var(--shadow);
             padding:12px 14px;font-size:12.5px;font-weight:400;color:#5c4326;line-height:1.55;
             text-align:left;opacity:0;visibility:hidden;transform:translateY(-4px);
             transition:.15s;z-index:60;pointer-events:none}
  .info.open .tip{opacity:1;visibility:visible;transform:translateY(0);pointer-events:auto}
  .info .tip b{color:var(--ink)}
  .info .tip a{color:var(--brand);font-weight:700;text-decoration:none;word-break:break-word}
  .info .tip a:hover{text-decoration:underline}
  .info .tip::before{content:"";position:absolute;top:-5px;right:8px;width:9px;height:9px;background:#fff;
                     border-left:1px solid var(--line);border-top:1px solid var(--line);transform:rotate(45deg)}

  /* crediti (footer dell'app) */
  .credits{flex:none;text-align:center;padding:9px 0 2px;font-size:11.5px;color:var(--soft)}
  .credits b{color:var(--muted);font-weight:700}
  .credits .u{color:var(--brand);font-weight:600}

  /* toast */
  #toast{position:fixed;left:50%;bottom:26px;transform:translateX(-50%) translateY(20px);
         background:#1c2330;color:#fff;padding:11px 18px;border-radius:11px;font-size:13.5px;font-weight:600;
         box-shadow:0 12px 32px rgba(0,0,0,.22);opacity:0;pointer-events:none;transition:.22s;z-index:50;
         display:flex;align-items:center;gap:9px}
  #toast.show{opacity:1;transform:translateX(-50%) translateY(0)}
  #toast.ok::before{content:"✓";color:#6ee7a8}
  .kbd{font-family:ui-monospace,Consolas,monospace;background:#eef1f6;border:1px solid var(--line);
       border-radius:5px;padding:1px 6px;font-size:11.5px;color:var(--muted)}

  /* config modal */
  .cfg-overlay{position:fixed;inset:0;background:rgba(33,26,48,.35);z-index:100;
               display:flex;align-items:center;justify-content:center;opacity:0;
               visibility:hidden;transition:.18s}
  .cfg-overlay.open{opacity:1;visibility:visible}
  .cfg-card{background:#fff;border-radius:16px;box-shadow:0 16px 48px rgba(33,26,48,.18);
            padding:26px 28px 22px;width:380px;max-width:92vw}
  .cfg-card h3{margin:0 0 16px;font-size:16px;font-weight:700;display:flex;align-items:center;gap:9px}
  .cfg-row{display:flex;flex-direction:column;gap:5px;margin-bottom:14px}
  .cfg-row label{font-size:12.5px;font-weight:600;color:var(--muted);text-transform:uppercase;
                 letter-spacing:.04em}
  .cfg-row input{border:1px solid var(--line);border-radius:9px;padding:9px 12px;font:inherit;
                 font-size:14px;color:var(--ink);background:#fcfcfe}
  .cfg-row input:focus{outline:none;border-color:var(--brand);box-shadow:0 0 0 3px rgba(124,58,158,.13)}
  .cfg-status{font-size:12.5px;font-weight:600;padding:7px 11px;border-radius:8px;margin-bottom:14px;
              display:none}
  .cfg-status.ok{display:block;background:#eaf7ef;color:var(--ok)}
  .cfg-status.fail{display:block;background:#fef2f2;color:#b91c1c}
  .cfg-btns{display:flex;gap:9px;align-items:center}
  .cfg-note{font-size:11.5px;color:var(--soft);margin-top:12px}
  .gear{background:none;border:1px solid var(--line);width:30px;height:30px;border-radius:8px;
        display:grid;place-items:center;font-size:15px;padding:0;cursor:pointer;transition:.12s;
        margin-left:6px;flex:none}
  .gear:hover{border-color:#cfd5e0;background:#fafbfd}
</style>
</head>
<body>
<div class="app">
  <header>
    <div class="logo"><img src="/assets/mascot_shield.png" alt="rizzo-pii"
         onerror="this.parentNode.textContent='🦔'"></div>
    <div>
      <h1>Rizzo PII <span class="ver">v__VERSION__</span></h1>
      <div class="tag" data-i18n="tagline">modello locale su CPU · GDPR compliant</div>
    </div>
    <span class="badge"><span class="dot"></span> <span data-i18n="badge">100% in locale</span></span>
    <div class="topright">
      <select id="lang" class="lang" title="Lingua / Language" aria-label="Lingua / Language">
        <option value="it">🇮🇹 Italiano</option>
        <option value="en">🇬🇧 English</option>
      </select>
      <button class="gear" id="gearBtn" title="Configurazione server" aria-label="Server config" onclick="openConfig()">⚙️</button>
      <span class="info" id="infoBtn" tabindex="0" role="button" aria-label="Avviso / info">⚠️<span class="tip" data-i18n="notice"></span></span>
    </div>
  </header>

  <div class="tabs">
    <div class="tab on" id="tab1" onclick="showTab(1)">
      <span class="num">1</span> <span data-i18n="tab1">Anonimizza</span> <span class="arrow">→</span>
    </div>
    <div class="tab" id="tab2" onclick="showTab(2)">
      <span class="num">2</span> <span data-i18n="tab2">Ripristina la risposta</span>
    </div>
  </div>

  <!-- ============ PANE 1: ANONIMIZZA ============ -->
  <div class="pane on" id="pane1">
    <div class="workspace">
      <!-- input -->
      <div class="card">
        <div class="hd"><h2 data-i18n="in_title">① Il tuo documento</h2>
          <div class="right hint" data-i18n="in_hint">incolla testo o trascina un PDF</div></div>
        <div class="bd">
          <textarea id="src" data-i18n-ph="src_ph" placeholder="Incolla qui il testo dell'atto, del contratto o della sentenza…&#10;&#10;Oppure trascina un PDF nell'area qui sotto."></textarea>
          <label class="drop" id="drop">
            <span class="ic">📄</span>
            <span id="dropTxt">Trascina un <b>PDF</b> qui, oppure <b>scegli un file</b></span>
            <input type="file" id="pdf" accept="application/pdf" hidden>
          </label>
          <div class="row">
            <button class="btn lg" id="go">🛡️ <span data-i18n="go">Anonimizza</span></button>
            <button class="ghost" id="clear" data-i18n="clear">Pulisci</button>
            <span class="hint"><span class="kbd">Ctrl</span>+<span class="kbd">Enter</span></span>
          </div>
        </div>
      </div>

      <!-- output -->
      <div class="card out">
        <div class="hd">
          <h2 data-i18n="out_title">② Risultato</h2>
          <div class="right">
            <div class="seg-tabs">
              <button class="on" id="vPrev" onclick="setView('prev')" data-i18n="v_prev">Anteprima</button>
              <button id="vText" onclick="setView('text')" data-i18n="v_text">Testo da copiare</button>
            </div>
          </div>
        </div>
        <div class="bd">
          <div class="view" id="viewPrev">
            <div class="empty" id="emptyPrev">
              <img src="/assets/mascot_doc.png" alt="" onerror="this.replaceWith(Object.assign(document.createElement('div'),{className:'big',textContent:'🕵️'}))">
              <div data-i18n="empty_prev">L'anteprima con le PII evidenziate apparirà qui.</div>
            </div>
            <div class="preview" id="prev" style="display:none"></div>
          </div>
          <textarea class="mono" id="anon" style="display:none" readonly
                    data-i18n-ph="anon_ph" placeholder="Il testo anonimizzato apparirà qui."></textarea>
          <div class="row">
            <button class="btn" id="copy">📋 <span data-i18n="copy">Copia per ChatGPT</span></button>
            <button class="ghost" id="dl">⬇️ <span data-i18n="dl">Scarica dizionario</span></button>
            <span class="hint" id="ulock"></span>
          </div>
        </div>
      </div>
    </div>

    <!-- dizionario: staccato, a tutta larghezza sotto le due colonne, scrollabile -->
    <div class="card dict" id="dictCard" style="display:none">
      <div class="hd">
        <h2 data-i18n="dict_title">Dizionario reversibile</h2>
        <div class="right hint" data-i18n="dict_hint">resta solo qui, in locale</div>
      </div>
      <div class="bd">
        <div class="meta" id="meta"></div>
        <div class="legend" id="legend"></div>
        <div class="tablewrap" id="tablewrap">
          <table><thead><tr><th data-i18n="th_id">ID</th><th data-i18n="th_val">Valore originale</th><th data-i18n="th_type">Tipo</th></tr></thead>
          <tbody id="maprows"></tbody></table>
        </div>
      </div>
    </div>
  </div>

  <!-- ============ PANE 2: RIPRISTINA ============ -->
  <div class="pane" id="pane2">
    <div class="callout">
      <span class="ic">💡</span>
      <div data-i18n="callout">Incolla qui la <b>risposta dell'LLM</b> (che contiene i placeholder come
      <span class="kbd">[FULLNAME_1]</span>): l'app rimette i valori veri usando il dizionario
      di questa sessione. Se hai chiuso e riaperto l'app, <b>carica il dizionario .json</b> che
      avevi salvato.</div>
    </div>
    <div class="grid">
      <div class="card">
        <div class="hd"><h2 data-i18n="r_title1">Risposta con i placeholder</h2>
          <div class="right">
            <label class="chip" style="cursor:pointer"><span data-i18n="loaddict">📁 Carica dizionario</span>
              <input type="file" id="dictFile" accept="application/json" hidden></label>
          </div></div>
        <div class="bd">
          <textarea id="rin" data-i18n-ph="rin_ph" placeholder="Incolla qui la risposta di ChatGPT…"></textarea>
          <div class="row">
            <button class="btn lg" id="rev">🔓 <span data-i18n="rev">Ripristina valori</span></button>
            <button class="ghost" id="rclear" data-i18n="clear">Pulisci</button>
            <span class="hint" id="dictInfo"></span>
          </div>
        </div>
      </div>
      <div class="card">
        <div class="hd"><h2 data-i18n="r_title2">Testo ripristinato</h2></div>
        <div class="bd">
          <div class="view"><div class="preview" id="rout">
            <div class="empty"><img src="/assets/mascot_doc.png" alt="" onerror="this.replaceWith(Object.assign(document.createElement('div'),{className:'big',textContent:'🔓'}))">
            <div data-i18n="empty_rout">Il testo con i valori reali apparirà qui.</div></div>
          </div></div>
          <div class="row"><button class="btn" id="rcopy">📋 <span data-i18n="rcopy">Copia testo ripristinato</span></button></div>
        </div>
      </div>
    </div>
  </div>

  <div class="credits">Realizzato da <b>Simone Rizzo</b> · sponsorizzato da <b>Rizzo AI Academy</b> · <span class="u">www.rizzoaiacademy.com</span></div>
</div>

<div id="toast"></div>

<!-- config modal -->
<div class="cfg-overlay" id="cfgOverlay">
  <div class="cfg-card">
    <h3>⚙️ <span data-i18n="cfg_title">Configurazione server</span></h3>
    <div class="cfg-row">
      <label data-i18n="cfg_host">Indirizzo</label>
      <input id="cfgHost" type="text" value="127.0.0.1" spellcheck="false">
    </div>
    <div class="cfg-row">
      <label data-i18n="cfg_port">Porta</label>
      <input id="cfgPort" type="number" min="1024" max="65535" value="5005">
    </div>
    <div class="cfg-status" id="cfgStatus"></div>
    <div class="cfg-btns">
      <button class="btn" id="cfgSave" onclick="saveConfig()">💾 <span data-i18n="cfg_save">Salva</span></button>
      <button class="ghost" id="cfgCheck" onclick="checkPort()"><span data-i18n="cfg_check">Verifica porta</span></button>
      <button class="ghost" onclick="closeConfig()" data-i18n="cfg_cancel">Annulla</button>
    </div>
    <div class="cfg-note" data-i18n="cfg_restart_note">Le modifiche avranno effetto al prossimo avvio.</div>
  </div>
</div>

<script>
const $ = id => document.getElementById(id);
let DATA = null;            // ultimo risultato analyze
let MAP = {};              // {placeholder -> valore} sessione corrente
const off = new Set();     // label nascoste nella preview
let L = 'it';              // lingua UI corrente

/* ---- i18n (IT default, EN opzionale) ---- */
const T = {
 it:{
  tagline:"modello locale su CPU · GDPR compliant", badge:"100% in locale",
  notice:"<b>Versione in sviluppo.</b> Il modello AI non è perfetto e può commettere errori: verifica sempre il risultato prima di usarlo. Queste sono le prime versioni e il progetto è completamente <b>open source</b>. Se ti è utile, <b>lascia una ⭐ alla repo</b> e contribuisci a migliorarlo: <a href=\"https://github.com/Rizzo-AI-Academy/rizzo-pii\" target=\"_blank\" rel=\"noopener\">apri la repo su GitHub ↗</a>",
  tab1:"Anonimizza", tab2:"Ripristina la risposta",
  in_title:"① Il tuo documento", in_hint:"incolla testo o trascina un PDF",
  src_ph:"Incolla qui il testo dell'atto, del contratto o della sentenza…\n\nOppure trascina un PDF nell'area qui sotto.",
  drop:"Trascina un <b>PDF</b> qui, oppure <b>scegli un file</b>",
  go:"Anonimizza", clear:"Pulisci",
  out_title:"② Risultato", v_prev:"Anteprima", v_text:"Testo da copiare",
  empty_prev:"L'anteprima con le PII evidenziate apparirà qui.",
  anon_ph:"Il testo anonimizzato apparirà qui.",
  copy:"Copia per ChatGPT", dl:"Scarica dizionario",
  dict_title:"Dizionario reversibile", dict_hint:"resta solo qui, in locale",
  th_id:"ID", th_val:"Valore originale", th_type:"Tipo",
  callout:"Incolla qui la <b>risposta dell'LLM</b> (che contiene i placeholder come <span class=\"kbd\">[FULLNAME_1]</span>): l'app rimette i valori veri usando il dizionario di questa sessione. Se hai chiuso e riaperto l'app, <b>carica il dizionario .json</b> che avevi salvato.",
  r_title1:"Risposta con i placeholder", loaddict:"📁 Carica dizionario",
  rin_ph:"Incolla qui la risposta di ChatGPT…",
  rev:"Ripristina valori", r_title2:"Testo ripristinato",
  empty_rout:"Il testo con i valori reali apparirà qui.", rcopy:"Copia testo ripristinato",
  st_ent:"entità", st_uniq:"valori unici", st_model:"dal modello",
  st_regex:"da regex/checksum", st_chars:"caratteri", analyzing:"Analizzo…",
  t_need_input:"Inserisci del testo o un PDF", t_error:"Errore",
  t_copied:"Testo anonimizzato copiato", t_need_anon:"Prima anonimizza un testo",
  t_nothing_dl:"Niente da scaricare", t_dl_ok:"Dizionario scaricato",
  t_paste_restore:"Incolla la risposta da ripristinare",
  t_no_dict:"Nessun dizionario: caricane uno .json", t_restored:"Valori ripristinati",
  t_nothing_copy:"Niente da copiare", t_restored_copied:"Testo ripristinato copiato",
  t_dict_loaded:"Dizionario caricato", t_json_invalid:"JSON non valido",
  t_drag_pdf:"Trascina un file PDF",
  pii_found:(n,u)=>n+" PII trovate · "+u+" valori unici",
  dict_n:n=>n+" ID nel dizionario",
  dict_loaded_n:n=>"dizionario caricato · "+n+" ID",
  dict_session:n=>"dizionario sessione · "+n+" ID",
  chars:n=>n.toLocaleString('it'),
  cfg_title:"Configurazione server", cfg_host:"Indirizzo", cfg_port:"Porta",
  cfg_check:"Verifica porta", cfg_save:"Salva", cfg_cancel:"Annulla",
  cfg_available:"Porta disponibile ✓", cfg_in_use:"Porta occupata ✗",
  cfg_saved:"Configurazione salvata (riavvia per applicare)",
  cfg_restart_note:"Le modifiche avranno effetto al prossimo avvio.",
 },
 en:{
  tagline:"local model on CPU · GDPR compliant", badge:"100% local",
  notice:"<b>Work in progress.</b> The AI model isn't perfect and can make mistakes: always double-check the result before relying on it. These are the very first versions and the project is fully <b>open source</b>. If you find it useful, <b>leave a ⭐ on the repo</b> and help improve it: <a href=\"https://github.com/Rizzo-AI-Academy/rizzo-pii\" target=\"_blank\" rel=\"noopener\">open the repo on GitHub ↗</a>",
  tab1:"Anonymize", tab2:"Restore the answer",
  in_title:"① Your document", in_hint:"paste text or drop a PDF",
  src_ph:"Paste here the text of the deed, contract or judgment…\n\nOr drop a PDF onto the area below.",
  drop:"Drop a <b>PDF</b> here, or <b>choose a file</b>",
  go:"Anonymize", clear:"Clear",
  out_title:"② Result", v_prev:"Preview", v_text:"Text to copy",
  empty_prev:"The preview with highlighted PII will appear here.",
  anon_ph:"The anonymized text will appear here.",
  copy:"Copy for ChatGPT", dl:"Download dictionary",
  dict_title:"Reversible dictionary", dict_hint:"stays here only, locally",
  th_id:"ID", th_val:"Original value", th_type:"Type",
  callout:"Paste here the <b>LLM's answer</b> (containing placeholders like <span class=\"kbd\">[FULLNAME_1]</span>): the app puts the real values back using this session's dictionary. If you closed and reopened the app, <b>load the .json dictionary</b> you saved.",
  r_title1:"Answer with placeholders", loaddict:"📁 Load dictionary",
  rin_ph:"Paste ChatGPT's answer here…",
  rev:"Restore values", r_title2:"Restored text",
  empty_rout:"The text with the real values will appear here.", rcopy:"Copy restored text",
  st_ent:"entities", st_uniq:"unique values", st_model:"from the model",
  st_regex:"from regex/checksum", st_chars:"characters", analyzing:"Analyzing…",
  t_need_input:"Enter some text or a PDF", t_error:"Error",
  t_copied:"Anonymized text copied", t_need_anon:"Anonymize a text first",
  t_nothing_dl:"Nothing to download", t_dl_ok:"Dictionary downloaded",
  t_paste_restore:"Paste the answer to restore",
  t_no_dict:"No dictionary: load a .json one", t_restored:"Values restored",
  t_nothing_copy:"Nothing to copy", t_restored_copied:"Restored text copied",
  t_dict_loaded:"Dictionary loaded", t_json_invalid:"Invalid JSON",
  t_drag_pdf:"Drop a PDF file",
  pii_found:(n,u)=>n+" PII found · "+u+" unique values",
  dict_n:n=>n+" IDs in the dictionary",
  dict_loaded_n:n=>"dictionary loaded · "+n+" IDs",
  dict_session:n=>"session dictionary · "+n+" IDs",
  chars:n=>n.toLocaleString('en'),
  cfg_title:"Server configuration", cfg_host:"Host", cfg_port:"Port",
  cfg_check:"Check port", cfg_save:"Save", cfg_cancel:"Cancel",
  cfg_available:"Port available ✓", cfg_in_use:"Port in use ✗",
  cfg_saved:"Config saved (restart to apply)",
  cfg_restart_note:"Changes take effect on next startup.",
 }
};
const tt=k=>T[L][k];

function routEmpty(){
  return '<div class="empty"><img src="/assets/mascot_doc.png" alt="" onerror="this.replaceWith(Object.assign(document.createElement(\'div\'),{className:\'big\',textContent:\'🔓\'}))"><div>'+tt('empty_rout')+'</div></div>';
}

function applyLang(l){
  L=(l==='en')?'en':'it';
  localStorage.setItem('pii_lang',L);
  document.documentElement.lang=L;$('lang').value=L;
  document.querySelectorAll('[data-i18n]').forEach(el=>{
    const v=T[L][el.getAttribute('data-i18n')]; if(v!=null) el.innerHTML=v;});
  document.querySelectorAll('[data-i18n-ph]').forEach(el=>{
    const v=T[L][el.getAttribute('data-i18n-ph')]; if(v!=null) el.placeholder=v;});
  if(!$('pdf').files.length) $('dropTxt').innerHTML=tt('drop');   // dropzone: solo se nessun file
  if(!$('rout')._raw) $('rout').innerHTML=routEmpty();
  if(DATA) render();
}

/* ---- scroll sincronizzato editor (sx) <-> anteprima/testo (dx) ---- */
let syncing=false;
function matchScroll(target,from){
  const rf=from.scrollHeight-from.clientHeight, rt=target.scrollHeight-target.clientHeight;
  target.scrollTop = rf>0 ? (from.scrollTop/rf)*rt : 0;
}
function linkScroll(a,b){
  a.addEventListener('scroll',()=>{
    if(syncing)return; syncing=true; matchScroll(b,a);
    setTimeout(()=>syncing=false,0);});   // reset robusto (rAF puo' non scattare in bg)
}

/* ---- colore deterministico per tipo di tag ---- */
function hue(s){let h=0;for(const c of s)h=(h*31+c.charCodeAt(0))%360;return h;}
function colors(label){const h=hue(label);
  return {bg:`hsl(${h} 56% 96%)`,bd:`hsl(${h} 42% 81%)`,tx:`hsl(${h} 40% 38%)`};}

function toast(msg,ok=true){const t=$('toast');t.textContent=msg;t.className='show'+(ok?' ok':'');
  clearTimeout(t._t);t._t=setTimeout(()=>t.className='',1800);}

/* ---- tabs ---- */
function showTab(n){
  $('tab1').classList.toggle('on',n===1);$('tab2').classList.toggle('on',n===2);
  $('pane1').classList.toggle('on',n===1);$('pane2').classList.toggle('on',n===2);
}

/* ---- analyze ---- */
async function run(){
  const file=$('pdf').files[0];const text=$('src').value.trim();
  if(!file&&!text){toast(tt('t_need_input'),false);return;}
  $('go').disabled=true;const old=$('go').innerHTML;
  $('go').innerHTML='<span class="spin"></span> '+tt('analyzing');
  try{
    let resp;
    if(file){const fd=new FormData();fd.append('pdf',file);
      resp=await fetch('/analyze',{method:'POST',body:fd});}
    else resp=await fetch('/analyze',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({text})});
    const d=await resp.json();
    if(!resp.ok){toast(d.error||tt('t_error'),false);return;}
    if(d.source_text&&file)$('src').value=d.source_text;
    DATA=d;MAP=d.mapping;off.clear();
    localStorage.setItem('pii_map',JSON.stringify(MAP));
    render();
    toast(T[L].pii_found(d.n_entities,d.n_unique));
  }catch(e){toast(tt('t_error')+': '+e.message,false);}
  finally{$('go').disabled=false;$('go').innerHTML=old;}
}

function render(){
  const d=DATA;
  $('dictCard').style.display='';            // mostra la card dizionario (sotto le due colonne)
  document.querySelector('.app').classList.add('has-result');  // -> scroll pagina, niente schiacciamento
  // preview evidenziata
  const prev=$('prev');prev.innerHTML='';prev.style.display='';$('emptyPrev').style.display='none';
  for(const s of d.segments){
    if(s.label){
      const c=colors(s.label);const sp=document.createElement('span');
      sp.className='ph'+(off.has(s.label)?' dim':'');
      sp.style.background=c.bg;sp.style.borderColor=c.bd;sp.style.color=c.tx;
      sp.title=`${s.t}\n(${s.src}${s.validated?' · checksum ✓':''})`;
      sp.innerHTML=s.ph.replace(/[\[\]]/g,'')+(s.validated?'<span class="ck">✓</span>':'');
      prev.appendChild(sp);
    }else prev.appendChild(document.createTextNode(s.t));
  }
  // testo da copiare
  $('anon').value=d.anonymized_text;
  // meta
  $('meta').innerHTML=
    `<span class="stat"><b>${d.n_entities}</b> ${tt('st_ent')}</span>`+
    `<span class="stat"><b>${d.n_unique}</b> ${tt('st_uniq')}</span>`+
    `<span class="stat"><b>${(d.by_source.modello||0)}</b> ${tt('st_model')}</span>`+
    `<span class="stat"><b>${(d.by_source.regex||0)}</b> ${tt('st_regex')}</span>`+
    `<span class="stat"><b>${T[L].chars(d.n_chars)}</b> ${tt('st_chars')}</span>`;
  // legenda cliccabile (toggle highlight)
  const lg=$('legend');lg.innerHTML='';
  for(const [k,v] of Object.entries(d.by_label)){
    const c=colors(k);const el=document.createElement('span');
    el.className='chip'+(off.has(k)?' off':'');
    el.innerHTML=`<span class="sw" style="background:${c.bd}"></span>${k}<span class="n">${v}</span>`;
    el.onclick=()=>{off.has(k)?off.delete(k):off.add(k);render();};
    lg.appendChild(el);
  }
  // dizionario
  const rows=$('maprows');rows.innerHTML='';
  const keys=Object.keys(d.mapping);
  $('tablewrap').style.display=keys.length?'':'none';
  for(const ph of keys){const lab=ph.slice(1,ph.lastIndexOf('_'));const c=colors(lab);
    const tr=document.createElement('tr');
    tr.innerHTML=`<td class="k" style="color:${c.tx}">${ph}</td>`+
      `<td class="v">${escapeHtml(d.mapping[ph])}</td>`+
      `<td><span class="chip" style="cursor:default"><span class="sw" style="background:${c.bd}"></span>${lab}</span></td>`;
    rows.appendChild(tr);}
  $('ulock').textContent=keys.length?T[L].dict_n(keys.length):'';
}

function escapeHtml(s){return s.replace(/[&<>"]/g,m=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[m]));}

/* ---- view toggle ---- */
function setView(v){
  const p=v==='prev';
  $('vPrev').classList.toggle('on',p);$('vText').classList.toggle('on',!p);
  $('viewPrev').style.display=p?'':'none';$('anon').style.display=p?'none':'';
  matchScroll(p?$('viewPrev'):$('anon'),$('src'));   // allinea la vista appena mostrata
}

/* ---- copy / download ---- */
$('copy').onclick=()=>{if(!DATA){toast(tt('t_need_anon'),false);return;}
  navigator.clipboard.writeText(DATA.anonymized_text).then(()=>toast(tt('t_copied')));};
$('dl').onclick=()=>{if(!DATA||!Object.keys(MAP).length){toast(tt('t_nothing_dl'),false);return;}
  const blob=new Blob([JSON.stringify(MAP,null,2)],{type:'application/json'});
  const a=document.createElement('a');a.href=URL.createObjectURL(blob);
  a.download='dizionario_anonimizzazione.json';a.click();URL.revokeObjectURL(a.href);
  toast(tt('t_dl_ok'));};

/* ---- reverse ---- */
function reverse(){
  const txt=$('rin').value;
  if(!txt.trim()){toast(tt('t_paste_restore'),false);return;}
  if(!Object.keys(MAP).length){toast(tt('t_no_dict'),false);return;}
  // placeholder piu' lunghi prima (evita FULLNAME_1 dentro FULLNAME_10)
  const keys=Object.keys(MAP).sort((a,b)=>b.length-a.length);
  let out=txt;
  for(const ph of keys){
    const inner=ph.slice(1,-1);                 // FULLNAME_1
    // tollerante: parentesi opzionali / spazi, eventuale grassetto markdown
    const rx=new RegExp('\\**\\[?\\s*'+inner.replace(/[.*+?^${}()|[\]\\]/g,'\\$&')+'\\s*\\]?\\**','g');
    out=out.replace(rx,MAP[ph].replace(/\$/g,'$$$$'));
  }
  const o=$('rout');o.textContent=out;o._raw=out;
  toast(tt('t_restored'));
}
$('rev').onclick=reverse;
$('rcopy').onclick=()=>{const o=$('rout');if(!o._raw){toast(tt('t_nothing_copy'),false);return;}
  navigator.clipboard.writeText(o._raw).then(()=>toast(tt('t_restored_copied')));};
$('rclear').onclick=()=>{$('rin').value='';$('rout').innerHTML=routEmpty();$('rout')._raw='';};

/* ---- carica dizionario da file (per sessioni diverse) ---- */
$('dictFile').onchange=e=>{const f=e.target.files[0];if(!f)return;
  const r=new FileReader();r.onload=()=>{try{MAP=JSON.parse(r.result);
    $('dictInfo').textContent=T[L].dict_loaded_n(Object.keys(MAP).length);
    toast(tt('t_dict_loaded'));}catch{toast(tt('t_json_invalid'),false);}};
  r.readAsText(f);};

/* ---- input helpers ---- */
$('go').onclick=run;
$('clear').onclick=()=>{$('src').value='';$('pdf').value='';$('dropTxt').innerHTML=tt('drop');
  DATA=null;$('prev').style.display='none';$('emptyPrev').style.display='';
  $('anon').value='';$('meta').innerHTML='';$('legend').innerHTML='';
  $('dictCard').style.display='none';$('ulock').textContent='';
  document.querySelector('.app').classList.remove('has-result');};
$('src').addEventListener('keydown',e=>{if((e.ctrlKey||e.metaKey)&&e.key==='Enter')run();});

/* pdf picker + dropzone */
const drop=$('drop');
$('pdf').onchange=e=>{const f=e.target.files[0];if(f)$('dropTxt').innerHTML=`📎 <b>${f.name}</b>`;};
['dragenter','dragover'].forEach(ev=>drop.addEventListener(ev,e=>{e.preventDefault();drop.classList.add('hot');}));
['dragleave','drop'].forEach(ev=>drop.addEventListener(ev,e=>{e.preventDefault();drop.classList.remove('hot');}));
drop.addEventListener('drop',e=>{const f=e.dataTransfer.files[0];
  if(f&&f.type==='application/pdf'){const dt=new DataTransfer();dt.items.add(f);$('pdf').files=dt.files;
    $('dropTxt').innerHTML=`📎 <b>${f.name}</b>`;}else toast(tt('t_drag_pdf'),false);});

/* scroll sincronizzato: editor (sx) <-> anteprima e testo (dx) */
linkScroll($('src'),$('viewPrev'));linkScroll($('viewPrev'),$('src'));
linkScroll($('src'),$('anon'));linkScroll($('anon'),$('src'));

/* lingua: selettore + applicazione iniziale (default IT, preferenza salvata) */
$('lang').onchange=e=>applyLang(e.target.value);
applyLang(localStorage.getItem('pii_lang')||'it');

/* avviso: popup a click (non hover), si chiude cliccando fuori o con Esc */
$('infoBtn').addEventListener('click',e=>{e.stopPropagation();$('infoBtn').classList.toggle('open');});
$('infoBtn').addEventListener('keydown',e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();$('infoBtn').classList.toggle('open');}});
document.addEventListener('click',e=>{if(!$('infoBtn').contains(e.target))$('infoBtn').classList.remove('open');});
document.addEventListener('keydown',e=>{if(e.key==='Escape'){$('infoBtn').classList.remove('open');closeConfig();}});

/* ---- config modal ---- */
async function openConfig(){
  const r=await fetch('/config');const d=await r.json();
  $('cfgHost').value=d.host||'127.0.0.1';
  $('cfgPort').value=d.port||5005;
  $('cfgStatus').className='cfg-status';$('cfgStatus').textContent='';
  $('cfgOverlay').classList.add('open');
}
function closeConfig(){$('cfgOverlay').classList.remove('open');}
$('cfgOverlay').addEventListener('click',e=>{if(e.target===$('cfgOverlay'))closeConfig();});
async function checkPort(){
  const h=$('cfgHost').value.trim(),p=parseInt($('cfgPort').value);
  if(!p||p<1024||p>65535){$('cfgStatus').className='cfg-status fail';$('cfgStatus').textContent=tt('cfg_in_use');return;}
  const r=await fetch(`/port-check?host=${encodeURIComponent(h)}&port=${p}`);
  const d=await r.json();
  $('cfgStatus').className=d.available?'cfg-status ok':'cfg-status fail';
  $('cfgStatus').textContent=d.available?tt('cfg_available'):tt('cfg_in_use');
}
async function saveConfig(){
  const h=$('cfgHost').value.trim(),p=parseInt($('cfgPort').value);
  if(!p||p<1024||p>65535){toast(tt('cfg_in_use'),false);return;}
  await fetch('/config',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({host:h,port:p})});
  toast(tt('cfg_saved'));
  closeConfig();
}

/* recupera dizionario da sessione precedente (dopo applyLang -> testo nella lingua giusta) */
try{const m=localStorage.getItem('pii_map');if(m){MAP=JSON.parse(m);
  if(Object.keys(MAP).length)$('dictInfo').textContent=T[L].dict_session(Object.keys(MAP).length);}}catch{}
</script>
</body>
</html>
"""

if __name__ == "__main__":
    import argparse as _ap
    _p = _ap.ArgumentParser(description="Rizzo PII — server locale di anonimizzazione.")
    _p.add_argument("--host", default=None, help="indirizzo su cui ascoltare (default da config/env/127.0.0.1)")
    _p.add_argument("--port", type=int, default=None, help="porta su cui ascoltare (default da config/env/5005)")
    _args = _p.parse_args()

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
