#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Genera una BANCA DI TEMPLATE di documenti legali italiani usando un LLM (Gemini),
MA con segnaposto al posto dei dati sensibili.

Principio: l'LLM scrive la PROSA giuridica realistica (atti, sentenze, contratti)
inserendo ESCLUSIVAMENTE i segnaposto {SLOT}; i dati veri (con checksum validi)
li inietta poi generate_synthetic_pii.py. Cosi':
  - le label BIO restano esatte (sappiamo dove sono i segnaposto),
  - i CF/IBAN/P.IVA sono matematicamente validi,
  - nessuna PII reale finisce nel dataset.

Sicurezza chiave API: NON si incolla in chiaro. Si legge da variabile d'ambiente.
  PowerShell:  $env:GEMINI_API_KEY = "la-tua-chiave-NUOVA"
  bash:        export GEMINI_API_KEY="la-tua-chiave-NUOVA"

Modello: default gemini-3.5-flash (override con la env var GEMINI_MODEL).

BACKEND LOCALE (nessuna chiave, niente esce dalla macchina): se e' impostata
LLM_BASE_URL i template li scrive un LLM locale via endpoint OpenAI-compatibile.
  export LLM_BASE_URL=http://127.0.0.1:8080/v1   # llama.cpp / Ollama / vLLM / LM Studio
  export LLM_MODEL=nome-del-modello

Uso:  python llm_template_bank.py --per-type 3
Output: legal_templates.json  (lista di {"id", "doc_type", "text"})
"""

import argparse
import io
import json
import os
import re
import sys
import time
import unicodedata
import urllib.request
import urllib.error
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", write_through=True)

# Segnaposto consentiti = quelli che gli iniettori di generate_synthetic_pii.py sanno
# riempire. Si DERIVA da li' invece di riscriverne la lista: era una copia a mano, e
# aveva gia' preso la deriva. Mancavano PROVINCE, NAMELIST, ORGLIST e MIXEDLIST, con
# l'effetto che la guardia scartava template che l'iniettore avrebbe riempito senza
# problemi - PROVINCE e' uno dei 22 tag, ed e' quello che nel training ha meno varieta'
# di contesto proprio perche' arriva quasi solo dai template.
# Una lista scritta a mano qui tornerebbe a divergere alla prossima aggiunta.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import generate_synthetic_pii as _gen  # noqa: E402

ALLOWED_SLOTS = set(_gen.SLOTS)

# breve legenda per i segnaposto il cui uso non e' ovvio dal nome (guida l'LLM a
# posizionarli nel contesto giusto, in molti documenti diversi -> varieta' strutturale)
SLOT_HINTS = """  {ORG}     = ragione sociale di una societa'/studio legale/banca (la PARTE, non il tribunale)
  {DOCID}   = codice identificativo di un atto: n. protocollo, n. repertorio/raccolta, n. sentenza
  {CATASTO} = dati catastali di un immobile (foglio, particella, subalterno)
  {CONTO}   = numero di conto corrente (diverso dall'IBAN)
  {CIG}     = Codice Identificativo Gara di un appalto pubblico
  {CUP}     = Codice Unico di Progetto di un investimento pubblico
  {POLIZZA} = numero di una polizza assicurativa
  {MATRICOLA} = matricola aziendale INPS/INAIL del datore di lavoro"""

DOC_TYPES = [
    "atto di citazione", "comparsa di costituzione e risposta", "sentenza civile",
    "decreto ingiuntivo", "contratto di locazione", "procura alle liti",
    "ricorso per decreto ingiuntivo", "verbale di udienza", "atto di diffida",
    "contratto di compravendita immobiliare",
]

# >>> incolla qui la tua chiave NUOVA (solo locale: non committare / non condividere questo file).
# Lasciala "" per usare invece la variabile d'ambiente GEMINI_API_KEY.
API_KEY = os.environ.get("GEMINI_API_KEY")

SLOT_RE = re.compile(r"\{(\w+)\}")
MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.5-flash")
ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}"

# --- backend alternativo: un LLM LOCALE (endpoint OpenAI-compatibile) ------------------
# Se LLM_BASE_URL e' impostata, i template vengono scritti da quel modello invece che da
# Gemini: nessuna chiave API, nessun prompt che esce dalla macchina. Coerente con lo scopo
# del progetto (e chi non ha una chiave Google puo' comunque contribuire dati).
#   export LLM_BASE_URL=http://127.0.0.1:8080/v1   # llama.cpp / Ollama / vLLM / LM Studio
#   export LLM_MODEL=nome-del-modello
LLM_BASE_URL = os.environ.get("LLM_BASE_URL")
LLM_MODEL = os.environ.get("LLM_MODEL", "local-model")
LLM_API_KEY = os.environ.get("LLM_API_KEY", "sk-no-key-required")

PROMPT = """Sei un giurista italiano. Scrivi un {doc_type} REALISTICO e completo (da 8 a 18 righe),
nel linguaggio tecnico-giuridico autentico usato nei tribunali italiani.

REGOLA ASSOLUTA: non inserire MAI dati reali o inventati (niente nomi, codici fiscali,
IBAN, indirizzi, importi scritti per esteso). Dove servirebbe un dato personale usa
ESCLUSIVAMENTE uno di questi segnaposto, scritti ESATTAMENTE cosi' tra parentesi graffe:

{slot_list}

Alcuni segnaposto, quando il tipo di documento lo richiede:
{slot_hints}

Puoi ripetere lo stesso segnaposto piu' volte. Non aggiungere segnaposto diversi da quelli elencati.

VIETATO scrivere nel testo un qualunque nome di persona o di citta', anche di fantasia
(es. "Mario Rossi", "Milano"): usa SEMPRE il segnaposto corrispondente. Non scrivere mai
un titolo seguito da un nome (es. "Sig. Bianchi", "avv. Verdi"): scrivi "il/la {{LAWYER}}",
"il Sig. {{FULLNAME}}", ecc.
Restituisci SOLO il testo del documento, senza titoli di contorno, senza commenti, senza markdown."""


def backend_name():
    """Descrizione del backend in uso, per i log."""
    return f"locale {LLM_MODEL} @ {LLM_BASE_URL}" if LLM_BASE_URL else f"Gemini {MODEL}"


def have_backend():
    """True se c'e' un modo per far scrivere i template (locale o Gemini)."""
    return bool(LLM_BASE_URL or API_KEY or os.environ.get("GEMINI_API_KEY"))


def call_llm(prompt, retries=3):
    """Fa scrivere un template al backend configurato: LLM locale se LLM_BASE_URL
    e' impostata, altrimenti Gemini."""
    if LLM_BASE_URL:
        return call_local_openai(prompt, retries)
    return call_gemini(prompt, retries)


def call_local_openai(prompt, retries=3):
    """Endpoint OpenAI-compatibile (llama.cpp server, Ollama, vLLM, LM Studio...)."""
    url = LLM_BASE_URL.rstrip("/")
    if not url.endswith("/chat/completions"):
        url += "/chat/completions"
    body = json.dumps({
        "model": LLM_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.9,
        "max_tokens": 1400,
        # Sui modelli "reasoning" serviti in locale l'intero output puo' finire nel campo
        # reasoning_content lasciando 'content' VUOTO (visto con gemma-4-12B su llama.cpp:
        # 175 s di pensiero e zero testo). Qui il pensiero si disattiva e, per sicurezza,
        # sotto si accetta reasoning_content come ripiego.
        "chat_template_kwargs": {"enable_thinking": False},
    }).encode()
    headers = {"Content-Type": "application/json",
               "Authorization": f"Bearer {LLM_API_KEY}"}
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, data=body, headers=headers)
            with urllib.request.urlopen(req, timeout=600) as r:
                data = json.load(r)
            msg = data["choices"][0]["message"]
            text = (msg.get("content") or msg.get("reasoning_content") or "").strip()
            if text:
                return text
            print("  risposta vuota dal modello locale")
        except urllib.error.HTTPError as e:
            print(f"  HTTP {e.code}: {e.read().decode()[:200]}")
        except (urllib.error.URLError, KeyError, IndexError) as e:
            print(f"  errore: {e}")
        time.sleep(2 * (attempt + 1))
    return None


def call_gemini(prompt, retries=3):
    key = API_KEY or os.environ.get("GEMINI_API_KEY")
    if not key:
        sys.exit("ERRORE: imposta API_KEY nel file oppure la variabile d'ambiente "
                 "GEMINI_API_KEY (chiave NUOVA, non quella incollata in chat).\n"
                 "  Oppure usa un LLM locale: export LLM_BASE_URL=http://127.0.0.1:8080/v1")
    url = ENDPOINT.format(model=MODEL, key=key)
    body = json.dumps({
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.9},
    }).encode()
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, data=body,
                                         headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=60) as r:
                data = json.load(r)
            return data["candidates"][0]["content"]["parts"][0]["text"].strip()
        except urllib.error.HTTPError as e:
            print(f"  HTTP {e.code}: {e.read().decode()[:200]}")
            time.sleep(2 * (attempt + 1))
        except (urllib.error.URLError, KeyError, IndexError) as e:
            print(f"  errore: {e}")
            time.sleep(2 * (attempt + 1))
    return None



# titoli che, seguiti da una maiuscola, segnalano un nome scritto inline. Solo le forme
# che introducono una PERSONA: "Spett." e "Spettabile" si rivolgono a un'organizzazione
# ("Spett.le Azienda ...") e da soli non dicono nulla su un nome.
TITLES = {"Sig", "Sig.ra", "Sigg", "Signor", "Signora", "Dott", "Dottor",
          "Dssa", "Avv", "Egr", "Egregio", "Ill", "Onorevole"}

# Parole che in italiano INTRODUCONO una persona. Due livelli, perche' non danno la stessa
# certezza:
#  - FORTI: dopo di queste una maiuscola e' un nome anche da sola ("il sottoscritto
#    Sferrazza"). Non introducono altro.
#  - RUOLI: possono essere seguite da un QUALIFICATORE maiuscolo che non e' una persona
#    ("il Giudice Unico", "di seguito denominato Locatore"). Qui serve una maiuscola in
#    piu' -- Nome + Cognome -- per parlare di nome inline.
# Sono classi chiuse che parlano di persone, non di domini: non vanno allungate quando si
# aggiunge un tipo di documento.
CONTESTO_FORTE = {
    "nato", "nata", "nati", "nate", "sottoscritto", "sottoscritta", "sottoscritti",
    "signor", "signora", "signori", "nome", "cognome",
}
CONTESTO_RUOLO = {
    "avvocato", "avvocatessa", "dottore", "dottoressa", "difeso", "difesa",
    "rappresentato", "rappresentata", "assistito", "assistita", "teste", "testimone",
    "giudice", "notaio", "perito", "curatore", "erede", "coniuge", "figlio", "figlia",
    "padre", "madre", "intestatario", "intestataria", "titolare", "beneficiario",
    "beneficiaria", "dipendente", "lavoratore", "lavoratrice", "paziente", "alunno",
    "alunna", "studente", "studentessa", "cliente", "contraente", "assicurato",
    "assicurata", "delegato", "delegata", "richiedente", "denunciante", "querelante",
    # gli altri modi italiani di dire "la persona che fa X". Stanno qui e non fra i
    # contesti forti perche' sono seguiti spesso da un qualificatore maiuscolo ("il
    # sottoscritto Agente accertatore", "il Venditore"), e perche' una corsa fatta solo
    # di queste parole e' vocabolario di ruolo, non un nome
    "agente", "operatore", "funzionario", "responsabile", "incaricato", "accertatore",
    "verbalizzante", "conducente", "proprietario", "trasgressore", "socio", "abbonato",
    "utente", "assegnatario", "candidato", "conduttore", "locatore", "locatrice",
    "venditore", "venditrice", "acquirente", "mandante", "mandatario", "cancelliere",
    "segretario", "presidente", "amministratore", "custode", "tutore", "genitore",
    "committente", "appaltatore", "fornitore", "debitore", "creditore", "passeggero",
}
TITLES_L = {t.lower() for t in TITLES}
# le ABBREVIAZIONI di titolo ("Sig.", "Avv.", "Dott.") stanno solo davanti a un nome, e
# restano forti; le stesse parole per esteso sono nomi di ruolo e possono essere seguite da
# un qualificatore ("il Dottore Commercialista"), quindi valgono come ruolo
CONTESTO_FORTE = (CONTESTO_FORTE | TITLES_L) - CONTESTO_RUOLO

# preposizioni della famiglia di "di": davanti a una maiuscola la rendono un complemento
# ("Giudice di Pace", "Corte dei Conti") e non un nome. Sostituiscono una lista di
# istituzioni scritta a mano: la relazione grammaticale vale per qualunque ufficio, la
# lista sarebbe rimasta indietro al primo dominio nuovo.
PREPOSIZIONI = {"di", "del", "dello", "della", "dei", "degli", "delle", "d"}

# I segnaposto vanno TOLTI prima di cercare i nomi, ma non lasciando un buco: se al loro
# posto resta uno spazio, chi introduce scavalca il segnaposto e si attacca alla parola
# dopo -- in "corrisposta dal Sig. {FULLNAME} al Venditore" il titolo "Sig." finisce a
# introdurre "Venditore" e il template viene scartato per un nome che invece era gia' un
# segnaposto. Un segno tutto maiuscolo al loro posto fa da barriera (e _cap lo ignora,
# perche' gli acronimi non sono nomi).
SEGNO_SLOT = "SEGNAPOSTO"

# parole di servizio che stanno fra chi introduce e il nome ("nato a", "difeso dal"):
# vanno saltate quando si cerca chi introduce, altrimenti la ricerca si ferma su di loro
FUNZIONALI = {
    "a", "ad", "al", "allo", "alla", "ai", "agli", "alle", "da", "dal", "dallo", "dalla",
    "dai", "dagli", "dalle", "di", "del", "dello", "della", "dei", "degli", "delle",
    "in", "nel", "nello", "nella", "nei", "negli", "nelle", "con", "e", "ed", "il", "lo",
    "la", "i", "gli", "le", "un", "una", "uno", "sig", "sigra", "che", "il/la",
} - CONTESTO_FORTE

_LESSICO_PERSONE = None


def _lessico_persone():
    """(nomi propri, cognomi) noti al progetto, minuscoli e tenuti SEPARATI.

    Sono le stesse liste con cui generate_synthetic_pii INIETTA le persone: se il modello
    scrive un nome inline, quasi sempre pesca da questo stesso repertorio di nomi comuni
    italiani. Riusarle qui evita di mantenere un secondo elenco a mano.

    Separati perche' non valgono lo stesso. Un nome proprio ("Giulia", "Aldo") in un
    documento e' quasi sempre una persona. Un cognome italiano e' spessissimo anche una
    parola comune -- Gentile, Pace, Costa, Conte, Villa, Monti, Fiore, Guerra, Bianco --
    e nei documenti amministrativi quelle parole compaiono con la maiuscola per motivi
    loro: "Gentile Cliente" apre ogni lettera commerciale, "Giudice di Pace" e' un
    ufficio. Un cognome vale quindi solo se qualcosa lo introduce come persona.

    Import pigro e stato del generatore casuale ripristinato: importare
    generate_synthetic_pii esegue random.seed(42) a import-time, e questo script pesca a
    caso i tipi di documento -- l'import non deve rendere deterministica quella scelta."""
    global _LESSICO_PERSONE
    if _LESSICO_PERSONE is None:
        import random as _r
        stato = _r.getstate()
        try:
            sys.path.insert(0, str(Path(__file__).resolve().parent))
            import generate_synthetic_pii as _g
            _LESSICO_PERSONE = ({w.lower() for w in _g.MALE_NAMES + _g.FEMALE_NAMES},
                                {w.lower() for w in _g.SURNAMES})
        except Exception as e:                       # pragma: no cover
            print(f"  (lessico dei nomi non caricato: {e}) -> guardia solo per contesto")
            _LESSICO_PERSONE = (set(), set())
        finally:
            _r.setstate(stato)
    return _LESSICO_PERSONE


def _cap(w):
    """La parola senza punto/elisione se ha l'iniziale maiuscola e non e' tutta maiuscola
    (gli acronimi non sono nomi di persona), altrimenti None. 'dell'Avv.' -> 'Avv'."""
    seg = w.split("'")[-1].strip(".")
    return seg if seg and seg[0].isupper() and not seg.isupper() else None


def _confrontabile(w):
    """(parola confrontabile, chiude_la_frase). Toglie l'elisione ("L'intestatario" ->
    "intestatario", altrimenti chi introduce non si riconosce) e la punteggiatura finale.
    Un punto finale e' un confine di frase, tranne nelle abbreviazioni di titolo: dopo un
    confine la maiuscola e' solo l'inizio della frase nuova ("assistito. Nonostante")."""
    seg = w.split("'")[-1]
    nudo = seg.rstrip(".:;!?,").lower()
    chiude = seg[-1:] in ".:;!?" and nudo not in TITLES_L
    return nudo, chiude


def _introduce(words, i):
    """Chi introduce la maiuscola in posizione i: la prima parola non funzionale nelle 3
    precedenti ("nato a X" -> "nato", "difeso dal X" -> "difeso"). None se prima c'e' un
    confine di frase, un segnaposto, o non si trova nulla."""
    for k in range(i - 1, max(-1, i - 4), -1):
        nudo, chiude = _confrontabile(words[k])
        if chiude or nudo == SEGNO_SLOT.lower():
            return None
        if nudo and nudo not in FUNZIONALI:
            return nudo
    return None


def find_stray_names(text):
    """Segnala i nomi di PERSONA scritti inline (PII non taggata -> template da scartare).

    La versione precedente considerava nome proprio qualunque coppia di maiuscole assente
    da un lessico giuridico scritto a mano (LEGAL_CAPITALIZED). Regge dentro il tribunale
    e crolla fuori: "Risorse Umane", "Ufficio Protocollo", "Servizio Clienti" vengono
    scartati come se fossero persone, cioe' si perdono i template buoni proprio nei domini
    nuovi -- e allungare quella lista a ogni dominio e' una rincorsa senza fine.

    Qui i segnali sono due, entrambi indipendenti dal dominio:
      1. LESSICO -- una delle parole e' un nome o un cognome noto al progetto;
      2. CONTESTO -- la corsa di maiuscole sta dove va una persona: da sola se la introduce
         un contesto forte, in coppia (Nome + Cognome) se la introduce un nome di ruolo.

    Si ragiona su CORSE di maiuscole consecutive, non su coppie: e' la corsa intera a
    essere un nome o una denominazione, e chi la introduce sta prima della corsa -- non fra
    le sue parole."""
    masked = re.sub(r"\{\w+\}", f" {SEGNO_SLOT} ", text)      # i segnaposto fanno barriera
    words = re.findall(r"[A-Za-zÀ-ÿ][A-Za-zÀ-ÿ'.]*", masked)
    nomi, cognomi = _lessico_persone()
    stray = []
    i = 0
    while i < len(words):
        if not _cap(words[i]):
            i += 1
            continue
        j = i
        # la corsa non attraversa un confine di frase: in "assistito dal Cancelliere. La
        # causa..." Cancelliere e La sono due frasi diverse, non un nome di due parole
        while (j + 1 < len(words) and _cap(words[j + 1])
                and not _confrontabile(words[j])[1]):
            j += 1
        corsa = words[i:j + 1]
        intro = _introduce(words, i)
        # una corsa fatta solo di titoli e nomi di ruolo e' vocabolario, non un nome: in
        # "il sottoscritto Sig. {FULLNAME}" il nome e' il segnaposto, non "Sig."
        if all(_cap(w).lower() in CONTESTO_FORTE | CONTESTO_RUOLO for w in corsa):
            i = j + 1
            continue
        # Una preposizione appena prima della corsa la rende un COMPLEMENTO di cio' che
        # viene prima, non un nome in apposizione: "Giudice di Pace", "agente di Polizia
        # Locale", "Responsabile del Procedimento" sono denominazioni di uffici, mentre una
        # persona segue il proprio ruolo senza preposizione in mezzo ("il paziente
        # Bianchi", "l'intestatario Ludovico Marinetti"). Il nome proprio resta valido
        # comunque: "la firma di Giulia" e' una persona anche da complemento.
        prec = _confrontabile(words[i - 1])[0] if i else ""
        complemento = prec in PREPOSIZIONI
        parole = [_cap(w).lower() for w in corsa]
        introdotta = intro in CONTESTO_FORTE or intro in CONTESTO_RUOLO
        if any(p in nomi for p in parole):                        # "Mario Rossi", "Giulia"
            stray.append(" ".join(corsa))
        elif complemento:
            pass
        elif any(p in cognomi for p in parole) and introdotta:    # "il paziente Bianchi"
            stray.append(" ".join(corsa))
        # il nome di una persona sta in 2-3 parole: una corsa piu' lunga e' la
        # denominazione di qualcos'altro (un'offerta commerciale, un ufficio, un titolo di
        # sezione), e chi la introduce puo' avere un altro senso -- in "l'offerta
        # sottoscritta Mobile Plan Pro" il participio non introduce nessun sottoscritto
        elif intro in CONTESTO_FORTE and len(corsa) <= 3:         # "sottoscritto Sferrazza"
            stray.append(f"{intro} {' '.join(corsa)}")
        elif intro in CONTESTO_RUOLO and 2 <= len(corsa) <= 3:    # "intestatario L. Marinetti"
            stray.append(f"{intro} {' '.join(corsa)}")
        i = j + 1
    return stray


def non_latin_char(text):
    """Primo carattere alfabetico NON latino, se presente. Un modello locale (piccolo o
    molto quantizzato) puo' infilare un ideogramma in mezzo alla prosa italiana; se il
    template passa, quel carattere finisce in ogni esempio che lo usa."""
    for ch in text:
        if ch.isalpha():
            try:
                if "LATIN" not in unicodedata.name(ch):
                    return ch
            except ValueError:
                return ch
    return None


def normalizza_escape(text):
    r"""Converte gli escape LETTERALI (i due caratteri '\' + 'n') in veri a capo.

    Il modello a volte scrive la sequenza invece dell'a capo. Non e' un problema
    estetico: iniettando un valore subito dopo, il tokenizzatore incolla la 'n'
    all'inizio dell'entita' ('\n09/06/1965' -> token 'n09'), che resta O mentre il
    resto dell'entita' diventa I-DATE. Ne esce una riga con BIO malformato (I- senza
    B-), inutilizzabile in training. Misurato: 1 template su 237 conteneva la
    sequenza e ha rotto 721 righe su 200.000."""
    return text.replace("\\r\\n", "\n").replace("\\n", "\n").replace("\\t", "\t")


# --- codici e date scritti inline -----------------------------------------------------
# Un CF/IBAN/telefono/data scritto nel testo invece che come segnaposto e' un'entita' che
# esiste nel testo ma NON nelle label: il modello impara a NON taggarla. La guardia sui
# nomi non lo vede (non sono parole). Serve soprattutto nei domini fuori dal tribunale,
# dove la prosa e' piena di numeri (bollette, referti, cedolini) e il modello e' piu'
# tentato di inventare un codice.
CIFRE_LUNGHE_RE = re.compile(r"\d{9,}")                       # telefoni, IBAN, carte
CODICE_MISTO_RE = re.compile(r"\b(?=[A-Z0-9]*\d)(?=[A-Z0-9]*[A-Z])[A-Z0-9]{11,}\b")
DATA_INLINE_RE = re.compile(r"\b\d{1,2}[/.\-]\d{1,2}[/.\-]\d{2,4}\b")
# Le citazioni di norme ("196/2003", "art. 2043 c.c.") sono numeri legittimi, non PII:
# nessuno dei pattern sopra le prende (due soli gruppi, cifre sotto la soglia).


def inline_code(text):
    """Primo codice o data scritto inline, se presente."""
    masked = re.sub(r"\{\w+\}", " ", text)
    for rx in (CIFRE_LUNGHE_RE, CODICE_MISTO_RE, DATA_INLINE_RE):
        m = rx.search(masked)
        if m:
            return m.group(0)
    return None


def clean_and_validate(text):
    """Pulisce il markdown, verifica i segnaposto e scarta i template con PII inline
    o con caratteri non latini."""
    if not text:
        return None
    text = re.sub(r"^```.*?\n|```$", "", text.strip(), flags=re.MULTILINE).strip()
    text = normalizza_escape(text)
    slots = set(SLOT_RE.findall(text))
    if not slots:
        return None                       # nessun segnaposto -> inutile
    if slots - ALLOWED_SLOTS:
        print(f"  scartato: segnaposto non consentiti {slots - ALLOWED_SLOTS}")
        return None
    stray = find_stray_names(text)
    if stray:
        print(f"  scartato: probabili nomi inline non taggati {sorted(set(stray))[:8]}")
        return None
    bad = non_latin_char(text)
    if bad:
        print(f"  scartato: carattere non latino {bad!r} (artefatto del modello)")
        return None
    codice = inline_code(text)
    if codice:
        print(f"  scartato: codice o data scritti inline invece che come segnaposto "
              f"({codice!r})")
        return None
    return text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-type", type=int, default=3, help="template per tipo documento")
    ap.add_argument("--out", default=str(ROOT / "dataset" / "synthetic" / "legal_templates.json"))
    ap.add_argument("--append", action="store_true", help="accoda al file esistente")
    args = ap.parse_args()

    slot_list = "\n".join(f"  {{{s}}}" for s in sorted(ALLOWED_SLOTS))
    templates = []
    if args.append and os.path.exists(args.out):
        templates = json.load(open(args.out, encoding="utf-8"))
        print(f"Accodo a {len(templates)} template esistenti")
    base = len(templates)            # quanti c'erano gia'
    tid = len(templates)

    total = len(DOC_TYPES) * args.per_type
    done = ok = skip = 0
    t0 = time.time()
    print(f"Genero {total} template ({len(DOC_TYPES)} tipi x {args.per_type}) "
          f"con [{backend_name()}]\n")

    def save():
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(templates, f, ensure_ascii=False, indent=2)

    for doc_type in DOC_TYPES:
        for _ in range(args.per_type):
            done += 1
            elapsed = time.time() - t0
            eta = (elapsed / done) * (total - done) if done else 0
            pct = 100 * done // total
            print(f"[{done:>3}/{total} | {pct:>3}%] OK={ok} scartati={skip} | "
                  f"trascorso {elapsed:>4.0f}s | ETA {eta:>4.0f}s | {doc_type} ...")
            raw = call_llm(PROMPT.format(doc_type=doc_type, slot_list=slot_list,
                                         slot_hints=SLOT_HINTS))
            text = clean_and_validate(raw)
            if text:
                templates.append({"id": tid, "doc_type": doc_type, "text": text})
                tid += 1
                ok += 1
                save()                # salvataggio incrementale: niente perso se interrotto
                print(f"        -> OK ({len(SLOT_RE.findall(text))} segnaposto) "
                      f"| totale buoni: {len(templates)}")
            else:
                skip += 1

    save()
    print(f"\n{'='*60}")
    print(f"FATTO in {time.time()-t0:.0f}s | tentativi {total} | nuovi buoni {ok} | "
          f"scartati {skip} | template totali nel file: {len(templates)} (era {base})")
    print(f"Salvati -> {args.out}")
    print("Ora: python generate_synthetic_pii.py  (li carica e inietta i dati validi)")


if __name__ == "__main__":
    main()
