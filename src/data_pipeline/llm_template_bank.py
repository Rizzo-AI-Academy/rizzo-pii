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
import random
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
    # --- atti giudiziari civili ---
    "atto di citazione", "comparsa di costituzione e risposta", "sentenza civile",
    "decreto ingiuntivo", "contratto di locazione", "procura alle liti",
    "ricorso per decreto ingiuntivo", "verbale di udienza", "atto di diffida",
    "contratto di compravendita immobiliare",
    # --- oltre il tribunale ---------------------------------------------------------
    # Gli identificativi italiani (CF, P.IVA, catasto, IBAN, carta) compaiono anche fuori
    # dagli atti civili, e il modello e' debole proprio sulle classi aperte -- ORG in testa
    # (F1 .983 su 145 esempi di validazione), poi ZIPCODE/STREET/CITY. Questi domini portano
    # ORG in contesti dove e' il soggetto naturale (appalti, banche, assicurazioni, condominio)
    # e strutture di documento che gli atti civili non hanno (moduli, informative, capitolati).
    "capitolato di appalto pubblico di lavori",
    "perizia tecnica di stima immobiliare",
    "atto di successione e dichiarazione di eredita'",
    "contratto di lavoro subordinato a tempo indeterminato",
    "lettera di contestazione disciplinare a un dipendente",
    "verbale di assemblea condominiale",
    "denuncia di sinistro assicurativo con controparte",
    "contratto di mutuo ipotecario bancario",
    "informativa privacy e consenso al trattamento dei dati (GDPR)",
    "ricorso tributario alla Corte di Giustizia Tributaria",
    "atto notarile di costituzione di societa' a responsabilita' limitata",
    "istanza di accesso agli atti amministrativi",
    "denuncia-querela per uso fraudolento di una carta di credito",
    "reclamo all'Arbitro Bancario Finanziario per operazione non autorizzata su carta",
    "verbale di sommarie informazioni testimoniali",
    "atto di citazione per risarcimento danni da sinistro stradale",
    "ricorso al Giudice di Pace in opposizione a sanzione amministrativa",
    "consenso informato al trattamento sanitario",
    "verbale di identificazione di persona sottoposta a controllo",
    "domanda di ammissione a prestazione previdenziale",
    # --- fuori dal mondo legale ------------------------------------------------------
    # I tipi sopra sono tutti atti, verbali, ricorsi, contratti: un unico mondo
    # linguistico. Aggiungere l'ennesimo atto rende poco -- il modello ha gia' detto
    # quello che si dice in un atto -- mentre una bolletta, un cedolino o una multa hanno
    # parole, formule e impaginato che negli atti non esistono. Sono i documenti in cui un
    # italiano incontra davvero i propri dati personali.
    "bolletta di fornitura di energia elettrica",
    "cedolino della retribuzione mensile di un dipendente",
    "fattura elettronica di un professionista con ritenuta d'acconto",
    "verbale di accertamento di violazione del codice della strada",
    "dichiarazione sostitutiva di certificazione (autocertificazione)",
    "domanda di partecipazione a un concorso pubblico",
    "modulo di apertura di un conto corrente bancario",
    "contratto di attivazione di una linea telefonica mobile",
    "contratto di noleggio di un autoveicolo a breve termine",
    "certificato di iscrizione scolastica con scheda di valutazione",
    "richiesta di prenotazione di una prestazione ambulatoriale",
    "reclamo per rimborso a un vettore di trasporto passeggeri",
    "lettera di sollecito di pagamento a un cliente moroso",
    "modulo di adesione a un'associazione con addebito SEPA",
]

# quanti tipi campionare per run, quando non specificato
DEFAULT_DOC_TYPES = 10

# --- chi scrive, e in che lingua ------------------------------------------------------
# Il prompt diceva "Sei un giurista italiano ... nel linguaggio dei tribunali": giusto per
# un atto di citazione, sbagliato per una bolletta. Chiedere a un giurista di scrivere una
# bolletta produce una bolletta in legalese, cioe' di nuovo le strutture che si volevano
# evitare aprendo un dominio nuovo -- e la struttura e' esattamente cio' che il modello
# impara. Ogni voce qui sotto dice chi scrive e con che lessico: e' la differenza fra un
# tipo di documento nuovo per davvero e un atto travestito.
REGISTRO_LEGALE = (
    "un giurista italiano",
    "il linguaggio tecnico-giuridico autentico usato nei tribunali italiani")

REGISTRI = {
    "bolletta di fornitura di energia elettrica": (
        "un addetto alla fatturazione di una societa' di vendita di energia elettrica",
        "il linguaggio amministrativo delle bollette: intestatario della fornitura, "
        "indirizzo di fornitura, periodo di riferimento, letture del contatore, voci di "
        "spesa in elenco (energia, trasporto, oneri, imposte), totale da pagare, scadenza "
        "e modalita' di pagamento"),
    "cedolino della retribuzione mensile di un dipendente": (
        "un consulente del lavoro che compila un cedolino paga",
        "il linguaggio delle buste paga: dati del datore e del lavoratore, qualifica e "
        "livello, voci di competenza e di trattenuta in elenco, imponibile "
        "previdenziale, ritenute, netto in busta, accredito"),
    "fattura elettronica di un professionista con ritenuta d'acconto": (
        "un commercialista che emette una fattura",
        "il linguaggio della fatturazione: cedente e committente, numero e data del "
        "documento, descrizione della prestazione, compenso, cassa previdenza, IVA, "
        "ritenuta d'acconto, netto a pagare, riferimenti di pagamento"),
    "verbale di accertamento di violazione del codice della strada": (
        "un agente di polizia locale che redige un verbale",
        "il linguaggio degli accertamenti stradali: luogo e ora del rilievo, veicolo e "
        "targa, dati del trasgressore e del proprietario, norma violata, sanzione, "
        "decurtazione di punti, termini e modi per il pagamento e per il ricorso"),
    "dichiarazione sostitutiva di certificazione (autocertificazione)": (
        "un cittadino che compila un'autocertificazione allo sportello",
        "il linguaggio dei moduli della pubblica amministrazione: consapevole delle "
        "sanzioni penali, dichiara, elenco puntato di stati e qualita' personali, luogo e "
        "data, firma, informativa sul trattamento dei dati"),
    "domanda di partecipazione a un concorso pubblico": (
        "un candidato che compila la domanda di ammissione a un concorso",
        "il linguaggio delle domande di concorso: bando di riferimento, dati anagrafici e "
        "di residenza, titoli di studio, requisiti dichiarati, allegati, recapiti per le "
        "comunicazioni"),
    "modulo di apertura di un conto corrente bancario": (
        "un impiegato di banca che compila la scheda di apertura di un rapporto",
        "il linguaggio bancario dell'anagrafica cliente: dati identificativi e documento "
        "esibito, residenza e domicilio, professione, dichiarazioni antiriciclaggio, "
        "condizioni economiche del conto, coordinate del rapporto"),
    "contratto di attivazione di una linea telefonica mobile": (
        "un addetto di un operatore telefonico che compila un contratto di attivazione",
        "il linguaggio dei contratti di telefonia: intestatario, offerta e canone, "
        "numerazione attivata, portabilita' da altro operatore, durata e recesso, "
        "domiciliazione dei pagamenti, recapiti"),
    "contratto di noleggio di un autoveicolo a breve termine": (
        "un addetto al banco di una societa' di autonoleggio",
        "il linguaggio dei noleggi: conducente e patente, veicolo e targa, ritiro e "
        "riconsegna con data e ora, chilometraggio, franchigia e coperture, garanzia sulla "
        "carta di credito, stato del veicolo"),
    "certificato di iscrizione scolastica con scheda di valutazione": (
        "una segreteria scolastica che rilascia un certificato di iscrizione",
        "il linguaggio della scuola italiana: istituto e anno scolastico, alunno e classe "
        "frequentata, esercente la responsabilita' genitoriale, discipline e valutazioni, "
        "note sulla frequenza, uso del certificato"),
    "richiesta di prenotazione di una prestazione ambulatoriale": (
        "un operatore di sportello che registra una prenotazione al CUP",
        "il linguaggio amministrativo delle prenotazioni sanitarie: assistito e tessera, "
        "medico richiedente, prestazione richiesta con codice, classe di priorita', sede e "
        "appuntamento, quota di partecipazione o esenzione, disdetta. Nessun dato clinico: "
        "solo la parte amministrativa della prenotazione"),
    "reclamo per rimborso a un vettore di trasporto passeggeri": (
        "un passeggero che scrive all'assistenza clienti di una compagnia di trasporto",
        "il linguaggio dei reclami al servizio clienti: riferimento della prenotazione e "
        "del titolo di viaggio, tratta e orari, disservizio subito, spese sostenute, "
        "rimborso richiesto e coordinate per l'accredito, termini di risposta"),
    "lettera di sollecito di pagamento a un cliente moroso": (
        "un addetto all'amministrazione di un'impresa che scrive un sollecito",
        "il linguaggio dell'amministrazione commerciale: riferimento alle fatture "
        "insolute, scadenze superate, importo dovuto, invito al pagamento entro un "
        "termine, coordinate per il versamento, avvertimento sulle azioni successive"),
    "modulo di adesione a un'associazione con addebito SEPA": (
        "una segreteria associativa che raccoglie le adesioni dei soci",
        "il linguaggio dei moduli di adesione: dati del socio, quota e periodicita', "
        "mandato di addebito diretto SEPA con identificativo del creditore, consensi al "
        "trattamento dei dati e alle comunicazioni, firma e data"),
}


def registro(doc_type):
    """Chi scrive il documento e con che lessico. Fuori dai tipi elencati vale il registro
    legale: i tipi giudiziari e contrattuali sono nati con quel prompt."""
    return REGISTRI.get(doc_type, REGISTRO_LEGALE)


def sample_doc_types(n=DEFAULT_DOC_TYPES):
    """Sottoinsieme casuale dei tipi di documento (n=0 -> tutti).

    Con 30 tipi, scrivere un template per OGNI tipo a ogni run triplicherebbe il costo
    rispetto a prima -- e per chi usa Gemini il costo e' in euro. Campionarne 10 per run
    tiene il costo invariato e fa si' che contributori diversi coprano domini diversi:
    l'unione delle contribuzioni copre tutto senza che nessuno paghi per tutto."""
    if not n or n >= len(DOC_TYPES):
        return list(DOC_TYPES)
    return random.sample(DOC_TYPES, n)

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

                                        # "Scrivi un {doc_type}" non concorda con i tipi
                                        # femminili ("un bolletta", "un fattura"): il tipo
                                        # va dopo i due punti, senza articolo da azzeccare
PROMPT = """Sei {persona}. Scrivi questo documento, REALISTICO e completo (da 8 a 18 righe): {doc_type}.
Usa {stile}.

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


# vocabolario di parole con la maiuscola LEGITTIME nei testi legali italiani:
# se resta una maiuscola fuori da questo set (e fuori dai segnaposto), e' un nome
# proprio "sciolto" scritto inline dall'LLM -> PII non taggata -> template da scartare.
LEGAL_CAPITALIZED = set("""
Il La Lo Gli Le Un Una Uno Con Per Si Che Chi In Da Del Della Dei Delle Dello Al Alla
Ai Alle Allo Nel Nella Nei Sul Sulla Tra Fra A Ad E Ed O Od Ma Se Come Quando Dove
Mentre Inoltre Pertanto Quindi Tutto Tutti Tutte Questa Questo Questi Queste Tale Tali
Detto Detta Predetto Suddetto Premesso Considerato Visto Vista Visti Viste Letto Letti
Letta Rilevato Ritenuto Dichiara Dichiarano Chiede Chiedono Voglia Vogliano Cosi Così
Ciò Essendo Avendo Ove Salvo Fermo Nonche Nonché Ovvero Stante Affinche Affinché
Tribunale Corte Appello Cassazione Giudice Giudici Giudicante Avvocato Avvocati
Procuratore Procura Pubblico Ministero Repubblica Italiana Italia Stato Regione
Provincia Comune Codice Civile Penale Procedura Costituzione Legge Leggi Decreto Decreti
Articolo Art Comma Commi Capo Sezione Sez Ruolo Generale Causa Cause Sentenza Sentenze
Ordinanza Ordinanze Ricorso Ricorrente Comparsa Atto Atti Citazione Liti Foro Udienza
Cancelleria Cancelliere Spettabile Spett Egregio Egr Signor Signora Sig Sigg Dottor
Dottore Dott Avv Ill Illustrissimo Onorevole Oggetto Premessa Fatto Diritto Conclusioni
Motivi Domanda Eccezione Memoria Verbale Notaio Repertorio Raccolta Parte Parti Attore
Convenuto Resistente Testimone Teste Perito Curatore Fallimento Societa Società
Gennaio Febbraio Marzo Aprile Maggio Giugno Luglio Agosto Settembre Ottobre Novembre
Dicembre Lunedi Lunedì Martedi Martedì Mercoledi Mercoledì Giovedi Giovedì Venerdi
Venerdì Sabato Domenica Euro
Costituzionale Ufficiale Gazzetta Ordinario Ordinaria Suprema Supremo Amministrativo
Amministrativa Regionale Nazionale Europea Europeo Unione Agenzia Entrate Comunale
Provinciale Locatore Locatrice Conduttore Conduttrice Venditore Venditrice Acquirente
Promittente Promissario Canone Deposito Cauzione Cauzionale Durata Oneri Onere Catasto
Catastale Particella Foglio Subalterno Mappale Rendita Fabbricati Fabbricato Terreni
Immobile Immobili Bene Beni Prezzo Acconto Saldo Designato Designata Condanna Condannato
Deciso Decisa Firma Firme Regolamento Mandato Delega Comparente Comparenti Contraenti
Contraente Contrattuale Mensile Annuale Banca Filiale Bonifico Pagamento Pagamenti
Scadenza Interessi Interesse Capitale Allegato Allegati Documento Documenti Conto
Corrente Vi Voi Vostra Vostro Vostri Vostre Tanto Nondimeno
""".split())


# titoli che, seguiti da una maiuscola, segnalano un nome scritto inline
TITLES = {"Sig", "Sig.ra", "Sigg", "Signor", "Signora", "Dott", "Dottor", "Dottore",
          "Dssa", "Avv", "Egr", "Egregio", "Spett", "Ill", "Onorevole", "Spettabile"}


def _is_name_cap(w):
    """True se la parola e' una maiuscola 'da nome': iniziale maiuscola, non tutta
    maiuscola (no acronimi), e non un termine giuridico noto. Gestisce le elisioni
    (es. 'dell'Avv' -> valuta 'Avv')."""
    seg = w.split("'")[-1].strip(".")
    return bool(seg) and seg[0].isupper() and not seg.isupper() and seg not in LEGAL_CAPITALIZED


def find_stray_names(text):
    """Segnala SOLO i pattern che indicano un vero nome proprio scritto inline:
    due maiuscole 'da nome' consecutive (Nome Cognome) o titolo + maiuscola (Sig. Rossi).
    Le singole parole giuridiche maiuscole sono testo normale e NON vengono toccate."""
    masked = re.sub(r"\{\w+\}", " ", text)          # togli i segnaposto
    words = re.findall(r"[A-Za-zÀ-ÿ][A-Za-zÀ-ÿ'.]*", masked)
    stray = []
    for a, b in zip(words, words[1:]):
        if a and a[-1] in ".:;!?":                              # confine di frase: non e' un nome
            continue
        if _is_name_cap(a) and _is_name_cap(b):                 # "Mario Rossi"
            stray.append(f"{a} {b}")
        elif a.rstrip(".") in TITLES and _is_name_cap(b):       # "Sig. Bianchi"
            stray.append(f"{a} {b}")
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
    return text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-type", type=int, default=3, help="template per tipo documento")
    ap.add_argument("--doc-types", type=int, default=DEFAULT_DOC_TYPES,
                    help=f"quanti tipi di documento campionare per run "
                         f"(0 = tutti i {len(DOC_TYPES)})")
    ap.add_argument("--types", default=None,
                    help="genera solo per i tipi di documento che contengono uno di questi "
                         "pezzi di testo, separati da virgola (es. --types bolletta,multa). "
                         "Serve per coprire un dominio alla volta senza rigenerare il resto")
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

    if args.types:
        pezzi = [p.strip().lower() for p in args.types.split(",") if p.strip()]
        doc_types = [d for d in DOC_TYPES if any(p in d.lower() for p in pezzi)]
        if not doc_types:
            sys.exit(f"ERRORE: nessun tipo di documento contiene {pezzi}. "
                     f"Tipi disponibili:\n  " + "\n  ".join(DOC_TYPES))
    else:
        doc_types = sample_doc_types(args.doc_types)
    total = len(doc_types) * args.per_type
    done = ok = skip = 0
    t0 = time.time()
    print(f"Genero {total} template ({len(doc_types)} tipi su {len(DOC_TYPES)} "
          f"x {args.per_type}) con [{backend_name()}]\n")

    def save():
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(templates, f, ensure_ascii=False, indent=2)

    for doc_type in doc_types:
        for _ in range(args.per_type):
            done += 1
            elapsed = time.time() - t0
            eta = (elapsed / done) * (total - done) if done else 0
            pct = 100 * done // total
            print(f"[{done:>3}/{total} | {pct:>3}%] OK={ok} scartati={skip} | "
                  f"trascorso {elapsed:>4.0f}s | ETA {eta:>4.0f}s | {doc_type} ...")
            persona, stile = registro(doc_type)
            raw = call_llm(PROMPT.format(doc_type=doc_type, slot_list=slot_list,
                                         slot_hints=SLOT_HINTS,
                                         persona=persona, stile=stile))
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
