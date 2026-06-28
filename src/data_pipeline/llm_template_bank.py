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
import urllib.request
import urllib.error
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", write_through=True)

# segnaposto consentiti = tassonomia coperta dagli iniettori in generate_synthetic_pii.py
ALLOWED_SLOTS = {
    "FULLNAME", "JUDGE", "LAWYER", "PLAINTIFF", "DEFENDANT", "WITNESS",
    "CF", "PIVA", "IBAN", "ADDRESS", "EMAIL", "PEC", "PHONE", "AMOUNT",
    "RG", "TRIBUNAL", "TARGA", "IDCARD", "DRIVING", "CITY", "DATE",
    "ORG", "DOCID", "CATASTO", "CONTO",
}

# breve legenda per i segnaposto il cui uso non e' ovvio dal nome (guida l'LLM a
# posizionarli nel contesto giusto, in molti documenti diversi -> varieta' strutturale)
SLOT_HINTS = """  {ORG}     = ragione sociale di una societa'/studio legale/banca (la PARTE, non il tribunale)
  {DOCID}   = codice identificativo di un atto: n. protocollo, n. repertorio/raccolta, n. sentenza
  {CATASTO} = dati catastali di un immobile (foglio, particella, subalterno)
  {CONTO}   = numero di conto corrente (diverso dall'IBAN)"""

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


def call_gemini(prompt, retries=3):
    key = API_KEY or os.environ.get("GEMINI_API_KEY")
    if not key:
        sys.exit("ERRORE: imposta API_KEY nel file oppure la variabile d'ambiente "
                 "GEMINI_API_KEY (chiave NUOVA, non quella incollata in chat).")
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


def clean_and_validate(text):
    """Pulisce il markdown, verifica i segnaposto e scarta i template con PII inline."""
    if not text:
        return None
    text = re.sub(r"^```.*?\n|```$", "", text.strip(), flags=re.MULTILINE).strip()
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
          f"con Gemini [{MODEL}]\n")

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
            raw = call_gemini(PROMPT.format(doc_type=doc_type, slot_list=slot_list,
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
