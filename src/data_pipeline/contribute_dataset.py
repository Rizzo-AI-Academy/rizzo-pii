#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Contribuisci dati sintetici al dataset community su Hugging Face.

    rizzoaiacademy/anonimizzazione-testi-italiano

Questo e' lo script che CHIUNQUE puo' eseguire per aiutare il progetto: genera
esempi PII sintetici (testo legale italiano + label BIO esatte, con checksum
matematicamente validi per CF/PIVA/IBAN) e li carica sul dataset come **Pull
Request**, cosi' un maintainer puo' revisionarli prima del merge.

Dati GENUINAMENTE NUOVI: con la TUA chiave Gemini lo script scrive NUOVI template
legali ad ogni esecuzione (prosa diversa, temperatura alta) e poi vi inietta i dati.
Cosi' ogni contributore produce testo nuovo, non solo nuovi valori sugli stessi
template. Principio "LLM autore, codice etichettatore" (CLAUDE.md / README.md):
l'LLM scrive solo la prosa con segnaposto, il codice inietta i dati -> label BIO
esatte, checksum validi, NESSUNA PII reale prodotta.

  ⚠️  NON contribuire MAI dati personali reali. Questo strumento esiste per
      proteggere le PII: gli esempi devono essere sempre sintetici.

--------------------------------------------------------------------------------
Prerequisiti
--------------------------------------------------------------------------------
  pip install -r requirements.txt              # include huggingface_hub
  hf auth login                                # oppure: export HF_TOKEN=hf_xxx
  export GEMINI_API_KEY=...                    # chiave Gemini (PowerShell: $env:GEMINI_API_KEY=...)
                                               # ottienila su https://aistudio.google.com/apikey

--------------------------------------------------------------------------------
Uso
--------------------------------------------------------------------------------
  # genera NUOVI template con Gemini + 5000 esempi e apre una PR sul dataset
  python src/data_pipeline/contribute_dataset.py --n 5000 --handle iltuonome

  # quanti NUOVI template per tipo di documento far scrivere a Gemini (default 2)
  python src/data_pipeline/contribute_dataset.py --n 5000 --handle iltuonome --per-type 3

  # solo locale, senza caricare nulla (per vedere cosa verrebbe inviato)
  python src/data_pipeline/contribute_dataset.py --n 2000 --handle iltuonome --no-upload

  # senza chiave Gemini: usa solo i template built-in (dati meno "nuovi")
  python src/data_pipeline/contribute_dataset.py --n 5000 --handle iltuonome --offline
"""

import argparse
import hashlib
import io
import json
import os
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "data_pipeline"))

# carica .env (GEMINI_API_KEY, eventuale HF_TOKEN) PRIMA di importare i moduli che
# leggono le env var a import-time (llm_template_bank legge GEMINI_API_KEY).
try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass

# riusa i generatori gia' nel repo (import => esegue random.seed(42); ri-seminiamo dopo).
# NB: llm_template_bank forza gia' UTF-8 su sys.stdout a import-time; NON ri-wrappare
# qui (un secondo TextIOWrapper, una volta GC-ato, chiuderebbe il buffer originale).
import generate_synthetic_pii as gen  # noqa: E402
import llm_template_bank as tb        # noqa: E402  (Gemini: scrittura nuovi template)

try:
    sys.stdout.reconfigure(encoding="utf-8")  # fallback se l'ordine di import cambia
except Exception:
    pass


# --- validatori checksum (stessi di src/inspect/validate_checksums.py, qui inline
#     per evitare l'effetto collaterale di import di quel modulo) -----------------
_ODD = {"0":1,"1":0,"2":5,"3":7,"4":9,"5":13,"6":15,"7":17,"8":19,"9":21,
        "A":1,"B":0,"C":5,"D":7,"E":9,"F":13,"G":15,"H":17,"I":19,"J":21,
        "K":2,"L":4,"M":18,"N":20,"O":11,"P":3,"Q":6,"R":8,"S":12,"T":14,
        "U":16,"V":10,"W":22,"X":25,"Y":24,"Z":23}


def iban_ok(i):
    r = i[4:] + i[:4]
    n = int("".join(str(ord(c) - 55) if c.isalpha() else c for c in r))
    return n % 97 == 1


def piva_ok(p):
    if len(p) != 11 or not p.isdigit():
        return False
    t = 0
    for i, c in enumerate(map(int, p[:10])):
        if i % 2 == 0:
            t += c
        else:
            x = c * 2
            t += x - 9 if x > 9 else x
    return (10 - t % 10) % 10 == int(p[10])


def cf_ok(c):
    if len(c) != 16:
        return False
    b = c[:15]
    t = sum((_ODD[ch] if i % 2 == 0 else (int(ch) if ch.isdigit() else ord(ch) - 65))
            for i, ch in enumerate(b))
    return chr(65 + t % 26) == c[15]


REPO_ID = "rizzoaiacademy/anonimizzazione-testi-italiano"
GENERATOR_VERSION = "1.0.0"   # versione del formato/generatore di questa contribuzione

MAX_N = 200_000               # tetto ragionevole per una singola contribuzione

# label "grezze" emesse dai generatori -> tag "coarse" su cui ragiona il contributore
# (rispecchia il TAG_MAP del training: ruoli/nome/cognome -> FULLNAME, ecc.)
BOOST_COARSE = {
    "GIVENNAME": "FULLNAME", "SURNAME": "FULLNAME", "GIUDICE": "FULLNAME",
    "AVVOCATO": "FULLNAME", "ATTORE": "FULLNAME", "CONVENUTO": "FULLNAME",
    "TESTIMONE": "FULLNAME", "IDCARDNUM": "ID_DOC", "DRIVERLICENSENUM": "ID_DOC",
    "PEC": "EMAIL", "CONTO": "IBAN", "RG": "DOCID",
}
# tag coarse -> segnaposto rappresentativi (per suggerire a Gemini di usarli spesso)
COARSE_TO_SLOT = {
    "FULLNAME": ["FULLNAME"], "ID_DOC": ["IDCARD", "DRIVING"], "IBAN": ["IBAN", "CONTO"],
    "DOCID": ["DOCID"], "ORG": ["ORG"], "CF": ["CF"], "PIVA": ["PIVA"],
    "CATASTO": ["CATASTO"], "AMOUNT": ["AMOUNT"], "TARGA": ["TARGA"],
    "EMAIL": ["EMAIL", "PEC"], "TELEPHONENUM": ["PHONE"], "PROVINCE": ["ADDRESS"],
    "ZIPCODE": ["ADDRESS"], "STREET": ["ADDRESS"], "CITY": ["CITY"], "DATE": ["DATE"],
}


def _coarse(label):
    return BOOST_COARSE.get(label, label)


def discover_slot_tags(samples=60):
    """Mappa segnaposto -> insieme dei tag coarse che produce (campionando i generatori)."""
    m = {}
    for name, fn in gen.SLOTS.items():
        tags = set()
        for _ in range(samples):
            for _txt, lbl in fn():
                if lbl:
                    tags.add(_coarse(lbl))
        m[name] = tags
    return m


def template_weights(templates, boost, slot_tags):
    """Peso di selezione per ogni template: il MAX dei boost dei tag che copre
    (cosi' i template che contengono i tag sotto-rappresentati vengono scelti piu' spesso)."""
    weights = []
    for t in templates:
        covered = set()
        for slot in gen.SLOT_RE.findall(t):
            covered |= slot_tags.get(slot, set())
        w = max((boost.get(tag, 1.0) for tag in covered), default=1.0)
        weights.append(w)
    return weights


def parse_boost(items):
    """--boost ORG=6 IBAN=4 -> {'ORG': 6.0, 'IBAN': 4.0}."""
    boost = {}
    for it in items or []:
        if "=" not in it:
            sys.exit(f"ERRORE: --boost vuole TAG=PESO, ricevuto '{it}'.")
        tag, val = it.split("=", 1)
        try:
            boost[tag.strip().upper()] = float(val)
        except ValueError:
            sys.exit(f"ERRORE: peso non numerico in '{it}'.")
    return boost


# --- tassonomia ammessa -------------------------------------------------------------
# ATTENZIONE, non ovvio: train_pii.py costruisce l'elenco delle label DAI DATI
# (`label_set` dall'unione train+validation, `num_labels=len(label_list)`), e
# normalize_labels() lascia passare invariata qualsiasi label che non sia in TAG_MAP.
# Conseguenza: una label sconosciuta in un file contribuito -- anche solo un errore di
# battitura come "CREDITCARD" o "FULLNAM" -- NON da' errore: diventa silenziosamente una
# CLASSE NUOVA del modello, con dati di training e zero support in validation. Qui la
# contribuzione viene fermata prima, cosi' un cambio di tassonomia resta una decisione
# esplicita dei maintainer e non un effetto collaterale di un upload.
TAG_FINALI = {
    "FULLNAME", "AGE", "GENDER", "DATE", "TIME", "STREET", "BUILDINGNUM", "ZIPCODE",
    "CITY", "PROVINCE", "EMAIL", "TELEPHONENUM", "CF", "PIVA", "ID_DOC", "IBAN",
    "CREDITCARDNUMBER", "AMOUNT", "TARGA", "ORG", "DOCID", "CATASTO",
}
# label "grezze" che TAG_MAP in train_pii.py sa rimappare su un tag finale
TAG_GREZZI = {
    "GIVENNAME", "SURNAME", "GIUDICE", "AVVOCATO", "ATTORE", "CONVENUTO", "TESTIMONE",
    "SEX", "TAXNUM", "PEC", "RG", "IDCARDNUM", "PASSPORTNUM", "DRIVERLICENSENUM",
    "SOCIALNUM", "CONTO",
    # codici di procedura/contratto: TAG_MAP li porta su DOCID (vedi TASSONOMIA_TAG.md)
    "CIG", "CUP", "POLIZZA", "MATRICOLA",
}
TAG_AMMESSI = TAG_FINALI | TAG_GREZZI


def _validate_record(rec):
    """Controlli di integrita' strutturale + checksum su una riga generata."""
    if len(rec["tokens"]) != len(rec["bio_labels"]):
        return "tokens/bio_labels di lunghezza diversa"
    for lab in rec["bio_labels"]:
        if lab != "O" and lab[2:] not in TAG_AMMESSI:
            return f"label BIO fuori tassonomia: '{lab}'"
    for e in rec["entities"]:
        # offset coerenti col testo
        if rec["source_text"][e["start"]:e["end"]] != e["value"]:
            return f"offset entita' incoerente: {e}"
        if e["label"] not in TAG_AMMESSI:
            return (f"label fuori tassonomia: '{e['label']}' (aggiungerebbe una classe "
                    f"nuova al modello; vedi docs/TASSONOMIA_TAG.md)")
        if e["label"] == "CF" and not cf_ok(e["value"]):
            return f"CF con checksum non valido: {e['value']}"
        if e["label"] == "PIVA" and not piva_ok(e["value"]):
            return f"PIVA con checksum non valido: {e['value']}"
        if e["label"] == "IBAN" and not iban_ok(e["value"]):
            return f"IBAN con checksum non valido: {e['value']}"
    return None


def gen_new_templates(per_type, boost):
    """Fa scrivere a Gemini NUOVI template legali (prosa con soli segnaposto).

    Ritorna la lista dei testi-template validi. Ogni esecuzione produce prosa
    diversa (temperatura alta) -> dati genuinamente nuovi. Scarta i template con
    PII inline o segnaposto non gestiti (stessa validazione di llm_template_bank).
    Se 'boost' e' presente, chiede a Gemini di usare spesso i segnaposto dei tag
    potenziati -> piu' esempi per i tag sotto-rappresentati."""
    if not tb.have_backend():
        sys.exit("ERRORE: nessun backend LLM configurato.\n"
                 "  (a) LLM LOCALE, senza chiave e senza far uscire nulla dalla macchina:\n"
                 "    export LLM_BASE_URL=http://127.0.0.1:8080/v1   # llama.cpp / Ollama / vLLM\n"
                 "    export LLM_MODEL=nome-del-modello\n"
                 "  (b) Gemini: chiave su https://aistudio.google.com/apikey e poi:\n"
                 "    export GEMINI_API_KEY=...        (PowerShell: $env:GEMINI_API_KEY=...)\n"
                 "  Oppure usa --offline per generare dai soli template built-in.")

    slot_list = "\n".join(f"  {{{s}}}" for s in sorted(tb.ALLOWED_SLOTS))
    # suggerimento mirato: i segnaposto dei tag potenziati, da usare spesso
    hint = ""
    boost_slots = sorted({s for tag in boost for s in COARSE_TO_SLOT.get(tag, [])})
    if boost_slots:
        hint = ("\n\nIMPORTANTE: in questo documento usa PIU' VOLTE, in modo naturale, "
                "i seguenti segnaposto: " + " ".join(f"{{{s}}}" for s in boost_slots) + ".")

    total = len(tb.DOC_TYPES) * per_type
    print(f"Scrivo {total} NUOVI template con [{tb.backend_name()}] "
          f"({len(tb.DOC_TYPES)} tipi x {per_type}) ...")

    out, done, ok = [], 0, 0
    for doc_type in tb.DOC_TYPES:
        for _ in range(per_type):
            done += 1
            prompt = tb.PROMPT.format(doc_type=doc_type, slot_list=slot_list,
                                      slot_hints=tb.SLOT_HINTS) + hint
            text = tb.clean_and_validate(tb.call_llm(prompt))
            if text:
                out.append(text)
                ok += 1
            print(f"  [{done:>3}/{total}] {doc_type:42s} {'OK' if text else 'scartato'}")
    print(f"Template nuovi validi: {ok}/{total}")
    if not out:
        sys.exit("ERRORE: nessun template valido dal modello. Riprova o usa --offline.")
    return out


def text_key(rec):
    """Impronta del testo esatto. Serve solo a non scrivere due volte la stessa riga
    identica: con valori pescati a caso non capita quasi mai, ma costa poco escluderlo."""
    return hashlib.blake2b(rec["source_text"].encode(), digest_size=8).digest()


def structure_key(rec):
    """Impronta della STRUTTURA della riga: template di origine + sequenza delle label
    nell'ordine in cui compaiono. E' la grana giusta per dire quante cose diverse c'e'
    dentro un file, e quindi quella su cui mettere un tetto.

    Non lo scheletro (il testo con le entita' sostituite dalle label): misurato, lo
    scheletro punisce i template migliori. Un template di bolletta, in cui OGNI valore
    iniettato e' etichettato, produce 1 solo scheletro su 500 righe -- tutte le sue righe
    hanno lo stesso testo a meno delle label -- mentre un atto che contiene {TRIBUNAL}, il
    cui valore iniettato NON e' etichettato, ne produce 343 perche' cambia il nome del
    tribunale. Filtrando per scheletro il primo avrebbe diritto a una riga e il secondo a
    venticinque: esattamente al rovescio.

    A parita' di struttura le righe hanno nomi, date, importi e indirizzi tutti diversi, ed
    e' variazione utile: insegna al modello che la label non dipende dal valore. Utile a
    dosi piccole, zavorra a dosi grandi -- da qui il tetto."""
    labels = "|".join(e["label"] for e in sorted(rec["entities"], key=lambda e: e["start"]))
    return hashlib.blake2b(f"{rec['template_id']}#{labels}".encode(),
                           digest_size=8).digest()


# Rifiuti consecutivi dopo i quali un template si considera esaurito e smette di essere
# pescato. Senza questo l'ultima parte della generazione e' quasi tutta lavoro buttato: i
# template le cui strutture sono gia' piene continuano a uscire (sono la maggioranza) e ogni
# loro riga viene costruita per essere scartata. Con l'uscita per template la generazione si
# ferma da sola quando sono esauriti tutti, e il numero di righe e' quello che la banca sa
# dare. La soglia e' un compromesso: troppo bassa dichiara esaurito un template che avrebbe
# ancora strutture rare, troppo alta spreca tentativi.
STOP_TEMPLATE = 2_000


def generate(n, seed, handle, per_type, offline, boost, out_path=None,
             max_per_structure=25):
    """Genera esempi sintetici tenendo solo le righe che aggiungono qualcosa.

    Perche' non semplicemente n righe: il testo di ogni riga e' unico (i valori cambiano
    sempre), quindi "0 duplicati" non dice niente sulla varieta'. Misurato su una
    contribuzione da 200.000 righe con 273 template: 57.093 scheletri distinti, cioe' il 71%
    del file ripeteva una riga gia' presente -- 800 MB di zavorra che allunga ogni download
    e ogni epoca di training senza insegnare niente di nuovo.

    Il tetto sta sulla STRUTTURA (template + sequenza delle label): al massimo
    max_per_structure righe possono condividerla. A parita' di struttura le righe hanno
    nomi, date, importi e indirizzi tutti diversi -- variazione utile al modello, che
    impara che la label non dipende dal valore -- ma utile a dosi piccole. Con 1 si tiene
    una riga per struttura, e il file diventa il puro inventario di cio' che la banca sa
    dire.

    Se la banca si esaurisce prima di arrivare a n, la generazione si ferma e lo dichiara:
    il numero di righe consegnabili e' un RISULTATO della banca, non una scelta di chi
    genera.

    Con out_path le righe vengono SCRITTE MANO A MANO invece di essere accumulate in
    memoria. Serve: MAX_N e' 200.000 e con template lunghi (documenti interi, non frasi)
    una riga pesa ~5 KB, quindi l'accumulo arriva a diversi GB di oggetti Python e la
    macchina va in swap prima di scrivere il primo byte. In streaming la memoria resta
    costante qualunque sia --n.

    Ritorna (righe_o_None, conteggi, n_template_nuovi, n_valide, n_da_template_nuovi):
    con out_path il primo elemento e' None (le righe sono sul file)."""
    # IMPORTANTE: ri-seminiamo DOPO l'import (gen imposta seed(42) a import-time).
    # Seed diverso per contributore => valori diversi => no duplicati nel dataset.
    random.seed(seed)

    new_templates = [] if offline else gen_new_templates(per_type, boost)
    # i built-in garantiscono comunque la copertura dei tag rari (CATASTO/DOCID/CONTO...);
    # la NOVITA' del testo viene dai template freschi di Gemini.
    templates = new_templates + gen.TEMPLATES + gen.load_external_templates()
    print(f"\nTemplate nel pool: {len(templates)} "
          f"({len(new_templates)} nuovi da {tb.backend_name()} + {len(gen.TEMPLATES)} built-in + "
          f"{len(templates) - len(new_templates) - len(gen.TEMPLATES)} locali)")

    # selezione pesata: i template che coprono i tag potenziati escono piu' spesso
    weights = template_weights(templates, boost, discover_slot_tags()) if boost else None
    if boost:
        print(f"Boost distribuzione tag: {boost}")
    idx_pool = list(range(len(templates)))

    n_new = len(new_templates)
    rows, label_counts, bad = ([] if out_path is None else None), {}, 0
    n_ok = n_da_nuovi = 0
    testi = set()                  # testi gia' scritti: nessuna riga identica due volte
    strutture = {}                 # struttura -> quante righe gia' scritte
    rip_testo = rip_struttura = tentativi = 0
    rifiuti = {}                   # tid -> rifiuti consecutivi (per l'uscita del template)
    attivi = list(idx_pool)
    pesi = list(weights) if weights else None
    out = None
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out = open(out_path, "w", encoding="utf-8")
    def esaurisci(tid):
        """Toglie dal sorteggio un template che non da' piu' niente di nuovo."""
        nonlocal attivi, pesi
        posizione = attivi.index(tid)
        attivi = attivi[:posizione] + attivi[posizione + 1:]
        if pesi is not None:
            pesi = pesi[:posizione] + pesi[posizione + 1:]

    def rifiuta(tid):
        rifiuti[tid] = rifiuti.get(tid, 0) + 1
        if rifiuti[tid] >= STOP_TEMPLATE:
            esaurisci(tid)

    while n_ok < n and attivi:
        tentativi += 1
        tid = random.choices(attivi, weights=pesi, k=1)[0]
        text, entities = gen.build_example(tid, templates)
        tokens, bio = gen.to_bio(text, entities)
        rec = {
            "source_text": text,
            "language": "it",
            "template_id": tid,
            "entities": entities,
            "tokens": tokens,
            "bio_labels": bio,
            # provenienza: sopravvive a merge/split, ignorata dal loader di training
            "meta": {"contributor": handle, "seed": seed,
                     "generator_version": GENERATOR_VERSION, "synthetic": True,
                     # True se la riga usa un template NUOVO scritto da Gemini in questo run
                     "new_template": tid < n_new},
        }
        err = _validate_record(rec)
        if err:
            bad += 1
            rifiuta(tid)
            continue
        testo = text_key(rec)
        if testo in testi:                          # la stessa riga identica, di nuovo
            rip_testo += 1
            rifiuta(tid)
            continue
        struttura = structure_key(rec)
        if strutture.get(struttura, 0) >= max_per_structure:
            rip_struttura += 1                      # questa struttura ha gia' le sue righe
            rifiuta(tid)
            continue
        testi.add(testo)
        strutture[struttura] = strutture.get(struttura, 0) + 1
        rifiuti[tid] = 0
        for e in entities:
            label_counts[e["label"]] = label_counts.get(e["label"], 0) + 1
        n_ok += 1
        n_da_nuovi += rec["meta"]["new_template"]
        if out is None:
            rows.append(rec)
        else:
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if n_ok % 25_000 == 0:
                print(f"  scritte {n_ok} righe ...", flush=True)
    if out is not None:
        out.close()

    if bad:
        print(f"  scartati {bad} esempi non validi (self-check)")
    print(f"  righe tenute: {n_ok} su {tentativi} tentativi "
          f"({100 * n_ok / max(1, tentativi):.1f}%) | scartate {rip_testo} righe identiche "
          f"a una gia' scritta e {rip_struttura} oltre il tetto di {max_per_structure} "
          f"per struttura")
    print(f"  strutture di etichette distinte: {len(strutture)} "
          f"({n_ok / max(1, len(strutture)):.1f} righe per struttura)")
    if n_ok < n:
        print(f"  BANCA ESAURITA: chieste {n} righe, ottenute {n_ok}. Tutti i "
              f"{len(templates)} template hanno smesso di produrre righe nuove "
              f"({STOP_TEMPLATE} rifiuti di fila ciascuno).\n  Per averne di piu' serve "
              f"allargare la banca di template (llm_template_bank.py), non alzare --n.")
    return rows, label_counts, n_new, n_ok, n_da_nuovi


def upload_pr(local_path, path_in_repo, handle, n, seed, repo_id):
    """Carica il file aprendo una Pull Request sul dataset HF."""
    try:
        from huggingface_hub import HfApi
    except ImportError:
        sys.exit("ERRORE: huggingface_hub non installato. Esegui: pip install -r requirements.txt")

    # token esplicito da HF_TOKEN (vince su un eventuale login in cache read-only)
    api = HfApi(token=os.environ.get("HF_TOKEN") or None)
    try:
        who = api.whoami()  # usa il token di `hf auth login` o HF_TOKEN
    except Exception:
        sys.exit("ERRORE: non sei autenticato su Hugging Face.\n"
                 "  Esegui:  hf auth login        (oppure: export HF_TOKEN=hf_xxx)")
    user = who.get("name", "?")
    print(f"Autenticato come: {user}")

    commit = (f"Contributo dati sintetici: {n} esempi (handle={handle}, seed={seed}, "
              f"gen v{GENERATOR_VERSION})")
    print(f"Apro una Pull Request su {repo_id} ...")
    res = api.upload_file(
        path_or_fileobj=str(local_path),
        path_in_repo=path_in_repo,
        repo_id=repo_id,
        repo_type="dataset",
        commit_message=commit,
        commit_description=(
            "Dati 100% sintetici generati con src/data_pipeline/contribute_dataset.py "
            "(principio 'LLM autore, codice etichettatore', checksum CF/PIVA/IBAN validi). "
            "Nessuna PII reale."),
        create_pr=True,
    )
    url = getattr(res, "pr_url", None) or getattr(res, "commit_url", None) or res
    print("\n✅ Pull Request creata. Un maintainer la revisionera' e la unira'.")
    print(f"   {url}")


def main():
    ap = argparse.ArgumentParser(
        description="Genera dati PII sintetici e contribuiscili al dataset HF (come PR).")
    ap.add_argument("-n", "--n", type=int, default=5000,
                    help="numero di esempi da generare (default 5000)")
    ap.add_argument("--handle", default=None,
                    help="il tuo nickname/handle (per tracciare il contributo)")
    ap.add_argument("--seed", type=int, default=None,
                    help="seed RNG (default: casuale -> dati diversi da altri contributori)")
    ap.add_argument("--per-type", type=int, default=2,
                    help="quanti NUOVI template per tipo documento far scrivere a Gemini (default 2)")
    ap.add_argument("--boost", nargs="*", metavar="TAG=PESO",
                    help="rinforza i tag sotto-rappresentati, es. --boost ORG=6 IBAN=4 CF=4")
    ap.add_argument("--offline", action="store_true",
                    help="non usare Gemini: genera dai soli template built-in (dati meno nuovi)")
    ap.add_argument("--no-upload", action="store_true",
                    help="genera solo in locale, non apre la PR")
    ap.add_argument("--upload-file", metavar="PATH",
                    help="non rigenerare: carica un .jsonl gia' prodotto (push veloce, niente Gemini)")
    ap.add_argument("--max-per-structure", type=int, default=25, metavar="K",
                    help="quante righe al massimo possono condividere la stessa STRUTTURA "
                         "(template + sequenza delle label). Default 25: qualche variante di "
                         "valori insegna al modello che la label non dipende dal valore, "
                         "oltre e' volume. Con 1 il file e' l'inventario delle strutture")
    ap.add_argument("--repo", default=REPO_ID, help="dataset di destinazione (override)")
    args = ap.parse_args()

    # scorciatoia: ri-carica un file gia' generato (utile per ritentare il push)
    if args.upload_file:
        p = Path(args.upload_file)
        if not p.is_file():
            sys.exit(f"ERRORE: file inesistente: {p}")
        n = sum(1 for _ in open(p, encoding="utf-8"))
        print(f"Carico file esistente ({n} righe): {p}")
        upload_pr(p, f"contributions/{p.name}", "upload", n, "-", args.repo)
        return

    if args.n < 1 or args.n > MAX_N:
        sys.exit(f"ERRORE: --n deve essere tra 1 e {MAX_N}.")
    if not args.handle:
        sys.exit("ERRORE: --handle obbligatorio (es. --handle iltuonome).")
    handle = "".join(c for c in args.handle if c.isalnum() or c in "-_").strip("-_")
    if not handle:
        sys.exit("ERRORE: --handle non valido (usa lettere/numeri/-/_).")

    seed = args.seed if args.seed is not None else random.SystemRandom().randrange(2**31)
    boost = parse_boost(args.boost)

    print("=" * 70)
    print("rizzo-pii — contribuzione dati sintetici al dataset community")
    print("⚠️  Solo dati SINTETICI: nessuna PII reale viene prodotta o caricata.")
    print("=" * 70)
    mode = ("built-in (offline)" if args.offline
            else f"{tb.backend_name()} (--per-type {args.per_type})")
    print(f"handle={handle}  n={args.n}  seed={seed}  template={mode}")

    # il nome definitivo contiene il numero di righe VALIDE, che si sa solo alla fine:
    # si scrive su un temporaneo e si rinomina.
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    tmp_path = ROOT / "dataset" / "contributions" / f".{handle}-{stamp}.parziale.jsonl"
    _, counts, n_new, n_ok, from_new = generate(
        args.n, seed, handle, args.per_type, args.offline, boost, out_path=tmp_path,
        max_per_structure=args.max_per_structure)
    if n_new:
        print(f"\n{from_new}/{n_ok} esempi da template NUOVI di {tb.backend_name()}.")
    print(f"Generati {n_ok} esempi validi. Entita' per label:")
    for label, c in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"  {label:18s} {c}")

    fname = f"{handle}-{stamp}-seed{seed}-n{n_ok}.jsonl"
    local_path = tmp_path.parent / fname
    tmp_path.replace(local_path)
    print(f"\nScritto in locale -> {local_path}")

    if args.no_upload:
        print("\n(--no-upload) Nessun caricamento. Rilancia senza --no-upload per aprire la PR.")
        return

    upload_pr(local_path, f"contributions/{fname}", handle, n_ok, seed, args.repo)


if __name__ == "__main__":
    main()
