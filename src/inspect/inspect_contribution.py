#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Ispeziona un file di contribuzione .jsonl: e' valido, e porta informazione o volume?

    python src/inspect/inspect_contribution.py FILE.jsonl
    python src/inspect/inspect_contribution.py FILE.jsonl --json        # per la coda
    python src/inspect/inspect_contribution.py FILE.jsonl --contro ALTRO.jsonl

Perche' serve, e a chi. Sul dataset comunitario le Pull Request di dati si accumulano piu' in
fretta di quanto si riesca a leggerle, e il numero di righe -- l'unica cosa che il nome del file
dichiara -- non dice se un contributo insegna qualcosa: 200.000 righe scritte da pochi template
ripetono la stessa struttura centinaia di volte, e a valle costano download e tempo di training
senza portare varieta'. Chi revisiona non ha modo di distinguerle da un file piccolo e ricco
senza rifare la misura a mano, file per file. Questo script la fa in pochi secondi, e con --json
si esegue su tutta la coda per ordinarla.

Due cose insieme, perche' separate non bastano:

  1. ERRORI DURI: ricalcola tutto da zero -- tokenizzazione, offset, coerenza BIO, checksum,
     tassonomia -- senza fidarsi dello script che ha generato il file. Un file che sbaglia qui
     non va unito: le label sbagliate insegnano al modello a sbagliare, e una label fuori
     tassonomia diventa in silenzio una classe nuova (train_pii.py costruisce l'elenco DAI
     DATI). Con errori duri l'uscita e' 1, cosi' vale da cancello in uno script.

  2. INFORMAZIONE: quante cose DIVERSE contiene. La grana giusta e' la STRUTTURA -- template di
     origine piu' sequenza delle label -- non il testo: i valori cambiano sempre per costruzione,
     quindi "0 duplicati" e' vero anche in un file che ripete una frase sola.

Sulla differenza fra struttura e scheletro (il testo con le entita' sostituite dalla label) c'e'
una misura da tenere presente, perche' e' controintuitiva: lo scheletro PUNISCE i template
migliori. Un template di bolletta in cui ogni valore iniettato e' etichettato produce 1 solo
scheletro su 500 righe; un atto che contiene un nome di tribunale NON etichettato ne produce
343, perche' basta che cambi quel nome. Ordinando per scheletri distinti il secondo sembra
quattro volte piu' ricco del primo, ed e' il contrario. Percio' qui la struttura si usa quando
il file porta template_id, e lo scheletro resta come ripiego dichiarato per i file che non ce
l'hanno.

Sola lettura: non modifica nulla, non carica nulla, e senza --confronta non tocca la rete.
La memoria non dipende dal peso del file: dei testi si tengono solo impronte brevi.
"""
import argparse
import hashlib
import io
import json
import re
import statistics
import sys
import unicodedata
import urllib.parse
import urllib.request
from collections import Counter
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", write_through=True)

ROOT = Path(__file__).resolve().parents[2]
REPO_ID = "rizzoaiacademy/anonimizzazione-testi-italiano"

# la stessa tokenizzazione dichiarata dal formato (docs/FORMATO_DATI.md) e usata da
# generate_synthetic_pii.to_bio(): se i token del file non si riproducono con questa, le
# bio_labels non sono allineate a niente di verificabile.
TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)

# Ripiego se le fonti nel repo non sono leggibili (file eseguito fuori dall'albero).
TAG_DI_SCORTA = {
    "FULLNAME", "AGE", "GENDER", "DATE", "TIME", "STREET", "BUILDINGNUM", "ZIPCODE", "CITY",
    "PROVINCE", "EMAIL", "TELEPHONENUM", "CF", "PIVA", "ID_DOC", "IBAN", "CREDITCARDNUMBER",
    "AMOUNT", "TARGA", "ORG", "DOCID", "CATASTO",
}


def tassonomia():
    """Le label ammesse, LETTE dal repo invece di riscritte qui.

    Una lista scritta a mano in questo file resterebbe indietro in silenzio: aggiungere forme
    nuove sotto un tag esistente (CIG, CUP, polizza, matricola -> DOCID) tocca solo TAG_MAP, e
    l'ispettore le segnalerebbe come fuori tassonomia pur essendo corrette -- cioe' bloccherebbe
    contributi buoni, che e' il danno peggiore per uno strumento di revisione.

    Si legge con una regex e non con un import: train_pii.py importa torch e transformers, che
    non c'entrano nulla con l'ispezione di un .jsonl e non sono installati nella CI dei test;
    contribute_dataset.py, importato, carica il .env e semina il generatore casuale.
    """
    ammesse, fonti = set(), []
    contrib = ROOT / "src" / "data_pipeline" / "contribute_dataset.py"
    if contrib.is_file():
        testo = contrib.read_text(encoding="utf-8")
        for nome in ("TAG_FINALI", "TAG_GREZZI"):
            m = re.search(nome + r"\s*=\s*\{(.*?)\}", testo, re.S)
            if m:
                ammesse |= set(re.findall(r'"([A-Z0-9_]+)"', m.group(1)))
                fonti.append(f"{nome} da contribute_dataset.py")
    train = ROOT / "src" / "training" / "train_pii.py"
    if train.is_file():
        m = re.search(r"^TAG_MAP\s*=\s*\{(.*?)^\}", train.read_text(encoding="utf-8"), re.S | re.M)
        if m:
            corpo = re.sub(r"#[^\n]*", "", m.group(1))          # via i commenti
            ammesse |= set(re.findall(r'"([A-Z0-9_]+)"\s*:', corpo))
            ammesse |= set(re.findall(r':\s*"([A-Z0-9_]+)"', corpo))
            fonti.append("TAG_MAP da train_pii.py")
    if not ammesse:
        return TAG_DI_SCORTA, ["lista di scorta interna (fonti del repo non leggibili)"]
    return ammesse, fonti


# --- checksum: gli stessi del repo, ricalcolati qui per non importare nulla ---------------
_ODD = {"0": 1, "1": 0, "2": 5, "3": 7, "4": 9, "5": 13, "6": 15, "7": 17, "8": 19, "9": 21,
        "A": 1, "B": 0, "C": 5, "D": 7, "E": 9, "F": 13, "G": 15, "H": 17, "I": 19, "J": 21,
        "K": 2, "L": 4, "M": 18, "N": 20, "O": 11, "P": 3, "Q": 6, "R": 8, "S": 12, "T": 14,
        "U": 16, "V": 10, "W": 22, "X": 25, "Y": 24, "Z": 23}


def cf_ok(c):
    c = c.replace(" ", "").upper()
    if len(c) != 16:
        return False
    try:
        t = sum((_ODD[ch] if i % 2 == 0 else (int(ch) if ch.isdigit() else ord(ch) - 65))
                for i, ch in enumerate(c[:15]))
    except KeyError:
        return False
    return chr(65 + t % 26) == c[15]


def piva_ok(p):
    p = p.replace(" ", "")
    if len(p) != 11 or not p.isdigit():
        return False
    t = 0
    for i, c in enumerate(map(int, p[:10])):
        t += c if i % 2 == 0 else (c * 2 - 9 if c * 2 > 9 else c * 2)
    return (10 - t % 10) % 10 == int(p[10])


def iban_ok(i):
    i = i.replace(" ", "").upper()
    if not re.fullmatch(r"IT\d{2}[A-Z]\d{22}", i):
        return False
    r = i[4:] + i[:4]
    return int("".join(str(ord(c) - 55) if c.isalpha() else c for c in r)) % 97 == 1


def luhn_ok(n):
    if any(c not in " -." for c in n if not c.isdigit()):
        return False
    d = [int(c) for c in n if c.isdigit()][::-1]
    if len(d) < 12:
        return False
    return sum(x if i % 2 == 0 else (x * 2 - 9 if x * 2 > 9 else x * 2)
               for i, x in enumerate(d)) % 10 == 0


CHECKSUM = {"CF": cf_ok, "PIVA": piva_ok, "TAXNUM": piva_ok, "IBAN": iban_ok,
            "CREDITCARDNUMBER": luhn_ok}


def impronta(testo):
    """Impronta breve. Su un file da un milione di righe i testi interi non stanno in memoria,
    e per contare i distinti bastano 8 byte per riga."""
    return hashlib.blake2b(testo.encode(), digest_size=8).digest()


def primo_non_latino(testo):
    """Un modello locale piccolo o molto quantizzato infila ideogrammi nella prosa italiana:
    se il template e' passato, quel carattere e' in ogni riga che lo usa."""
    for ch in testo:
        if ch.isalpha():
            try:
                if "LATIN" not in unicodedata.name(ch):
                    return ch
            except ValueError:
                return ch
    return None


def struttura(rec):
    """template_id + sequenza delle label: la grana su cui si misura la varieta'."""
    labels = "|".join(e["label"] for e in sorted(rec["entities"], key=lambda e: e["start"]))
    return impronta(f"{rec.get('template_id')}#{labels}")


def scheletro(rec):
    """Il testo con ogni entita' sostituita dalla sua label (ripiego senza template_id)."""
    testo, pezzi, pos = rec["source_text"], [], 0
    for e in sorted(rec["entities"], key=lambda e: e["start"]):
        if e["start"] < pos:
            continue
        pezzi.append(testo[pos:e["start"]])
        pezzi.append("{" + e["label"] + "}")
        pos = e["end"]
    pezzi.append(testo[pos:])
    return impronta(" ".join("".join(pezzi).split()))


def ispeziona(path, ammesse):
    """Un solo passaggio sul file. Ritorna un dizionario di numeri e un Counter di errori."""
    errori, label = Counter(), Counter()
    strutture, scheletri, testi = Counter(), Counter(), set()
    n = con_template_id = lung_tot = ent_tot = 0
    lung_max = 0
    for numero, riga in enumerate(open(path, encoding="utf-8"), 1):
        if not riga.strip():
            continue
        try:
            r = json.loads(riga)
        except json.JSONDecodeError as e:
            errori[f"JSON non valido (riga {numero}): {e.msg}"] += 1
            continue
        n += 1
        testo = r.get("source_text")
        if not isinstance(testo, str):
            errori["source_text mancante o non testo"] += 1
            continue
        for campo in ("language", "tokens", "bio_labels", "entities"):
            if campo not in r:
                errori[f"campo mancante: {campo}"] += 1
        if r.get("language") != "it":
            errori[f"language != it: {r.get('language')!r}"] += 1
        if r.get("tokens") is not None and TOKEN_RE.findall(testo) != r["tokens"]:
            errori["tokens non riproducibili dalla regex del formato"] += 1
        if len(r.get("tokens") or []) != len(r.get("bio_labels") or []):
            errori["len(tokens) != len(bio_labels)"] += 1

        entita = r.get("entities") or []
        for e in entita:
            if testo[e["start"]:e["end"]] != e["value"]:
                errori["offset dell'entita' incoerente col testo"] += 1
            if e["label"] not in ammesse:
                errori[f"label fuori tassonomia: {e['label']}"] += 1
            label[e["label"]] += 1
            verifica = CHECKSUM.get(e["label"])
            if verifica and not verifica(e["value"]):
                errori[f"checksum non valido ({e['label']})"] += 1
        ordinate = sorted(entita, key=lambda e: e["start"])
        for a, b in zip(ordinate, ordinate[1:]):
            if a["end"] > b["start"]:
                errori["entita' sovrapposte"] += 1

        atteso = None
        for lab in r.get("bio_labels") or []:
            if lab == "O":
                atteso = None
                continue
            if not re.fullmatch(r"[BI]-\S+", lab):
                errori[f"label BIO malformata: {lab}"] += 1
                atteso = None
                continue
            if lab.startswith("I-") and atteso != lab[2:]:
                errori["I- senza B- dello stesso tag"] += 1
            atteso = lab[2:]

        bad = primo_non_latino(testo)
        if bad:
            errori[f"carattere non latino nel testo: {bad!r}"] += 1

        testi.add(impronta(testo))
        scheletri[scheletro(r)] += 1
        if r.get("template_id") is not None:
            con_template_id += 1
            strutture[struttura(r)] += 1
        lung_tot += len(testo)
        lung_max = max(lung_max, len(testo))
        ent_tot += len(entita)

    grana = "struttura" if con_template_id == n and n else "scheletro"
    gruppi = strutture if grana == "struttura" else scheletri
    conteggi = sorted(gruppi.values(), reverse=True) or [0]
    return {
        "righe": n,
        "grana": grana,
        "grana_plurale": "strutture" if grana == "struttura" else "scheletri",
        "gruppi_distinti": len(gruppi),
        "righe_per_gruppo_mediana": statistics.median(conteggi),
        "righe_per_gruppo_max": conteggi[0],
        "quota_nei_10_gruppi_piu_frequenti": (sum(conteggi[:10]) / n) if n else 0,
        "scheletri_distinti": len(scheletri),
        "testi_duplicati": n - len(testi),
        "entita": sum(label.values()),
        "label_distinte": len(label),
        "label": dict(label.most_common()),
        "lunghezza_media": (lung_tot / n) if n else 0,
        "lunghezza_max": lung_max,
        "entita_per_riga": (ent_tot / n) if n else 0,
        "righe_senza_template_id": n - con_template_id,
        "errori_duri": sum(errori.values()),
    }, errori, testi


def impronte_di(path):
    with open(path, encoding="utf-8") as f:
        return {impronta(json.loads(r)["source_text"]) for r in f if r.strip()}


def impronte_dal_dataset(quanti=3):
    """Le impronte dei testi di alcune contribuzioni GIA' unite, per misurare le
    sovrapposizioni: un file che ripete righe gia' presenti non aggiunge niente."""
    api = f"https://huggingface.co/api/datasets/{REPO_ID}"
    with urllib.request.urlopen(api, timeout=60) as r:
        siblings = [s["rfilename"] for s in json.load(r).get("siblings", [])]
    out = {}
    for nome in [f for f in siblings if f.endswith(".jsonl")][:quanti]:
        url = (f"https://huggingface.co/datasets/{REPO_ID}/resolve/main/"
               f"{urllib.parse.quote(nome)}")
        try:
            with urllib.request.urlopen(url, timeout=300) as resp:
                righe = resp.read().decode().splitlines()
            out[nome] = {impronta(json.loads(r)["source_text"]) for r in righe if r.strip()}
        except Exception as e:                       # rete, non correttezza del file
            out[nome] = e
    return out


def stampa(path, m, errori, fonti):
    print(f"file: {path}")
    print(f"tassonomia: {len(m['_ammesse'])} label ammesse ({'; '.join(fonti)})")
    print()
    print(f"ERRORI DURI: {m['errori_duri']}")
    if errori:
        for k, v in errori.most_common(15):
            print(f"  {v:>7} x {k}")
        if len(errori) > 15:
            print(f"  (e altri {len(errori) - 15} tipi di errore)")
    else:
        print("  nessuno.")
    print()
    print("INFORMAZIONE")
    print(f"  righe: {m['righe']}")
    print(f"  {m['grana_plurale']} distinte: {m['gruppi_distinti']}"
          f"  ({100 * m['gruppi_distinti'] / max(m['righe'], 1):.1f}% delle righe)")
    if m["grana"] == "scheletro":
        print(f"    grana di ripiego: {m['righe_senza_template_id']} righe non portano "
              f"template_id (lo scheletro sottostima i template che etichettano tutto)")
    else:
        print(f"  scheletri distinti: {m['scheletri_distinti']} (per confronto)")
    print(f"  righe per {m['grana']}: mediana "
          f"{m['righe_per_gruppo_mediana']:.0f}, max {m['righe_per_gruppo_max']}")
    print(f"  quota di righe nelle 10 {m['grana_plurale']} piu' frequenti: "
          f"{100 * m['quota_nei_10_gruppi_piu_frequenti']:.0f}%")
    print(f"  testi duplicati nel file: {m['testi_duplicati']}")
    print()
    print(f"COMPOSIZIONE  ({m['entita']} entita' su {m['label_distinte']} label, "
          f"{m['entita_per_riga']:.1f} per riga; testo medio {m['lunghezza_media']:.0f} "
          f"caratteri, max {m['lunghezza_max']})")
    for k, v in list(m["label"].items())[:8]:
        print(f"  {k:20s} {v:7d}  {100 * v / max(m['entita'], 1):5.1f}%")
    rare = list(m["label"].items())[-4:]
    if len(m["label"]) > 12:
        print(f"  ... meno rappresentate: {', '.join(f'{k} ({v})' for k, v in rare)}")


def sintesi(m):
    """Le due righe da incollare nel corpo della Pull Request: sono quelle che permettono a chi
    revisiona di giudicare senza rifare la misura."""
    return (f"{m['righe']} righe, **{m['gruppi_distinti']} {m['grana_plurale']} distinte** "
            f"(max {m['righe_per_gruppo_max']} righe per {m['grana']}), "
            f"{m['entita']} entita' su {m['label_distinte']} label, "
            f"{m['testi_duplicati']} testi duplicati, {m['errori_duri']} errori di validita'.")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("path", help="il file .jsonl da ispezionare")
    ap.add_argument("--contro", metavar="FILE",
                    help="misura la sovrapposizione dei testi con un altro file locale")
    ap.add_argument("--confronta", action="store_true",
                    help="scarica alcune contribuzioni gia' unite e misura le sovrapposizioni")
    ap.add_argument("--json", action="store_true",
                    help="stampa i numeri come JSON (per ispezionare una coda di PR)")
    args = ap.parse_args()

    if not Path(args.path).is_file():
        sys.exit(f"ERRORE: file inesistente: {args.path}")

    ammesse, fonti = tassonomia()
    m, errori, mie = ispeziona(args.path, ammesse)
    m["_ammesse"] = sorted(ammesse)

    sovrapposizioni = {}
    if args.contro:
        altre = impronte_di(args.contro)
        sovrapposizioni[args.contro] = len(mie & altre)
    if args.confronta:
        for nome, altre in impronte_dal_dataset().items():
            sovrapposizioni[nome] = (f"non scaricato ({altre})"
                                     if isinstance(altre, Exception) else len(mie & altre))

    if args.json:
        fuori = {k: v for k, v in m.items() if k != "_ammesse"}
        fuori["errori"] = dict(errori.most_common())
        fuori["sovrapposizioni"] = sovrapposizioni
        fuori["sintesi"] = sintesi(m)
        print(json.dumps(fuori, ensure_ascii=False, indent=1))
    else:
        stampa(args.path, m, errori, fonti)
        if sovrapposizioni:
            print("\nSOVRAPPOSIZIONE dei testi con altri file")
            for nome, quanti in sovrapposizioni.items():
                print(f"  {nome.split('/')[-1]}: {quanti}")
        print(f"\nSINTESI (da incollare nella PR)\n  {sintesi(m)}")

    # uscita 1 SOLO sugli errori duri: la soglia oltre la quale un contributo e' troppo
    # ripetitivo e' una decisione di chi mantiene il progetto, non di questo script.
    return 1 if m["errori_duri"] else 0


if __name__ == "__main__":
    sys.exit(main())
