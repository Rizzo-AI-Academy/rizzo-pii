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


def _validate_record(rec):
    """Controlli di integrita' strutturale + checksum su una riga generata."""
    if len(rec["tokens"]) != len(rec["bio_labels"]):
        return "tokens/bio_labels di lunghezza diversa"
    for e in rec["entities"]:
        # offset coerenti col testo
        if rec["source_text"][e["start"]:e["end"]] != e["value"]:
            return f"offset entita' incoerente: {e}"
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


def generate(n, seed, handle, per_type, offline, boost):
    """Genera n esempi sintetici. Con Gemini usa template NUOVI scritti al volo.
    Ritorna (righe, conteggi, n_template_nuovi)."""
    # IMPORTANTE: ri-seminiamo DOPO l'import (gen imposta seed(42) a import-time).
    # Seed diverso per contributore => valori diversi => no duplicati nel dataset.
    random.seed(seed)

    new_templates = [] if offline else gen_new_templates(per_type, boost)
    # i built-in garantiscono comunque la copertura dei tag rari (CATASTO/DOCID/CONTO...);
    # la NOVITA' del testo viene dai template freschi di Gemini.
    templates = new_templates + gen.TEMPLATES + gen.load_external_templates()
    print(f"\nTemplate nel pool: {len(templates)} "
          f"({len(new_templates)} nuovi da Gemini + {len(gen.TEMPLATES)} built-in + "
          f"{len(templates) - len(new_templates) - len(gen.TEMPLATES)} locali)")

    # selezione pesata: i template che coprono i tag potenziati escono piu' spesso
    weights = template_weights(templates, boost, discover_slot_tags()) if boost else None
    if boost:
        print(f"Boost distribuzione tag: {boost}")
    idx_pool = list(range(len(templates)))

    n_new = len(new_templates)
    rows, label_counts, bad = [], {}, 0
    for _ in range(n):
        tid = random.choices(idx_pool, weights=weights, k=1)[0]
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
            continue
        for e in entities:
            label_counts[e["label"]] = label_counts.get(e["label"], 0) + 1
        rows.append(rec)

    if bad:
        print(f"  scartati {bad} esempi non validi (self-check)")
    return rows, label_counts, n_new


def write_local(rows, out_path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return out_path


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

    rows, counts, n_new = generate(args.n, seed, handle, args.per_type, args.offline, boost)
    if n_new:
        from_new = sum(1 for r in rows if r["meta"]["new_template"])
        print(f"\n{from_new}/{len(rows)} esempi da template NUOVI di Gemini.")
    print(f"Generati {len(rows)} esempi validi. Entita' per label:")
    for label, c in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"  {label:18s} {c}")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    fname = f"{handle}-{stamp}-seed{seed}-n{len(rows)}.jsonl"
    local_path = write_local(rows, ROOT / "dataset" / "contributions" / fname)
    print(f"\nScritto in locale -> {local_path}")

    if args.no_upload:
        print("\n(--no-upload) Nessun caricamento. Rilancia senza --no-upload per aprire la PR.")
        return

    upload_pr(local_path, f"contributions/{fname}", handle, len(rows), seed, args.repo)


if __name__ == "__main__":
    main()
