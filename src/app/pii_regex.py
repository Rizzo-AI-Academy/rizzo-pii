# -*- coding: utf-8 -*-
"""Rete REGEX + CHECKSUM che affianca il modello.

Copre i campi a forma molto specifica (EMAIL, TELEFONO, IBAN, CF, PIVA, carta di
credito, importi, targhe). Su quegli span e' piu' affidabile del modello: evita
la frammentazione di CF/IBAN/carta in piu' pezzi, e le entita' validate
matematicamente hanno priorita' nel merge.

Modulo separato da app.py apposta: app.py carica il modello all'import (secondi
di attesa, GPU/CPU), qui c'e' solo logica pura -> si testa in millisecondi.
"""

import re


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


def detect_regex(text, relax_strict=False):
    """Entita' della rete regex. validated=True solo quando il checksum passa.

    relax_strict=True  -> il testo viene da OCR. Gli errori di riconoscimento
    (0/O, 1/l, 5/S, 8/B) rompono i checksum: un IBAN storpiato non passerebbe
    mod-97 e resterebbe IN CHIARO nel testo mandato all'LLM. Su testo OCR la
    forma basta e il checksum degrada a semplice indicatore: piu' falsi positivi
    visibili, zero PII perse in silenzio. Le entita' cosi' ammesse sono marcate
    source="regex-ocr" perche' l'UI le mostri come "da verificare".
    """
    ents = []
    for label, rx, validator, strict in DETECTORS:
        for m in rx.finditer(text):
            ok = validator(m.group(0)) if validator else False
            if validator and strict and not ok and not relax_strict:
                continue
            ents.append({
                "label": label,
                "start": m.start(),
                "end": m.end(),
                "score": 1.0 if ok else 0.9,
                "validated": ok,
                "source": "regex-ocr" if (validator and strict and not ok) else "regex",
            })
    return ents
