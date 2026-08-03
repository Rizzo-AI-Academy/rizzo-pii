# -*- coding: utf-8 -*-
"""
Rete REGEX + CHECKSUM che affianca il modello (EMAIL, TELEFONO, IBAN, CF, PIVA,
carta di credito, importo, targa, URL).

Come `pdf_export`, questo modulo lavora solo su stringhe: nessun import di
torch/transformers/fitz, quindi e' testabile in isolamento e senza il modello.
`app.py` ne ri-esporta i nomi, il comportamento pubblico non cambia.

Le regex accettano i separatori con cui gli identificativi sono STAMPATI nei
documenti, non solo la forma compatta: un IBAN su una fattura o su una carta
intestata e' raggruppato a quattro, una carta di credito e' separata da spazi,
trattini o punti, un telefono da spazi, punti o trattini. I validatori
normalizzano gia' i separatori: se la regex non gliene passa mai uno il
checksum non viene nemmeno interrogato e, dove `strict=True`, il valore resta
in chiaro senza alcun fallback.
"""

import re


def iban_ok(s):
    # Si normalizzano gli stessi separatori che la regex ammette: spazio (anche
    # non-breaking, e gli a-capo del testo estratto da un PDF), punto, trattino.
    s = re.sub(r"[\s.\-]", "", s).upper()
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
    # CF OMOCODICO: quando due contribuenti collidono, l'Agenzia sostituisce le
    # cifre da destra con una lettera (0->L 1->M 2->N 3->P 4->Q 5->R 6->S 7->T
    # 8->U 9->V), quindi la forma sopra non lo trova. Voce separata e non classe
    # allargata sulla precedente: qui la forma da sola e' troppo generica (sette
    # posizioni che accettano lettere), percio' strict=True e il checksum
    # diventa obbligatorio. `cf_ok` gia' calcola l'omocodia. La forma canonica
    # matcha entrambe le voci, ma sullo stesso span: `_merge` tiene il candidato
    # validato e scarta il duplicato.
    ("CF",
     re.compile(r"\b[A-Za-z]{6}[\dLMNPQRSTUVlmnpqrstuv]{2}[A-Za-z]"
                r"[\dLMNPQRSTUVlmnpqrstuv]{2}[A-Za-z]"
                r"[\dLMNPQRSTUVlmnpqrstuv]{3}[A-Za-z]\b"),
     cf_ok, True),
    # IBAN: forma compatta, oppure raggruppata a quattro come e' stampata su
    # fatture, carte intestate e home banking. Il gruppo intermedio e quello
    # finale devono contenere almeno una cifra: senza quel vincolo il match
    # goloso inghiottirebbe la parola successiva ("...0005 1332 pago"), il
    # mod-97 fallirebbe e con strict=True l'IBAN si perderebbe del tutto.
    ("IBAN",
     re.compile(r"\b[A-Za-z]{2}\d{2}"
                r"(?:[A-Za-z0-9]{11,30}"
                r"|[\s.\-][A-Za-z0-9]{4}"
                r"(?:[\s.\-](?=[A-Za-z0-9]{0,3}\d)[A-Za-z0-9]{4}){1,6}"
                r"(?:[\s.\-](?=[A-Za-z0-9]{0,3}\d)[A-Za-z0-9]{1,4})?)\b"),
     iban_ok, True),
    # Carta: al gruppo dei separatori si aggiunge il punto ("4111.1111...").
    # Qui NON si usa \s: a differenza dell'IBAN il vincolo e' il solo Luhn (una
    # sequenza qualunque lo supera una volta su dieci) e con l'a-capo una
    # colonna di numeri in tabella diventerebbe un candidato.
    # Il match ora finisce per forza su una CIFRA: con il separatore in coda
    # ("(?:\d[ .\-]?){13,19}") il punto che chiude la frase entrerebbe nello
    # span e finirebbe dentro il placeholder.
    ("CREDITCARDNUMBER",
     re.compile(r"(?<!\d)\d(?:[ .\-]?\d){12,18}(?!\d)"),
     luhn_ok, True),
    ("PIVA",
     re.compile(r"(?<!\d)\d{11}(?!\d)"),
     piva_ok, True),
    # Telefono: al gruppo dei separatori si aggiunge il trattino, usatissimo sia
    # sul cellulare ("333-123-4567") sia sul fisso ("010-2471234").
    ("TELEPHONENUM",
     re.compile(r"(?<![\w.])(?:\+39[\s.\-]?)?(?:3\d{2}[\s.\-]?\d{3}[\s.\-]?\d{3,4}"
                r"|0\d{1,3}[\s.\-]?\d{5,8})(?![\w])"),
     None, True),
    ("AMOUNT",
     re.compile(r"(?:€|EUR|euro)\s?\d{1,3}(?:[.\s]\d{3})*(?:,\d{2})?"
                r"|\d{1,3}(?:\.\d{3})*,\d{2}\s?(?:€|EUR|euro)", re.IGNORECASE),
     None, True),
    ("TARGA",
     re.compile(r"\b[A-Za-z]{2}\s?\d{3}\s?[A-Za-z]{2}\b"),
     None, True),
    # URL: tre forme, dalla piu' esplicita alla piu' rischiosa.
    #   1) con schema (http/https/ftp)     -> sempre un URL
    #   2) www.<dominio>                   -> sempre un URL
    #   3) dominio nudo, ma SOLO con un TLD della lista chiusa qui sotto.
    # Il dominio nudo generico (r"\w+\.\w{2,}") non si puo' usare: nel legalese
    # italiano prenderebbe "p.iva", "n.ro", "S.r.l." e simili. Con la lista chiusa
    # "p.iva" non matcha ("iva" non e' un TLD) e i falsi positivi crollano.
    # Prezzo del compromesso: un dominio con TLD esotico resta in chiaro.
    ("URL",
     re.compile(r"(?:https?|ftp)://[^\s<>\"']+"
                r"|www\.[A-Za-z0-9\-._~%]+\.[A-Za-z]{2,}(?:/[^\s<>\"']*)?"
                r"|\b(?:[A-Za-z0-9](?:[A-Za-z0-9\-]*[A-Za-z0-9])?\.)+"
                r"(?:it|com|net|org|eu|info|io|dev|app|gov|edu|cloud|online|site|blog)"
                r"\b(?:/[^\s<>\"']*)?", re.IGNORECASE),
     None, True),
]

# Punteggiatura che chiude la frase e non fa parte dell'URL: "vedi https://x.it/pagina."
_URL_TRAIL = ".,;:!?)]}»\"'"


def detect_regex(text):
    """Entita' della rete regex. validated=True solo quando il checksum passa."""
    ents = []
    for label, rx, validator, strict in DETECTORS:
        for m in rx.finditer(text):
            start, end = m.start(), m.end()
            if label == "URL":
                while end > start and text[end - 1] in _URL_TRAIL:
                    end -= 1
                if end <= start:
                    continue
            ok = validator(m.group(0)) if validator else False
            if validator and strict and not ok:
                continue
            ents.append({
                "label": label,
                "start": start,
                "end": end,
                "score": 1.0 if ok else 0.9,
                "validated": ok,
                "source": "regex",
            })
    return ents


