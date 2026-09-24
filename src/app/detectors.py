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
    # IBAN, forma compatta: copre qualsiasi paese, anche fuori dal registro ISO.
    # Il formato di stampa a gruppi NON e' una regex ma `detect_iban()` qui sotto,
    # perche' per sapere dove finisce serve la lunghezza prevista per il paese.
    ("IBAN",
     re.compile(r"\b[A-Za-z]{2}\d{2}[A-Za-z0-9]{11,30}\b"),
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
    # il trattino e' un separatore quanto lo spazio ("AB-123-CD" nei moduli e negli
    # export): senza, la targa scritta cosi' non viene vista da nessuno dei due lati,
    # perche' nemmeno il modello la riconosce, e resta in chiaro.
    ("TARGA",
     re.compile(r"\b[A-Za-z]{2}[\s-]?\d{3}[\s-]?[A-Za-z]{2}\b"),
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
    # DOCID: il codice di un atto e' scritto sempre dopo la sua sigla ("R.G. 1234/2024",
    # "Prot. 123/2024", "Rep. 45"). E' la sigla a renderlo riconoscibile: il numero da
    # solo ("1234/2024") non si distingue da una frazione o da un articolo di legge,
    # quindi la si pretende sempre e la si include nello span.
    # Il "n." fra la sigla e il numero e' opzionale e vale per TUTTE le sigle: si scrive
    # "Prot. 456/2024" quanto "Prot. n. 456/2024", e negli atti la seconda e' la forma
    # piu' frequente. Tenendolo attaccato alla sola parola "protocollo" - com'era - la
    # forma abbreviata piu' comune restava in chiaro.
    ("DOCID",
     re.compile(r"\b(?:R\.?G\.?\s*N\.?R\.?|R\.?G\.?|RG|Prot\.?|protocollo"
                r"|Rep\.?|repertorio)"
                r"(?:\s*(?:n\.?|num\.?|nro\.?))?"
                r"\s*\d{1,8}(?:[/\-]\d{2,4})?\b",
                re.IGNORECASE),
     None, True),
    # Date numeriche: forma specifica ma SENZA validatore possibile (una data non ha
    # checksum). Sta in SOFT_REGEX_LABELS -> in fusione non eredita la priorita' della
    # rete regex, quindi il modello puo' sovrascriverla: "06-11-2014" dentro un
    # riferimento di laboratorio resta un falso positivo accettabile, non un verdetto.
    ("DATE",
     re.compile(r"(?<!\d)(?:0?[1-9]|[12]\d|3[01])[/.\-]"
                r"(?:0?[1-9]|1[0-2])[/.\-](?:19|20)\d{2}"
                r"(?:\s+\d{1,2}[:.]\d{2})?(?!\d)"),
     None, True),
]

# Label della rete regex SENZA validatore forte: la forma da sola non basta a dire
# "e' certamente questo campo". In _merge non ereditano la priorita' della rete regex,
# cosi' il modello puo' sovrascriverle.
SOFT_REGEX_LABELS = {"DATE"}

# Punteggiatura che chiude la frase e non fa parte dell'URL: "vedi https://x.it/pagina."
_URL_TRAIL = ".,;:!?)]}»\"'"


# ISO 13616: lunghezza dell'IBAN per paese. Serve a sapere DOVE finisce quando e'
# scritto a gruppi: indovinare il confine significa inghiottire le parole vicine o
# fermarsi a meta' lasciando la coda dell'IBAN in chiaro.
IBAN_LEN = {c[:2]: int(c[2:]) for c in (
    "AD24 AE23 AL28 AT20 AZ28 BA20 BE16 BG22 BH22 BI27 BR29 BY28 CH21 CR22 CY28 CZ24 "
    "DE22 DJ27 DK18 DO28 EE20 EG29 ES24 FI18 FK18 FO18 FR27 GB22 GE22 GI23 GL18 GR27 "
    "GT28 HN28 HR21 HU28 IE22 IL23 IQ23 IS26 IT27 JO30 KW30 KZ20 LB28 LC32 LI21 LT20 "
    "LU20 LV21 LY25 MC27 MD24 ME22 MK19 MN20 MR27 MT31 MU30 NI28 NL18 NO15 OM23 PK24 "
    "PL28 PS29 PT25 QA29 RO24 RS22 RU33 SA24 SC31 SD18 SE24 SI19 SK24 SM27 SO23 ST25 "
    "SV28 TL23 TN24 TR26 UA29 VA22 VG24 XK20 YE30").split()}

# sigla e cifre di controllo attaccate: sono sempre stampate insieme, e pretenderlo
# evita che "il 17 luglio 2008" o "RO 48 2897 ..." si aggancino come IBAN.
_IBAN_HEAD = re.compile(r"(?<![A-Za-z0-9])[A-Za-z]{2}\d{2}")
_IBAN_MAX_SEP = 3                      # separatori di fila (PDF giustificati, tabelle)
_IBAN_RUN = 5                          # caratteri per gruppo: piu' lunghi sono parole


def _iban_char(c):
    """ASCII: senza isascii() la scansione inghiotte "eta'", "piu'", ..."""
    return c.isascii() and c.isalnum()


def detect_iban(text):
    """IBAN nel formato di stampa a gruppi ("IT60 X054 2811 ...").

    Dalla sigla del paese consuma esattamente i caratteri previsti da IBAN_LEN,
    tollerando i separatori; poi decide il mod-97.

    Uno scanner e non una regex perche' il confine destro non e' deducibile dalla
    forma: senza la lunghezza attesa il match o inghiotte la parola dopo, o si ferma
    a meta' e lascia la coda dell'IBAN in chiaro sotto un placeholder rassicurante.
    """
    ents = []
    for m in _IBAN_HEAD.finditer(text):
        n, i = IBAN_LEN.get(m.group()[:2].upper()), m.start()
        if not n or (i and text[i - 1].isalnum()):
            continue
        chars, sep, run, nl, grp, start = [], 0, 0, 0, False, i
        while i < len(text) and len(chars) < n:
            c = text[i]
            if _iban_char(c):
                chars.append(c)
                run += 1
                sep = 0
            elif sep < _IBAN_MAX_SEP and run <= _IBAN_RUN and (c in ".-" or c.isspace()):
                sep, run, grp = sep + 1, 0, True
                nl += c.isspace() and c not in " \t "
                if nl > 1:
                    break              # un IBAN va a capo una volta; una colonna a ogni cella
            else:
                break
            i += 1
        if len(chars) < n or (i < len(text) and _iban_char(text[i])):
            continue                    # troncato, oppure il codice prosegue oltre
        if grp:
            g = [x for x in re.split(r"[\s.\-]+", text[start:i]) if x]
            if len(set(map(len, g[:-1]))) > 1 or len(g[-1]) > len(g[0]):
                continue                # gruppi disuguali: e' prosa, non un IBAN stampato
        if iban_ok(text[start:i]):
            ents.append({"label": "IBAN", "start": start, "end": i,
                         "score": 1.0, "validated": True, "source": "regex"})
    # due candidati sovrapposti: uno e' l'artefatto di una scansione partita da un codice
    # vicino, ma quale dei due non e' decidibile dal testo. Si maschera l'unione, perche'
    # scartarne uno lascia in chiaro un pezzo dell'altro sotto un [IBAN_1] rassicurante.
    ents.sort(key=lambda e: e["start"])
    for a, b in zip(ents, ents[1:]):
        if b["start"] < a["end"]:
            b["start"], b["end"] = a["start"], max(a["end"], b["end"])
            a["end"] = a["start"]       # assorbito in b
    return [e for e in ents if e["end"] > e["start"]]


# Dalla piu' lunga: le stesse lunghezze che accetta luhn_ok, cosi' un PAN non viene
# tagliato a una lunghezza piu' corta che passa il Luhn per caso (con 17 fuori dalla
# lista, di un PAN di 17 cifre se ne coprivano 16 e l'ultima restava in chiaro).
_CARD_LEN = (19, 18, 17, 16, 15, 14, 13)
_CARD_CODA = 4                     # cifre di coda ammesse: la scadenza, "12/26" o "12 26"


def _card_scadenza(coda):
    """La coda deve somigliare a una scadenza: due-quattro cifre, col mese fra 1 e 12.
    E' il vincolo che tiene bassi i falsi positivi: senza, un numero lungo qualsiasi
    trova per caso un prefisso che passa il Luhn. Una sola cifra non e' mai una
    scadenza, e ammetterla faceva passare 9 code su 10."""
    return 2 <= len(coda) <= 4 and 1 <= int(coda[:2]) <= 12


def _card_senza_scadenza(text, start, end):
    """La scadenza attaccata al numero ("4111 1111 1111 1111 12/26") entra nel match:
    il Luhn fallisce, strict lo scarta e la carta resta IN CHIARO. Qui si riprova
    tagliando la coda, ma solo su un separatore gia' presente e solo se quel che resta
    fuori e' un mese plausibile."""
    s = text[start:end]
    d = [i for i, c in enumerate(s) if c.isdigit()]
    for n in _CARD_LEN:
        if not 0 < len(d) - n <= _CARD_CODA:
            continue
        taglio = d[n - 1] + 1
        if (s[taglio] in " .-" and _card_scadenza(re.sub(r"\D", "", s[taglio:]))
                and luhn_ok(s[:taglio])):
            return start + taglio, True
    return end, False


def detect_regex(text):
    """Entita' della rete regex. validated=True solo quando il checksum passa."""
    ents = detect_iban(text)
    for label, rx, validator, strict in DETECTORS:
        for m in rx.finditer(text):
            start, end = m.start(), m.end()
            if label == "URL":
                while end > start and text[end - 1] in _URL_TRAIL:
                    end -= 1
                if end <= start:
                    continue
            ok = validator(m.group(0)) if validator else False
            if not ok and label == "CREDITCARDNUMBER":
                end, ok = _card_senza_scadenza(text, start, end)
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


