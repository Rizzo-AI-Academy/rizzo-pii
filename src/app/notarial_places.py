"""Luoghi documentali riconoscibili da formule notarili esplicite.

Regole locali e senza stato: non inferiscono residenza, nascita o ubicazione
del bene dalla sola uguaglianza dei valori. Non alterano gli span del NER.
"""

import re


_ADDRESS = {"CITY": "comune", "PROVINCE": "provincia", "STREET": "indirizzo",
            "BUILDINGNUM": "civico", "ZIPCODE": "cap"}
_OPENING = re.compile(
    r"\brepubblica\s+italiana\b[\s\S]{0,100}?\bil\s+giorno\b"
    r"[^;!?]{1,200};\s*(?:a|in)\s+(?P<address>[^;!?]{1,300}?)\s*;\s*"
    r"davanti\s+a\s+me\b[\s\S]{0,220}?\bnotai[oa]\b", re.I)
_DISTRICT = re.compile(
    r"\bcollegio\s+notarile\s+(?:(?:dei\s+)?distretti\s+(?:riuniti\s+)?di|di)\s+", re.I)
_SEAT = re.compile(r"\bcon\s+sede\s+(?:a|ad|in)\s+", re.I)
_NOTARY = re.compile(r"\b(?:notaio|notarile)\b", re.I)
_COMPANY = re.compile(r"\b(?:societ[àa]|azienda|impresa|sede\s+legale|s\.?\s*r\.?\s*l\.?|"
                      r"s\.?\s*p\.?\s*a\.?|s\.?\s*a\.?\s*s\.?|s\.?\s*n\.?\s*c\.?)\b", re.I)
_REGISTRATION = re.compile(r"\b(?:registrat[oaie]|trascritt[oaie])\s+(?:a|ad|in)\s+", re.I)
_AUTHORITY = re.compile(
    r"\b(?:rilasciat[oaie]|emess[oaie]|presentat[oaie]|depositat[oaie]|protocollat[oaie])"
    r"\s+(?:al|dal|presso\s+il)\s+comune\s+di\s+", re.I)
_NAME = r"(?:[A-ZÀ-ÖØ-Þ][\w'’.-]*|[Dd][ei]|[Dd]el(?:la)?|[Dd]a)"
_PROFESSIONAL = re.compile(
    r"\b(?i:notaio|perito\s+industriale|ingegnere|ing\.|geometra|geom\.|architetto|arch\.)"
    r"\s+(?:(?i:dott(?:or|ore)?\.?)\s+)?"
    r"(?:" + _NAME + r"\s+){1,8}(?i:di)\s*$")
_CITY_WORD = r"(?:[A-ZÀ-ÖØ-Þ][\w'’-]*|del|della|di|dei|degli|delle|sul|sulla)"
_PROFESSIONAL_PROVINCE = re.compile(_PROFESSIONAL.pattern[:-1] +
                                    r"(?:" + _CITY_WORD + r"\s*){1,8}\(\s*$")
_ROLE_STOP = re.compile(
    r"\b(?:nat[oaie]|nascita|residente|residenti|residenza|res\.te|domicili[oa]|"
    r"domiciliat[oaie]|sede\s+legale|immobil[ei]|fabbricat[oi]|terren[oi]|ben[ei]|"
    r"appartament[oi]|unit[àa]\s+immobiliar[ei]|"
    r"sono\s+presenti|comparenti|con\s+sede|registrat[oaie]|trascritt[oaie]|"
    r"rilasciat[oaie]|presentat[oaie]|protocollat[oaie])\b"
    r"|\b(?:res(?:id)?\.|dom\.(?:to|ta)?)(?=\s|:|$)", re.I)
_NARRATIVE = re.compile(r"\b(?:che|dove|presso|vicino|sopra|sotto|nato|residente|via|per|con)\b", re.I)
_STREET = re.compile(r"\b(?:via|viale|piazza|corso|strada|largo|vicolo)\b", re.I)
_ABBREVIATIONS = {"n", "nr", "num", "civ", "dott", "dr", "sig", "avv", "ing",
                  "geom", "arch", "s", "r", "l", "p", "spa", "srl"}


def _boundary(value):
    for match in re.finditer(r"[;!?]|\n\s*\n|\.(?=\s|$)", value):
        if match.group() == ".":
            previous = re.search(r"(\w+)$", value[:match.start()])
            if previous and (previous.group(1).casefold() in _ABBREVIATIONS
                             or len(previous.group(1)) == 1):
                continue
        return True
    return False


def _negated(text, start):
    return bool(re.search(
        r"\b(?:non|né|nè)\s+(?:(?:è|risulta|essere|più|stat[oaie]|mai|ancora|"
        r"venne|viene|fu|era|risultava|sia|fosse)\s+){0,5}$|\bmai\s*$",
        text[max(0, start - 75):start], re.I))


def _last(pattern, text, start, distance=350):
    matches = list(pattern.finditer(text, max(0, start - distance), start))
    return matches[-1] if matches else None


def _location_tail(text, match, start, max_length=150):
    if match is None or _negated(text, match.start()):
        return None
    tail = text[match.end():start]
    if len(tail) > max_length or _boundary(tail) or _ROLE_STOP.search(tail):
        return None
    return tail


def _seat_component(text, tail, start, end, label):
    """La sede non si estende a toponimi narrativi o numeri di altri campi."""
    if label == "CITY":
        return bool(re.fullmatch(r"(?:comune\s+di\s+)?", tail, re.I))
    streets = list(_STREET.finditer(tail))
    lead = tail[:streets[0].start()] if streets else tail
    if label == "PROVINCE" and not streets:
        lead = re.sub(r"\(\s*$|\bprov(?:incia|\.)?\s*(?:di\s+)?$", "", lead, flags=re.I)
    if label == "ZIPCODE" and not streets:
        lead = re.sub(r"\bCAP\s*:?\s*$", "", lead, flags=re.I)
    # Il prefisso della via può contenere solo un toponimo e connettivi di
    # indirizzo. "Imola e Ada Neri in via..." apre invece un nuovo soggetto.
    if not re.fullmatch(r"\s*(?:(?:comune\s+di\s+)?(?:" + _CITY_WORD +
                        r"\s*){1,8}(?:\([A-Z]{2}\))?)?\s*,?\s*(?:(?:in|alla|nella)\s+)?", lead):
        return False
    if label == "STREET":
        return bool(_STREET.match(text[start:end]) or
                    re.search(r"\b(?:via|viale|piazza|corso|strada|largo|vicolo)\s+$", tail, re.I))
    if label == "BUILDINGNUM":
        return bool(_STREET.search(tail) and re.search(r"\b(?:n[°º.]?|civ(?:ico)?\.?)\s*$", tail, re.I))
    if label == "ZIPCODE":
        return bool(re.search(r"\bCAP\s*:?\s*$", tail, re.I))
    return bool(re.search(r"\(\s*$|\bprov(?:incia|\.)?\s*(?:di\s+)?$", tail, re.I))


def classify_place(text, start, end, label):
    """Restituisce ``(categoria, motivo)`` soltanto per formule esplicite.

    Le località di un ufficio e del professionista restano distinte dalla sede
    societaria e dall'indirizzo del bene. I numeri documentali non vengono mai
    promossi a civici dal solo contesto di registrazione.
    """
    if (not isinstance(text, str) or type(start) is not int or type(end) is not int
            or not 0 <= start < end <= len(text)):
        raise ValueError("Intervallo del luogo non valido.")
    if label not in _ADDRESS:
        return None
    suffix = _ADDRESS[label]

    if label == "BUILDINGNUM" and re.fullmatch(r"\d{1,3}", text[start:end]):
        marker = _last(_REGISTRATION, text, start)
        tail = _location_tail(text, marker, start, 90)
        if (tail is not None and re.fullmatch(r"(?:" + _CITY_WORD + r"\s+){1,8}", tail)
                and re.match(r"\s+(?:il|in\s+data)\b", text[end:end + 20], re.I)):
            return ("luoghi_documentali.ufficio_unita",
                    "Numero dell'ufficio nella formula di registrazione, non un civico.")

    # Tutta la formula è necessaria: "il giorno" da solo non identifica un atto.
    for match in _OPENING.finditer(text, max(0, start - 550), min(len(text), end + 600)):
        left, right = match.span("address")
        if left <= start < end <= right and not _ROLE_STOP.search(text[left:start]):
            return ("luoghi_documentali.stipula_" + suffix,
                    "Luogo di stipula nella formula introduttiva dell'atto.")

    if label in {"CITY", "PROVINCE"}:
        match = _last(_DISTRICT, text, start)
        tail = _location_tail(text, match, start, 220)
        if (tail is not None and not _NARRATIVE.search(tail)
                and re.fullmatch(r"[\w\s'’(),-]*", tail)):
            return ("luoghi_documentali.distretto_notarile",
                    "Località del collegio o distretto notarile esplicito.")

    match = _last(_SEAT, text, start)
    tail = _location_tail(text, match, start)
    if tail is not None:
        before = text[max(0, match.start() - 350):match.start()]
        notaries = list(_NOTARY.finditer(before))
        if notaries:
            context = before[notaries[-1].end():]
            if (not _boundary(context) and not _COMPANY.search(before) and not _ROLE_STOP.search(context)
                    and _seat_component(text, tail, start, end, label)):
                return ("luoghi_documentali.sede_notaio_" + suffix,
                        "Sede del notaio indicata esplicitamente, distinta dalla sede societaria.")

    # Per gli uffici si classificano soltanto località e provincia. Date, numeri
    # di registrazione e dati catastali richiedono le rispettive regole.
    if label in {"CITY", "PROVINCE"}:
        candidates = [m for m in (_last(_REGISTRATION, text, start), _last(_AUTHORITY, text, start))
                      if m is not None]
        match = max(candidates, key=lambda m: m.start()) if candidates else None
        tail = _location_tail(text, match, start, 90)
        if tail is not None:
            if label == "CITY":
                accepted = bool(re.fullmatch(r"(?:comune\s+di\s+)?", tail, re.I))
            else:
                accepted = bool(re.fullmatch(r"(?:" + _CITY_WORD + r"\s*){1,8}\(\s*", tail))
            if accepted:
                return ("luoghi_documentali.ufficio_" + suffix,
                        "Località dell'ufficio di registrazione o dell'autorità amministrativa.")

        prefix = text[max(0, start - 230):start]
        match = (_PROFESSIONAL if label == "CITY" else _PROFESSIONAL_PROVINCE).search(prefix)
        if match is not None and not _negated(prefix, match.start()):
            context = prefix[match.start():]
            if not _ROLE_STOP.search(context):
                return ("luoghi_documentali.sede_professionista_" + suffix,
                        "Località associata esplicitamente al professionista nell'atto.")
    return None
