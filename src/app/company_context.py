"""Sedi d'impresa e registri espliciti, senza inferire la natura legale della sede.

Il riconoscimento NER e gli offset non vengono cambiati. Una semplice «sede»
non viene promossa a sede legale e non si eredita il ruolo da altri soggetti.
"""

import re


_ADDRESS = {"CITY": "comune", "PROVINCE": "provincia", "STREET": "indirizzo",
            "BUILDINGNUM": "civico", "ZIPCODE": "cap"}
_SEAT = re.compile(r"\b(?:con\s+)?sede\s+(?:(?P<legal>legale)\s+)?"
                   r"(?:in|ad?|nel\s+Comune\s+di)\s+|\bcorrente\s+(?:in|ad?)\s+", re.I)
_COMPANY = re.compile(r"\b(?:societ[àa]|impresa(?:\s+individuale)?|azienda|ditta|agenzia\s+immobiliare)\b|"
                      r"\b(?:s\.?\s*r\.?\s*l\.?|s\.?\s*a\.?\s*s\.?|"
                      r"s\.?\s*n\.?\s*c\.?|s\.?\s*p\.?\s*a\.?)(?=\W|$)", re.I)
_PERSON = re.compile(r"\b(?:notai[oa]|notarile|signor[aei]?|sig\.?|nat[oaie]|resident[ei]|"
                     r"domiciliat[oaie]|avvocat[oa]|perito|ingegnere|geometra|studio)\b", re.I)
_INSTITUTION = re.compile(r"\b(?:notaio|notarile|tribunale|comune|registro|universit[àa]|"
                          r"ministero|studio|avvocat[oa]|perito|ingegnere|geometra|"
                          r"agenzia\s+delle\s+entrate)\b", re.I)
_REGISTRY = re.compile(r"\b(?:(?P<registro>Registro\s+(?:delle\s+)?Imprese)|"
                       r"(?P<camera>C\.?\s*C\.?\s*I\.?\s*A\.?\s*A\.?|Camera\s+di\s+Commercio))"
                       r"\s+di\s+", re.I)
_STOP = re.compile(r"\b(?:capitale\s+sociale|cod(?:ice|\.)?\s*fisc(?:ale|\.)?|"
                   r"partita\s+IVA|P\.?\s*IVA|Registro\s+(?:delle\s+)?Imprese|"
                   r"Camera\s+di\s+Commercio|C\.?\s*C\.?\s*I\.?\s*A\.?\s*A\.?|"
                   r"R\.?\s*E\.?\s*A\.?|residenza|resident[ei]|domicilio|domiciliat[oaie]|"
                   r"nat[oaie]|nascita|immobil[ei]|fabbricat[oi]|confinant[ei]|"
                   r"sede|corrente|notai[oa]|signor[aei]?|rappresentante)\b|"
                   r"\b(?:res(?:id)?\.|res\.te|dom\.(?:to|ta)?)(?=\s|:|$)", re.I)
_STREET = re.compile(r"\b(?:via|viale|piazza|corso|strada|largo|vicolo|contrada|contr[àa]['’]?)\b", re.I)
_CITY_WORD = r"(?:[A-ZÀ-ÖØ-Þ][\w'’.-]*|di|del|dello|della|dei|degli|delle|sul|sulla)"
_CITY_PREFIX = re.compile(r"\s*(?:(?:[Cc]omune\s+di\s+)?(?:" + _CITY_WORD +
                          r"\s*){1,8}(?:\([A-Z]{2}\))?)?\s*,?\s*(?:(?:in|alla|nella)\s+)?")
_ABBREVIATIONS = {"s", "r", "l", "a", "n", "c", "p", "spa", "srl", "sas", "snc",
                  "civ", "num", "nr", "loc", "fraz", "prov", "sig", "dott"}


def _boundary(value):
    for match in re.finditer(r"[;!?]|\n[ \t]*\n|\.(?=\s|$)", value):
        if match.group() == ".":
            word = re.search(r"(\w+)$", value[:match.start()])
            if word and word.group(1).casefold() in _ABBREVIATIONS:
                continue
        return True
    return False


def _negated(text, start):
    return bool(re.search(r"\b(?:non|né|nè)\s+(?:(?:è|risulta|essere|più|stat[oaie]|"
                          r"mai|ancora|viene|venne|fu|era|ha|aveva)\s+){0,5}$|\bmai\s*$",
                          text[max(0, start - 70):start], re.I))


def _company_subject(text, anchor, entities):
    left = max(0, anchor.start() - 300)
    candidates = list(_COMPANY.finditer(text, left, anchor.start()))
    candidates = [(m.start(), m.end()) for m in candidates]
    for e in entities:
        if (e.get("label") == "ORG" and left <= e["start"] < e["end"] <= anchor.start()
                and not _INSTITUTION.search(text[e["start"]:e["end"]])):
            candidates.append((e["start"], e["end"]))
    if not candidates:
        return False
    _, end = max(candidates, key=lambda item: item[1])
    between = text[end:anchor.start()]
    if _boundary(between) or _PERSON.search(between):
        return False
    for e in entities:
        if e.get("label") == "FULLNAME" and end <= e["start"] < e["end"] <= anchor.start():
            lead = text[end:e["start"]]
            # «Impresa individuale ... di Nome» appartiene alla denominazione;
            # «società ... e Nome con sede» introduce invece un altro soggetto.
            if re.search(r"\b(?:e|ed|o|oppure)\s*$", lead, re.I):
                return False
    return True


def _address_component(text, anchor, start, end, label):
    tail = text[anchor.end():start]
    if len(tail) > 220 or _boundary(tail) or _STOP.search(tail):
        return False
    if label == "CITY":
        return bool(re.fullmatch(r"(?:Comune\s+di\s+)?", tail, re.I))
    streets = list(_STREET.finditer(tail))
    lead = tail[:streets[0].start()] if streets else tail
    if not streets and label == "PROVINCE":
        lead = re.sub(r"\(\s*$|\bprov(?:incia|\.)?\s*(?:di\s+)?$", "", lead, flags=re.I)
    if not streets and label == "ZIPCODE":
        lead = re.sub(r"\bCAP\s*:?\s*$", "", lead, flags=re.I)
    if (re.search(r"\b(?:e|ed|poi|che|presso|per|con)\b", lead, re.I)
            or not _CITY_PREFIX.fullmatch(lead)):
        return False
    if label == "STREET":
        return bool(_STREET.match(text[start:end]) or
                    re.search(_STREET.pattern + r"\s+$", tail, re.I))
    if label == "BUILDINGNUM":
        return bool(streets and re.search(r"\b(?:n[°º.]?|civ(?:ico)?\.?)\s*$", tail, re.I))
    if label == "PROVINCE":
        return bool(re.search(r"\(\s*$|\bprov(?:incia|\.)?\s*(?:di\s+)?$", tail, re.I))
    return bool(re.search(r"\bCAP\s*:?\s*$", tail, re.I))


def classify_company(text, entity, entities=()):
    """Categoria e motivazione per sede o Registro Imprese, altrimenti None.

    I nomi societari riconosciuti dal NER sono una prova del soggetto, non una
    chiave per sostituire tutte le occorrenze uguali. ``entities`` rimane intatto.
    """
    start, end, label = entity.get("start"), entity.get("end"), entity.get("label")
    if (type(start) is not int or type(end) is not int or not 0 <= start < end <= len(text)):
        raise ValueError("Intervallo della sede non valido.")
    if label not in _ADDRESS:
        return None
    suffix = _ADDRESS[label]
    if label in {"CITY", "PROVINCE"}:
        registries = list(_REGISTRY.finditer(text, max(0, start - 180), start))
        if registries:
            anchor = registries[-1]
            tail = text[anchor.end():start]
            valid = not tail if label == "CITY" else bool(
                re.fullmatch(r"(?:" + _CITY_WORD + r"\s*){1,8}\(\s*", tail))
            if valid and not _negated(text, anchor.start()):
                kind = "camera_commercio" if anchor.group("camera") else "registro_imprese"
                return ("luoghi_documentali." + kind + "_" + suffix,
                        "Località dell'ente camerale o del Registro delle Imprese, distinta dalla sede dell'impresa.")
    anchors = list(_SEAT.finditer(text, max(0, start - 350), start))
    if not anchors:
        return None
    anchor = anchors[-1]
    if (_negated(text, anchor.start()) or not _company_subject(text, anchor, entities)
            or any(e.get("label") in {"FULLNAME", "ORG"}
                   and anchor.end() <= e["start"] < e["end"] <= start for e in entities)
            or not _address_component(text, anchor, start, end, label)):
        return None
    legal = anchor.group("legal") is not None
    category = "sede_societaria." + ("" if legal else "sede_") + suffix
    reason = ("Sede legale esplicita del soggetto societario." if legal else
              "Sede dell'impresa indicata esplicitamente; natura legale non specificata.")
    return category, reason
