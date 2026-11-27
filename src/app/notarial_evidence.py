"""Evidenze notarili locali per span esistenti, senza correggerne gli offset.

Il tipo NER originale rimane disponibile: una particella etichettata come civico
può avere funzione catastale senza diventare un nuovo span. Le regole non fanno
I/O, non conservano testo e non estendono automaticamente i frammenti numerici.
"""

import re


_ADDRESS = {"CITY": "comune", "STREET": "indirizzo", "BUILDINGNUM": "civico",
            "ZIPCODE": "cap", "PROVINCE": "provincia"}
_NUMBER = r"(?:n(?:\.\s*ri\.?|[°º.]|umero)?|nn\.)"
_PARCEL_PREFIX = re.compile(r"\bparticell[ae]\s*(?:" + _NUMBER + r"\s*)?:?\s*$", re.I)
_CAT_START = re.compile(r"\bCatasto\s+(?:Fabbricati|Terreni)\b\s*:?", re.I)
_CAT_STOP = re.compile(r"\b(?:Articolo|NOTAIO|registrat[oaie]|trascritt[oaie]|resident[ei]|residenza|"
                       r"domiciliat[oaie]|domicilio|sede|corrente\s+(?:ad?|in)|societ[àa]|impresa|nat[oaie]|permesso|confinant[ei])"
                       r"\b|\bconfini\b|\n[ \t]*\n", re.I)
_SUB_ROW = re.compile(r"\b(?P<parcel>\d{1,8})\s+sub\.?\s*(?P<sub>\d{1,6})\b", re.I)
_STREET = re.compile(r"\b(?:via|viale|piazza|corso|strada|vicolo|largo|contrada|contrà)\b|\bcontra['’](?=\s)", re.I)
_PROPERTY = r"(?:fabbricat[oi]|immobil[ei]|appartament[oi]|terren[oi]|ben[ei]|compless[oi]|unit[àa]\s+immobiliar[ei])"
_LOCATION = re.compile(r"\b" + _PROPERTY + r"\b[^;.!?]{0,240}?\b"
                       r"(?:sit[oaie0]|ubicat[oaie]|post[oaie])\s+"
                       r"(?:(?:in|nel)\s+Comune\s+di|in|a)\s+", re.I)
_OTHER_ROLE = re.compile(r"\b(?:resident[ei]|residenza|domicilio|domiciliat[oaie]|nat[oaie]|"
                         r"notaio|tribunale\s+di|confinant[ei]|confini|proveniente|sede\s+legale|"
                         r"studio\s+(?:notarile|professionale|contabile))\b", re.I)
_NAME = re.compile(r"\bResidence\s*[\"“«](?P<name>[^\"”»]{1,100})[\"”»]", re.I)
_DECIMAL = re.compile(r"(?<![\w.,])\d+(?:[.,]\d+)+(?!\w|[.,]\d)")
_CADASTRAL_MEASURE = re.compile(r"\b(?:vani|mq|m²|m\.\s*q\.)\s*[:=]?\s*$", re.I)
_SECTION = re.compile(r"(?:^|\n)[ \t]*(?:Articolo\s+)?(?P<num>\d+(?:\.\d+)+)\s*\)", re.I)
_ARTICLE = re.compile(r"\bArticolo\s+(?P<num>\d+)\b", re.I)
_MONTH = r"(?:gennaio|febbraio|marzo|aprile|maggio|giugno|luglio|agosto|settembre|ottobre|novembre|dicembre)"
_LAW = r"\b(?:D\.?\s*Lgs\.?|D\.?\s*L\.?|D\.?\s*P\.?\s*R\.?|L\.|legge|decreto\s+(?:legislativo|legge))"
_LAW_YEAR = re.compile(_LAW + r"\s+(?P<date>(?P<day>\d{1,2})\s+(?P<month_year>" + _MONTH +
                       r"\s+(?P<year>\d{4})))\b", re.I)
_LAW_IDENTIFIER = re.compile(
    _LAW + r"\s+(?:(?P<code>\d{1,6}/(?P<code_year>\d{4}))|"
    r"\d{1,2}\s+" + _MONTH + r"\s+\d{4}\s*,?\s*"
    + _NUMBER + r"\s*(?P<number>\d{1,6}(?:/\d{4})?))\b", re.I)
_CADASTRAL_CLASS = re.compile(r"\bcat(?:egoria)?\.?\s*:?\s*(?P<code>[A-F]/\d{1,2})\b", re.I)
_CONSTRUCTION = re.compile(r"\b(?:costruit[oaie]|edificat[oaie])\s+(?:in\s+data\s+)?"
                           r"anteriore\s+(?:all['’]|al|a)\s*(?P<date>\d{1,2}\s+" + _MONTH +
                           r"\s+\d{4})\b", re.I)
_BOUNDARY = re.compile(r"\bconfini\s*:|\bdel\s+garage\s*:", re.I)
_BOUNDARY_STOP = re.compile(r"\b(?:Articolo|NOTAIO|registrat[oaie]|trascritt[oaie]|resident[ei]|residenza|"
                            r"domiciliat[oaie]|domicilio|sede\s+legale|nat[oaie])\b", re.I)
_BOUNDARY_PARCEL = re.compile(r"\bparticella\s*(?:" + _NUMBER + r"\s*)?\d+\b", re.I)
_ABBREVIATIONS = {"n", "nr", "civ", "loc", "fraz", "v", "ri", "s", "c", "sub", "cat", "cl", "sup", "m", "q"}


def _span(text, entity):
    start, end = entity.get("start"), entity.get("end")
    if (type(start) is not int or type(end) is not int
            or not 0 <= start < end <= len(text)):
        raise ValueError("Intervallo non valido.")
    return start, end


def _broken(text, start, end, *, semicolons=True):
    section = text[start:end]
    if re.search(r"[!?]|\n[ \t]*\n" + (r"|;" if semicolons else ""), section):
        return True
    for match in re.finditer(r"\.(?=\s|$)", section):
        previous = re.search(r"(\w+)$", section[:match.start()])
        if not previous or previous.group(1).casefold() not in _ABBREVIATIONS:
            return True
    return False


def _cat_section(text, start):
    anchors = list(_CAT_START.finditer(text, max(0, start - 3000), start))
    if not anchors:
        return None
    anchor = anchors[-1]
    if _CAT_STOP.search(text, anchor.end(), start):
        return None
    return anchor


def _cat_row(text, anchor, start):
    rows = list(_SUB_ROW.finditer(text, max(anchor.end(), start - 500), start))
    if not rows:
        return None
    row = rows[-1]
    if _broken(text, row.end(), start):
        return None
    return row


def cadastral_decimal_bounds(text, start, end):
    """Intero valore decimale soltanto in un campo metrico catastale esplicito.

    Lo span può essere un frammento del valore. La funzione propone i confini,
    senza modificare entità o testo; il chiamante gestisce eventuali overlap.
    """
    if (type(start) is not int or type(end) is not int
            or not 0 <= start < end <= len(text)):
        raise ValueError("Intervallo non valido.")
    for match in _DECIMAL.finditer(text, max(0, start - 24), min(len(text), end + 24)):
        if not (match.start() <= start < end <= match.end()) or "," not in match.group():
            continue
        if not re.fullmatch(r"(?:\d+|\d{1,3}(?:\.\d{3})+),\d+", match.group()):
            continue  # sequenze OCR malformate non dimostrano un unico decimale
        # La finestra non deve troncare un numero più lungo.
        after = text[match.end():match.end() + 2]
        if ((match.start() and text[match.start() - 1] in "0123456789.,")
                or after[:1].isdigit() or (after[:1] in {".", ","} and after[1:2].isdigit())):
            continue
        anchor = _cat_section(text, match.start())
        row = _cat_row(text, anchor, match.start()) if anchor else None
        if row and _CADASTRAL_MEASURE.search(text[max(row.end(), match.start() - 30):match.start()]):
            return match.span()
    return None


def _normative_evidence(text, start, end, label):
    """Legge l'intera formula anche se il modello ha restituito solo giorno/anno."""
    window_end = min(len(text), end + 100)
    for match in _LAW_YEAR.finditer(text, max(0, start - 110), window_end):
        if re.search(r"\n[ \t]*\n", match.group()):
            continue
        accepted = ("date", "day", "month_year", "year") if label == "DATE" else ("year",)
        if any(match.span(group) == (start, end) for group in accepted):
            return {"category": "riferimenti.normativo", "ambiguous": False,
                    "context_reason": "Data o componente esplicita della data di una disposizione normativa.",
                    "evidence_type": "law_date"}
    if label == "DOCID":
        for match in _LAW_IDENTIFIER.finditer(text, max(0, start - 130), window_end):
            if (not re.search(r"\n[ \t]*\n", match.group())
                    and any(match.span(group) == (start, end) for group in ("code", "number"))):
                return {"category": "riferimenti.normativo", "ambiguous": False,
                        "context_reason": "Numero identificativo di una disposizione normativa esplicita.",
                        "evidence_type": "law_identifier"}
    return None


def _boundary_street(text, start, end, entities):
    anchors = list(_BOUNDARY.finditer(text, max(0, start - 500), start))
    if not anchors:
        return None
    anchor = anchors[-1]
    if (_BOUNDARY_STOP.search(text, anchor.end(), start)
            or _broken(text, anchor.end(), start, semicolons=False)
            or any(e.get("label") in {"FULLNAME", "ORG"}
                   and anchor.end() <= e["start"] < e["end"] <= start for e in entities)):
        return None
    parcels = list(_BOUNDARY_PARCEL.finditer(text, anchor.end(), start))
    if parcels and _STREET.search(text, parcels[-1].end(), end):
        return "immobile.confine_indirizzo", "Indirizzo nella descrizione esplicita dei confini catastali del bene."
    return None


def evidence(text, entity):
    """Restituisce una decisione semantica forte o None, senza mutare l'entità."""
    start, end = _span(text, entity)
    label = entity.get("label")
    if label in {"DATE", "DOCID", "BUILDINGNUM"}:
        normative = _normative_evidence(text, start, end, label)
        if normative is not None:
            return normative
    if label == "DATE":
        for match in _CONSTRUCTION.finditer(text, max(0, start - 130), end):
            negated = re.search(r"\b(?:non|né|nè)\s+(?:(?:è|stat[oaie]|risulta|essere|mai|fu|"
                                r"viene|venne|più)\s+){0,5}$", text[max(0, match.start() - 70):match.start()], re.I)
            if match.span("date") == (start, end) and not negated:
                return {"category": "date_documentali.data", "ambiguous": False,
                        "context_reason": "Data limite nella dichiarazione esplicita di costruzione del bene.",
                        "evidence_type": "construction_threshold"}
        return None
    if label != "BUILDINGNUM":
        return None
    # Il punto nel numero di sottoparagrafo non è un separatore decimale.
    # La struttura deve essere completa e ancorata all'inizio della riga;
    # si mantengono gli span originali, inclusi frammenti e sola parentesi.
    for match in _SECTION.finditer(text, max(0, start - 30), min(len(text), end + 20)):
        if match.start("num") <= start < end <= match.end():
            return {"category": "riferimenti.articolo", "ambiguous": False,
                    "context_reason": "Componente della numerazione esplicita di sottoparagrafo, non un civico.",
                    "evidence_type": "article_number"}
    measure = cadastral_decimal_bounds(text, start, end)
    if measure == (start, end):
        return {"category": "immobile.catasto", "ambiguous": False,
                "context_reason": "Valore completo del campo metrico nella riga catastale del bene.",
                "evidence_type": "cadastral_measure"}
    for match in _DECIMAL.finditer(text, max(0, start - 24), min(len(text), end + 24)):
        if match.start() <= start < end <= match.end() and (start, end) != match.span():
            return {"category": "altri.buildingnum", "ambiguous": True,
                    "context_reason": "Il NER ha riconosciuto un frammento di un numero: verificare lo span.",
                    "evidence_type": "numeric_fragment"}
    for pattern in (_ARTICLE,):
        for match in pattern.finditer(text, max(0, start - 30), min(len(text), end + 20)):
            if match.span("num") == (start, end):
                return {"category": "riferimenti.articolo", "ambiguous": False,
                        "context_reason": "Numerazione esplicita di articolo o sottoparagrafo, non un civico.",
                        "evidence_type": "article_number"}
    anchor = _cat_section(text, start)
    if anchor:
        for match in _CADASTRAL_CLASS.finditer(text, max(anchor.end(), start - 30), min(len(text), end + 15)):
            if match.start("code") <= start < end <= match.end("code"):
                return {"category": "immobile.catasto", "ambiguous": False,
                        "context_reason": "Componente della categoria catastale esplicita nella scheda del bene.",
                        "evidence_type": "cadastral_class"}
    if not text[start:end].isdigit():
        return None
    prefix = text[max(0, start - 70):start]
    if _PARCEL_PREFIX.search(prefix):
        return {"category": "immobile.catasto", "ambiguous": False,
                "context_reason": "Numero preceduto dall'indicazione esplicita di particella catastale.",
                "evidence_type": "cadastral_parcel"}
    if anchor:
        # La riga comprende anche il numero da classificare: niente associazione
        # per valore fra particelle uguali in punti diversi del documento.
        for row in _SUB_ROW.finditer(text, max(anchor.end(), start - 24), min(len(text), end + 32)):
            if row.span("parcel") == (start, end) or row.span("sub") == (start, end):
                return {"category": "immobile.catasto", "ambiguous": False,
                        "context_reason": "Numero di particella o subalterno in una riga catastale esplicita.",
                        "evidence_type": "cadastral_row"}
    return None


def property_context(text, entity, entities=()):
    """Funzione del bene; entities mantiene la barriera dei nuovi soggetti NER."""
    start, end = _span(text, entity)
    label = entity.get("label")
    if label == "STREET":
        boundary = _boundary_street(text, start, end, entities)
        if boundary is not None:
            return boundary
    if label in {"CITY", "STREET", "ORG"}:
        for match in _NAME.finditer(text, max(0, start - 120), min(len(text), end + 120)):
            if (match.start() <= start < end <= match.end()
                    and start < match.end("name") and end > match.start("name")):
                return "immobile.denominazione", "Denominazione esplicita di un complesso Residence, non un comune o una via."
    if label not in _ADDRESS:
        return None
    anchors = list(_LOCATION.finditer(text, max(0, start - 700), start))
    if anchors:
        anchor = anchors[-1]
        before = text[anchor.start():anchor.end()]
        negated = re.search(r"\b(?:non|né|nè)\s+(?:(?:è|risulta|essere|più|stat[oaie]|mai|fu|viene)\s+){0,5}"
                            r"(?:sit[oaie]|ubicat[oaie]|post[oaie])\b", before, re.I)
        negated = negated or re.search(r"\b(?:non|né|nè)\s+(?:(?:è|risulta|essere|più|stat[oaie]|mai|fu|viene|un|una|il|la)\s+){0,5}$",
                                       text[max(0, anchor.start() - 45):anchor.start()], re.I)
        if (not negated and not _OTHER_ROLE.search(before)
                and not _OTHER_ROLE.search(text, anchor.end(), start)
                and not any(e.get("label") in {"FULLNAME", "ORG"}
                            and anchor.end() <= e["start"] < e["end"] <= start for e in entities)
                and not _broken(text, anchor.end(), start)):
            return "immobile." + _ADDRESS[label], "Ubicazione esplicita del fabbricato o bene descritto."
    anchor = _cat_section(text, start)
    if not anchor:
        return None
    # Il Comune censuario è riconoscibile dal campo Comune e dalla voce foglio.
    if label in {"CITY", "PROVINCE"}:
        prefix = text[max(anchor.end(), start - 100):start]
        suffix = text[end:min(len(text), end + 100)]
        if (re.search(r"\bComune\s+di\s+[^;.!?]{0,60}$", prefix, re.I)
                and re.match(r"[^;.!?]{0,40}\bfoglio\b", suffix, re.I)):
            return "immobile." + _ADDRESS[label], "Comune o provincia del censimento catastale esplicito."
    row = _cat_row(text, anchor, start)
    if not row:
        return None
    if any(e.get("label") in {"FULLNAME", "ORG"}
           and row.end() <= e["start"] < e["end"] <= start for e in entities):
        return None
    content = text[row.end():end]
    if label == "STREET" and _STREET.search(content):
        return "immobile.indirizzo", "Indirizzo riportato nella riga catastale del bene."
    if label == "BUILDINGNUM" and _STREET.search(content):
        prefix = text[row.end():start]
        if re.search(r"(?:\b" + _NUMBER + r"|\bciv\.)\s*$", prefix, re.I):
            return "immobile.civico", "Numero civico esplicito dell'indirizzo nella riga catastale."
    return None
