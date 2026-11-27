"""Formule documentali esplicite, senza modificare NER, valori o offset.

Le regole leggono solo finestre limitate del testo in memoria. Non deducono una
data documentale dal solo «in data» e non interpretano numeri isolati.
"""

import re


_NUMBER = r"(?:n\.\s*ri|nn\.?|n(?:\.?[°º]\.?|\.)?|numer[oi]|num\.)"
_REFERENCE = (r"(?:repertorio|rep\.?|raccolta|racc\.?|(?:protocollo|prot\.?)"
              r"(?:\s+(?:pratica|gener(?:ale)?\.?))?|progr(?:essivo)?\.?|R\.?E\.?A\.?)")
_REFERENCE_PREFIX = re.compile(r"\b" + _REFERENCE + r"\s*(?:con\s+)?(?:" + _NUMBER + r"\s*)?$", re.I)
_REFERENCE_VALUE = re.compile(r"^\s*" + _REFERENCE + r"(?=\s|$)", re.I)
_REFERENCE_SUFFIX = re.compile(r"^\s*" + _REFERENCE + r"(?=\s|$)", re.I)
_REGISTRATION_NUMBER = re.compile(
    r"\b(?:registrat[oaie]|trascritt[oaie])\b(?P<between>.{0,280}?)"
    r"\b(?:al|ai)\s+" + _NUMBER + r"\s*$", re.I | re.S)
_DATED_DOCUMENT_NUMBER = re.compile(
    r"\b(?:permesso\s+di\s+costruire|(?:SCIA|DIA)(?:\s+in\s+variante)?|"
    r"richiesta|protocollat[oaie])\s+(?:in\s+data|il)\s+"
    r"(?P<date>.{1,120}?)\s+(?:(?:al|ai)\s+)?" + _NUMBER + r"\s*$", re.I | re.S)
_WORDS = r"(?:[^\W\d_]+(?:['’\-][^\W\d_]+)*\s+){1,16}"
_DOCUMENT = (r"(?:atto(?:\s+di\s+compravendita)?|rogito|contratto(?:\s+preliminare)?|verbale|perizia|relazione|"
             r"certificato(?:\s+di\s+nascita)?|estratto(?:\s+di\s+nascita)?|"
             r"permesso\s+di\s+costruire|(?:SCIA|DIA)(?:\s+in\s+variante)?|richiesta|"
             r"iscrizione|trascrizione|decreto|ordinanza|licenza|concessione)")
_DATE_PREFIXES = tuple(re.compile(pattern, re.I) for pattern in (
    r"\b" + _DOCUMENT + r"\s+(?:del|in\s+data|il)\s*$",
    r"\b(?:registrat[oaie]|trascritt[oaie])\s+(?:a|in|presso)\s+"
    + r"(?P<place>" + _WORDS + r")(?:\d{1,2}\s+)?(?:in\s+data|il)\s*$",
    r"\b(?:presentat[oaie]|depositat[oaie]|rilasciat[oaie])\s+(?:al|dal|presso\s+il)"
    r"\s+Comune\s+di\s+(?P<place>" + _WORDS + r")(?:in\s+data|il)\s*$",
    r"\b(?:protocollat[oaie]|redatt[oaie]|registrat[oaie]|trascritt[oaie]|"
    r"depositat[oaie]|rilasciat[oaie]|emess[oaie]|stipulat[oaie]|presentat[oaie])"
    r"\s+(?:in\s+data|il)\s*$",
))
_NEGATION = re.compile(r"\b(?:non(?:\s+(?:è|era|risulta|essere|stat[oaie]|fu|viene|venne|più|mai|ancora|"
                       r"emess[oaie]|eseguit[oaie]|dispost[oaie]|effettuat[oaie]|previst[oaie]|rilasciat[oaie]|presentat[oaie]|"
                       r"il|la|un|una)){0,6}|né|nè)\s*$", re.I)
_BIRTH = re.compile(r"\b(?:nat[oaie0]|nascita)\b", re.I)
_CONTEXT_CHANGE = re.compile(r"\b(?:non|né|nè|nat[oaie0]|nascita|residente|residenti|"
                             r"residenza|domiciliat[oaie]|domicilio|matrimonio|decesso|"
                             r"decedut[oaie]|scadenza|decorrenza|immobil[ei]|confinant[ei]|"
                             r"proprietari[oaie]|dichiara|acquista|vende)\b", re.I)
_ADDRESS = re.compile(r"\b(?:via|viale|piazza|corso|civico|indirizzo)\b", re.I)
_ABBREVIATIONS = {"n", "nn", "num", "rep", "racc", "prot", "sig", "dott", "notaio",
                  "u", "t", "p", "g", "progr", "gener", "rea"}
_NUMBER_WORD = (r"(?:mille|mila|cento|cent|venti|vent|trenta|trent|quaranta|quarant|cinquanta|cinquant|"
                r"sessanta|sessant|settanta|settant|ottanta|ottant|novanta|novant|diciannove|"
                r"diciassette|diciotto|sedici|quindici|quattordici|tredici|dodici|undici|dieci|"
                r"nove|otto|sette|sei|cinque|quattro|tré|tre|due|uno|un|zero)+")
_DATE_FRAGMENT = re.compile(
    r"(?:\s|[()/.,'’\-]|\d+|\b(?:" + _NUMBER_WORD + r"|primo|gennaio|febbraio|marzo|aprile|"
    r"maggio|giugno|luglio|agosto|settembre|ottobre|novembre|dicembre|di|del|e)\b)*", re.I)
_PERMIT = re.compile(r"\b(?:permesso\s+di\s+costruire|(?:DIA|SCIA)(?:\s+in\s+variante)?)\b", re.I)
_INVOICE = re.compile(r"\bfattur[ae](?:\s+emess[ae])?\b", re.I)
_NUMBER_END = re.compile(r"\b" + _NUMBER + r"\s*$", re.I)
_IDENTIFIER = re.compile(r"\d+(?:[/-]\d+)*")
_REFERENCE_TOKENS = re.compile(r"\b(?:" + _NUMBER + r"|U\.\s*T\.|P\.\s*G\.|"
                               r"Prot\.(?:\s*Gener\.?)?|Progr\.)(?=\s|\d|$)", re.I)
_FINANCIAL_DATE = re.compile(r"\bbonifico(?:\s+bancario)?"
                             r"(?:\s+(?:eseguit[oa]|dispost[oa]|effettuat[oa]))?"
                             r"\s+(?:in\s+data|del|il)\s*$", re.I)
_CAPARRA_DATE = re.compile(r"\bcaparra(?:\s+confirmatoria)?"
                           r"(?:\s*(?:,|ed?|e)\s*(?:ora\s+)?imputat[oa]\s+(?:a|in\s+conto)\s+prezzo)?"
                           r"\s*\)?\s+(?:in\s+data|il)\s*$", re.I)
_AMOUNT_CUE = re.compile(r"\b(?:euro|EUR)\s+\d[\d.,]*\b", re.I)
_CREDIT_SUFFIX = re.compile(r"^\s+con\s+accredito\s+(?:sul|su|in)\s+"
                           r"(?:conto\s+corrente|c/c|IBAN)\b", re.I)
_WORK_DATE = re.compile(r"\binizio\s+(?:dei\s+)?lavori\s+(?:in\s+data|il)\s*$", re.I)


def _reference_sequence(fragment):
    """Solo sigle e numeri: la prosa non prolunga la clausola del permesso."""
    if not _bounded_clause(fragment) or _CONTEXT_CHANGE.search(fragment):
        return False
    residue = _REFERENCE_TOKENS.sub(" ", fragment)
    residue = _IDENTIFIER.sub(" ", residue)
    return not residue.strip(" \t\r\n,.:()-/")


def _invoice_context(prefix):
    matches = list(_INVOICE.finditer(prefix))
    if not matches:
        return None
    marker = matches[-1]
    tail = prefix[marker.end():]
    if _negated(prefix, marker.start()) or not _bounded_clause(tail) or _CONTEXT_CHANGE.search(tail):
        return None
    residue = re.sub(r"\bin\s+data\b|\bodierna\b", " ", tail, flags=re.I)
    residue = _REFERENCE_TOKENS.sub(" ", residue)
    return tail if _date_spelling(residue, allow_empty=True) else None


def _special_documentary(prefix, value, label, suffix=""):
    """Clausole aggiuntive con struttura verificata, senza NER o stato."""
    invoice = _invoice_context(prefix)
    if invoice is not None:
        if _NUMBER_END.search(invoice) and _IDENTIFIER.fullmatch(value.strip()):
            return "riferimenti.documentale", "Numero di fattura esplicitamente introdotto da n., non una data."
        if label == "DATE" and re.search(r"(?:\bin\s+data|\bdel)\s*$", invoice) and _date_spelling(value):
            return "date_documentali.data", "Data nella clausola esplicita di fatturazione."
    if label == "DATE":
        financial = _FINANCIAL_DATE.search(prefix)
        if financial and not _negated(prefix, financial.start()):
            return "economici.data", "Data del bonifico bancario esplicitamente indicato."
        caparra = _CAPARRA_DATE.search(prefix)
        if caparra and _CREDIT_SUFFIX.match(suffix):
            before = prefix[max(0, caparra.start() - 160):caparra.start()]
            amounts = list(_AMOUNT_CUE.finditer(before))
            if amounts:
                amount = amounts[-1]
                clause = before[amount.start():] + prefix[caparra.start():]
                negation_context = before[max(0, amount.start() - 70):]
                if (_bounded_clause(clause) and not _CONTEXT_CHANGE.search(clause)
                        and not re.search(r"\b(?:non|né|nè|mai)\b", negation_context, re.I)
                        and not _negated(prefix, caparra.start())):
                    return "economici.data", "Data della caparra con importo e accredito espliciti."
        work = _WORK_DATE.search(prefix)
        if work and not _negated(prefix, work.start()):
            return "date_documentali.data", "Data di inizio lavori esplicitamente indicata."
    permits = list(_PERMIT.finditer(prefix))
    if not permits:
        return None
    marker = permits[-1]
    if _negated(prefix, marker.start()):
        return None
    tail = prefix[marker.end():]
    if label == "DOCID" and _NUMBER_END.search(tail) and _reference_sequence(tail):
        return "riferimenti.documentale", "Numero del permesso o della pratica edilizia esplicitamente indicata."
    date_intro = re.search(r"\bin\s+data\s*$", tail, re.I)
    if label == "DATE" and date_intro and _reference_sequence(tail[:date_intro.start()]):
        return "date_documentali.data", "Data del provvedimento edilizio dopo i suoi riferimenti numerici."
    return None


def _bounded_clause(fragment):
    if re.search(r"[;!?]|\n[ \t]*\n", fragment):
        return False
    for match in re.finditer(r"\.(?=\s|$)", fragment):
        previous = re.search(r"(\w+)$", fragment[:match.start()])
        if not previous or previous.group(1).casefold() not in _ABBREVIATIONS:
            return False
    return True


def _negated(prefix, match_start):
    return bool(_NEGATION.search(prefix[max(0, match_start - 65):match_start]))


def _date_spelling(fragment, *, allow_empty=False):
    if not _DATE_FRAGMENT.fullmatch(fragment):
        return False
    return allow_empty or bool(re.search(r"\d|\b(?:" + _NUMBER_WORD + r"|primo)\b", fragment, re.I))


def documentary(text, start, end, label):
    """Restituisce (categoria, motivazione) solo per formule sufficientemente forti.

    La chiamata non cambia lo span e non classifica gli altri tipi di entità.
    None lascia la decisione al classificatore generale e alla revisione.
    """
    if label not in {"DATE", "DOCID"}:
        return None
    prefix = text[max(0, start - 360):start]
    value = text[start:end]
    # Una sigla dopo il numero non deve scavalcare la negazione del documento
    # esplicito nella medesima clausola (per esempio un permesso non emesso).
    for domain in (_PERMIT, _INVOICE):
        markers = list(domain.finditer(prefix))
        if markers:
            marker = markers[-1]
            if _negated(prefix, marker.start()) and _bounded_clause(prefix[marker.end():]):
                return None
    special = _special_documentary(prefix, value, label, text[end:end + 180])
    if special is not None:
        return special
    if label == "DOCID":
        explicit = _REFERENCE_PREFIX.search(prefix)
        if explicit and not _negated(prefix, explicit.start()):
            return "riferimenti.documentale", "Sigla documentale esplicita prima del numero."
        if _REFERENCE_VALUE.match(value):
            if not _NEGATION.search(prefix):
                return "riferimenti.documentale", "Sigla documentale contenuta nell'occorrenza."
        if (_REFERENCE_SUFFIX.match(text[end:end + 35])
                and re.fullmatch(r"\s*\d[\d\s/.-]*\s*", value)
                and not _NEGATION.search(prefix)):
            return "riferimenti.documentale", "Repertorio esplicito dopo il numero."
        for match in _REGISTRATION_NUMBER.finditer(prefix):
            between = match.group("between")
            if (_bounded_clause(between) and not _negated(prefix, match.start())
                    and not _CONTEXT_CHANGE.search(between) and not _ADDRESS.search(between)):
                return "riferimenti.documentale", "Numero nella clausola di registrazione o trascrizione."
        dated = _DATED_DOCUMENT_NUMBER.search(prefix)
        if (dated and not _negated(prefix, dated.start())
                and _bounded_clause(dated.group("date"))
                and _date_spelling(dated.group("date"))):
            return "riferimenti.documentale", "Numero del provvedimento o protocollo dopo una data esplicita."
        return None
    for pattern in _DATE_PREFIXES:
        match = pattern.search(prefix)
        if match and _bounded_clause(match.group()) and not _negated(prefix, match.start()):
            # Il complemento di luogo non può assorbire una nuova residenza,
            # nascita o evento solo perché contiene parole prima di «in data».
            place = match.groupdict().get("place")
            if place and _CONTEXT_CHANGE.search(place):
                continue
            return "date_documentali.data", "Formula documentale esplicita prima della data."
    # La formula di apertura richiede sia l'intestazione sia il notaio successivo.
    # «Nato il giorno ...» non diventa la data dell'atto per prossimità al notaio.
    opening = re.search(r"\bil\s+giorno\b(?P<date>[^;!?]{0,160})$", prefix, re.I)
    if opening and not _BIRTH.search(prefix[max(0, opening.start() - 90):]):
        fragment = opening.group("date")
        if not _bounded_clause(fragment) or not _date_spelling(fragment, allow_empty=True):
            return None
        if not _date_spelling(value):
            return None
        heading = text[max(0, start - 800):start]
        suffix = text[end:end + 512]
        if (re.search(r"\bREPUBBLICA\s+ITALIANA\b", heading, re.I)
                and re.search(r"\b(?:davanti|avanti)\s+a\s+me\b.{0,240}?\bnotaio\b", suffix, re.I | re.S)
                and not _negated(prefix, opening.start())):
            return "date_documentali.data", "Data nella formula notarile di apertura dell'atto."
    return None
