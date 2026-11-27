# -*- coding: utf-8 -*-
"""Classificazione contestuale locale, successiva al NER e indipendente dal modello.

Gli offset restano quelli del testo ricevuto (code point Python, fine esclusa).
Le decisioni si riferiscono a occurrence_id, mai al valore testuale. Le regole
non aggiungono entita' e si astengono quando manca un contesto riconoscibile.
"""

import re

from notarial_rules import documentary
from notarial_places import classify_place
from notarial_evidence import evidence, property_context
from company_context import classify_company


TAXONOMY_VERSION = 4
TAXONOMY = []


def _category(identifier, title, labels=(), parent=None):
    TAXONOMY.append({"id": identifier, "parent": parent, "title": title,
                     "labels": list(labels)})


for _id, _title in (
    ("identificativi", "Identità e identificativi"), ("nascita", "Nascita"),
    ("residenza", "Residenza"), ("domicilio", "Domicilio"),
    ("immobile", "Immobile oggetto di perizia"),
    ("sede_societaria", "Sedi societarie e d'impresa"),
    ("date_documentali", "Date documentali"), ("recapiti", "Recapiti"),
    ("luoghi_documentali", "Luoghi dell'atto, degli uffici e dei professionisti"),
    ("economici", "Dati economici"), ("riferimenti", "Riferimenti degli atti"),
    ("altri", "Altri dati / contesto da verificare"),
):
    _category(_id, _title)

_DIRECT = {
    "FULLNAME": ("identificativi.nome_completo", "Nome e cognome"),
    "CF": ("identificativi.codice_fiscale", "Codice fiscale"),
    "ID_DOC": ("identificativi.documento", "Documento d'identità"),
    "PIVA": ("identificativi.partita_iva", "Partita IVA"),
    "ORG": ("identificativi.organizzazione", "Denominazione societaria"),
    "EMAIL": ("recapiti.email", "Email"),
    "TELEPHONENUM": ("recapiti.telefono", "Telefono"),
    "AMOUNT": ("economici.importo", "Importo"),
    "IBAN": ("economici.iban", "IBAN / conto"),
    "CREDITCARDNUMBER": ("economici.carta", "Carta di pagamento"),
    "CATASTO": ("immobile.catasto", "Dati catastali"),
}
for _label, (_id, _title) in _DIRECT.items():
    _labels = (_label, "BUILDINGNUM") if _label == "CATASTO" else (_label,)
    _category(_id, _title, _labels, _id.split(".")[0])
_category("recapiti.pec", "PEC", ("EMAIL",), "recapiti")
_category("nascita.luogo", "Luogo di nascita", ("CITY", "PROVINCE"), "nascita")
_category("nascita.data", "Data di nascita", ("DATE",), "nascita")
_category("date_documentali.data", "Data documentale", ("DATE",), "date_documentali")
_category("riferimenti.giudiziario", "Riferimento giudiziario", ("DOCID",), "riferimenti")
_category("riferimenti.documentale", "Protocollo / repertorio / altro atto", ("DOCID", "DATE"),
          "riferimenti")
_category("riferimenti.articolo", "Articolo / paragrafo del documento", ("BUILDINGNUM",), "riferimenti")
_category("riferimenti.normativo", "Riferimento normativo", ("BUILDINGNUM", "DOCID", "DATE"), "riferimenti")
_category("immobile.denominazione", "Denominazione del fabbricato / complesso", ("CITY", "STREET", "ORG"), "immobile")
_category("immobile.confine_indirizzo", "Indirizzo di confine del bene", ("STREET",), "immobile")
_category("economici.data", "Data di pagamento / operazione", ("DATE",), "economici")

_ADDRESS = {"CITY": ("comune", "Comune"), "STREET": ("indirizzo", "Via / indirizzo"),
            "BUILDINGNUM": ("civico", "Numero civico"), "ZIPCODE": ("cap", "CAP"),
            "PROVINCE": ("provincia", "Provincia")}
for _parent in ("residenza", "domicilio", "immobile", "sede_societaria"):
    for _label, (_suffix, _title) in _ADDRESS.items():
        _title = ("Sede legale — " + _title) if _parent == "sede_societaria" else _title
        _category(f"{_parent}.{_suffix}", _title, (_label,), _parent)
for _label, (_suffix, _title) in _ADDRESS.items():
    _category(f"sede_societaria.sede_{_suffix}", "Sede dichiarata — " + _title, (_label,), "sede_societaria")
for _place, _title in (("stipula", "Luogo di stipula"), ("sede_notaio", "Sede del notaio"),
                       ("ufficio", "Ufficio / autorità del documento"),
                       ("sede_professionista", "Luogo del professionista")):
    for _label, (_suffix, _field) in _ADDRESS.items():
        _category(f"luoghi_documentali.{_place}_{_suffix}", _title + " — " + _field,
                  (_label,), "luoghi_documentali")
_category("luoghi_documentali.distretto_notarile", "Distretto notarile", ("CITY", "PROVINCE"),
          "luoghi_documentali")
_category("luoghi_documentali.registro_imprese_comune", "Registro delle Imprese — Comune", ("CITY",),
          "luoghi_documentali")
_category("luoghi_documentali.registro_imprese_provincia", "Registro delle Imprese — Provincia", ("PROVINCE",),
          "luoghi_documentali")
_category("luoghi_documentali.camera_commercio_comune", "Camera di Commercio — Comune", ("CITY",),
          "luoghi_documentali")
_category("luoghi_documentali.camera_commercio_provincia", "Camera di Commercio — Provincia", ("PROVINCE",),
          "luoghi_documentali")
_category("luoghi_documentali.ufficio_unita", "Numero / unità dell'ufficio di registrazione", ("BUILDINGNUM",),
          "luoghi_documentali")

_OTHER = {
    "AGE": "Età", "GENDER": "Genere", "TIME": "Ora", "DATE": "Data senza contesto",
    "CITY": "Comune senza contesto", "STREET": "Indirizzo senza contesto",
    "BUILDINGNUM": "Civico senza contesto", "ZIPCODE": "CAP senza contesto",
    "PROVINCE": "Provincia senza contesto", "TARGA": "Targa", "URL": "Indirizzo web",
    "IPADDR": "Indirizzo IP",
}
for _label, _title in _OTHER.items():
    _category("altri." + _label.lower(), _title, (_label,), "altri")
_category("altri.non_classificato", "Tipo non classificato", (), "altri")

_BY_ID = {c["id"]: c for c in TAXONOMY}
_LEAVES = {c["id"] for c in TAXONOMY if c["parent"] is not None}

# Gli indicatori aprono un intervallo; un altro indicatore, una nuova persona,
# una frase o un paragrafo lo chiudono. Le schede con intestazione esplicita
# possono proseguire oltre gli a-capo, solo lungo campi di indirizzo riconoscibili.
_PROPERTY = r"(?:immobil[ei]|fabbricat[oi]|terren[oi]|ben[ei]|appartament[oi]|unit[àa]\s+immobiliar[ei])"
_DOCUMENT = (r"(?:atto(?:\s+di\s+nascita)?|rogito|contratto|sentenza|udienza|verbale|perizia|relazione|"
             r"certificato(?:\s+di\s+nascita)?|estratto(?:\s+di\s+nascita)?|visura|"
             r"iscrizione|trascrizione|pignoramento|sopralluogo|decreto|ordinanza|"
             r"incarico|nomina|permesso|concessione|licenza|istanza)")
_RULES = (
    ("nascita", r"\b(?:nat[oaie0]|nascita)\b"),
    ("residenza", r"\b(?:residente|residenti|resldente|residenza|res\.te|resid\.|res\.)(?=[\s:;/,]|$)"),
    ("domicilio", r"\b(?:domiciliat[oaie]|domicilio(?:\s+eletto)?|dom\.to|dom\.ta|dom\.)(?=[\s:;/,]|$)"),
    ("immobile", r"\b" + _PROPERTY + r"\s+(?:(?:oggetto\s+di\s+(?:perizia|stima))\s+)?"
                 r"(?:sit[oaie0]|ubicat[oaie]|post[oaie])\b"
                 r"|\b(?:ubicazione|indirizzo)\s+(?:dell['’]\s*|del\s+|dei\s+)?" + _PROPERTY + r"\b"),
    ("sede_societaria", r"\bsede\s+legale\b"),
    ("date_documentali", r"\b(?:" + _DOCUMENT + r"\s+(?:del|in\s+data|il)|"
                          r"(?:stipulat[oa]|registrat[oa]|depositat[oa]|redatt[oa]|"
                          r"emess[oa]|rilasciat[oa])\s+(?:il|in\s+data)|"
                          r"trascritt[oaie]|data\s+(?:(?:dell['’]\s*|del\s+|di\s+))?"
                          r"(?:documento|" + _DOCUMENT + r"))\b"),
)
_ANCHOR = re.compile("|".join(f"(?P<{role}>{pattern})" for role, pattern in _RULES), re.I)
_BREAK = re.compile(r"[;!?]|\n[ \t]*\n|\.(?=\s|$)")
_ABBREVIATIONS = {"sig", "sigg", "dott", "avv", "geom", "arch", "ing", "dr", "n",
                  "num", "nr", "civ", "prov", "loc", "fraz", "s", "r", "l", "p",
                  "res", "resid", "dom", "v", "srl", "spa"}
_FORM_FIELD = re.compile(r"^(?:comune|citt[àa]|indirizzo|via|piazza|civico|numero\s+civico|"
                         r"n\.|cap|provincia|prov\.|localit[àa]|frazione|luogo|data)\s*:"
                         r"|^(?:comune|citt[àa]|provincia)\s+di(?:\s|$)", re.I)
_FOREIGN_CONTEXT = re.compile(r"\b(?:tribunale|corte\s+d['’]appello|conservatoria|"
                              r"agenzia\s+delle\s+entrate|matrimonio|decesso|decedut[oa]|"
                              r"scadenza|decorrenza|confinant[ei]|confini|proveniente)\b", re.I)


def _negated(text, anchor_start):
    return bool(re.search(r"\bnon\s+(?:(?:è|risulta|essere|più|stat[oaie]|mai|fu|viene|un|una|il|la)\s+){0,5}$|\b(?:né|nè)\s*$",
                          text[max(0, anchor_start - 45):anchor_start], re.I))


def _has_boundary(text, start, end, *, paragraphs=True):
    for match in _BREAK.finditer(text, start, end):
        if not paragraphs and match.group().startswith("\n"):
            continue
        if match.group() == ".":
            previous = re.search(r"([\w]+)$", text[start:match.start()])
            if previous and previous.group(1).casefold() in _ABBREVIATIONS:
                continue
        return True
    return False


def _anchors(text):
    anchors = []
    for match in _ANCHOR.finditer(text):
        role = match.lastgroup
        prefix = text[max(0, match.start() - 45):match.start()]
        # "Certificato di nascita del ..." descrive il documento, non la nascita.
        if role == "nascita" and re.search(r"\b(?:atto|certificato|estratto)\s+di\s*$", prefix, re.I):
            continue
        uncertain = bool(re.search(r"\b(?:ivi|ivi\s+pure|colà)\s*$",
                                   prefix, re.I))
        uncertain = uncertain or _negated(text, match.start())
        if anchors:
            previous = anchors[-1]
            connector = text[previous[1]:match.start()].strip().casefold()
            if previous[2] != role and connector in {"e", "ed", "o", "e/o", "/", "ovvero"}:
                uncertain = True
        anchors.append((match.start(), match.end(), role, uncertain))
    return anchors


def _form_scope(text, anchor, start, entities, cache):
    """None per la prosa, bool per una scheda con intestazione autonoma.

    Non si estende indiscriminatamente il ruolo a tutto il paragrafo: ogni
    riga successiva deve essere un campo etichettato o soli componenti di
    indirizzo già riconosciuti. Si verifica tutta la riga corrente, perché
    la prosa successiva all'entità può smentire l'interpretazione del campo.
    """
    a_start, a_end, role, _ = anchor
    line_start = text.rfind("\n", 0, a_start) + 1
    line_end = text.find("\n", a_end)
    if line_end == -1 or start < line_end:
        return None
    prefix = text[line_start:a_start].strip()
    suffix = text[a_end:line_end].strip()
    if (suffix not in {"", ":"} or not re.fullmatch(
            r"(?:\d+[.)]?\s*)?(?:(?:dati|dati\s+di|dati\s+della)\s*)?", prefix, re.I)):
        return None
    if role not in {"residenza", "domicilio", "immobile", "sede_societaria", "nascita"}:
        return None
    row_start = line_end + 1
    allowed = set(_ADDRESS) | ({"DATE"} if role == "nascita" else set())
    row_end = text.find("\n", start)
    if row_end == -1:
        row_end = len(text)
    previous = cache.get(a_start)
    if previous is not None:
        row_start, valid = previous
        if not valid or row_start == row_end:
            return valid
    for line in text[row_start:row_end].splitlines(keepends=True):
        if line.strip() and not _FORM_FIELD.match(line.strip()):
            # Nessuna inferenza basata sul valore: si tolgono soltanto gli span
            # riconosciuti di questa riga per controllarne la struttura restante.
            residue = line
            for e in reversed(entities):
                if e["label"] in allowed and row_start <= e["start"] < row_start + len(line):
                    left, right = e["start"] - row_start, min(e["end"] - row_start, len(line))
                    residue = residue[:left] + residue[right:]
            residue = re.sub(r"\b(?:n\.|civ\.|cap|via|viale|piazza|corso)\s*", "", residue, flags=re.I)
            if residue.strip(" \t\r\n,.:;()-/"):
                cache[a_start] = (row_end, False)
                return False
        row_start += len(line)
    cache[a_start] = (row_end, True)
    return True


def classify(text, entities):
    """Copia e ordina gli span senza cambiare riconoscimento, offset o valori.

    Il punteggio del NER non e' una probabilita' della classificazione contestuale.
    Ogni ambiguita' richiede una decisione esplicita prima dell'esportazione.
    """
    if not isinstance(text, str):
        raise ValueError("Il testo deve essere una stringa.")
    result = []
    for entity in entities:
        if not isinstance(entity, dict):
            raise ValueError("Entità non valida.")
        start, end = entity.get("start"), entity.get("end")
        if (type(start) is not int or type(end) is not int or not 0 <= start < end <= len(text)
                or not isinstance(entity.get("label"), str)):
            raise ValueError("Intervallo o tipo dell'entità non valido.")
        result.append(dict(entity))
    result.sort(key=lambda e: (e["start"], e["end"], e["label"]))
    anchors = _anchors(text)
    names = [e for e in result if e["label"] in {"FULLNAME", "ORG"}]
    addresses = [e for e in result if e["label"] in _ADDRESS]
    form_scopes = {}  # sola durata della chiamata; ogni riga della scheda è letta una volta
    overlaps = set()
    previous_end, previous_index = -1, -1
    for i, entity in enumerate(result):
        if entity["start"] < previous_end:
            overlaps.update((previous_index, i))
        if entity["end"] > previous_end:
            previous_end, previous_index = entity["end"], i

    for i, entity in enumerate(result):
        label, start, end = entity["label"], entity["start"], entity["end"]
        category = _DIRECT.get(label, ("altri." + label.lower(), ""))[0]
        if category not in _BY_ID:
            category = "altri.non_classificato"
        ambiguous = label in _ADDRESS or label == "DATE" or category == "altri.non_classificato"
        reason = "Tipo riconosciuto; nessuna distinzione contestuale richiesta."
        if ambiguous:
            reason = "Contesto assente o non riconosciuto: revisione necessaria."
        active = next((a for a in reversed(anchors) if a[1] <= start), None)
        if active and (label in _ADDRESS or label == "DATE"):
            a_start, a_end, role, uncertain = active
            form_scope = _form_scope(text, active, start, result, form_scopes)
            interrupted = (form_scope is False or
                           _has_boundary(text, a_end, start, paragraphs=form_scope is not True))
            # "Via del Tribunale" è un indirizzo, non un cambio di ufficio.
            interrupted = interrupted or any(
                not any(e["start"] <= m.start() and m.end() <= e["end"] for e in addresses)
                for m in _FOREIGN_CONTEXT.finditer(text, a_end, start))
            negated = _negated(text, a_start)
            new_person = any(a_end <= n["start"] and n["end"] <= start for n in names)
            # Una formula dopo un soggetto diverso non eredita il suo indirizzo.
            if not interrupted and not new_person and not negated:
                if role in {"residenza", "domicilio", "immobile", "sede_societaria"} and label in _ADDRESS:
                    category = role + "." + _ADDRESS[label][0]
                    ambiguous = uncertain
                elif role == "nascita" and label in {"CITY", "PROVINCE", "DATE"}:
                    category = "nascita." + ("data" if label == "DATE" else "luogo")
                    ambiguous = uncertain
                elif role == "date_documentali" and label == "DATE":
                    category, ambiguous = "date_documentali.data", uncertain
                if not ambiguous:
                    reason = "Indicatore esplicito: " + role.replace("_", " ") + "."
                elif uncertain:
                    reason = "Indicatori relativi o conflittuali: revisione necessaria."
            elif negated:
                reason = "Indicatore negato: ruolo da verificare."
            elif interrupted:
                reason = "Cambio di frase, sezione o contesto: ruolo da verificare."
            elif new_person:
                reason = "Nuovo soggetto dopo l'indicatore: ruolo da verificare."
        if label == "EMAIL":
            prefix = text[max(0, start - 35):start]
            if re.search(r"\bPEC\b[^;.!?\n]*$", prefix, re.I):
                category = "recapiti.pec"
        if label == "DOCID":
            value = text[start:end]
            prefix = text[max(0, start - 45):start]
            judicial = re.search(r"\b(?:R\.?\s*G\.?|sentenza|procedimento)\b", value, re.I)
            judicial = judicial or re.search(r"\b(?:R\.?\s*G\.?\s*(?:E\.?)?|sentenza|procedimento)"
                                             r"\s*(?:n\.?\s*)?$", prefix, re.I)
            category = "riferimenti." + ("giudiziario" if judicial else "documentale")
            explicit = judicial
            ambiguous = not bool(explicit)
            reason = ("Sigla o riferimento documentale esplicito." if explicit else
                      "Riferimento senza sigla riconoscibile: revisione necessaria.")
        decision = (documentary(text, start, end, label) or classify_company(text, entity, result)
                    or classify_place(text, start, end, label)
                    or property_context(text, entity, result))
        if decision is not None:
            proposed, explanation = decision
            conflicting = (ambiguous and proposed == category and
                           reason == "Indicatori relativi o conflittuali: revisione necessaria.")
            if not conflicting:
                category, ambiguous, reason = proposed, False, explanation
        correction = evidence(text, entity)
        if correction is not None:
            category, ambiguous, reason = (correction["category"], correction["ambiguous"],
                                            correction["context_reason"])
            entity["evidence_type"] = correction["evidence_type"]
        if i in overlaps:
            ambiguous, reason = True, "Intervalli sovrapposti: revisione necessaria."
        entity.update(occurrence_id=f"e{i + 1:06d}", category=category,
                      ambiguous=ambiguous, context_reason=reason)
    return result


def policy_label(entity):
    """Compatibilità dei tag legacy dopo una correzione semantica esplicita.

    Il tipo NER resta invariato. Un numero di particella non eredita la
    preferenza dei civici soltanto perché il modello lo ha chiamato così.
    Le scelte contestuali e individuali continuano a prevalere.
    """
    category, label = entity["category"], entity["label"]
    if label == "BUILDINGNUM":
        if category == "immobile.catasto":
            return "CATASTO"
        if category in {"riferimenti.articolo", "riferimenti.normativo"}:
            return "DOCID"
        if category == "luoghi_documentali.ufficio_unita":
            return "DOCID"
    if label == "DATE" and category == "riferimenti.documentale":
        return "DOCID"
    if category == "immobile.denominazione":
        return "ORG"
    return label


def select(entities, selected_categories=None, overrides=None, classifications=None):
    """Applica scelte per categoria e override per ID, senza alterare gli span.

    None seleziona tutte le categorie; [] nessuna. Un padre seleziona i suoi figli.
    Correzione o override esplicito risolvono la revisione di quella occorrenza.
    """
    entities = list(entities)
    if any(not isinstance(e, dict) or not isinstance(e.get("occurrence_id"), str)
           or not e["occurrence_id"] for e in entities):
        raise ValueError("ID delle occorrenze mancanti o duplicati.")
    ids = {e["occurrence_id"] for e in entities}
    if len(ids) != len(entities):
        raise ValueError("ID delle occorrenze mancanti o duplicati.")
    if selected_categories is not None:
        if not isinstance(selected_categories, (list, tuple, set)):
            raise ValueError("La selezione delle categorie deve essere una lista.")
        if any(not isinstance(c, str) or c not in _BY_ID for c in selected_categories):
            raise ValueError("Categoria selezionata sconosciuta.")
        selected_categories = set(selected_categories)
    overrides = {} if overrides is None else overrides
    classifications = {} if classifications is None else classifications
    if not isinstance(overrides, dict) or not isinstance(classifications, dict):
        raise ValueError("Le correzioni devono essere oggetti per ID occorrenza.")
    if not overrides.keys() <= ids or not classifications.keys() <= ids:
        raise ValueError("ID occorrenza sconosciuto.")
    if any(type(v) is not bool for v in overrides.values()):
        raise ValueError("La scelta della singola occorrenza deve essere booleana.")
    if any(not isinstance(c, str) or c not in _LEAVES for c in classifications.values()):
        raise ValueError("La correzione richiede una sottocategoria valida.")
    result = []
    for entity in entities:
        e = dict(entity)
        identifier = e["occurrence_id"]
        category = classifications.get(identifier, e.get("category"))
        if not isinstance(category, str) or category not in _LEAVES:
            raise ValueError("Sottocategoria dell'occorrenza non valida.")
        allowed = _BY_ID[category]["labels"]
        if identifier in classifications and allowed and e["label"] not in allowed:
            raise ValueError("Sottocategoria incompatibile con il tipo riconosciuto.")
        e["category"] = category
        selected = selected_categories is None
        current = category
        while selected_categories is not None and current is not None:
            selected = selected or current in selected_categories
            current = _BY_ID[current]["parent"]
        e["selected"] = overrides.get(identifier, selected)
        reviewed = identifier in overrides or identifier in classifications
        e["review_status"] = "reviewed" if reviewed else ("pending" if e.get("ambiguous") else "automatic")
        result.append(e)
    return result
