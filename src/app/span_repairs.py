"""Riparazione circoscritta di span numerici prima degli ID della revisione.

Il testo canonico resta immutato. Si completa un decimale soltanto se la
struttura catastale ne dimostra i confini e non si invadono altre entità.
I metadati di provenienza contengono offset e tipi, mai valori originali.
"""

from notarial_evidence import cadastral_decimal_bounds


def normalize(text, entities):
    """Copia span validi e completa i soli decimali catastali dimostrabili.

    Frammenti BUILDINGNUM dello stesso valore diventano una sola occorrenza.
    Una collisione con un altro tipo lascia gli span originali da revisionare.
    L'operazione precede placeholder, alias geometrici e scelte dell'utente.
    """
    if not isinstance(text, str):
        raise ValueError("Il testo deve essere una stringa.")
    spans = []
    for entity in entities:
        if not isinstance(entity, dict):
            raise ValueError("Entità non valida.")
        start, end = entity.get("start"), entity.get("end")
        if (type(start) is not int or type(end) is not int or not 0 <= start < end <= len(text)
                or not isinstance(entity.get("label"), str)):
            raise ValueError("Intervallo o tipo dell'entità non valido.")
        spans.append(dict(entity))
    spans.sort(key=lambda e: (e["start"], e["end"]))
    if any(a["end"] > b["start"] for a, b in zip(spans, spans[1:])):
        raise ValueError("La riparazione richiede gli span non sovrapposti della fusione NER.")
    proposals = {}
    for entity in spans:
        if entity["label"] != "BUILDINGNUM":
            continue
        bounds = cadastral_decimal_bounds(text, entity["start"], entity["end"])
        if bounds is None or bounds == (entity["start"], entity["end"]):
            continue
        left, right = bounds
        inside = [(i, e) for i, e in enumerate(spans) if e["start"] < right and left < e["end"]]
        if any(e["label"] != "BUILDINGNUM" or not left <= e["start"] < e["end"] <= right
               for _, e in inside):
            continue
        proposals[bounds] = inside
    consumed, repaired = set(), []
    for (left, right), inside in sorted(proposals.items()):
        first = dict(inside[0][1])
        first.update(start=left, end=right, span_repair="cadastral_decimal",
                     recognition_spans=[{k: e[k] for k in ("start", "end", "label")} for _, e in inside])
        repaired.append(first)
        consumed.update(i for i, _ in inside)
    result = [e for i, e in enumerate(spans) if i not in consumed] + repaired
    result.sort(key=lambda e: (e["start"], e["end"]))
    if any(a["end"] > b["start"] for a, b in zip(result, result[1:])):
        raise ValueError("Riparazione degli intervalli non univoca: revisione necessaria.")
    return result
