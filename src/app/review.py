"""Decisioni per occorrenza e sostituzioni sullo snapshot immutabile.

Nessun modello, I/O, cache o valore originale nei risultati di esportazione
irreversibili. La revisione locale dei valori deve essere richiesta esplicitamente.
"""

from collections import Counter

import contextual


def validate_spans(text, entities):
    end = 0
    ids = set()
    for e in entities:
        start, stop = e["start"], e["end"]
        if (type(start) is not int or type(stop) is not int
                or not end <= start < stop <= len(text)
                or e["occurrence_id"] in ids):
            raise ValueError("Intervalli delle occorrenze non validi o sovrapposti.")
        ids.add(e["occurrence_id"])
        end = stop


def assign_placeholders(entities):
    """Numerazione stabile per occorrenza; l'uguaglianza non prova la coreferenza."""
    counters = Counter()
    result = []
    for entity in entities:
        e = dict(entity)
        counters[e["label"]] += 1
        e["ph"] = f"[{e['label']}_{counters[e['label']]}]"
        result.append(e)
    return result


def render(text, entities, *, categories=None, overrides=None,
           classifications=None, excluded=(), mapping_enabled=False,
           local_review=False, n_chunks=0):
    validate_spans(text, entities)
    chosen = contextual.select(entities, categories, overrides, classifications)
    for e in chosen:
        if (categories is None and contextual.policy_label(e) in excluded
                and e["occurrence_id"] not in (overrides or {})):
            e["selected"] = False
    segments, output, mapping, occurrences = [], [], {}, []
    by_label, by_source = Counter(), Counter()
    pos = 0
    for e in chosen:
        value = text[e["start"]:e["end"]]
        if e["start"] > pos:
            plain = text[pos:e["start"]]
            segments.append({"t": plain})
            output.append(plain)
        segment = {k: e[k] for k in ("label", "ph", "occurrence_id", "category",
                                    "selected", "ambiguous", "review_status")}
        segment.update(src=e["source"], validated=e["validated"])
        if local_review or mapping_enabled or not e["selected"]:
            segment["t"] = value
        segments.append(segment)
        output.append(e["ph"] if e["selected"] else value)
        if e["selected"]:
            by_label[e["label"]] += 1
            by_source[e["source"]] += 1
            if mapping_enabled:
                mapping[e["ph"]] = value
        occurrence = {k: v for k, v in e.items() if k not in ("word", "value")}
        if local_review:
            occurrence["value"] = value
            # Offset Python: niente slicing JS (UTF-16) e nessuna copia nei log
            # o nei risultati irreversibili. Il contesto vive solo nella revisione.
            left, right = max(0, e["start"] - 90), min(len(text), e["end"] + 90)
            occurrence["context_excerpt"] = (("…" if left else "") + text[left:right]
                                             + ("…" if right < len(text) else ""))
        occurrences.append(occurrence)
        pos = e["end"]
    output.append(text[pos:])
    if pos < len(text):
        segments.append({"t": text[pos:]})
    unresolved = [e["occurrence_id"] for e in chosen
                  if e["review_status"] == "pending"]
    result = {
        "segments": segments, "anonymized_text": "".join(output),
        "occurrences": occurrences, "mapping": mapping,
        "mapping_enabled": mapping_enabled, "n_chunks": n_chunks,
        "n_chars": len(text), "n_entities": sum(by_label.values()),
        "n_unique": len({e["ph"] for e in chosen if e["selected"]}), "by_label": dict(by_label),
        "by_source": dict(by_source), "excluded_tags": sorted(excluded),
        "unresolved": unresolved, "offset_unit": "unicode_codepoint",
        "verification": {"status": "review_required" if unresolved else "ready",
                         "complete_anonymization": False},
    }
    if local_review:
        result["review_text"] = text
    return result, chosen
