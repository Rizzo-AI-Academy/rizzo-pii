# -*- coding: utf-8 -*-
"""Ripristino del mapping di anonimizzazione senza rompere la sintassi JSON.

``/analyze`` restituisce ``anonymized_text`` con i placeholder (``[AMOUNT_1]``)
e un ``mapping`` {placeholder: valore} i cui valori sono le stringhe originali
del documento (es. ``"1.500,00"``). Una sostituzione di sottostringa su un
payload JSON produce ``""1.500,00""`` (sintassi rotta) quando il placeholder e'
dentro una stringa quotata, e non normalizza gli importi in formato italiano.

Questo modulo restituisce una copia del payload con i placeholder sostituiti:

- su strutture JSON (dict/list, o stringa JSON anche con fence ```json)
  la sintassi resta valida;
- una foglia che e' ESATTAMENTE un placeholder di tipo monetario (``AMOUNT``)
  diventa il numero normalizzato (``"1.500,00"`` -> ``1500.0``);
- una foglia che contiene il placeholder come sottostringa resta una stringa
  (``"Totale: [AMOUNT_1]"`` -> ``"Totale: 1.500,00"``);
- un testo non-JSON usa la sostituzione diretta (compatibilita').

Nessuna dipendenza oltre la stdlib: importabile e testabile in CI senza
torch/transformers/flask. Le chiavi del mapping e del payload non vengono mai
toccate; le chiavi del mapping non vengono mai sostituite nei valori.
"""

import json
import re
from typing import Any, Mapping

# placeholder emessi da /analyze: [LABEL_n] (n progressivo per occorrenza)
_PH_RE = re.compile(r"\[([A-Z]+)(?:_\d+)?\]")
# etichette monetarie del vocabolario del progetto (detectors.py: AMOUNT)
_MONEY_LABELS = frozenset({"AMOUNT"})
# numero in formato italiano: 1.234,56 | 1234,56 | 1234.56 (€/EUR/euro tollerati)
_NUM_IT_RE = re.compile(
    r"[+-]?\d{1,3}(?:\.\d{3})+(?:,\d+)?|[+-]?\d+(?:[.,]\d+)?"
)


def restore(payload: Any, mapping: Mapping[str, str]) -> Any:
    """Ripristina i placeholder di ``mapping`` in ``payload``.

    ``payload`` puo' essere una stringa (testo libero o JSON, anche con fence
    ```json```) oppure una struttura dict/list. Il risultato mantiene il tipo
    di ingresso: stringa JSON -> stringa JSON ri-serializzata, struttura ->
    struttura copiata, testo -> testo sostituito.
    """
    if not mapping:
        return payload
    if isinstance(payload, str):
        stripped = _strip_json_fence(payload)
        try:
            data = json.loads(stripped)
        except (ValueError, TypeError):
            return _replace_text(payload, mapping)
        if isinstance(data, (dict, list)):
            restored = _restore_structure(data, mapping)
            return json.dumps(restored, ensure_ascii=False)
        # JSON scalare (es. la stringa nuda "[AMOUNT_1]"): ripristina il valore
        # e lo ri-serializza (stringa o numero, mai sintassi rotta)
        return json.dumps(_restore_leaf(data, mapping), ensure_ascii=False)
    if isinstance(payload, (dict, list)):
        return _restore_structure(payload, mapping)
    return payload


def _strip_json_fence(text: str) -> str:
    """Rimuove una fence markdown ```json ... ``` attorno al payload."""
    return re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.IGNORECASE)


def _restore_structure(node: Any, mapping: Mapping[str, str]) -> Any:
    """Copia ricorsiva con i placeholder sostituiti solo nei valori (foglie)."""
    if isinstance(node, dict):
        return {k: _restore_structure(v, mapping) for k, v in node.items()}
    if isinstance(node, list):
        return [_restore_structure(v, mapping) for v in node]
    if isinstance(node, str):
        return _restore_leaf(node, mapping)
    return node


def _restore_leaf(value: str, mapping: Mapping[str, str]) -> Any:
    """Foglia stringa: placeholder intero -> valore (numero se AMOUNT);
    placeholder come sottostringa -> sostituzione in-stringa; altrimenti uguale."""
    if value.strip() in mapping:
        raw = mapping[value.strip()]
        if _ph_tag(value) in _MONEY_LABELS:
            num = _num_from_it(raw)
            if num is not None:
                return num
        return raw
    if any(k in value for k in mapping):
        return _replace_text(value, mapping)
    return value


def _ph_tag(placeholder: str) -> str:
    """Etichetta del placeholder ("[AMOUNT_3]" -> "AMOUNT"); "" se non placeholder."""
    m = _PH_RE.fullmatch(placeholder.strip())
    return m.group(1) if m else ""


def _num_from_it(value: str) -> float | None:
    """Parte numerica di una stringa in formato italiano -> float.

    Virgola = decimali, punto = migliaia (``1.500,00`` -> ``1500.0``); tollera
    prefissi/suffissi (``€ 12.500,00``, ``1.500,00 EUR``). ``1234.56`` (punto
    decimale senza migliaia) resta ``1234.56``. None se non c'e' numero.
    """
    s = value.strip()
    if not s:
        return None
    m = _NUM_IT_RE.search(s)
    if not m:
        return None
    s = m.group(0)
    if "," in s:
        s = s.replace(".", "").replace(",", ".")
    elif re.fullmatch(r"[+-]?\d{1,3}(?:\.\d{3})+", s):
        s = s.replace(".", "")
    try:
        return float(s)
    except ValueError:
        return None


def _replace_text(text: str, mapping: Mapping[str, str]) -> str:
    """Sostituzione diretta (testo libero / placeholder in-stringa).

    Le chiavi piu' lunghe prima: evita collisioni tipo ``[ORG_1]`` dentro
    ``[ORG_10]``.
    """
    for ph in sorted(mapping, key=len, reverse=True):
        text = text.replace(ph, mapping[ph])
    return text
