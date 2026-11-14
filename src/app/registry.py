"""Registro degli pseudonimi di un job CSV.

Stesso formato `[TAG_N]` e stessa normalizzazione di analyze(), cosi' il dizionario
scaricato si carica nella tab "Ripristina" come quello di sempre. Il registro rende la
numerazione persistente *per la durata di un job* (anche su piu' CSV): lo stesso
valore riceve lo stesso pseudonimo in righe, colonne e file diversi.

Chiave di coerenza = valore normalizzato (strip, spazi compressi, casefold),
SENZA il tag: se un valore e' gia' stato visto, il suo pseudonimo vince anche se
altrove viene etichettato con un tag diverso ("il primo tag vince").

Niente su disco: vive in memoria e muore con il job. Memoria: circa 150-250 byte
per valore unico (1M valori ~ 200 MB) se il dizionario e' attivo; con
mapping_enabled=False non si conserva il valore originale (resta solo la chiave
normalizzata, necessaria per la coerenza).

Modulo solo-stdlib: niente torch, importabile nei test di CI.
"""
import re
import threading

_WS = re.compile(r"\s+")


def norm(s):
    """Stessa normalizzazione di app._norm (dedup dei placeholder)."""
    return _WS.sub(" ", s.strip()).casefold()


class Registry:
    def __init__(self, mapping_enabled=True):
        self.mapping_enabled = mapping_enabled
        self._seen = {}        # norm(valore) -> placeholder
        self._counters = {}    # tag -> ultimo N assegnato
        self._mapping = {}     # placeholder -> valore originale (prima grafia vista)
        self._lock = threading.Lock()

    def assign(self, label, value):
        """Pseudonimo di `value`; se nuovo, gli assegna `[label_N]`."""
        key = norm(value)
        with self._lock:
            ph = self._seen.get(key)
            if ph is None:
                n = self._counters.get(label, 0) + 1
                self._counters[label] = n
                ph = f"[{label}_{n}]"
                self._seen[key] = ph
                if self.mapping_enabled:
                    self._mapping[ph] = value
            return ph

    def lookup(self, value):
        return self._seen.get(norm(value))

    @property
    def mapping(self):
        """{placeholder: valore}, formato del dizionario scaricabile della UI."""
        with self._lock:
            return dict(self._mapping)

    def __len__(self):
        return len(self._seen)
