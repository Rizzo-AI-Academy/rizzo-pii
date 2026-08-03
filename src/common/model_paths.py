# -*- coding: utf-8 -*-
"""Risoluzione della cartella del modello, in un posto solo.

Perche' esiste questo modulo: la stessa logica ("prendi la versione piu' alta fra
models/rizzo-pii-0.3B-v*") viveva copiata in quattro punti — app.py, test_pii.py,
evaluate_pii.py, export_onnx.py. Le copie sono divergute: tre avevano la guardia sul
match del regex, quella INLINE dentro app.py no, e faceva crashare l'app all'import
appena in models/ compariva una cartella non versionata.

E' esattamente il caso che l'export ONNX produce: `export_onnx.py` scrive il bundle in
`models/<nome>-onnx/`, che il glob raccoglie ma il regex della versione non riconosce.

Nessuna dipendenza oltre la stdlib: lo importano sia gli script di training sia l'app.
"""
import os
import re
from pathlib import Path

MODEL_GLOB = "rizzo-pii-0.3B-v*"
# Ancorato in fondo: il nome deve FINIRE con la versione. E' voluto — serve proprio a
# distinguere "…-v1.3.0" (checkpoint) da "…-v1.3.0-onnx" (bundle esportato).
VERSION_RE = re.compile(r"-v([0-9][0-9.]*)$")


def _version_key(name):
    """(1,3,0) per 'rizzo-pii-0.3B-v1.3.0'; None se il nome non e' versionato."""
    m = VERSION_RE.search(name)
    if not m:
        return None
    try:
        return tuple(int(x) for x in m.group(1).split("."))
    except ValueError:          # es. "-v1..2": cifre attese, punteggiatura sballata
        return None


def versioned_dirs(models_dir):
    """[(path, chiave_versione)] dei soli checkpoint versionati, ordinati dal piu' vecchio.

    Il filtro sul match NON e' ridondante rispetto al glob: il glob accetta qualsiasi
    suffisso dopo la 'v', quindi da solo lascia passare anche le cartelle sorelle
    (-onnx, e domani -gguf o -backup) che non sono checkpoint.
    """
    out = []
    for p in Path(models_dir).glob(MODEL_GLOB):
        if not p.is_dir():
            continue
        key = _version_key(p.name)
        if key is not None:
            out.append((p, key))
    out.sort(key=lambda pk: pk[1])
    return out


def resolve_model_dir(models_dir, pinned=None):
    """Cartella del modello da usare, come stringa.

    Precedenza: PII_MODEL_DIR (env) > versione richiesta con `pinned` > versione piu'
    alta presente > vecchio percorso non versionato > legacy. L'ultimo ripiego puo'
    non esistere: e' il chiamante a decidere cosa farne (l'app mostra un errore utile).
    """
    if os.environ.get("PII_MODEL_DIR"):
        return os.environ["PII_MODEL_DIR"]

    models_dir = Path(models_dir)
    if pinned:
        cand = models_dir / f"rizzo-pii-0.3B-v{pinned}"
        if cand.is_dir():
            return str(cand)

    found = versioned_dirs(models_dir)
    if found:
        return str(found[-1][0])

    base = models_dir / "rizzo-pii-0.3B"
    return str(base if base.exists() else models_dir / "pii_model_legacy")
