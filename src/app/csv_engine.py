"""Anonimizzazione CSV in streaming.

Modulo solo-stdlib (csv, json, io, os, re, collections): NON importa torch/app.
Il motore di rilevamento entra solo tramite `detect_fn`, iniettato da app.py
(`detect_spans_batch`: modello + rete regex/checksum insieme, lo stesso motore di
analyze()), cosi' i test girano in CI senza dipendenze.

Piano per colonna (`plan`, lista allineata all'header, un dict per colonna):
  {"mode": "anonymize"}   la cella passa dal motore dell'app (modello + regex):
                          ogni PII trovata -> pseudonimo `[TAG_N]` del registro
  {"mode": "preserve"}    cella identica, byte per byte
Non c'e' una modalita' "solo regex" ne' un trattamento speciale delle date: una data
in una colonna anonymize si anonimizza come in analyze(), a meno che i tag DATE/TIME
siano esclusi. Per tenere intatta una colonna (es. timestamp di un log) la si mette
in preserve. Proposta di default: `default_plan(header)` = tutto anonymize.
Celle vuote (o di soli spazi): restano identiche.
Celle JSON (oggetto/array valido): si analizzano le foglie stringa che sono VALORI
(non chiavi) e si sostituiscono in modo testuale ancorato dentro la cella originale,
che resta byte-identica salvo i valori. Altrimenti la cella e' testo.
I tag esclusi (`excluded`) vengono passati a detect_fn e comunque filtrati qui.

Coerenza: tutte le sostituzioni passano da `registry.Registry` (uno per job, anche
su piu' file): lo stesso valore ha lo stesso pseudonimo in righe, colonne e file.
Cache: LRU dei risultati per testo, max CACHE_SIZE voci / CACHE_MAX_CHARS caratteri.

Affidabilita': output scritto come `<out>.partial`, rinominato solo a fine run
riuscita; su eccezione/cancel il .partial viene cancellato. Righe malformate
(es. numero di campi diverso dall'header, errore di parsing) -> saltate e contate
(solo il numero di riga nel report), oppure eccezione se strict=True.
Nessun valore delle celle in log, eccezioni o report.

Fedelta': stesso delimitatore, stessa codifica (BOM mantenuto), stessi fine riga,
ordine righe/colonne invariato, virgolette QUOTE_MINIMAL.

Note di implementazione:
- La scrittura non usa csv.writer: su Python < 3.12 con lineterminator "\\n" un
  campo con un "\\r" isolato non viene quotato e il file risultante si spezza.
- Le righe vuote (nessun campo) non sono record: si saltano senza contarle.
"""
import codecs
import csv
import io
import json
import os
import re
import time
from collections import OrderedDict

MODES = ("anonymize", "preserve")
CACHE_SIZE = 100_000


class CsvError(ValueError):
    """Errore del motore CSV: il messaggio non contiene MAI valori delle celle
    (si puo' mostrare all'utente). Ogni altra eccezione va trattata come opaca."""


SNIFF_BYTES = 64 * 1024          # prima lettura della testa
SNIFF_MAX_BYTES = 4 * 1024 * 1024  # se header+preview non ci stanno, si raddoppia fino a qui
CACHE_MAX_TEXT = 4096            # testi piu' lunghi non entrano in cache (memoria limitata)
CACHE_MAX_CHARS = 32 * 1024 * 1024  # tetto ai caratteri in cache, oltre a CACHE_SIZE voci
MAX_SKIPPED_LINES = 1000
FIELD_SIZE_LIMIT = 16 * 1024 * 1024
_DELIMS = (",", ";", "\t")
_UTF8_BOM = b"\xef\xbb\xbf"


class Cancelled(Exception):
    pass


# --------------------------------------------------------------------------- sniff

def _decode_head(data, eof):
    """Decodifica la testa; ValueError senza contenuto se non e' UTF-8."""
    if data.startswith((codecs.BOM_UTF16_LE, codecs.BOM_UTF16_BE)):
        raise CsvError("codifica non supportata (UTF-16): salva il CSV come UTF-8")
    encoding = "utf-8-sig" if data.startswith(_UTF8_BOM) else "utf-8"
    if b"\x00" in data:
        raise CsvError("il file contiene byte NUL: non e' un CSV UTF-8")
    try:
        text = codecs.getincrementaldecoder(encoding)().decode(data, final=eof)
    except UnicodeDecodeError as e:
        raise CsvError(f"il file non e' UTF-8 valido (byte {e.start} della testa)") from None
    return encoding, text


def _newline(text):
    """Primo fine riga fuori dalle virgolette."""
    inq = False
    for i, c in enumerate(text):
        if c == '"':
            inq = not inq
        elif c == "\n" and not inq:
            return "\r\n" if i and text[i - 1] == "\r" else "\n"
    return "\r\n" if "\r\n" in text else "\n"


def _parse(text, delimiter, eof):
    rows = []
    try:
        for r in csv.reader(io.StringIO(text, newline=""), delimiter=delimiter):
            if r:
                rows.append(r)
    except csv.Error:
        return None
    if not eof and rows:
        rows.pop()  # l'ultimo record puo' essere troncato a meta'
    return rows


def _score(rows):
    """(coerenza del numero di campi con l'header, numero di colonne)."""
    if not rows:
        return (0.0, 0)
    n = len(rows[0])
    if n < 2:
        return (0.0, n)
    body = rows[1:] or rows
    return (sum(len(r) == n for r in body) / len(body), n)


def _delimiter(text, eof):
    parsed = {d: _parse(text, d, eof) for d in _DELIMS}
    scores = {d: _score(rows) if rows is not None else (0.0, 0) for d, rows in parsed.items()}
    try:
        guess = csv.Sniffer().sniff(text[:SNIFF_BYTES], delimiters="".join(_DELIMS)).delimiter
    except csv.Error:
        guess = None
    if guess in scores and scores[guess][0] == 1.0:
        return guess
    best = max(_DELIMS, key=lambda d: scores[d])
    return best if scores[best][1] >= 2 else ","


def sniff(path, n_preview=20):
    """Ispeziona l'inizio del file senza leggerlo tutto.

    Ritorna {"encoding": "utf-8"|"utf-8-sig", "delimiter": ","|";"|"\\t",
             "newline": "\\r\\n"|"\\n", "header": [str], "preview": [[str]]}.
    Solleva CsvError (messaggio senza contenuto delle celle) se non e' UTF-8/CSV.
    """
    size = os.path.getsize(path)
    if size == 0:
        raise CsvError("il file e' vuoto")
    want = SNIFF_BYTES
    with open(path, "rb") as f:
        while True:
            f.seek(0)
            data = f.read(want)
            eof = len(data) >= size
            encoding, text = _decode_head(data, eof)
            if encoding == "utf-8-sig" and text.startswith("﻿"):
                text = text[1:]
            delimiter = _delimiter(text, eof)
            rows = _parse(text, delimiter, eof)
            if rows is None:
                rows = []
            if eof or len(rows) > n_preview or want >= SNIFF_MAX_BYTES:
                break
            want *= 2
    if not rows:
        raise CsvError("intestazione CSV non trovata (riga troppo lunga o file non CSV)")
    return {
        "encoding": encoding,
        "delimiter": delimiter,
        "newline": _newline(text),
        "header": rows[0],
        "preview": rows[1:1 + n_preview],
    }


# --------------------------------------------------------------------------- piano

def default_plan(header):
    """Piano di partenza: ogni colonna passa dal motore dell'app. L'utente mette in
    preserve cio' che vuole intatto (es. timestamp, ID tecnici)."""
    return [{"mode": "anonymize"} for _ in header]


# --------------------------------------------------------------------------- span

def _non_overlapping(spans):
    """Span (start, end, label) ordinate e senza sovrapposizioni (vince la prima)."""
    out, last = [], -1
    for s, e, lab in sorted(spans):
        if s >= last and e > s:
            out.append((s, e, lab))
            last = e
    return out


def _clean(spans, n, excluded):
    norm = []
    for sp in spans or ():
        if isinstance(sp, dict):
            s, e, lab = sp["start"], sp["end"], sp["label"]
        else:
            s, e, lab = sp[0], sp[1], sp[2]
        if lab in excluded:
            continue
        s, e = max(0, int(s)), min(n, int(e))
        if e > s:
            norm.append((s, e, lab))
    return tuple(_non_overlapping(norm))


def _apply(text, spans, registry, counts):
    if not spans:
        return text
    parts, pos = [], 0
    for s, e, lab in spans:
        parts.append(text[pos:s])
        parts.append(registry.assign(lab, text[s:e]))
        counts[lab] = counts.get(lab, 0) + 1
        pos = e
    parts.append(text[pos:])
    return "".join(parts)


# --------------------------------------------------------------------------- JSON

def _json_string_tokens(raw):
    """(start, end) dei letterali stringa che sono VALORI (non chiavi) in un JSON
    gia' validato. start/end includono le virgolette."""
    out, i, n = [], 0, len(raw)
    while i < n:
        if raw[i] != '"':
            i += 1
            continue
        j = i + 1
        while raw[j] != '"':
            j += 2 if raw[j] == "\\" else 1
        end = j + 1
        k = end
        while k < n and raw[k] in " \t\r\n":
            k += 1
        if not (k < n and raw[k] == ":"):
            out.append((i, end))
        i = end
    return out


def _raw_offsets(inner):
    """Per ogni carattere decodificato, l'offset del suo inizio nel letterale grezzo."""
    offs, i, n = [], 0, len(inner)
    while i < n:
        offs.append(i)
        if inner[i] != "\\":
            i += 1
        elif inner[i + 1] != "u":
            i += 2
        else:
            cp = int(inner[i + 2:i + 6], 16)
            nxt = inner[i + 6:i + 12]
            if (0xD800 <= cp < 0xDC00 and nxt[:2] == "\\u"
                    and 0xDC00 <= int(nxt[2:6], 16) < 0xE000):
                i += 12
            else:
                i += 6
    offs.append(n)
    return offs


def _json_leaves(cell):
    """Lista di (raw_start, raw_end, testo) per le foglie stringa, o None se non JSON."""
    st = cell.strip()
    if not st or st[0] not in "{[" or st[-1] not in "}]":
        return None
    try:
        obj = json.loads(cell)
    except ValueError:
        return None
    if not isinstance(obj, (dict, list)):
        return None
    leaves = []
    for s, e in _json_string_tokens(cell):
        value = json.loads(cell[s:e])
        if value.strip():
            leaves.append((s, e, value))
    return leaves


def _apply_json(cell, leaves, results, registry, counts):
    parts, pos = [], 0
    for (s, e, value), spans in zip(leaves, results):
        if not spans:
            continue
        inner = cell[s + 1:e - 1]
        if inner == value:
            new_inner = _apply(inner, spans, registry, counts)
        else:
            offs = _raw_offsets(inner)
            pieces, p = [], 0
            for a, b, lab in spans:
                pieces.append(inner[p:offs[a]])
                pieces.append(registry.assign(lab, value[a:b]))
                counts[lab] = counts.get(lab, 0) + 1
                p = offs[b]
            pieces.append(inner[p:])
            new_inner = "".join(pieces)
        parts.append(cell[pos:s + 1])
        parts.append(new_inner)
        pos = e - 1
    parts.append(cell[pos:])
    return "".join(parts)


# --------------------------------------------------------------------------- output

def _row_writer(delimiter, newline):
    special = re.compile("[" + re.escape(delimiter) + '"\r\n]')

    def fmt(fields):
        if len(fields) == 1 and fields[0] == "":
            return '""' + newline
        return delimiter.join(
            '"' + f.replace('"', '""') + '"' if special.search(f) else f for f in fields
        ) + newline

    return fmt


class _Cache:
    def __init__(self, size, max_chars=CACHE_MAX_CHARS):
        self.size = size
        self.max_chars = max_chars
        self.chars = 0
        self.data = OrderedDict()

    def get(self, key):
        v = self.data.get(key)
        if v is not None:
            self.data.move_to_end(key)
        return v

    def put(self, key, value):
        if self.size <= 0 or len(key) > CACHE_MAX_TEXT or key in self.data:
            return
        self.data[key] = value
        self.chars += len(key)
        while self.data and (len(self.data) > self.size or self.chars > self.max_chars):
            old, _ = self.data.popitem(last=False)
            self.chars -= len(old)


# --------------------------------------------------------------------------- process

def _validate_plan(plan, ncols):
    if len(plan) != ncols:
        raise CsvError(f"il piano ha {len(plan)} colonne, il CSV {ncols}")
    for i, step in enumerate(plan):
        mode = step.get("mode") if isinstance(step, dict) else None
        if mode not in MODES:
            raise CsvError(f"modalita' non valida per la colonna {i + 1} (valide: {', '.join(MODES)})")


def process_file(in_path, out_path, plan, *, registry, detect_fn, excluded=(),
                 dialect=None, progress=None, cancel=None, strict=False,
                 batch_size=64):
    """Anonimizza in streaming `in_path` -> `out_path` (via .partial + rename).

    plan: [{"mode": "anonymize"|"preserve"}, ...] allineato all'header.
    detect_fn(texts: list[str], excluded: set) -> list[list[(start, end, label)]]
        il motore dell'app (modello + rete regex) su un batch di testi; obbligatorio
        se almeno una colonna e' anonymize (altrimenti CsvError).
    excluded: tag da non sostituire (passati a detect_fn e filtrati comunque).
    dialect: dict come ritornato da sniff() (se None si chiama sniff).
    progress(rows_done: int, bytes_done: int, bytes_total: int) chiamata periodicamente.
    cancel(): -> True per interrompere (eccezione Cancelled, .partial rimosso).

    Ritorna report: {"rows_in", "rows_out", "rows_skipped", "skipped_lines": [int, ...] (max 1000),
                     "by_label": {tag: n}, "seconds"}.
    """
    t0 = time.monotonic()
    excluded = set(excluded or ())
    batch_size = max(1, int(batch_size))
    modes = [step.get("mode") if isinstance(step, dict) else None for step in plan]
    if "anonymize" in modes and detect_fn is None:
        raise CsvError("le colonne da anonimizzare richiedono il motore di rilevamento "
                       "(modello non disponibile)")
    if dialect is None:
        dialect = sniff(in_path)
    delimiter, newline, encoding = dialect["delimiter"], dialect["newline"], dialect["encoding"]
    if csv.field_size_limit() < FIELD_SIZE_LIMIT:
        csv.field_size_limit(FIELD_SIZE_LIMIT)

    total = os.path.getsize(in_path)
    partial = out_path + ".partial"
    fmt = _row_writer(delimiter, newline)
    cache = _Cache(CACHE_SIZE)
    counts = {}
    rep = {"rows_in": 0, "rows_out": 0, "rows_skipped": 0, "skipped_lines": []}

    def detect(texts):
        res = detect_fn(list(texts), set(excluded))
        if len(res) != len(texts):
            raise CsvError("il motore di rilevamento ha restituito un numero di risultati "
                           "diverso dai testi")
        return res

    try:
        with open(in_path, "rb") as raw, open(partial, "w", encoding=encoding, newline="") as out:
            text_in = io.TextIOWrapper(raw, encoding=encoding, newline="")
            reader = csv.reader(text_in, delimiter=delimiter)
            try:
                header = next(reader)
            except StopIteration:
                raise CsvError("il file e' vuoto") from None
            _validate_plan(plan, len(header))
            ncols = len(header)
            out.write(fmt(header))

            pending = []   # righe in attesa: lista di celle (str) o job
            todo = {}      # testo -> None, testi da rilevare nel prossimo flush

            def flush():
                if cancel is not None and cancel():
                    raise Cancelled()
                results = {}
                texts = list(todo)
                for k in range(0, len(texts), batch_size):
                    chunk = texts[k:k + batch_size]
                    for t, spans in zip(chunk, detect(chunk)):
                        clean = _clean(spans, len(t), excluded)
                        results[t] = clean
                        cache.put(t, clean)
                todo.clear()
                for cells in pending:
                    row = []
                    for c in cells:
                        if isinstance(c, str):
                            row.append(c)
                        elif c[0] == "text":
                            _, text, spans = c
                            spans = spans if spans is not None else results[text]
                            row.append(_apply(text, spans, registry, counts))
                        else:  # json
                            _, cell, leaves, found = c
                            res = [f if f is not None else results[lv[2]]
                                   for lv, f in zip(leaves, found)]
                            row.append(_apply_json(cell, leaves, res, registry, counts))
                    out.write(fmt(row))
                rep["rows_out"] += len(pending)
                pending.clear()
                if progress is not None:
                    progress(rep["rows_in"], min(raw.tell(), total), total)

            def lookup(text):
                hit = cache.get(text)
                if hit is None:
                    todo[text] = None
                return hit

            def skip(line):
                if strict:
                    raise CsvError(f"riga malformata alla linea {line}")
                rep["rows_skipped"] += 1
                if len(rep["skipped_lines"]) < MAX_SKIPPED_LINES:
                    rep["skipped_lines"].append(line)

            stuck = 0
            while True:
                start_line = reader.line_num + 1
                try:
                    row = next(reader)
                except StopIteration:
                    break
                except csv.Error:
                    rep["rows_in"] += 1
                    skip(start_line)
                    stuck = stuck + 1 if reader.line_num < start_line else 0
                    if stuck > 2:
                        raise CsvError(f"CSV illeggibile dalla linea {start_line}") from None
                    continue
                except UnicodeDecodeError:
                    raise CsvError(f"il file non e' UTF-8 valido (vicino alla linea {start_line})") from None
                if not row:
                    continue  # riga vuota: non e' un record
                rep["rows_in"] += 1
                if len(row) != ncols:
                    skip(start_line)
                    continue
                cells = []
                for cell, mode in zip(row, modes):
                    if mode == "preserve" or not cell.strip():
                        cells.append(cell)
                        continue
                    leaves = _json_leaves(cell)
                    if leaves is None:
                        cells.append(("text", cell, lookup(cell)))
                    else:
                        found = [lookup(lv[2]) for lv in leaves]
                        cells.append(("json", cell, leaves, found))
                pending.append(cells)
                if len(todo) >= batch_size or len(pending) >= batch_size:
                    flush()
            flush()
            text_in.detach()
        os.replace(partial, out_path)
    except BaseException:
        try:
            os.remove(partial)
        except OSError:
            pass
        raise

    if progress is not None:
        progress(rep["rows_in"], total, total)
    rep["by_label"] = counts
    rep["seconds"] = round(time.monotonic() - t0, 3)
    return rep
