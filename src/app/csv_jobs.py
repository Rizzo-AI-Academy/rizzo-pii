# -*- coding: utf-8 -*-
"""Job CSV: upload a pezzi, coda, avanzamento e download (Blueprint Flask).

Un CSV puo' pesare GB: non passa da un solo POST (MAX_CONTENT_LENGTH globale resta
50 MB) ma viene caricato a pezzi (`File.slice` lato browser, CHUNK_SIZE = 8 MB) in un
file temporaneo nella config dir dell'app (`<config_dir>/csv_jobs/<istanza>/`).
Il motore (`csv_engine.process_file`) lavora in streaming file -> file; l'output si
scarica con una risposta HTTP in streaming. A fine job (DELETE) input e output
vengono cancellati; quel che resta di un processo morto si spazza all'avvio
successivo.

Tutto locale: nessun valore delle celle finisce in log, eccezioni o messaggi
d'errore (le anteprime tornano solo all'UI che ha caricato il file, come /preview).

Endpoint (JSON salvo i download):

  POST   /csv/uploads                      {"name", "size"} -> {"upload_id", "chunk_size"}
  PUT    /csv/uploads/<uid>?offset=N       corpo grezzo (<= CHUNK_SIZE), pezzi in ordine
                                           -> {"received", "size", "complete"}
  POST   /csv/uploads/<uid>/finalize       sniff + piano di default (tutte le colonne
                                           "anonymize"):
                                           {"upload_id","name","size","header","preview",
                                            "delimiter","encoding","newline","plan","modes",
                                            "large","large_threshold"}
  DELETE /csv/uploads/<uid>                scarta un upload non ancora usato da un job
  POST   /csv/jobs                         {"files": [{"upload_id", "plan"?}], "strict"?,
                                            plan = [{"mode": "anonymize"|"preserve"}] per colonna
                                            "exclude_tags"?, "include_mapping"?}
                                           -> {"job_id", "state": "queued", "position"}
  GET    /csv/jobs/<jid>                   stato (vedi Job.status)
  POST   /csv/jobs/<jid>/cancel            interrompe (o toglie dalla coda)
  GET    /csv/jobs/<jid>/files/<i>         output i-esimo, `<nome>.anon.csv` (streaming)
  GET    /csv/jobs/<jid>/files/<i>/preview header + prime righe dell'output (anteprima UI)
  GET    /csv/jobs/<jid>/dictionary        `dizionario_anonimizzazione.json` {"[TAG_N]": valore}
  DELETE /csv/jobs/<jid>                   cancella input/output temporanei e il registro

Un job alla volta: un solo thread worker consuma una coda FIFO. Gli pseudonimi sono
coerenti fra tutti i file di un job (un `registry.Registry` per job); con il
dizionario disattivato il registro non conserva i valori e /dictionary risponde 404.

Ogni colonna o si anonimizza (sempre lo stesso motore di analyze(): modello + rete
regex/checksum, tag esclusi compresi) o resta intatta. `large` segnala all'UI i file
oltre `large_threshold` byte (soglia passata da app.py, oggi = MAX_CONTENT_LENGTH):
l'analisi col modello richiede tempo, conviene disattivare le colonne senza PII.

Il modulo non importa torch ne' app: `detect_fn` arriva per iniezione da app.py
(`create_blueprint(get_detect_fn=...)`), cosi' i test girano con un motore finto.
"""

import json
import os
import queue
import re
import secrets
import shutil
import sys
import threading
import time
from pathlib import Path

from flask import Blueprint, Response, jsonify, request, send_file

import registry as registry_mod
import server_config

CHUNK_SIZE = 8 * 1024 * 1024          # pezzo suggerito al client
CHUNK_MAX = CHUNK_SIZE + 64 * 1024    # tolleranza: oltre -> 413
DISK_MARGIN = 256 * 1024 * 1024       # spazio da lasciare libero oltre a input+output
PREVIEW_ROWS = 20
MAX_SKIPPED_SHOWN = 50
OK_EXT = (".csv", ".tsv", ".txt")
LOCK_NAME = ".lock"

STATES = ("queued", "running", "done", "error", "cancelled")
IDLE_TTL = 2 * 3600                   # upload/job non toccati da 2 h -> cancellati
REAP_EVERY = 300                      # il worker controlla ogni 5 min


# --------------------------------------------------------------------------- #
# Cartella dell'istanza + pulizia dei resti di processi morti
# --------------------------------------------------------------------------- #
def default_root():
    return server_config.config_dir() / "csv_jobs"


def _try_lock(fh):
    """Lock esclusivo non bloccante sul file aperto; True se acquisito."""
    try:
        if sys.platform == "win32":
            import msvcrt
            fh.seek(0)
            msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        return True
    except OSError:
        return False


def sweep_stale(root):
    """Cancella le cartelle di istanze non piu' vive (lock libero). Ritorna quante.

    Ogni processo tiene aperto e bloccato `<istanza>/.lock`: se il lock si prende,
    il processo che l'ha creata e' morto e la cartella e' un avanzo."""
    root = Path(root)
    if not root.is_dir():
        return 0
    n = 0
    for d in root.iterdir():
        if not d.is_dir():
            continue
        lock = d / LOCK_NAME
        stale = True
        if lock.exists():
            try:
                with open(lock, "a+b") as fh:
                    stale = _try_lock(fh)      # chiudere il file rilascia il lock
            except OSError:
                stale = False                  # Windows: aperto da un'altra istanza
        if stale:
            shutil.rmtree(d, ignore_errors=True)
            n += 1
    return n


class _Instance:
    """Cartella privata di questo processo, creata pigramente al primo upload."""

    def __init__(self, root):
        self.root = Path(root)
        self._dir = None
        self._fh = None
        self._lock = threading.Lock()

    @property
    def dir(self):
        with self._lock:
            if self._dir is None:
                d = self.root / f"{os.getpid()}-{secrets.token_hex(4)}"
                d.mkdir(parents=True, exist_ok=True)
                fh = open(d / LOCK_NAME, "a+b")
                _try_lock(fh)
                self._dir, self._fh = d, fh
            return self._dir

    def close(self):
        """Rilascia il lock (la cartella diventa spazzabile). Per test/arresto."""
        with self._lock:
            if self._fh is not None:
                self._fh.close()
                self._fh = None


# --------------------------------------------------------------------------- #
# Stato in memoria
# --------------------------------------------------------------------------- #
def _new_id():
    return secrets.token_hex(8)


def _safe_display_name(name):
    """Nome file per l'UI e il download: solo il basename, niente separatori."""
    base = os.path.basename(str(name or "").replace("\\", "/")).strip()
    base = re.sub(r'[\x00-\x1f<>:"/\\|?*]', "_", base)
    return base[:200] or "file.csv"


def _out_name(name):
    stem, ext = os.path.splitext(name)
    return f"{stem or 'file'}.anon{ext if ext.lower() in ('.csv', '.tsv') else '.csv'}"


def _safe_error(exc, safe=ValueError):
    """Messaggio d'errore senza contenuto delle celle.

    `safe` = classe degli errori del motore (csv_engine.CsvError), il cui messaggio
    per contratto non contiene valori. Tutto il resto, comprese le ValueError del
    modello/tokenizer o di altre librerie, OSError con path ecc.: solo il tipo."""
    if isinstance(exc, safe):
        return str(exc)[:300]
    return f"Errore interno ({type(exc).__name__})"


class Upload:
    def __init__(self, uid, name, size, path):
        self.id = uid
        self.name = name
        self.size = size
        self.path = path
        self.received = 0
        self.dialect = None       # risultato di sniff(), dopo finalize
        self.job_id = None        # assegnato quando entra in un job
        self.lock = threading.Lock()
        self.touched = time.time()


class JobFile:
    def __init__(self, upload, plan, out_path):
        self.upload = upload
        self.plan = plan
        self.out_path = out_path
        self.state = "queued"
        self.rows_done = 0
        self.bytes_done = 0
        self.bytes_total = upload.size
        self.report = None
        self.error = None

    def status(self):
        rep = self.report or {}
        skipped = rep.get("skipped_lines") or []
        return {
            "name": self.upload.name,
            "out_name": _out_name(self.upload.name),
            "state": self.state,
            "rows_done": rep.get("rows_in", self.rows_done),
            "rows_out": rep.get("rows_out"),
            "rows_skipped": rep.get("rows_skipped", 0),
            "skipped_lines": skipped[:MAX_SKIPPED_SHOWN],
            "by_label": rep.get("by_label", {}),
            "bytes_done": self.bytes_total if self.state == "done" else self.bytes_done,
            "bytes_total": self.bytes_total,
            "error": self.error,
        }


class Job:
    def __init__(self, jid, files, mapping_enabled, excluded, strict, workdir):
        self.id = jid
        self.files = files
        self.mapping_enabled = mapping_enabled
        self.excluded = set(excluded)
        self.strict = strict
        self.workdir = workdir
        self.registry = registry_mod.Registry(mapping_enabled=mapping_enabled)
        self.state = "queued"
        self.error = None
        self.cancel = threading.Event()
        self.deleted = False
        self.created = time.time()
        self.started = None
        self.finished = None
        self.touched = time.time()

    def status(self, position=None):
        files = [f.status() for f in self.files]
        out = {
            "job_id": self.id,
            "state": self.state,
            "error": self.error,
            "files": files,
            "rows_done": sum(f["rows_done"] or 0 for f in files),
            "rows_skipped": sum(f["rows_skipped"] or 0 for f in files),
            "bytes_done": sum(f["bytes_done"] for f in files),
            "bytes_total": sum(f["bytes_total"] for f in files),
            "mapping_enabled": self.mapping_enabled,
            "dictionary_size": len(self.registry) if self.mapping_enabled else 0,
            "unique_values": len(self.registry),
            "excluded_tags": sorted(self.excluded),
            "strict": self.strict,
            "seconds": round((self.finished or time.time()) - self.started, 2)
            if self.started else 0,
        }
        if position is not None:
            out["position"] = position
        return out


# --------------------------------------------------------------------------- #
# Blueprint
# --------------------------------------------------------------------------- #
def create_blueprint(get_prefs, get_detect_fn=lambda: None, engine=None, root=None,
                     valid_tags=None, large_threshold=None, url_prefix="/csv"):
    """get_prefs() -> (excluded_tags, mapping_enabled) : default del server.
    get_detect_fn() -> detect_fn | None, risolto a ogni job (il modello puo' caricarsi dopo).
    large_threshold() -> byte | None: oltre questa taglia finalize segnala "large"
    (None = nessun avviso). E' una callable per seguire la config dell'app.
    engine: modulo con l'API di csv_engine (iniettabile nei test).
    root: cartella dei temporanei (default <config_dir>/csv_jobs)."""
    if engine is None:
        import csv_engine as engine
    safe_exc = getattr(engine, "CsvError", ValueError)
    root = Path(root) if root is not None else default_root()
    sweep_stale(root)
    inst = _Instance(root)

    bp = Blueprint("csv_jobs", __name__, url_prefix=url_prefix)
    uploads, jobs = {}, {}
    pending = []                       # job id in coda, in ordine
    lock = threading.Lock()
    q = queue.Queue()
    worker = {"thread": None}

    def err(msg, code=400, **extra):
        return jsonify({"error": msg, **extra}), code

    # ------------------------------------------------------------------ worker
    def _ensure_worker():
        with lock:
            t = worker["thread"]
            if t is None or not t.is_alive():
                t = threading.Thread(target=_loop, name="csv-jobs", daemon=True)
                worker["thread"] = t
                t.start()

    def _loop():
        while True:
            try:
                jid = q.get(timeout=REAP_EVERY)
            except queue.Empty:
                _reap()
                continue
            with lock:
                job = jobs.get(jid)
                if jid in pending:
                    pending.remove(jid)
            if job is None:
                continue
            try:
                _run(job)
            finally:
                if job.deleted:
                    _purge(job)

    def _run(job):
        if job.cancel.is_set():
            job.state = "cancelled"
            for f in job.files:
                f.state = "cancelled"
            return
        job.state = "running"
        job.started = time.time()
        detect_fn = get_detect_fn()
        try:
            for jf in job.files:
                if job.cancel.is_set():
                    raise engine.Cancelled()
                jf.state = "running"

                def progress(rows, done, total, jf=jf):
                    jf.rows_done, jf.bytes_done = rows, done
                    if total:
                        jf.bytes_total = total

                try:
                    jf.report = engine.process_file(
                        str(jf.upload.path), str(jf.out_path), jf.plan,
                        registry=job.registry, detect_fn=detect_fn,
                        excluded=job.excluded, dialect=jf.upload.dialect,
                        progress=progress, cancel=job.cancel.is_set, strict=job.strict)
                except engine.Cancelled:
                    jf.state = "cancelled"
                    raise
                except Exception as e:  # noqa: BLE001 - il messaggio passa da _safe_error
                    jf.state = "error"
                    jf.error = _safe_error(e, safe_exc)
                    raise
                jf.state = "done"
            job.state = "done"
        except engine.Cancelled:
            job.state = "cancelled"
            for f in job.files:
                if f.state == "queued":
                    f.state = "cancelled"
        except Exception as e:  # noqa: BLE001
            job.state = "error"
            job.error = _safe_error(e, safe_exc)
            # solo il tipo nel log del backend: mai il contenuto delle celle
            print(f"[csv_jobs] job {job.id} fallito: {type(e).__name__}", file=sys.stderr)
            for f in job.files:
                if f.state == "queued":
                    f.state = "cancelled"
        finally:
            job.finished = time.time()

    def _purge(job):
        """Via file temporanei (input + output) e registro."""
        for jf in job.files:
            shutil.rmtree(jf.upload.path.parent, ignore_errors=True)
            with lock:
                uploads.pop(jf.upload.id, None)
        shutil.rmtree(job.workdir, ignore_errors=True)
        job.registry = registry_mod.Registry(mapping_enabled=False)
        with lock:
            jobs.pop(job.id, None)

    def _job_or_404(jid):
        with lock:
            job = jobs.get(jid)
        if job is not None:
            job.touched = time.time()
        return job

    def _reap(now=None):
        """Via upload e job abbandonati (tab chiusa, niente DELETE): input/output
        temporanei e registro non restano per tutta la vita del processo."""
        limit = (now or time.time()) - IDLE_TTL
        with lock:
            old_jobs = [j for j in jobs.values()
                        if j.state not in ("queued", "running") and j.touched < limit]
            old_ups = [u for u in uploads.values() if u.job_id is None and u.touched < limit]
            for u in old_ups:
                uploads.pop(u.id, None)
        for j in old_jobs:
            _purge(j)
        for u in old_ups:
            shutil.rmtree(u.path.parent, ignore_errors=True)
        return len(old_jobs) + len(old_ups)

    def _reserved_bytes():
        """Byte che upload e job gia' aperti devono ancora scrivere (resto
        dell'input + output non ancora prodotto)."""
        with lock:
            ups = list(uploads.values())
            done = {jid for jid, j in jobs.items() if j.state != "queued" and j.state != "running"}
        return sum((u.size - u.received) + (0 if u.job_id in done else u.size) for u in ups)

    def _position(jid):
        with lock:
            return pending.index(jid) + 1 if jid in pending else None

    # ----------------------------------------------------------------- uploads
    @bp.route("/uploads", methods=["POST"])
    def upload_create():
        data = request.get_json(silent=True) or {}
        name = _safe_display_name(data.get("name"))
        try:
            size = int(data.get("size"))
        except (TypeError, ValueError):
            return err("Dimensione del file non valida.")
        if size <= 0:
            return err("Il file è vuoto.")
        if not name.lower().endswith(OK_EXT):
            return err("Formato non supportato: servono file .csv.")
        base = inst.dir
        _reap()
        _ensure_worker()                       # fa anche da spazzino periodico
        # input + output (circa la stessa taglia) + margine + quel che gli altri
        # upload/job aperti devono ancora scrivere
        free = shutil.disk_usage(base).free
        if 2 * size + DISK_MARGIN + _reserved_bytes() > free:
            return err("Spazio su disco insufficiente per elaborare questo file.", 507)
        uid = _new_id()
        d = base / f"u_{uid}"
        d.mkdir()
        path = d / "input.csv"
        path.touch()
        with lock:
            uploads[uid] = Upload(uid, name, size, path)
        return jsonify({"upload_id": uid, "chunk_size": CHUNK_SIZE, "name": name,
                        "size": size})

    def _upload(uid):
        with lock:
            up = uploads.get(uid)
        if up is not None:
            up.touched = time.time()
        return up

    @bp.route("/uploads/<uid>", methods=["PUT"])
    def upload_chunk(uid):
        up = _upload(uid)
        if up is None:
            return err("Upload non trovato.", 404)
        if up.job_id is not None or up.dialect is not None:
            return err("Upload già completato.", 409)
        try:
            offset = int(request.args.get("offset", ""))
        except ValueError:
            return err("Offset mancante o non valido.")
        clen = request.content_length
        if clen is not None and clen > CHUNK_MAX:
            return err("Pezzo troppo grande.", 413)
        body = request.get_data(cache=False)
        if len(body) > CHUNK_MAX:
            return err("Pezzo troppo grande.", 413)
        if not body:
            return err("Pezzo vuoto.")
        with up.lock:
            if offset != up.received:
                return err("Offset fuori sequenza.", 409, received=up.received)
            if offset + len(body) > up.size:
                return err("Il file supera la dimensione dichiarata.", 400,
                           received=up.received)
            with open(up.path, "ab") as fh:
                fh.write(body)
            up.received += len(body)
            received = up.received
        return jsonify({"received": received, "size": up.size,
                        "complete": received == up.size})

    def _plan(up):
        return engine.default_plan(up.dialect["header"])

    def _threshold():
        try:
            t = large_threshold() if large_threshold is not None else None
            return int(t) if t else None
        except (TypeError, ValueError):
            return None

    @bp.route("/uploads/<uid>/finalize", methods=["POST"])
    def upload_finalize(uid):
        up = _upload(uid)
        if up is None:
            return err("Upload non trovato.", 404)
        with up.lock:
            if up.received != up.size:
                return err("Upload incompleto.", 409, received=up.received)
            try:
                up.dialect = engine.sniff(str(up.path), n_preview=PREVIEW_ROWS)
                plan = _plan(up)
            except NotImplementedError:
                return err("Motore CSV non disponibile.", 501)
            except Exception as e:  # noqa: BLE001
                up.dialect = None
                return err(_safe_error(e, safe_exc), 422)
        d = up.dialect
        threshold = _threshold()
        return jsonify({
            "upload_id": uid, "name": up.name, "size": up.size,
            "header": d["header"], "preview": d.get("preview", [])[:PREVIEW_ROWS],
            "delimiter": d.get("delimiter"), "encoding": d.get("encoding"),
            "newline": d.get("newline"), "plan": plan, "modes": list(engine.MODES),
            "large": bool(threshold) and up.size > threshold,
            "large_threshold": threshold,
        })

    @bp.route("/uploads/<uid>", methods=["DELETE"])
    def upload_delete(uid):
        up = _upload(uid)
        if up is None:
            return jsonify({"ok": True})
        if up.job_id is not None:
            return err("Upload in uso da un job: cancella il job.", 409)
        with lock:
            uploads.pop(uid, None)
        shutil.rmtree(up.path.parent, ignore_errors=True)
        return jsonify({"ok": True})

    # -------------------------------------------------------------------- jobs
    def _check_plan(plan, header):
        if not isinstance(plan, list) or len(plan) != len(header):
            return "Il piano deve avere una voce per colonna."
        for c in plan:
            if not isinstance(c, dict) or c.get("mode") not in engine.MODES:
                return "Modalità di colonna non valida."
        return None

    @bp.route("/jobs", methods=["POST"])
    def job_create():
        data = request.get_json(silent=True) or {}
        items = data.get("files")
        if not isinstance(items, list) or not items:
            return err("Nessun file nel job.")
        default_excl, default_map = get_prefs()
        raw_excl = data.get("exclude_tags")
        excluded = (server_config.parse_tag_list(raw_excl) if raw_excl is not None
                    else list(default_excl))
        if valid_tags is not None:
            unknown = [t for t in excluded if t not in valid_tags]
            if unknown:
                return err(f"Tag sconosciuti: {', '.join(unknown)}")
        raw_map = data.get("include_mapping")
        mapping = (server_config.parse_bool(raw_map, default_map) if raw_map is not None
                   else default_map)
        strict = bool(data.get("strict", False))

        chosen, seen = [], set()
        for it in items:
            if not isinstance(it, dict):
                return err("Voce di file non valida.")
            uid = it.get("upload_id")
            up = _upload(uid)
            if up is None:
                return err("Upload non trovato.", 404)
            if uid in seen:
                return err("Lo stesso file compare due volte.")
            seen.add(uid)
            if up.dialect is None:
                return err("Upload non ancora finalizzato.", 409)
            if up.job_id is not None:
                return err("Upload già usato da un altro job.", 409)
            plan = it.get("plan")
            if plan is None:
                plan = _plan(up)
            msg = _check_plan(plan, up.dialect["header"])
            if msg:
                return err(f"{up.name}: {msg}")
            chosen.append((up, plan))

        jid = _new_id()
        workdir = inst.dir / f"j_{jid}"
        workdir.mkdir()
        with lock:
            if any(up.job_id is not None for up, _ in chosen):   # gara fra due POST
                shutil.rmtree(workdir, ignore_errors=True)
                return err("Upload già usato da un altro job.", 409)
            for up, _ in chosen:
                up.job_id = jid
            files = [JobFile(up, plan, workdir / f"out_{i}.csv")
                     for i, (up, plan) in enumerate(chosen)]
            job = Job(jid, files, mapping, excluded, strict, workdir)
            jobs[jid] = job
            pending.append(jid)
            position = len(pending)
        q.put(jid)
        _ensure_worker()
        return jsonify({"job_id": jid, "state": "queued", "position": position}), 202

    @bp.route("/jobs/<jid>", methods=["GET"])
    def job_status(jid):
        job = _job_or_404(jid)
        if job is None:
            return err("Job non trovato.", 404)
        return jsonify(job.status(_position(jid)))

    @bp.route("/jobs/<jid>/cancel", methods=["POST"])
    def job_cancel(jid):
        job = _job_or_404(jid)
        if job is None:
            return err("Job non trovato.", 404)
        job.cancel.set()
        with lock:
            if jid in pending:        # in coda: lo si chiude subito, il worker lo salta
                pending.remove(jid)
                job.state = "cancelled"
                for f in job.files:
                    f.state = "cancelled"
        return jsonify(job.status())

    @bp.route("/jobs/<jid>/files/<int:i>", methods=["GET"])
    def job_file(jid, i):
        job = _job_or_404(jid)
        if job is None:
            return err("Job non trovato.", 404)
        if not 0 <= i < len(job.files):
            return err("File non trovato.", 404)
        jf = job.files[i]
        if jf.state != "done" or not jf.out_path.exists():
            return err("Output non ancora pronto.", 409)
        return send_file(str(jf.out_path), mimetype="text/csv", as_attachment=True,
                         download_name=_out_name(jf.upload.name), conditional=False,
                         max_age=0)

    @bp.route("/jobs/<jid>/files/<int:i>/preview", methods=["GET"])
    def job_file_preview(jid, i):
        """Prime righe dell'output anonimizzato (header + PREVIEW_ROWS righe): legge
        solo la testa del file, come l'anteprima dell'input."""
        job = _job_or_404(jid)
        if job is None:
            return err("Job non trovato.", 404)
        if not 0 <= i < len(job.files):
            return err("File non trovato.", 404)
        jf = job.files[i]
        if jf.state != "done" or not jf.out_path.exists():
            return err("Output non ancora pronto.", 409)
        try:
            d = engine.sniff(str(jf.out_path), n_preview=PREVIEW_ROWS)
        except Exception as e:  # noqa: BLE001
            return err(_safe_error(e, safe_exc), 422)
        return jsonify({"header": d["header"], "preview": d.get("preview", [])[:PREVIEW_ROWS],
                        "rows_out": (jf.report or {}).get("rows_out")})

    @bp.route("/jobs/<jid>/dictionary", methods=["GET"])
    def job_dictionary(jid):
        job = _job_or_404(jid)
        if job is None:
            return err("Job non trovato.", 404)
        if not job.mapping_enabled:
            return err("Dizionario disattivato: l'anonimizzazione è definitiva.", 404)
        if job.state != "done":
            return err("Il job non è terminato.", 409)
        body = json.dumps(job.registry.mapping, ensure_ascii=False, indent=2)
        return Response(body, mimetype="application/json", headers={
            "Content-Disposition": 'attachment; filename="dizionario_anonimizzazione.json"',
            "Cache-Control": "no-store"})

    @bp.route("/jobs/<jid>", methods=["DELETE"])
    def job_delete(jid):
        job = _job_or_404(jid)
        if job is None:
            return jsonify({"ok": True})
        job.deleted = True
        job.cancel.set()
        with lock:
            queued = jid in pending
            if queued:
                pending.remove(jid)
        if queued or job.state != "running":
            if queued:
                job.state = "cancelled"
            _purge(job)                # in corso: ci pensa il worker a fine file
        return jsonify({"ok": True})

    # per i test / diagnostica
    bp.csv_state = {"uploads": uploads, "jobs": jobs, "instance": inst, "root": root,
                    "reap": _reap}
    return bp
