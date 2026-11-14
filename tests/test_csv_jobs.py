# -*- coding: utf-8 -*-
"""Job CSV (`csv_jobs.py`): upload a pezzi, finalize, coda, download, pulizia.

Si monta il Blueprint su una Flask minima (niente torch/fitz/modello) con un motore
finto che rispetta il contratto di `csv_engine` e un `detect_fn` finto. Se il motore
vero e' implementato, `RealEngine` rifa' il giro completo con quello.
Si salta tutto se Flask non c'e' (CI senza pip install).

Tutti i valori sono SINTETICI.
"""

import csv
import io
import json
import os
import re
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "app"))

try:
    import flask  # noqa: F401
    import csv_jobs
except ImportError:                     # CI senza Flask
    csv_jobs = None


# --------------------------------------------------------------------------- #
# Motore finto: stesso contratto di csv_engine, minimale
# --------------------------------------------------------------------------- #
class _Cancelled(Exception):
    pass


def _fake_sniff(path, n_preview=20):
    with open(path, "rb") as fh:
        raw = fh.read(65536)
    if b"\x00" in raw:
        raise ValueError("Il file non sembra un CSV di testo.")
    enc = "utf-8-sig" if raw.startswith(b"\xef\xbb\xbf") else "utf-8"
    text = raw.decode(enc)
    delim = ";" if text.split("\n", 1)[0].count(";") > text.split("\n", 1)[0].count(",") else ","
    rows = list(csv.reader(io.StringIO(text), delimiter=delim))
    return {"encoding": enc, "delimiter": delim, "newline": "\n",
            "header": rows[0], "preview": rows[1:1 + n_preview]}


def _fake_default_plan(header):
    return [{"mode": "anonymize"} for _ in header]


def _fake_process(in_path, out_path, plan, *, registry, detect_fn=None, excluded=(),
                  dialect=None, progress=None, cancel=None, strict=False, batch_size=64):
    dialect = dialect or _fake_sniff(in_path)
    total = os.path.getsize(in_path)
    part = out_path + ".partial"
    rows_in = rows_out = skipped = 0
    lines, by_label = [], {}
    try:
        with open(in_path, encoding=dialect["encoding"], newline="") as fi, \
                open(part, "w", encoding=dialect["encoding"], newline="") as fo:
            r = csv.reader(fi, delimiter=dialect["delimiter"])
            w = csv.writer(fo, delimiter=dialect["delimiter"], lineterminator="\n")
            header = next(r)
            w.writerow(header)
            for n, row in enumerate(r, start=2):
                if cancel and cancel():
                    raise _Cancelled()
                rows_in += 1
                if len(row) != len(header):
                    if strict:
                        raise ValueError(f"Riga {n}: numero di campi errato.")
                    skipped += 1
                    lines.append(n)
                    continue
                out = []
                for cell, p in zip(row, plan):
                    if cell and p["mode"] == "anonymize":
                        spans = detect_fn([cell], set(excluded))[0]
                        for s, e, lab in sorted(spans, reverse=True):
                            cell = cell[:s] + registry.assign(lab, cell[s:e]) + cell[e:]
                            by_label[lab] = by_label.get(lab, 0) + 1
                    out.append(cell)
                w.writerow(out)
                rows_out += 1
                if progress:
                    progress(rows_in, 0, total)
                if FAKE.delay:
                    time.sleep(FAKE.delay)
        os.replace(part, out_path)
    except BaseException:
        if os.path.exists(part):
            os.remove(part)
        raise
    return {"rows_in": rows_in, "rows_out": rows_out, "rows_skipped": skipped,
            "skipped_lines": lines, "by_label": by_label, "seconds": 0.0}


FAKE = SimpleNamespace(
    MODES=("anonymize", "preserve"),
    sniff=_fake_sniff, default_plan=_fake_default_plan, process_file=_fake_process,
    Cancelled=_Cancelled, delay=0.0)

PATTERNS = (("TELEFONO", re.compile(r"\+39 ?\d{3} ?\d{6,7}")),
            ("EMAIL", re.compile(r"[\w.]+@[\w.]+\.\w+")))


def fake_detect(texts, excluded):
    """Solo telefoni ed e-mail (finto): span non sovrapposte per ogni testo."""
    out = []
    for t in texts:
        spans = [(m.start(), m.end(), lab) for lab, rx in PATTERNS if lab not in excluded
                 for m in rx.finditer(t)]
        out.append(sorted(spans))
    return out


CSV_A = ("nome,email,note,segreto\r\n"
         "Mario,mario.rossi@example.com,chiamare +39 333 1234567,x1\r\n"
         "Anna,anna.bianchi@example.com,nessuna,x2\r\n"
         "Mario,Mario.Rossi@example.com,,x3\r\n")
CSV_B = ("utente;email\n"
         "u1;anna.bianchi@example.com\n"
         "u2;luca.verdi@example.com\n")


# --------------------------------------------------------------------------- #
@unittest.skipIf(csv_jobs is None, "Flask non installato")
class _Base(unittest.TestCase):
    engine = FAKE
    prefs = ((), True)
    threshold = None                     # large_threshold (byte); None = nessun avviso

    def setUp(self):
        FAKE.delay = 0.0
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name) / "csv_jobs"
        app = flask.Flask("t")
        app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024
        self.bp = csv_jobs.create_blueprint(lambda: self.prefs, lambda: fake_detect,
                                            engine=self.engine, root=self.root,
                                            large_threshold=lambda: self.threshold)
        app.register_blueprint(self.bp)
        self.c = app.test_client()

    def tearDown(self):
        FAKE.delay = 0.0
        for jid in list(self.bp.csv_state["jobs"]):
            self.c.delete(f"/csv/jobs/{jid}")
        self.bp.csv_state["instance"].close()
        self.tmp.cleanup()

    # helpers ---------------------------------------------------------------
    def upload(self, name, text, chunk=7):
        data = text.encode("utf-8")
        r = self.c.post("/csv/uploads", json={"name": name, "size": len(data)})
        self.assertEqual(r.status_code, 200, r.get_json())
        uid = r.get_json()["upload_id"]
        for off in range(0, len(data), chunk):
            r = self.c.put(f"/csv/uploads/{uid}?offset={off}", data=data[off:off + chunk],
                           content_type="application/octet-stream")
            self.assertEqual(r.status_code, 200, r.get_json())
        self.assertTrue(r.get_json()["complete"])
        r = self.c.post(f"/csv/uploads/{uid}/finalize", json={})
        self.assertEqual(r.status_code, 200, r.get_json())
        return r.get_json()

    def start(self, fins, **opts):
        files = [{"upload_id": f["upload_id"], "plan": f["plan"]} for f in fins]
        r = self.c.post("/csv/jobs", json={"files": files, **opts})
        self.assertEqual(r.status_code, 202, r.get_json())
        return r.get_json()["job_id"]

    def wait(self, jid, states=("done", "error", "cancelled"), timeout=10):
        t0 = time.time()
        while time.time() - t0 < timeout:
            st = self.c.get(f"/csv/jobs/{jid}").get_json()
            if st["state"] in states:
                return st
            time.sleep(0.02)
        self.fail(f"job ancora {st['state']}")


class Upload(_Base):
    def test_chunked_upload_and_preview(self):
        fin = self.upload("clienti.csv", CSV_A)
        self.assertEqual(fin["header"], ["nome", "email", "note", "segreto"])
        self.assertEqual(len(fin["preview"]), 3)
        self.assertEqual(fin["delimiter"], ",")
        self.assertEqual(fin["plan"], [{"mode": "anonymize"}] * 4)
        self.assertEqual(fin["modes"], ["anonymize", "preserve"])
        self.assertFalse(fin["large"])
        self.assertIsNone(fin["large_threshold"])
        up = self.bp.csv_state["uploads"][fin["upload_id"]]
        self.assertEqual(up.path.read_bytes(), CSV_A.encode())   # byte-identico

    def test_large_flag_follows_threshold(self):
        self.threshold = len(CSV_B.encode()) + 1
        small = self.upload("utenti.csv", CSV_B)
        self.assertFalse(small["large"])
        self.assertEqual(small["large_threshold"], self.threshold)
        big = self.upload("clienti.csv", CSV_A)            # CSV_A > CSV_B
        self.assertTrue(big["large"])
        self.assertEqual(big["large_threshold"], self.threshold)

    def test_out_of_order_and_oversize(self):
        r = self.c.post("/csv/uploads", json={"name": "a.csv", "size": 10})
        uid = r.get_json()["upload_id"]
        r = self.c.put(f"/csv/uploads/{uid}?offset=5", data=b"abc")
        self.assertEqual(r.status_code, 409)
        self.assertEqual(r.get_json()["received"], 0)
        r = self.c.put(f"/csv/uploads/{uid}?offset=0", data=b"x" * 11)
        self.assertEqual(r.status_code, 400)
        r = self.c.put(f"/csv/uploads/{uid}?offset=0", data=b"x" * (csv_jobs.CHUNK_MAX + 1))
        self.assertEqual(r.status_code, 413)
        r = self.c.post(f"/csv/uploads/{uid}/finalize", json={})
        self.assertEqual(r.status_code, 409)              # incompleto

    def test_rejects_non_csv_and_bad_names(self):
        r = self.c.post("/csv/uploads", json={"name": "atto.pdf", "size": 10})
        self.assertEqual(r.status_code, 400)
        r = self.c.post("/csv/uploads", json={"name": "../../x.csv", "size": 3})
        self.assertEqual(r.get_json()["name"], "x.csv")

    def test_sniff_error_has_no_values(self):
        data = b"a,b\n\x00segreto\n"
        r = self.c.post("/csv/uploads", json={"name": "bin.csv", "size": len(data)})
        uid = r.get_json()["upload_id"]
        self.c.put(f"/csv/uploads/{uid}?offset=0", data=data)
        r = self.c.post(f"/csv/uploads/{uid}/finalize", json={})
        self.assertEqual(r.status_code, 422)
        self.assertNotIn("segreto", r.get_data(as_text=True))

    def test_delete_upload_removes_file(self):
        fin = self.upload("a.csv", CSV_A)
        up = self.bp.csv_state["uploads"][fin["upload_id"]]
        self.c.delete(f"/csv/uploads/{fin['upload_id']}")
        self.assertFalse(up.path.parent.exists())


class Jobs(_Base):
    def test_job_runs_to_done_and_downloads(self):
        a = self.upload("clienti.csv", CSV_A)
        b = self.upload("utenti.csv", CSV_B)
        jid = self.start([a, b])
        st = self.wait(jid)
        self.assertEqual(st["state"], "done", st)
        self.assertEqual([f["rows_done"] for f in st["files"]], [3, 2])
        self.assertEqual(st["bytes_done"], st["bytes_total"])
        self.assertEqual(st["files"][0]["out_name"], "clienti.anon.csv")

        r = self.c.get(f"/csv/jobs/{jid}/files/0")
        self.assertEqual(r.status_code, 200)
        self.assertIn("clienti.anon.csv", r.headers["Content-Disposition"])
        out_a = r.get_data(as_text=True)
        r.close()
        self.assertEqual(out_a.splitlines(), [
            "nome,email,note,segreto",
            "Mario,[EMAIL_1],chiamare [TELEFONO_1],x1",
            "Anna,[EMAIL_2],nessuna,x2",
            "Mario,[EMAIL_1],,x3"])
        r = self.c.get(f"/csv/jobs/{jid}/files/1")
        out_b = r.get_data(as_text=True)
        r.close()
        # pseudonimi coerenti fra i file dello stesso job
        self.assertEqual(out_b.splitlines()[1], "u1;[EMAIL_2]")
        self.assertEqual(out_b.splitlines()[2], "u2;[EMAIL_3]")

        r = self.c.get(f"/csv/jobs/{jid}/dictionary")
        self.assertEqual(r.status_code, 200)
        self.assertIn("dizionario_anonimizzazione.json", r.headers["Content-Disposition"])
        self.assertEqual(json.loads(r.get_data(as_text=True)), {
            "[EMAIL_1]": "mario.rossi@example.com", "[EMAIL_2]": "anna.bianchi@example.com",
            "[TELEFONO_1]": "+39 333 1234567", "[EMAIL_3]": "luca.verdi@example.com"})
        self.assertEqual(st["dictionary_size"], 4)

    def test_output_preview(self):
        a = self.upload("clienti.csv", CSV_A)
        jid = self.start([a])
        self.assertEqual(self.c.get(f"/csv/jobs/{jid}/files/5/preview").status_code, 404)
        self.wait(jid)
        r = self.c.get(f"/csv/jobs/{jid}/files/0/preview")
        self.assertEqual(r.status_code, 200, r.get_json())
        d = r.get_json()
        self.assertEqual(d["header"], ["nome", "email", "note", "segreto"])
        self.assertEqual(d["preview"][0], ["Mario", "[EMAIL_1]", "chiamare [TELEFONO_1]", "x1"])
        self.assertEqual(len(d["preview"]), 3)
        self.assertNotIn("mario.rossi", json.dumps(d))

    def test_preserved_column_left_untouched(self):
        a = self.upload("clienti.csv", CSV_A)
        plan = [{"mode": "anonymize"}, {"mode": "preserve"},
                {"mode": "anonymize"}, {"mode": "anonymize"}]
        st = self.wait(self.start([dict(a, plan=plan)]))
        self.assertEqual(st["state"], "done", st)
        r = self.c.get(f"/csv/jobs/{st['job_id']}/files/0")
        lines = r.get_data(as_text=True).splitlines()
        r.close()
        self.assertEqual(lines[1], "Mario,mario.rossi@example.com,chiamare [TELEFONO_1],x1")

    def test_mapping_off_has_no_dictionary(self):
        a = self.upload("a.csv", CSV_A)
        jid = self.start([a], include_mapping=False)
        st = self.wait(jid)
        self.assertEqual(st["state"], "done")
        self.assertFalse(st["mapping_enabled"])
        self.assertEqual(st["dictionary_size"], 0)
        r = self.c.get(f"/csv/jobs/{jid}/dictionary")
        self.assertEqual(r.status_code, 404)
        self.assertNotIn("example.com", r.get_data(as_text=True))
        self.assertEqual(self.bp.csv_state["jobs"][jid].registry.mapping, {})

    def test_server_pref_mapping_off_and_excluded(self):
        self.prefs = (["TELEFONO"], False)
        a = self.upload("a.csv", CSV_A)
        jid = self.start([a])
        st = self.wait(jid)
        self.assertFalse(st["mapping_enabled"])
        self.assertEqual(st["excluded_tags"], ["TELEFONO"])
        r = self.c.get(f"/csv/jobs/{jid}/files/0")
        self.assertIn("chiamare +39 333 1234567", r.get_data(as_text=True))
        r.close()

    def test_bad_plan_rejected(self):
        a = self.upload("a.csv", CSV_A)
        r = self.c.post("/csv/jobs", json={"files": [
            {"upload_id": a["upload_id"], "plan": [{"mode": "preserve"}]}]})
        self.assertEqual(r.status_code, 400)
        for old in ("boh", "pseudonymize", "drop", "regex", "model"):
            r = self.c.post("/csv/jobs", json={"files": [
                {"upload_id": a["upload_id"], "plan": [{"mode": old}] * 4}]})
            self.assertEqual(r.status_code, 400, old)

    def test_upload_used_once(self):
        a = self.upload("a.csv", CSV_A)
        self.start([a])
        r = self.c.post("/csv/jobs", json={"files": [{"upload_id": a["upload_id"]}]})
        self.assertEqual(r.status_code, 409)

    def test_skipped_rows_and_strict(self):
        bad = CSV_A + "riga,rotta\r\n"
        a = self.upload("a.csv", bad)
        st = self.wait(self.start([a]))
        self.assertEqual(st["files"][0]["rows_skipped"], 1)
        self.assertEqual(st["files"][0]["skipped_lines"], [5])
        b = self.upload("b.csv", bad)
        st = self.wait(self.start([b], strict=True))
        self.assertEqual(st["state"], "error")
        self.assertIn("Riga 5", st["error"])

    def test_internal_error_message_has_no_values(self):
        def boom(*a, **k):
            raise RuntimeError("mario.rossi@example.com")
        self.engine.process_file, orig = boom, self.engine.process_file
        try:
            a = self.upload("a.csv", CSV_A)
            st = self.wait(self.start([a]))
        finally:
            self.engine.process_file = orig
        self.assertEqual(st["state"], "error")
        self.assertNotIn("example.com", json.dumps(st))

    def test_cancel_running_job(self):
        FAKE.delay = 0.05
        big = "nome,email\n" + "".join(f"n{i},u{i}@example.com\n" for i in range(400))
        a = self.upload("big.csv", big, chunk=4096)
        jid = self.start([a])
        self.wait(jid, states=("running",))
        r = self.c.post(f"/csv/jobs/{jid}/cancel")
        self.assertEqual(r.status_code, 200)
        st = self.wait(jid)
        self.assertEqual(st["state"], "cancelled")
        self.assertLess(st["rows_done"], 400)
        job = self.bp.csv_state["jobs"][jid]
        self.assertFalse(job.files[0].out_path.exists())
        self.assertFalse(Path(str(job.files[0].out_path) + ".partial").exists())
        self.assertEqual(self.c.get(f"/csv/jobs/{jid}/files/0").status_code, 409)

    def test_queue_fifo_and_cancel_queued(self):
        FAKE.delay = 0.02
        big = "nome,email\n" + "".join(f"n{i},u{i}@example.com\n" for i in range(60))
        a = self.upload("a.csv", big, chunk=4096)
        b = self.upload("b.csv", CSV_B)
        c = self.upload("c.csv", CSV_B.replace("u1", "u9"))
        j1, j2, j3 = self.start([a]), self.start([b]), self.start([c])
        st2 = self.c.get(f"/csv/jobs/{j2}").get_json()
        self.assertEqual(st2["state"], "queued")
        self.c.post(f"/csv/jobs/{j2}/cancel")
        self.assertEqual(self.wait(j1)["state"], "done")
        self.assertEqual(self.wait(j3)["state"], "done")
        self.assertEqual(self.c.get(f"/csv/jobs/{j2}").get_json()["state"], "cancelled")

    def test_delete_job_cleans_temp_files(self):
        a = self.upload("a.csv", CSV_A)
        jid = self.start([a])
        self.wait(jid)
        inst_dir = self.bp.csv_state["instance"].dir
        self.assertTrue(any(inst_dir.glob("j_*/out_0.csv")))
        r = self.c.delete(f"/csv/jobs/{jid}")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(sorted(p.name for p in inst_dir.iterdir()), [csv_jobs.LOCK_NAME])
        self.assertEqual(self.c.get(f"/csv/jobs/{jid}").status_code, 404)
        self.assertNotIn(a["upload_id"], self.bp.csv_state["uploads"])

    def test_delete_running_job(self):
        FAKE.delay = 0.05
        big = "nome,email\n" + "".join(f"n{i},u{i}@example.com\n" for i in range(400))
        a = self.upload("big.csv", big, chunk=4096)
        jid = self.start([a])
        self.wait(jid, states=("running",))
        inst_dir = self.bp.csv_state["instance"].dir
        self.c.delete(f"/csv/jobs/{jid}")
        t0 = time.time()
        while jid in self.bp.csv_state["jobs"] and time.time() - t0 < 10:
            time.sleep(0.02)
        self.assertEqual(sorted(p.name for p in inst_dir.iterdir()), [csv_jobs.LOCK_NAME])

    def test_idle_jobs_and_uploads_are_reaped(self):
        a = self.upload("a.csv", CSV_A)
        jid = self.start([a])
        self.wait(jid)
        b = self.upload("b.csv", CSV_B)          # upload mai usato da un job
        inst_dir = self.bp.csv_state["instance"].dir
        reap = self.bp.csv_state["reap"]
        self.assertEqual(reap(), 0)              # appena toccati: restano
        self.assertEqual(reap(time.time() + csv_jobs.IDLE_TTL + 1), 2)
        self.assertEqual(self.c.get(f"/csv/jobs/{jid}").status_code, 404)
        self.assertNotIn(b["upload_id"], self.bp.csv_state["uploads"])
        self.assertEqual(sorted(p.name for p in inst_dir.iterdir()), [csv_jobs.LOCK_NAME])

    def test_non_dict_file_item_is_400(self):
        for files in (["abc"], [["abc"]], [None]):
            r = self.c.post("/csv/jobs", json={"files": files})
            self.assertEqual(r.status_code, 400, files)

    def test_foreign_value_error_is_opaque(self):
        """Una ValueError che non viene dal motore (es. dal tokenizer) puo' contenere
        testo dell'utente: all'UI arriva solo il tipo."""
        self.assertEqual(csv_jobs._safe_error(ValueError("Mario Rossi"), safe=KeyError),
                         "Errore interno (ValueError)")
        self.assertEqual(csv_jobs._safe_error(KeyError("x"), safe=KeyError), "'x'")


@unittest.skipIf(csv_jobs is None, "Flask non installato")
class Sweep(unittest.TestCase):
    def test_sweep_removes_dead_instances_only(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            dead = root / "123-dead"
            (dead / "u_x").mkdir(parents=True)
            (dead / csv_jobs.LOCK_NAME).touch()
            (root / "nolock").mkdir()
            live = csv_jobs._Instance(root)
            live_dir = live.dir
            (live_dir / "u_y").mkdir()
            n = csv_jobs.sweep_stale(root)
            self.assertEqual(n, 2)
            self.assertFalse(dead.exists())
            self.assertTrue((live_dir / "u_y").exists())
            live.close()
            self.assertEqual(csv_jobs.sweep_stale(root), 1)   # lock rilasciato -> avanzo


# --------------------------------------------------------------------------- #
def _real_engine():
    if csv_jobs is None:
        return None
    import csv_engine
    try:
        with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as f:
            f.write("a,b\n1,2\n")
        csv_engine.sniff(f.name)
    except NotImplementedError:
        return None
    finally:
        os.unlink(f.name)
    return csv_engine


REAL = _real_engine()


@unittest.skipIf(REAL is None, "csv_engine non ancora implementato (o Flask assente)")
class RealEngine(_Base):
    engine = REAL

    def test_round_trip(self):
        a = self.upload("clienti.csv", CSV_A)
        plan = [{"mode": "preserve"}, {"mode": "anonymize"},
                {"mode": "anonymize"}, {"mode": "preserve"}]
        jid = self.start([dict(a, plan=plan)])
        st = self.wait(jid)
        self.assertEqual(st["state"], "done", st)
        r = self.c.get(f"/csv/jobs/{jid}/files/0")
        out = r.get_data(as_text=True)
        r.close()
        self.assertNotIn("example.com", out)
        self.assertEqual(out.splitlines()[0], "nome,email,note,segreto")
        self.assertIn("x1", out)                           # colonna preservata
        self.assertIn("[EMAIL_1]", out)
        self.assertIn("[TELEFONO_1]", out)
        mp = json.loads(self.c.get(f"/csv/jobs/{jid}/dictionary").get_data(as_text=True))
        self.assertEqual(mp["[EMAIL_1]"], "mario.rossi@example.com")


if __name__ == "__main__":
    unittest.main()
