# -*- coding: utf-8 -*-
"""Motore CSV in streaming (src/app/csv_engine.py).

Solo stdlib: il motore dell'app (modello + rete regex) e' sostituito da un
`detect_fn` finto a regex, SOLO per i test, cosi' si gira in CI senza torch.
Tutti i dati sono SINTETICI.
"""
import csv
import json
import os
import random
import re
import sys
import tempfile
import tracemalloc
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "app"))
import csv_engine  # noqa: E402
from csv_engine import Cancelled, CsvError, default_plan, process_file, sniff  # noqa: E402
from registry import Registry  # noqa: E402

A = {"mode": "anonymize"}
P = {"mode": "preserve"}

# Motore finto (test-only): poche regex che imitano detect_spans_batch.
_FAKE = [
    ("EMAIL", re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")),
    ("DATE", re.compile(r"\d{4}-\d{2}-\d{2}(?:[T ][\d:.]+(?:Z|[+-]\d{2}:\d{2})?)?")),
    ("IPADDR", re.compile(r"(?<![\d.])\d{1,3}(?:\.\d{1,3}){3}(?![\d.])")),
    ("TELEPHONENUM", re.compile(r"\b3\d{2} ?\d{6,7}\b")),
    ("FULLNAME", re.compile(r"Mario Rossi|Anna Bianchi")),
]


def fake_detect(texts, excluded):
    out = []
    for t in texts:
        spans, taken = [], []
        for lab, rx in _FAKE:
            if lab in excluded:
                continue
            for m in rx.finditer(t):
                s, e = m.span()
                if all(e <= a or s >= b for a, b in taken):
                    taken.append((s, e))
                    spans.append((s, e, lab))
        out.append(sorted(spans))
    return out


class Base(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = self._tmp.name

    def tearDown(self):
        self._tmp.cleanup()

    def write(self, name, data, encoding="utf-8"):
        p = os.path.join(self.dir, name)
        with open(p, "wb") as f:
            f.write(data.encode(encoding) if isinstance(data, str) else data)
        return p

    def read_bytes(self, p):
        with open(p, "rb") as f:
            return f.read()

    def rows(self, p, delimiter=",", encoding="utf-8"):
        with open(p, encoding=encoding, newline="") as f:
            return list(csv.reader(f, delimiter=delimiter))

    def run_engine(self, data, plan, name="in.csv", registry=None, detect_fn=fake_detect, **kw):
        src = self.write(name, data)
        out = os.path.join(self.dir, "out_" + name)
        reg = registry if registry is not None else Registry()
        rep = process_file(src, out, plan, registry=reg, detect_fn=detect_fn, **kw)
        return out, rep, reg


class SniffTest(Base):
    def test_delimiters(self):
        for d in (",", ";", "\t"):
            p = self.write("s.csv", f"a{d}b{d}c\n1{d}x, y{d}3\n4{d}5{d}6\n")
            info = sniff(p)
            self.assertEqual(info["delimiter"], d)
            self.assertEqual(info["header"], ["a", "b", "c"])
            self.assertEqual(info["encoding"], "utf-8")
            self.assertEqual(info["newline"], "\n")

    def test_semicolon_with_commas_in_text(self):
        p = self.write("s.csv", "Nome;Note\r\nMario;uno, due, tre\r\nAnna;a, b\r\n")
        info = sniff(p)
        self.assertEqual(info["delimiter"], ";")
        self.assertEqual(info["newline"], "\r\n")

    def test_bom_and_multiline_preview(self):
        p = self.write("s.csv", "﻿a,b\r\n\"riga1\r\nriga2\",x\r\n3,4\r\n")
        info = sniff(p)
        self.assertEqual(info["encoding"], "utf-8-sig")
        self.assertEqual(info["header"], ["a", "b"])
        self.assertEqual(info["preview"], [["riga1\r\nriga2", "x"], ["3", "4"]])
        self.assertEqual(info["newline"], "\r\n")

    def test_preview_limit_and_head_only(self):
        body = "a,b\n" + "".join(f"{i},v{i}\n" for i in range(200_000))
        p = self.write("big.csv", body)
        info = sniff(p, n_preview=5)
        self.assertEqual(len(info["preview"]), 5)
        self.assertEqual(info["preview"][0], ["0", "v0"])

    def test_rejects_non_utf8_without_content(self):
        p = self.write("l1.csv", "nome,citta\nSegreto,Città\n", encoding="latin-1")
        with self.assertRaises(CsvError) as cm:
            sniff(p)
        self.assertNotIn("Segreto", str(cm.exception))
        p = self.write("u16.csv", "a,b\nSegreto,2\n", encoding="utf-16")
        with self.assertRaises(CsvError) as cm:
            sniff(p)
        self.assertNotIn("Segreto", str(cm.exception))

    def test_rejects_empty(self):
        with self.assertRaises(CsvError):
            sniff(self.write("e.csv", ""))


class PlanTest(unittest.TestCase):
    def test_modes_constant(self):
        self.assertEqual(csv_engine.MODES, ("anonymize", "preserve"))

    def test_default_plan_all_anonymize(self):
        header = ["Data", "Email", "Evento", "Note"]
        self.assertEqual(default_plan(header), [{"mode": "anonymize"}] * 4)
        self.assertEqual(default_plan([]), [])

    def test_csv_error_is_value_error(self):
        self.assertTrue(issubclass(CsvError, ValueError))

    def test_removed_api(self):
        for name in ("propose_plan", "regex_spans", "_infer_tag"):
            self.assertFalse(hasattr(csv_engine, name), name)


class ProcessTest(Base):
    def test_delimiters_roundtrip(self):
        for d in (",", ";", "\t"):
            data = (f"id{d}email{d}note\n1{d}mario.rossi@example.com{d}"
                    f"\"ciao, scrivi a anna@example.org\"\n")
            out, rep, _ = self.run_engine(data, [P, A, A], name=f"d{ord(d)}.csv")
            rows = self.rows(out, delimiter=d)
            self.assertEqual(rows[0], ["id", "email", "note"])
            self.assertEqual(rows[1], ["1", "[EMAIL_1]", "ciao, scrivi a [EMAIL_2]"])
            self.assertEqual(rep["rows_in"], 1)
            self.assertEqual(rep["rows_out"], 1)
            self.assertEqual(rep["by_label"], {"EMAIL": 2})

    def test_line_endings_and_bom_preserved(self):
        out, _, _ = self.run_engine("﻿a,b\r\n1,x@y.it\r\n", [P, A])
        self.assertEqual(self.read_bytes(out), "﻿a,b\r\n1,[EMAIL_1]\r\n".encode("utf-8"))
        out, _, _ = self.run_engine("a,b\n1,x@y.it\n", [P, A], name="lf.csv")
        self.assertEqual(self.read_bytes(out), b"a,b\n1,[EMAIL_1]\n")

    def test_empty_values_untouched(self):
        seen = []

        def fn(texts, excluded):
            seen.extend(texts)
            return fake_detect(texts, excluded)

        data = "a,b,c\n,,\n1, ,x@y.it\n"
        out, _, reg = self.run_engine(data, [A, A, A], detect_fn=fn)
        self.assertEqual(self.rows(out), [["a", "b", "c"], ["", "", ""], ["1", " ", "[EMAIL_1]"]])
        self.assertEqual(len(reg), 1)
        self.assertNotIn("", seen)
        self.assertNotIn(" ", seen)

    def test_text_without_pii_unchanged(self):
        out, rep, _ = self.run_engine("a\nnessun dato personale\n", [A])
        self.assertEqual(self.rows(out)[1], ["nessun dato personale"])
        self.assertEqual(rep["by_label"], {})

    def test_consistency_across_columns_and_files(self):
        reg = Registry()
        a = "Attore,Destinatari\nmario@example.com,\"anna@example.org, mario@example.com\"\n"
        b = "Owner;Note\r\nanna@example.org;inviato a MARIO@example.com da Mario Rossi\r\n"
        out_a, _, _ = self.run_engine(a, [A, A], name="a.csv", registry=reg)
        out_b, _, _ = self.run_engine(b, [A, A], name="b.csv", registry=reg)
        self.assertEqual(self.rows(out_a)[1], ["[EMAIL_1]", "[EMAIL_2], [EMAIL_1]"])
        self.assertEqual(self.rows(out_b, ";")[1],
                         ["[EMAIL_2]", "inviato a [EMAIL_1] da [FULLNAME_1]"])
        self.assertEqual(reg.mapping["[EMAIL_1]"], "mario@example.com")

    def test_preserved_columns_byte_exact(self):
        ts = ["2024-03-05T10:11:12.123456+01:00", "2024-03-05 10:11:12 CET",
              "2024-03-05T09:11:12Z", "05/03/2024 10:11", "1709629872.5"]
        data = "Data,Valore,Email\n" + "".join(f"{t},{i * 1.5},u{i}@x.it\n" for i, t in enumerate(ts))
        out, _, _ = self.run_engine(data, [P, P, A])
        lines = self.read_bytes(out).decode("utf-8").split("\n")
        for i, t in enumerate(ts):
            self.assertEqual(lines[i + 1], f"{t},{i * 1.5},[EMAIL_{i + 1}]")

    def test_preserve_never_touched_even_with_pii(self):
        data = "a\nmario@example.com 2024-03-12\n"
        out, _, _ = self.run_engine(data, [P], detect_fn=None)
        self.assertEqual(self.read_bytes(out), data.encode())

    def test_timestamp_in_anonymize_column_goes_to_engine(self):
        seen = []

        def fn(texts, excluded):
            seen.extend(texts)
            return fake_detect(texts, excluded)

        ts = "2024-03-05T10:11:12Z"
        data = f"Data,Evento\n{ts},login\n"
        out, rep, _ = self.run_engine(data, [A, P], detect_fn=fn)
        self.assertIn(ts, seen)  # nessun trattamento speciale delle date
        self.assertEqual(self.rows(out)[1], ["[DATE_1]", "login"])
        self.assertEqual(rep["by_label"], {"DATE": 1})
        # con DATE escluso resta in chiaro
        out, _, _ = self.run_engine(data, [A, P], name="x.csv", excluded={"DATE"})
        self.assertEqual(self.rows(out)[1], [ts, "login"])

    def test_multiline_and_json_cells(self):
        obj = {"user": "mario@example.com", "n": 3, "nested": {"to": ["anna@example.org", "ok"]},
               "msg": "Mario Rossi ha scritto\nda 10.1.2.3"}
        cell = json.dumps(obj, ensure_ascii=False, indent=1)  # multilinea, spazi originali
        data = ("id,payload,testo\r\n1," + '"' + cell.replace('"', '""') + '"'
                + ",\"riga uno\nMario Rossi\"\r\n")
        out, rep, _ = self.run_engine(data, [P, A, A])
        rows = self.rows(out)
        self.assertEqual(len(rows), 2)
        new = rows[1][1]
        parsed = json.loads(new)
        self.assertEqual(parsed["user"], "[EMAIL_1]")
        self.assertEqual(parsed["n"], 3)
        self.assertEqual(parsed["nested"]["to"], ["[EMAIL_2]", "ok"])
        self.assertEqual(parsed["msg"], "[FULLNAME_1] ha scritto\nda [IPADDR_1]")
        expected = cell.replace("mario@example.com", "[EMAIL_1]").replace(
            "anna@example.org", "[EMAIL_2]").replace("Mario Rossi", "[FULLNAME_1]").replace(
            "10.1.2.3", "[IPADDR_1]")
        self.assertEqual(new, expected)  # struttura byte-identica salvo i valori
        self.assertEqual(rows[1][2], "riga uno\n[FULLNAME_1]")
        self.assertEqual(rep["by_label"], {"EMAIL": 2, "FULLNAME": 2, "IPADDR": 1})

    def test_json_keys_untouched_and_escapes(self):
        cell = ('{"mario@example.com": "x", "v": "a\\"b mario@example.com", '
                '"u": "\\u00e8 anna@example.org"}')
        data = "j\n\"" + cell.replace('"', '""') + "\"\n"
        out, _, _ = self.run_engine(data, [A])
        parsed = json.loads(self.rows(out)[1][0])
        self.assertIn("mario@example.com", parsed)
        self.assertEqual(parsed["v"], 'a"b [EMAIL_1]')
        self.assertEqual(parsed["u"], "è [EMAIL_2]")

    def test_excluded_tags_filtered_even_if_engine_returns_them(self):
        def ignores_excluded(texts, excluded):
            return fake_detect(texts, set())

        data = "a,b\nmario@example.com,\"Mario Rossi mario@example.com 10.0.0.1\"\n"
        out, rep, _ = self.run_engine(data, [A, A], excluded={"EMAIL"},
                                      detect_fn=ignores_excluded)
        self.assertEqual(self.rows(out)[1],
                         ["mario@example.com", "[FULLNAME_1] mario@example.com [IPADDR_1]"])
        self.assertNotIn("EMAIL", rep["by_label"])

    def test_excluded_passed_to_detect_fn(self):
        got = []

        def fn(texts, excluded):
            got.append(excluded)
            return fake_detect(texts, excluded)

        self.run_engine("a\nMario Rossi\n", [A], detect_fn=fn, excluded=["FULLNAME"])
        self.assertEqual(got, [{"FULLNAME"}])

    def test_detect_fn_required_for_anonymize(self):
        with self.assertRaises(CsvError) as cm:
            self.run_engine("a,b\nsegreto@example.com,1\n", [A, P], detect_fn=None)
        self.assertNotIn("segreto", str(cm.exception))
        self.assertEqual(sorted(os.listdir(self.dir)), ["in.csv"])
        # tutto preserve: detect_fn non serve
        out, _, _ = self.run_engine("a\n1\n", [P], name="p.csv", detect_fn=None)
        self.assertEqual(self.read_bytes(out), b"a\n1\n")

    def test_detect_fn_wrong_result_count(self):
        with self.assertRaises(CsvError):
            self.run_engine("a\nx\n", [A], detect_fn=lambda texts, excluded: [])

    def test_detect_fn_batched(self):
        calls = []

        def fn(texts, excluded):
            calls.append(len(texts))
            return fake_detect(texts, excluded)

        data = "t\n" + "".join(f"Mario Rossi {i}\n" for i in range(100))
        out, _, _ = self.run_engine(data, [A], detect_fn=fn, batch_size=16)
        self.assertTrue(all(n <= 16 for n in calls))
        self.assertLessEqual(len(calls), 8)
        self.assertEqual(sum(calls), 100)
        self.assertEqual(self.rows(out)[5], ["[FULLNAME_1] 4"])

    def test_cache_avoids_repeated_detection(self):
        seen = []

        def fn(texts, excluded):
            seen.extend(texts)
            return fake_detect(texts, excluded)

        data = "t,u\n" + "Mario Rossi,Mario Rossi\n" * 50
        self.run_engine(data, [A, A], detect_fn=fn, batch_size=8)
        self.assertEqual(seen, ["Mario Rossi"])

    def test_output_reparses(self):
        data = 'a,b,c\n"x,1","y ""q""",z@z.it\n"\n",,\n"a\rb",;,"multi\nriga mario@example.com"\n'
        out, rep, _ = self.run_engine(data, [P, P, A])
        rows = self.rows(out)
        self.assertEqual(len(rows), 4)
        self.assertTrue(all(len(r) == 3 for r in rows))
        self.assertEqual(rows[1], ["x,1", 'y "q"', "[EMAIL_1]"])
        self.assertEqual(rows[2], ["\n", "", ""])
        self.assertEqual(rows[3], ["a\rb", ";", "multi\nriga [EMAIL_2]"])
        self.assertEqual(rep["rows_out"], 3)


class RobustnessTest(Base):
    SECRET = "segretissimo@example.com"

    def test_malformed_row_skipped_and_strict(self):
        data = f"a,b\n1,2\n3,{self.SECRET},extra\n4,5\n"
        out, rep, _ = self.run_engine(data, [P, A])
        self.assertEqual(rep["rows_skipped"], 1)
        self.assertEqual(rep["skipped_lines"], [3])
        self.assertEqual(rep["rows_in"], 3)
        self.assertEqual(rep["rows_out"], 2)
        self.assertEqual(len(self.rows(out)), 3)
        self.assertNotIn(self.SECRET, json.dumps(rep))
        with self.assertRaises(CsvError) as cm:
            self.run_engine(data, [P, A], name="strict.csv", strict=True)
        self.assertNotIn(self.SECRET, str(cm.exception))
        self.assertIn("3", str(cm.exception))
        self.assertFalse(os.path.exists(os.path.join(self.dir, "out_strict.csv")))
        self.assertFalse(os.path.exists(os.path.join(self.dir, "out_strict.csv.partial")))

    def test_cancel_removes_partial(self):
        data = "a\n" + "".join(f"u{i}@y.it\n" for i in range(1000))
        n = {"c": 0}

        def cancel():
            n["c"] += 1
            return n["c"] > 2

        with self.assertRaises(Cancelled):
            self.run_engine(data, [A], cancel=cancel, batch_size=10)
        self.assertEqual(sorted(os.listdir(self.dir)), ["in.csv"])

    def test_exception_in_detect_fn_cleans_up(self):
        def boom(texts, excluded):
            raise RuntimeError("gpu esplosa")

        with self.assertRaises(RuntimeError):
            self.run_engine("a\nciao\n", [A], detect_fn=boom)
        self.assertEqual(sorted(os.listdir(self.dir)), ["in.csv"])

    def test_no_cell_values_in_errors(self):
        for plan in ([P], [P, {"mode": "boh"}], [P, {"mode": "drop"}],
                     [P, {"mode": "pseudonymize"}], [P, {"mode": "regex"}]):
            with self.assertRaises(CsvError) as cm:
                self.run_engine(f"a,b\n{self.SECRET},1\n", plan)
            self.assertNotIn(self.SECRET, str(cm.exception))
            self.assertNotIn("segretissimo", str(cm.exception))

    def test_progress_reports_rows_and_bytes(self):
        data = "a\n" + "x@y.it\n" * 500
        calls = []
        self.run_engine(data, [A], progress=lambda r, b, t: calls.append((r, b, t)), batch_size=50)
        self.assertGreater(len(calls), 1)
        self.assertEqual(calls[-1][0], 500)
        self.assertEqual(calls[-1][1], calls[-1][2])
        self.assertEqual(calls[-1][2], len(data))
        self.assertEqual([c[0] for c in calls], sorted(c[0] for c in calls))

    def test_large_file_streaming_bounded_memory(self):
        rnd = random.Random(7)
        # pochi valori unici: la memoria deve dipendere da loro, non dal numero di righe
        users = [f"utente{i}@example.com" for i in range(60)]
        ips = [f"10.{i // 250}.{i % 250}.{i % 7}" for i in range(500)]
        n = 200_000
        src = os.path.join(self.dir, "big.csv")
        with open(src, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f, lineterminator="\r\n")
            w.writerow(["Data", "Attore", "Indirizzo IP", "Destinatari", "Evento"])
            for i in range(n):
                w.writerow([f"2024-01-01T00:00:{i % 60:02d}.{i:06d}Z", rnd.choice(users),
                            rnd.choice(ips), f"{rnd.choice(users)}, {rnd.choice(users)}", f"ev{i}"])
        size = os.path.getsize(src)
        out = os.path.join(self.dir, "big_out.csv")
        reg = Registry()
        tracemalloc.start()
        try:
            rep = process_file(src, out, [P, A, A, A, P], registry=reg, detect_fn=fake_detect)
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        self.assertEqual(rep["rows_in"], n)
        self.assertEqual(rep["rows_out"], n)
        self.assertEqual(rep["rows_skipped"], 0)
        self.assertLess(peak, 12 * 1024 * 1024, f"picco {peak} su file di {size} byte")
        self.assertLess(peak, size / 2)
        count = 0
        with open(out, encoding="utf-8", newline="") as f:
            r = csv.reader(f)
            self.assertEqual(next(r), ["Data", "Attore", "Indirizzo IP", "Destinatari", "Evento"])
            for i, row in enumerate(r):
                self.assertEqual(len(row), 5)
                self.assertEqual(row[0], f"2024-01-01T00:00:{i % 60:02d}.{i:06d}Z")
                self.assertEqual(row[4], f"ev{i}")  # nessuna riga persa/duplicata/riordinata
                self.assertNotIn("@", row[1] + row[3])
                self.assertTrue(row[2].startswith("[IPADDR_"), row[2])
                count += 1
        self.assertEqual(count, n)


class CacheTest(unittest.TestCase):
    def test_lru_bounded_by_entries_and_chars(self):
        c = csv_engine._Cache(3, max_chars=10)
        for t in ("a", "b", "c", "d"):
            c.put(t, ())
        self.assertEqual(list(c.data), ["b", "c", "d"])
        c.get("b")
        c.put("123456789", ())
        self.assertEqual(list(c.data), ["b", "123456789"])  # c, d via: LRU + tetto
        c.put("x" * (csv_engine.CACHE_MAX_TEXT + 1), ())
        self.assertEqual(len(c.data), 2)


if __name__ == "__main__":
    unittest.main()
