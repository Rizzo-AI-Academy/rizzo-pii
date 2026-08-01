# -*- coding: utf-8 -*-
"""
Test del pacchetto di detector "cyber" (src/app/detectors_cyber.py) e della sua
integrazione in analyze() (src/app/app.py).

Il MODELLO e' sostituito da uno stub (dipendenza esterna, non il codice sotto esame):
restano reali le regex, i validatori, la keeplist e il merge. Girano quindi senza
torch/transformers e senza scaricare il checkpoint.

    python -m unittest discover -s tests

Tutti i valori usati qui sono inventati: nessun IOC reale, nessun indirizzo o dominio
di un caso vero (vedi CONTRIBUTING.md).
"""

import sys
import types
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "app"))

import detectors_cyber as cyber  # noqa: E402


class _FakeNlp:
    """Pipeline di token-classification finta: non trova nulla, cosi' il test misura
    solo la rete regex, la keeplist e il merge."""

    def __init__(self):
        self.model = types.SimpleNamespace(config=types.SimpleNamespace(
            label2id={"O": 0, "B-FULLNAME": 1, "I-FULLNAME": 2}))

    def __call__(self, texts, *args, **kwargs):
        return [[] for _ in texts] if isinstance(texts, list) else []


def _install_stubs():
    torch = types.ModuleType("torch")
    torch.cuda = types.SimpleNamespace(is_available=lambda: False)
    sys.modules.setdefault("torch", torch)
    transformers = types.ModuleType("transformers")
    transformers.pipeline = lambda *a, **k: _FakeNlp()
    sys.modules.setdefault("transformers", transformers)
    fitz = types.ModuleType("fitz")
    fitz.open = lambda *a, **k: None
    sys.modules.setdefault("fitz", fitz)


_install_stubs()

import app  # noqa: E402


def find(label, text):
    """Match del detector `label` sul testo, gia' filtrati dal loro validatore."""
    out = []
    for lab, rx, validator, strict in cyber.DETECTORS:
        if lab != label:
            continue
        for m in rx.finditer(text):
            ok = validator(m.group(0)) if validator else True
            if validator and strict and not ok:
                continue
            out.append(m.group(0))
    return out


class TestIp(unittest.TestCase):

    def test_ipv4_cidr_and_defanged_forms(self):
        self.assertEqual(find("IP", "il C2 e' 203.0.113.42"), ["203.0.113.42"])
        self.assertEqual(find("IP", "range 10.0.0.0/8 interno"), ["10.0.0.0/8"])
        self.assertEqual(find("IP", "defangato 198[.]51[.]100[.]7 nel report"),
                         ["198[.]51[.]100[.]7"])
        self.assertEqual(find("IP", "oppure 198(.)51(.)100(.)7 qui"), ["198(.)51(.)100(.)7"])

    def test_ip_at_the_end_of_a_sentence_is_detected(self):
        # regressione: con un lookahead (?![\w.]) il punto finale faceva perdere
        # l'indirizzo -> falso negativo, cioe' un leak
        self.assertEqual(find("IP", "Il server e' 192.168.1.10."), ["192.168.1.10"])

    def test_ipv6_is_detected_and_lookalikes_are_rejected(self):
        self.assertEqual(find("IP", "host 2001:db8::1 raggiungibile"), ["2001:db8::1"])
        self.assertEqual(find("IP", "alle ore 15:30:45 esatte"), [])
        self.assertEqual(find("IP", "scheda 00:1A:2B:3C:4D:5E"), [])   # e' un MAC

    def test_invalid_octets_and_version_numbers_are_rejected(self):
        self.assertEqual(find("IP", "valore 999.1.1.1 non valido"), [])
        self.assertEqual(find("IP", "aggiornato a v1.2.3.4 stabile"), [])
        self.assertEqual(find("IP", "build 1.2.3.4.5 interna"), [])


class TestNetworkAndFiles(unittest.TestCase):

    def test_mac_both_notations(self):
        self.assertEqual(find("MAC", "MAC 00:1A:2B:3C:4D:5E"), ["00:1A:2B:3C:4D:5E"])
        self.assertEqual(find("MAC", "MAC 001a.2b3c.4d5e cisco"), ["001a.2b3c.4d5e"])

    def test_url_keeps_the_whole_defanged_address(self):
        text = "payload da hxxps://consegna(.)example/.github/rel.zip e basta"
        self.assertEqual(find("URL", text), ["hxxps://consegna(.)example/.github/rel.zip"])

    def test_url_does_not_swallow_the_closing_punctuation(self):
        self.assertEqual(find("URL", "vedi (https://acme.com/a) e poi"),
                         ["https://acme.com/a"])
        self.assertEqual(find("URL", "vedi https://acme.com/a."), ["https://acme.com/a"])

    def test_domain_ignores_email_hosts_and_file_extensions(self):
        self.assertEqual(find("DOMAIN", "host dc01.corp.local nel dominio"),
                         ["dc01.corp.local"])
        self.assertEqual(find("DOMAIN", "scrivi a mario.rossi@studiolegale.it"), [])
        self.assertEqual(find("DOMAIN", "in allegato relazione.zip firmata"), [])

    def test_hash_lengths_and_digit_only_rejection(self):
        sha256 = "b" * 63 + "a"
        self.assertEqual(find("HASH", f"campione {sha256} noto"), [sha256])
        self.assertEqual(find("HASH", "codice " + "1" * 64 + " interno"), [])
        self.assertEqual(find("HASH", "troncato " + "a" * 63 + " qui"), [])


class TestIdentifiers(unittest.TestCase):

    def test_bitcoin_address_needs_a_valid_checksum(self):
        self.assertEqual(find("WALLET", "wallet 1BvBMSEYstWetqTFn5Au4m4GFg7xJaNVN2 attivo"),
                         ["1BvBMSEYstWetqTFn5Au4m4GFg7xJaNVN2"])
        self.assertEqual(find("WALLET", "wallet 1BvBMSEYstWetqTFn5Au4m4GFg7xJaNVN3 attivo"), [])

    def test_ethereum_address_by_form(self):
        addr = "0xa1b2c3d4e5f60718a1b2c3d4e5f60718a1b2c3d4"
        self.assertEqual(find("WALLET", f"deployer {addr} on-chain"), [addr])

    def test_cloud_identifiers(self):
        self.assertEqual(find("CLOUDID", "bucket s3://acme-backups/db.sql, esposto"),
                         ["s3://acme-backups/db.sql"])
        self.assertEqual(find("CLOUDID", "ruolo arn:aws:iam::123456789012:role/admin."),
                         ["arn:aws:iam::123456789012:role/admin"])
        self.assertEqual(find("CLOUDID", "istanza i-0abc1234def567890 spenta"),
                         ["i-0abc1234def567890"])

    def test_paths_that_carry_an_identity(self):
        self.assertEqual(find("PATH", r"profilo C:\Users\m.rossi\Desktop\nota.docx,"),
                         [r"C:\Users\m.rossi\Desktop\nota.docx"])
        self.assertEqual(find("PATH", r"share \\FS01\condivisa, montata"),
                         [r"\\FS01\condivisa"])
        self.assertEqual(find("PATH", "chiave /home/mrossi/.ssh/id_rsa."),
                         ["/home/mrossi/.ssh/id_rsa"])
        self.assertEqual(find("PATH", "file /etc/passwd letto"), [])   # non identifica nessuno

    def test_accounts_and_asn(self):
        self.assertEqual(find("USER", r"accesso con ACME\m.rossi alle"), [r"ACME\m.rossi"])
        self.assertEqual(find("USER", "utenza svc_backup disabilitata"), ["svc_backup"])
        self.assertEqual(find("ASN", "annuncia il prefisso AS64496 dal 2019"), ["AS64496"])
        self.assertEqual(find("ASN", "sigla AS0 inesistente"), [])


class TestKeepList(unittest.TestCase):

    def matches(self, text):
        return [m.group(0) for rx in cyber.KEEP_PATTERNS for m in rx.finditer(text)]

    def test_public_references_are_recognized(self):
        for ref in ("CVE-2024-3094", "CWE-506", "CAPEC-137", "RFC 7231", "T1059.001", "TA0002"):
            self.assertIn(ref, self.matches(f"vedi {ref} nel dettaglio"), ref)

    def test_generic_ids_are_not_protected(self):
        # una keeplist troppo larga non produce una parola illeggibile: produce un dato
        # sensibile lasciato in chiaro. Gli id ATT&CK di software/gruppi (S0154, G0016)
        # sono forme troppo generiche per essere protette.
        self.assertEqual(self.matches("codice S0154 e gruppo G0016"), [])


class TestIntegration(unittest.TestCase):
    """analyze() con il pacchetto attivo: composizione, keeplist e merge reali."""

    REPORT = (
        "Il beacon dell'host C:\\Users\\m.rossi\\AppData\\Local\\svc.exe contatta "
        "203.0.113.42 e il dominio consegna.example, sfruttando CVE-2024-3094 "
        "(tecnica T1059.001). Campione a2b3c4d5e6f70819a2b3c4d5e6f70819. "
        "Scrivere a soc@acme.com."
    )

    def setUp(self):
        app.enable_packs(["cyber"])

    def tearDown(self):
        app.enable_packs([])

    def test_pack_is_opt_in(self):
        app.enable_packs([])
        out = app.analyze("Il C2 e' 203.0.113.42 e il dominio consegna.example.")
        self.assertIn("203.0.113.42", out["anonymized_text"])      # core: nessun detector IP
        self.assertEqual(out["n_entities"], 0)

    def test_technical_identifiers_are_masked(self):
        out = app.analyze(self.REPORT)
        for secret in ("203.0.113.42", "consegna.example", "m.rossi",
                       "a2b3c4d5e6f70819a2b3c4d5e6f70819", "soc@acme.com"):
            self.assertNotIn(secret, out["anonymized_text"], secret)
        self.assertLessEqual({"IP", "DOMAIN", "PATH", "HASH", "EMAIL"}, set(out["by_label"]))

    def test_legal_prose_gets_no_cyber_false_positives(self):
        # il vincolo politico della PR: acceso il pacchetto, un atto resta un atto
        legal = ("Vista la sentenza n. 1234/2024 del Tribunale di Roma, R.G. 5678/2023, "
                 "il sottoscritto C.F. RSSMRA85H12F205Y versa € 12.500,00 sull'IBAN "
                 "IT60X0542811101000000123456 entro il 12/06/2025, tel. 06 5551234.")
        labels = {s["label"] for s in app.analyze(legal)["segments"] if s.get("label")}
        self.assertEqual(labels & {"IP", "DOMAIN", "URL", "PATH", "USER", "HASH",
                                   "MAC", "WALLET", "CLOUDID", "ASN"}, set())

    def test_public_references_survive(self):
        out = app.analyze(self.REPORT)
        self.assertIn("CVE-2024-3094", out["anonymized_text"])
        self.assertIn("T1059.001", out["anonymized_text"])

    def test_url_wins_over_the_domain_it_contains(self):
        out = app.analyze("Payload da https://consegna.example/rel.zip scaricato.")
        labels = [s["label"] for s in out["segments"] if s.get("label")]
        self.assertEqual(labels, ["URL"])
        self.assertEqual(out["mapping"]["[URL_1]"], "https://consegna.example/rel.zip")

    def test_email_is_not_broken_into_a_domain(self):
        out = app.analyze("Contatto: soc@acme.com per la risposta.")
        self.assertEqual(out["mapping"], {"[EMAIL_1]": "soc@acme.com"})

    def test_restore_is_exact_for_masked_technical_values(self):
        out = app.analyze(self.REPORT)
        restored = out["anonymized_text"]
        for placeholder, value in sorted(out["mapping"].items(), key=lambda kv: -len(kv[0])):
            restored = restored.replace(placeholder, value)
        self.assertEqual(restored, self.REPORT)


if __name__ == "__main__":
    unittest.main()
