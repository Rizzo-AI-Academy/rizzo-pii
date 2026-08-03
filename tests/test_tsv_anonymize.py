# -*- coding: utf-8 -*-
"""Regressione issue #54: paste Excel/TSV multi-riga non deve sfasare le celle.

Causa reale: il modello spezza le entita' a meta' token (DATE '7/07/20' dentro
'27/07/2026', FULLNAME 'DI' dentro 'VERDI'). La sostituzione era gia' a singolo
pass sugli offset: senza snap ai confini \\S+ i frammenti lasciavano pezzi in
chiaro nella stessa cella.
"""
import os
import re
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[1]
APP_DIR = ROOT / "src" / "app"
sys.path.insert(0, str(APP_DIR))

# Esempio esatto da https://github.com/Rizzo-AI-Academy/rizzo-pii/issues/54
ISSUE_TSV = (
    "Estrazione al\tDitta\tRagione sociale\tLavoratore\t"
    "Codice rilevazione presenze\tCognome\tNome\tSesso\tCodice fiscale\t"
    "Categoria lavoratore\tData assunzione\tData fine rapporto\n"
    "27/07/2026\t345\tAZIENDA SRL\t542\t8\tROSSI\tGIACOMO\tMaschio\t"
    "JLEKQE43E50K215N\tDipendente\t01/03/2013\t\n"
    "27/07/2026\t345\tAZIENDA SRL\t773\t35\tVERDI\tPAOLO\tMaschio\t"
    "JLEKQE43E50K415N\tDipendente\t14/10/2024\t\n"
    "27/07/2026\t345\tAZIENDA SRL\t783\t13\tROSSI\tSIMONE\tMaschio\t"
    "JLEKQE43E50K615N\tCollaboratore\t01/02/2019\t\n"
    "27/07/2026\t345\tAZIENDA SRL\t783\t14\tVERDI\tALBERTO\tMaschio\t"
    "JLEKQE43E50K415N\tTitolare/socio\t01/02/2019\t\n"
    "27/07/2026\t345\tAZIENDA SRL\t821\t12\tROSSI\tFRANCESCO\tMaschio\t"
    "JLEKQE43E50K715N\tDipendente\t31/08/2010\t\n"
    "27/07/2026\t345\tAZIENDA SRL\t1362\t23\tVERDI\tFLORIENZO\tMaschio\t"
    "JLEKQE43E50K915N\tDipendente\t05/09/2022\t\n"
    "27/07/2026\t345\tAZIENDA SRL\t1444\t20\tROSSI\tBULBASAUR\tMaschio\t"
    "JLEKQE43E50K715N\tCollaboratore\t01/02/2019\t\n"
    "27/07/2026\t345\tAZIENDA SRL\t1444\t21\tVERDI\tFABRIZIO\tMaschio\t"
    "JLEKQE43E50K235N\tTitolare/socio\t01/02/2019\t\n"
    "27/07/2026\t345\tAZIENDA SRL\t1458\t1\tROSSI\tPICKACKU\tMaschio\t"
    "JLEKQE43E50K285N\tDipendente\t02/08/2016\t\n"
    "27/07/2026\t345\tAZIENDA SRL\t1703\t37\tVERDI\tSTEVANZIO\tMaschio\t"
    "JLEKQE43E50K215N\tDipendente\t09/12/2024\t\n"
)

# Colonne non-PII che devono restare identiche (indice 0-based sulla riga dati).
NON_PII_COLS = {
    1: "Ditta",
    2: "Ragione sociale",
    4: "Codice rilevazione presenze",
    9: "Categoria lavoratore",
}


def _load_app():
    """Importa app mockando il pipeline HF (i test unitari non servono i pesi)."""
    if "app" in sys.modules:
        return sys.modules["app"]
    mock_nlp = MagicMock(return_value=[])
    with patch("transformers.pipeline", return_value=mock_nlp):
        import app as pii_app  # noqa: WPS433
    return pii_app


def _fragmented_model_ents(text):
    """Riproduce i frammenti tipici del bug #54 (offset reali sul TSV dell'issue)."""
    ents = []

    def add(label, start, end, score=0.95):
        ents.append({
            "label": label, "start": start, "end": end,
            "score": score, "validated": False, "source": "modello",
        })

    # Per ogni occorrenza di date/cognomi/codici: span parziali come dal modello.
    for m in re.finditer(r"27/07/2026", text):
        add("DATE", m.start(), m.start() + 1, 1.0)          # '2'
        add("DATE", m.start() + 1, m.start() + 8, 0.95)     # '7/07/20'
    for m in re.finditer(r"01/03/2013", text):
        add("DATE", m.start(), m.start() + 1, 1.0)
        add("DATE", m.start() + 1, m.start() + 8, 0.99)
    for m in re.finditer(r"14/10/2024", text):
        add("DATE", m.start(), m.start() + 1, 1.0)
        add("DATE", m.start() + 1, m.start() + 8, 0.99)
    for m in re.finditer(r"01/02/2019", text):
        add("DATE", m.start(), m.start() + 1, 1.0)
        add("DATE", m.start() + 1, m.start() + 8, 0.99)
        add("DATE", m.end() - 1, m.end(), 0.6)              # '9' residuo
    for m in re.finditer(r"\bVERDI\b", text):
        add("FULLNAME", m.start() + 3, m.end(), 0.52)       # 'DI'
    for m in re.finditer(r"\bPAOLO\b", text):
        add("FULLNAME", m.start(), m.end(), 0.99)
    for m in re.finditer(r"\bGIACOMO\b", text):
        add("FULLNAME", m.start(), m.end(), 0.99)
    for m in re.finditer(r"\b1362\b", text):
        add("BUILDINGNUM", m.start() + 2, m.start() + 3, 0.93)  # '6'
    for m in re.finditer(r"\b1444\b", text):
        add("BUILDINGNUM", m.start() + 2, m.start() + 3, 0.63)  # '4'
    return ents


def _assert_tsv_alignment(original, anonymized):
    orig_rows = [r for r in original.splitlines() if r.strip()]
    anon_rows = [r for r in anonymized.splitlines() if r.strip()]
    assert len(orig_rows) == len(anon_rows), (len(orig_rows), len(anon_rows))
    for i, (o, a) in enumerate(zip(orig_rows, anon_rows)):
        oc, ac = o.split("\t"), a.split("\t")
        assert len(oc) == len(ac), f"riga {i}: {len(oc)} col -> {len(ac)}: {a!r}"
        if i == 0:
            continue
        for idx in NON_PII_COLS:
            assert oc[idx] == ac[idx], (
                f"riga {i} col {NON_PII_COLS[idx]}: {oc[idx]!r} -> {ac[idx]!r}"
            )
        for cell in ac:
            assert not re.search(r"[A-Za-z0-9]\[[A-Z_]", cell), cell
            assert not re.search(r"\][A-Za-z0-9]", cell), cell


class SnapToTokenSpansTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = _load_app()

    def test_expands_partial_date_and_fullname_fragments(self):
        text = "27/07/2026\tVERDI\t1362"
        ents = [
            {"label": "DATE", "start": 0, "end": 1, "score": 1.0,
             "validated": False, "source": "modello"},
            {"label": "DATE", "start": 1, "end": 8, "score": 0.9,
             "validated": False, "source": "modello"},
            {"label": "FULLNAME", "start": 14, "end": 16, "score": 0.5,
             "validated": False, "source": "modello"},  # 'DI' in VERDI
            {"label": "BUILDINGNUM", "start": 19, "end": 20, "score": 0.9,
             "validated": False, "source": "modello"},  # '6' in 1362
        ]
        # Verifica indici rispetto al testo reale
        self.assertEqual(text[14:16], "DI")
        self.assertEqual(text[19:20], "6")
        snapped = self.app._snap_to_token_spans(ents, text)
        self.assertEqual(text[snapped[0]["start"]:snapped[0]["end"]], "27/07/2026")
        self.assertEqual(text[snapped[1]["start"]:snapped[1]["end"]], "27/07/2026")
        self.assertEqual(text[snapped[2]["start"]:snapped[2]["end"]], "VERDI")
        self.assertEqual(text[snapped[3]["start"]:snapped[3]["end"]], "1362")

    def test_preserves_multiword_entity(self):
        text = "residente in Via Roma 10"
        # "Via Roma" come STREET (due token)
        start, end = text.index("Via"), text.index("Roma") + 4
        ents = [{"label": "STREET", "start": start, "end": end, "score": 0.9,
                 "validated": False, "source": "modello"}]
        snapped = self.app._snap_to_token_spans(ents, text)
        self.assertEqual(text[snapped[0]["start"]:snapped[0]["end"]], "Via Roma")


class Issue54TsvAlignmentTests(unittest.TestCase):
    """Con entita' frammentate come nel bug, dopo snap+merge l'allineamento regge."""

    @classmethod
    def setUpClass(cls):
        cls.app = _load_app()

    def test_fragmented_spans_preserve_columns_and_non_pii(self):
        text = ISSUE_TSV
        fragments = _fragmented_model_ents(text)

        with patch.object(self.app, "detect_model", return_value=(fragments, 1)):
            # Prima del fix (senza snap) il testo sarebbe sfasato: lo dimostriamo
            # applicando solo merge sui frammenti grezzi.
            broken = self.app._merge([dict(e) for e in fragments] + self.app.detect_regex(text), text)
            broken_anon = self._rebuild(text, broken)
            self.assertTrue(
                any(re.search(r"[A-Za-z0-9]\[[A-Z_]", cell) or re.search(r"\][A-Za-z0-9]", cell)
                    for row in broken_anon.splitlines() for cell in row.split("\t")),
                "il fixture deve riprodurre lo sfasamento pre-fix",
            )

            res = self.app.analyze(text)
            _assert_tsv_alignment(text, res["anonymized_text"])
            self.assertNotIn("]26", res["anonymized_text"])
            self.assertNotIn("VER[", res["anonymized_text"])
            self.assertNotIn("13[", res["anonymized_text"])

    @staticmethod
    def _rebuild(text, kept):
        """Ricostruzione minimal come analyze(), senza mapping."""
        anon, pos = [], 0
        for e in kept:
            if e["start"] > pos:
                anon.append(text[pos:e["start"]])
            anon.append(f"[{e['label']}_X]")
            pos = e["end"]
        if pos < len(text):
            anon.append(text[pos:])
        return "".join(anon)


@unittest.skipUnless(
    (ROOT / "models" / "rizzo-pii-0.3B-main").is_dir()
    or os.environ.get("PII_MODEL_DIR"),
    "modello PII non presente in locale",
)
class Issue54IntegrationTests(unittest.TestCase):
    """Smoke con il modello reale (se scaricato)."""

    @classmethod
    def setUpClass(cls):
        # Forza reimport con il modello vero se i test precedenti hanno mockato.
        if "app" in sys.modules:
            del sys.modules["app"]
        if not os.environ.get("PII_MODEL_DIR"):
            os.environ["PII_MODEL_DIR"] = str(ROOT / "models" / "rizzo-pii-0.3B-main")
        import app as pii_app  # noqa: WPS433
        cls.app = pii_app

    def test_issue_tsv_with_real_model(self):
        res = self.app.analyze(ISSUE_TSV)
        _assert_tsv_alignment(ISSUE_TSV, res["anonymized_text"])
        self.assertNotIn("VER[", res["anonymized_text"])
        self.assertNotIn("]26", res["anonymized_text"])


class RegexChecksumStillWorksTests(unittest.TestCase):
    """Lo snap non deve rompere CF/IBAN della rete regex."""

    @classmethod
    def setUpClass(cls):
        cls.app = _load_app()

    def test_cf_and_iban_still_detected(self):
        text = ("Il sig. Mario Rossi, C.F. RSSMRA85H12F205Z, IBAN "
                "IT60X0542811101000000123456.")
        with patch.object(self.app, "detect_model", return_value=([], 0)):
            res = self.app.analyze(text)
        self.assertIn("[CF_1]", res["anonymized_text"])
        self.assertIn("[IBAN_1]", res["anonymized_text"])
        self.assertEqual(res["mapping"]["[CF_1]"], "RSSMRA85H12F205Z")
        self.assertIn("IT60", res["mapping"]["[IBAN_1]"])
        # Lo snap non deve mangiare la virgola dopo il CF (solo source=modello).
        self.assertIn("[CF_1],", res["anonymized_text"])

    def test_snap_skips_regex_spans(self):
        text = "CF RSSMRA85H12F205Z, fine"
        cf = text.index("RSSMRA85H12F205Z")
        ents = [{
            "label": "CF", "start": cf, "end": cf + 16,
            "score": 1.0, "validated": True, "source": "regex",
        }]
        snapped = self.app._snap_to_token_spans(ents, text)
        self.assertEqual(text[snapped[0]["start"]:snapped[0]["end"]], "RSSMRA85H12F205Z")


if __name__ == "__main__":
    unittest.main()
