# -*- coding: utf-8 -*-
"""Regressione issue #54: un paste da Excel/TSV non deve sfasare le celle.

Il modello etichetta i sotto-token e a volte ne copre solo una parte: dentro
`27/07/2026` marca `7/07/20`, dentro `VERDI` marca `DI`, dentro `1362` marca `6`.
Sostituire il frammento cosi' com'e' produce `2[DATE_1]`, `VER[FULLNAME_1]`,
`13[BUILDINGNUM_1]2`: pezzi di PII ancora in chiaro nella cella, e - dato che il
placeholder e' piu' lungo dell'originale - colonne che non tornano piu'.

Il rimedio e' l'allineamento ai confini di parola dentro `_merge` (#35): una span
che taglia una parola a meta' viene estesa fino a coprirla, e le span che cosi' si
sovrappongono vengono fuse. Qui si verifica l'EFFETTO su tutto il documento -
colonne allineate, celle non-PII intatte, nessun placeholder attaccato a un
residuo - non la funzione che lo ottiene.

Il fixture e' quello dell'issue, con i frammenti iniettati a mano: cosi' il test
non dipende da cosa il modello predice oggi.

Nomi, codici fiscali e date sono SINTETICI.

Questo modulo importa `app.py`, che tira dentro torch/transformers/flask/fitz:
se mancano, il test si salta invece di fallire. La CI non li installa apposta
(minuti e gigabyte per zero copertura sull'inferenza), quindi li' non gira; in
locale, con l'ambiente dell'app, si'.
"""

import re
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "app"))

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

# Colonne senza PII: devono uscire identiche a come sono entrate.
NON_PII_COLS = {
    1: "Ditta",
    2: "Ragione sociale",
    4: "Codice rilevazione presenze",
    9: "Categoria lavoratore",
}


def _carica_app():
    """Importa app col pipeline HF finto: qui i pesi non servono."""
    if "app" in sys.modules:
        return sys.modules["app"]
    with patch("transformers.pipeline", return_value=MagicMock(return_value=[])):
        import app as pii_app
    return pii_app


try:                                    # torch/transformers/flask/fitz presenti?
    APP = _carica_app()
except Exception:                       # noqa: BLE001 - qualunque import mancante
    APP = None


def _frammenti_del_modello(text):
    """I frammenti tipici del bug #54, agli offset reali del TSV dell'issue."""
    ents = []

    def add(label, start, end, score=0.95):
        ents.append({"label": label, "start": start, "end": end,
                     "score": score, "validated": False, "source": "modello"})

    for m in re.finditer(r"27/07/2026", text):
        add("DATE", m.start(), m.start() + 1, 1.0)          # '2'
        add("DATE", m.start() + 1, m.start() + 8, 0.95)     # '7/07/20'
    for pattern in (r"01/03/2013", r"14/10/2024"):
        for m in re.finditer(pattern, text):
            add("DATE", m.start(), m.start() + 1, 1.0)
            add("DATE", m.start() + 1, m.start() + 8, 0.99)
    for m in re.finditer(r"01/02/2019", text):
        add("DATE", m.start(), m.start() + 1, 1.0)
        add("DATE", m.start() + 1, m.start() + 8, 0.99)
        add("DATE", m.end() - 1, m.end(), 0.6)              # '9' residuo
    for m in re.finditer(r"\bVERDI\b", text):
        add("FULLNAME", m.start() + 3, m.end(), 0.52)       # 'DI' dentro VERDI
    for pattern in (r"\bPAOLO\b", r"\bGIACOMO\b"):
        for m in re.finditer(pattern, text):
            add("FULLNAME", m.start(), m.end(), 0.99)
    for m in re.finditer(r"\b1362\b", text):
        add("BUILDINGNUM", m.start() + 2, m.start() + 3, 0.93)   # '6' dentro 1362
    for m in re.finditer(r"\b1444\b", text):
        add("BUILDINGNUM", m.start() + 2, m.start() + 3, 0.63)   # '4' dentro 1444
    return ents


@unittest.skipIf(APP is None, "serve l'ambiente dell'app (torch/transformers/flask/fitz)")
class AllineamentoTsv(unittest.TestCase):
    """Con le entita' frammentate come nel bug, il documento resta allineato."""

    def setUp(self):
        self.frammenti = _frammenti_del_modello(ISSUE_TSV)
        # il fixture deve davvero contenere frammenti a meta' parola, altrimenti
        # il test passerebbe per il motivo sbagliato
        self.assertTrue(any(ISSUE_TSV[e["start"]:e["end"]] == "DI"
                            for e in self.frammenti))

    def _anonimizza(self):
        with patch.object(APP, "detect_model", return_value=(self.frammenti, 1)):
            return APP.analyze(ISSUE_TSV)["anonymized_text"]

    def test_colonne_e_celle_non_pii_restano_intatte(self):
        anonimizzato = self._anonimizza()
        righe_orig = [r for r in ISSUE_TSV.splitlines() if r.strip()]
        righe_anon = [r for r in anonimizzato.splitlines() if r.strip()]
        self.assertEqual(len(righe_orig), len(righe_anon))

        for i, (o, a) in enumerate(zip(righe_orig, righe_anon)):
            celle_o, celle_a = o.split("\t"), a.split("\t")
            with self.subTest(riga=i):
                self.assertEqual(len(celle_o), len(celle_a),
                                 "il numero di colonne non deve cambiare")
                if i == 0:
                    continue            # intestazione
                for idx, nome in NON_PII_COLS.items():
                    self.assertEqual(celle_o[idx], celle_a[idx],
                                     "colonna %s alterata" % nome)

    def test_nessun_placeholder_attaccato_a_un_residuo(self):
        """`2[DATE_1]` o `VER[FULLNAME_1]` = meta' PII ancora leggibile."""
        anonimizzato = self._anonimizza()
        for riga in anonimizzato.splitlines():
            for cella in riga.split("\t"):
                with self.subTest(cella=cella):
                    self.assertIsNone(re.search(r"[A-Za-z0-9]\[[A-Z_]", cella))
                    self.assertIsNone(re.search(r"\][A-Za-z0-9]", cella))


if __name__ == "__main__":
    unittest.main()
