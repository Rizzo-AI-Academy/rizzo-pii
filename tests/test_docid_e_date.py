# -*- coding: utf-8 -*-
"""DOCID e DATE: le due voci della rete regex senza checksum.

Nessuna delle due si puo' validare (il numero di un atto e una data non hanno
cifra di controllo), quindi l'unico argine ai falsi positivi e' il contesto:

  - DOCID pretende la SIGLA davanti al numero, e la include nello span. Senza,
    "1234/2024" non si distingue da una frazione o da un articolo di legge.
  - DATE controlla che giorno e mese esistano e che l'anno sia 19xx/20xx, ed e'
    in SOFT_REGEX_LABELS: e' un'ipotesi che il modello puo' sovrascrivere, non
    un verdetto come lo sono IBAN o CF col checksum in mano.

Tutti i valori sono SINTETICI.
"""

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "app"))

import detectors  # noqa: E402


def trovati(text, label):
    return [text[e["start"]:e["end"]]
            for e in detectors.detect_regex(text) if e["label"] == label]


class Docid(unittest.TestCase):

    # (testo, valore atteso). Il "n." fra sigla e numero e' opzionale su tutte
    # le sigle: negli atti "Prot. n. 456/2024" e' piu' comune di "Prot. 456/2024".
    RICONOSCIUTI = [
        ("Nel proc. R.G. 1234/2024 il giudice ha disposto.", "R.G. 1234/2024"),
        ("Iscritto al R.G.N.R. 123/2023 della Procura.", "R.G.N.R. 123/2023"),
        ("Prot. 456/2024 del Comune.", "Prot. 456/2024"),
        ("Prot. n. 456/2024 del Comune.", "Prot. n. 456/2024"),
        ("Vedi protocollo num. 789 in atti.", "protocollo num. 789"),
        ("Rep. n. 45 del notaio incaricato.", "Rep. n. 45"),
        ("Atto a repertorio 8891/2022 registrato.", "repertorio 8891/2022"),
    ]

    # numeri che NON sono codici di atto: senza la sigla davanti non si redigono
    IGNORATI = [
        "Il termine di 30 giorni decorre dalla notifica.",
        "Art. 1234 del codice civile.",
        "La quota di 2/3 spetta all'erede.",
        "Nato nella Rep. Ceca nel 1980.",       # "Rep." non seguito da numero
    ]

    def test_riconosce_le_sigle_degli_atti(self):
        for testo, atteso in self.RICONOSCIUTI:
            with self.subTest(atteso):
                self.assertEqual(trovati(testo, "DOCID"), [atteso])

    def test_il_numero_nudo_non_basta(self):
        for testo in self.IGNORATI:
            with self.subTest(testo[:30]):
                self.assertEqual(trovati(testo, "DOCID"), [])


class Date(unittest.TestCase):

    RICONOSCIUTE = [
        ("Udienza fissata al 12/06/2025 in aula 3.", "12/06/2025"),
        ("Fattura del 3.7.1999 saldata.", "3.7.1999"),
        ("Contratto stipulato il 01-12-2020.", "01-12-2020"),
        ("Deposito 28/02/2024 09:15 allo sportello.", "28/02/2024 09:15"),
    ]

    # giorni, mesi e anni impossibili: la forma da sola ingannerebbe
    IGNORATE = [
        "Rif. 45/13/2025 interno.",          # giorno 45, mese 13
        "Codice 12/06/1899 di archivio.",    # anno fuori da 19xx/20xx
        "Scala 1/2/3 del disegno.",
    ]

    def test_riconosce_le_date_plausibili(self):
        for testo, atteso in self.RICONOSCIUTE:
            with self.subTest(atteso):
                self.assertEqual(trovati(testo, "DATE"), [atteso])

    def test_scarta_giorni_mesi_e_anni_impossibili(self):
        for testo in self.IGNORATE:
            with self.subTest(testo[:30]):
                self.assertEqual(trovati(testo, "DATE"), [])

    def test_date_e_soft(self):
        """DATE non deve avere la priorita' della rete regex sul modello."""
        self.assertIn("DATE", detectors.SOFT_REGEX_LABELS)


if __name__ == "__main__":
    unittest.main()
