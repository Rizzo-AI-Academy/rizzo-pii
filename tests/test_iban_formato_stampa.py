# -*- coding: utf-8 -*-
"""IBAN nel formato di STAMPA (a gruppi) — `detectors.detect_iban()`.

L'IBAN e' l'unico identificativo della rete che non si riconosce con una regex:
il confine destro non e' deducibile dalla forma, perche' i caratteri dopo la
sigla del paese sono alfanumerici come le parole intorno. Da qui i due modi di
sbagliare, ed e' il secondo quello grave:

  - inghiottire la parola vicina    -> il mod-97 fallisce e l'IBAN resta in chiaro;
  - fermarsi a meta'                -> esce un `[IBAN_1]` che ne copre solo un pezzo,
                                       e la coda resta leggibile nel documento
                                       "anonimizzato". Peggio di non mascherare,
                                       perche' l'utente crede di aver finito.

Percio' la regola sotto e' binaria: **o esce per intero, o non esce**. Ogni caso
qui e' un IBAN come lo stampano fatture, carte intestate, home banking ed export.

Tutti i valori sono SINTETICI, con mod-97 calcolato apposta.
"""

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "app"))

import detectors  # noqa: E402

IB = "IT60X0542811101000000123456"


def gruppi(k, sep=" "):
    """L'IBAN spezzato in gruppi da k, come lo si stampa."""
    return sep.join(IB[i:i + k] for i in range(0, len(IB), k))


# a fine riga il PDF spezza: il testo estratto ha l'a-capo dentro l'IBAN
A_CAPO = "\n".join(gruppi(4).rsplit(" ", 1))


def iban_span(text):
    """Il pezzo di testo che verrebbe mascherato come IBAN, o None."""
    ents = [e for e in detectors.detect_regex(text) if e["label"] == "IBAN"]
    if not ents:
        return None
    e = max(ents, key=lambda x: x["end"] - x["start"])
    return text[e["start"]:e["end"]]


# (nome, valore che deve essere rilevato per INTERO)
INTERI = [
    ("compatto", IB),
    ("minuscolo", IB.lower()),
    ("gruppi_di_4", gruppi(4)),
    ("gruppi_di_4_minuscolo", gruppi(4).lower()),
    ("separato_da_trattini", gruppi(4, "-")),
    ("spazio_non_breaking", gruppi(4, " ")),
    ("spazi_multipli", gruppi(4, "   ")),          # PDF giustificato
    ("gruppi_di_5", gruppi(5)),
    ("spezzato_a_fine_riga", A_CAPO),
    ("separatori_misti", "IT60-X054 2811-1010.0000 0123 456"),
    # il registro ISO delle lunghezze non copre tutti i paesi: la forma compatta
    # deve continuare a funzionare anche per quelli che non ci sono
    ("paese_fuori_registro_iso", "MA64011519000001205000534921"),
]

# (nome, testo che NON deve produrre un IBAN)
NESSUNO = [
    ("checksum_sbagliato",
     "Codice IT99 X999 9999 9999 9999 9999 999 da verificare."),
    ("data_in_prosa",
     "Come da mandato il 17 luglio 2008 conferito a Mario Rossi."),
    # l'alfabeto IBAN e' ASCII: senza il vincolo, "eta'"/"piu'" entrano nel match
    ("parole_accentate",
     "Prot. LV94 età più attivita del 2025."),
    ("sigla_e_numero_in_prosa",
     "Il SA 67 ha deliberato in data 21/12/2008 la ratifica."),
    ("gruppi_di_lunghezza_disuguale",
     "Prot. IT60 X0542 811 1010 0000 0123 456 in atti."),
    # una colonna di tabella e' fatta di celle, non di un IBAN andato a capo 5 volte
    ("colonna_di_tabella",
     "Codici\nAT29\n2914\n1777\n6317\n0669\n"),
    # il prefisso fin qui passa il mod-97, ma e' mezzo IBAN: mascherarlo lascerebbe
    # la coda in chiaro sotto il placeholder
    ("meta_iban_che_passa_il_mod97",
     "Codice IT96 3B8J I6WA Q9YI    KB8S ASPB 8JO in atti."),
    ("codice_interno_che_ne_ha_la_forma",
     "Codice commessa GR14 0172 7402 1280 riferimento interno."),
    ("il_codice_prosegue_oltre",
     "Codice %s7890 interno." % IB),
]


class IbanFormatoDiStampa(unittest.TestCase):

    def test_esce_per_intero(self):
        for nome, valore in INTERI:
            with self.subTest(nome):
                testo = "Bonifico su %s al ricorrente." % valore
                self.assertEqual(iban_span(testo), valore)

    def test_non_esce_affatto(self):
        for nome, testo in NESSUNO:
            with self.subTest(nome):
                self.assertIsNone(iban_span(testo))

    def test_un_codice_vicino_non_tronca_liban(self):
        """Un altro codice accanto non deve far finire l'IBAN a meta'.

        E' il caso realistico: la fattura ha la partita IVA due parole prima
        dell'IBAN. Se la scansione parte dal codice sbagliato ne esce un
        `[IBAN_1]` con la coda ancora leggibile.
        """
        # un gruppo interno di questo IBAN ha la forma di una sigla di paese
        HU = "HU26 ED40 LPFN GL54 NKH1 X8UD SV92"
        casi = [
            ("P. IVA IT77781880495 bonifico su %s entro 30 giorni." % gruppi(4),
             gruppi(4)),
            ("Si dispone il pagamento sul conto AL56 %s intestato." % A_CAPO,
             A_CAPO),
            ("Bonifico su %s al ricorrente." % HU, HU),
        ]
        for testo, atteso in casi:
            with self.subTest(atteso[:12]):
                self.assertIn(atteso, iban_span(testo) or "")


if __name__ == "__main__":
    unittest.main()
