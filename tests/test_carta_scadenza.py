# -*- coding: utf-8 -*-
"""Carta con la scadenza attaccata — `detectors._card_senza_scadenza()`.

Il modo normale di scrivere una carta è numero **e scadenza di seguito**: il detector è
avido, ingloba il mese, il Luhn fallisce sulla sequenza allungata e `strict` scarta tutto,
lasciando il PAN in chiaro. Quando il Luhn fallisce si riprova tagliando la coda.

I due modi di sbagliare del taglio, ed è il secondo quello grave:

  - tagliare dove non si doveva  -> nasce un falso positivo su un numero lungo qualsiasi;
  - tagliare troppo corto        -> esce un `[CREDITCARDNUMBER_1]` che copre solo una parte
                                    del PAN e le cifre restanti sono leggibili accanto al
                                    segnaposto. Peggio che non mascherare, perché l'utente
                                    crede di aver finito.

Tutti i numeri sono SINTETICI, con il Luhn calcolato apposta. `4111 1111 1111 1111` è il
PAN di test riservato già usato dal resto del repo.
"""

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "app"))

import detectors  # noqa: E402

CARTA = "4111 1111 1111 1111"          # PAN di test riservato
PAN17 = "60112222333344447"            # 17 cifre, Luhn valido, sintetico
# 16 cifre col Luhn valido, ma è un numero di pratica. Scelto perché con le code qui sotto
# il numero INTERO non è a sua volta Luhn-valido: solo così il taglio viene davvero tentato
# ed è la guardia a doverlo respingere. (Una coda di due zeri conserva sempre il Luhn, per
# quello non compare: sarebbe un caso irraggiungibile, non un test.)
NON_CARTA = "9010000000000009"


def carte(testo):
    return [testo[e["start"]:e["end"]] for e in detectors.detect_regex(testo)
            if e["label"] == "CREDITCARDNUMBER"]


class CartaConScadenza(unittest.TestCase):
    def test_scadenza_attaccata_il_pan_esce_intero(self):
        for coda in ("12/26", "12 26", "1226", "03/28"):
            for pan in (CARTA, CARTA.replace(" ", "-"), CARTA.replace(" ", "")):
                testo = "Pagamento con carta %s %s." % (pan, coda)
                self.assertEqual(carte(testo), [pan], testo)

    def test_senza_scadenza_niente_cambia(self):
        for pan in (CARTA, CARTA.replace(" ", "-"), CARTA.replace(" ", "")):
            testo = "Carta %s intestata al ricorrente." % pan
            self.assertEqual(carte(testo), [pan], testo)

    def test_il_pan_non_viene_tagliato_a_meta(self):
        """17 cifre: se la lista delle lunghezze si ferma a 16, l'ultima resta in chiaro."""
        testo = "Carta %s 12/26 intestata." % PAN17
        self.assertEqual(carte(testo), [PAN17], testo)

    def test_una_cifra_sola_non_e_una_scadenza(self):
        """Coda di una cifra: senza il vincolo passa 9 volte su 10 e nasce un falso positivo."""
        testo = "Rif. pratica %s 9 del fascicolo." % NON_CARTA
        self.assertEqual(carte(testo), [], testo)

    def test_mese_fuori_scala(self):
        for coda in ("13", "99"):
            testo = "Rif. %s %s del fascicolo." % (NON_CARTA, coda)
            self.assertEqual(carte(testo), [], testo)


if __name__ == "__main__":
    unittest.main()
