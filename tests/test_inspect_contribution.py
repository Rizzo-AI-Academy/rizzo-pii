# -*- coding: utf-8 -*-
"""L'ispettore delle contribuzioni: respinge i file guasti e misura la varieta'.

Un controllo che accetta tutto e' peggio di nessun controllo, perche' fa credere che la coda
sia stata verificata. Quindi qui, per ogni tipo di errore duro, si costruisce un file che lo
contiene e si pretende che venga contato -- non basta che i file buoni passino.

La parte sulla varieta' verifica la misura controintuitiva su cui si regge lo strumento: lo
SCHELETRO (testo con le entita' sostituite dalla label) sottostima proprio i template migliori,
quelli che etichettano ogni valore iniettato, e per questo la grana usata e' la STRUTTURA.

Tutti i valori sono SINTETICI: i codici fiscali e gli IBAN qui sotto hanno checksum validi ma
non appartengono a nessuno.
"""

import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "inspect"))

import inspect_contribution as ic  # noqa: E402

TOKEN = ic.TOKEN_RE.findall


def riga(testo, entita, template_id=1):
    """Una riga valida: token e BIO ricalcolati dal testo, come fa il formato."""
    tokens = TOKEN(testo)
    bio = ["O"] * len(tokens)
    # posizione di ogni token nel testo, per assegnare le label agli indici giusti
    pos, offsets = 0, []
    for t in tokens:
        i = testo.index(t, pos)
        offsets.append((i, i + len(t)))
        pos = i + len(t)
    for e in entita:
        primo = True
        for k, (a, b) in enumerate(offsets):
            if a >= e["start"] and b <= e["end"]:
                bio[k] = ("B-" if primo else "I-") + e["label"]
                primo = False
    return {"source_text": testo, "language": "it", "template_id": template_id,
            "entities": entita, "tokens": tokens, "bio_labels": bio}


def ent(testo, valore, label):
    i = testo.index(valore)
    return {"value": valore, "label": label, "start": i, "end": i + len(valore)}


# checksum validi e verificati: la lettera di controllo di RSSMRA85M01H501 e' Q, non Z --
# la Z e' l'errore che si trova negli esempi scritti a mano, ed e' il caso invalido qui sotto.
CF_VALIDO = "RSSMRA85M01H501Q"
CF_NON_VALIDO = "RSSMRA85M01H501Z"
IBAN_VALIDO = "IT60X0542811101000000123456"


def buona(nome="Mario", tid=1):
    t = f"Il ricorrente {nome} Rossi, C.F. {CF_VALIDO}, chiede il bonifico su {IBAN_VALIDO}."
    return riga(t, [ent(t, nome, "GIVENNAME"), ent(t, "Rossi", "SURNAME"),
                    ent(t, CF_VALIDO, "CF"), ent(t, IBAN_VALIDO, "IBAN")], tid)


def scrivi(righe):
    f = tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False, encoding="utf-8")
    for r in righe:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
    f.close()
    return f.name


def ispeziona(righe):
    m, errori, _ = ic.ispeziona(scrivi(righe), ic.tassonomia()[0])
    return m, errori


class FileValido(unittest.TestCase):

    def test_nessun_errore_e_numeri_giusti(self):
        m, errori = ispeziona([buona("Mario"), buona("Giulia"), buona("Anna")])
        self.assertEqual(dict(errori), {})
        self.assertEqual(m["errori_duri"], 0)
        self.assertEqual(m["righe"], 3)
        self.assertEqual(m["entita"], 12)
        self.assertEqual(m["testi_duplicati"], 0)

    def test_i_duplicati_veri_si_contano(self):
        m, _ = ispeziona([buona("Mario"), buona("Mario")])
        self.assertEqual(m["testi_duplicati"], 1)

    def test_la_tassonomia_si_legge_dal_repo(self):
        ammesse, fonti = ic.tassonomia()
        self.assertIn("CF", ammesse)          # tag finale
        self.assertIn("GIVENNAME", ammesse)   # label grezza rimappata da TAG_MAP
        self.assertIn("CIG", ammesse)         # forma aggiunta sotto DOCID: il caso della deriva
        self.assertTrue(any("train_pii" in f or "contribute_dataset" in f for f in fonti),
                        "la tassonomia deve venire dal repo, non da una lista scritta a mano")


class ErroriDuri(unittest.TestCase):
    """Ogni errore che rende un file inutilizzabile deve essere CONTATO, non ignorato."""

    def test_offset_incoerente(self):
        r = buona()
        r["entities"][0]["start"] += 3        # la span non copre piu' il valore
        _, errori = ispeziona([r])
        self.assertIn("offset dell'entita' incoerente col testo", errori)

    def test_label_fuori_tassonomia(self):
        r = buona()
        r["entities"][0]["label"] = "CREDITCARD"   # il refuso classico di CREDITCARDNUMBER
        _, errori = ispeziona([r])
        self.assertTrue(any("fuori tassonomia" in k for k in errori))

    def test_checksum_non_valido(self):
        t = f"Codice fiscale {CF_NON_VALIDO} del richiedente."
        r = riga(t, [ent(t, CF_NON_VALIDO, "CF")])        # lettera di controllo sbagliata
        _, errori = ispeziona([r])
        self.assertTrue(any("checksum non valido" in k for k in errori))

    def test_token_non_riproducibili(self):
        r = buona()
        r["tokens"] = r["tokens"][:-1]
        _, errori = ispeziona([r])
        self.assertIn("tokens non riproducibili dalla regex del formato", errori)
        self.assertIn("len(tokens) != len(bio_labels)", errori)

    def test_bio_i_senza_b(self):
        r = buona()
        r["bio_labels"] = ["I-CF"] + ["O"] * (len(r["tokens"]) - 1)
        _, errori = ispeziona([r])
        self.assertIn("I- senza B- dello stesso tag", errori)

    def test_entita_sovrapposte(self):
        t = f"Il ricorrente Mario Rossi paga."
        e1 = ent(t, "Mario Rossi", "FULLNAME")
        e2 = ent(t, "Rossi", "SURNAME")
        _, errori = ispeziona([riga(t, [e1, e2])])
        self.assertIn("entita' sovrapposte", errori)

    def test_lingua_sbagliata(self):
        r = buona()
        r["language"] = "en"
        _, errori = ispeziona([r])
        self.assertTrue(any("language != it" in k for k in errori))

    def test_carattere_non_latino(self):
        t = "Il ricorrente 漢 chiede il pagamento."
        _, errori = ispeziona([riga(t, [])])
        self.assertTrue(any("non latino" in k for k in errori))

    def test_json_non_valido(self):
        f = tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False, encoding="utf-8")
        f.write(json.dumps(buona()) + "\n{non json}\n")
        f.close()
        _, errori, _ = ic.ispeziona(f.name, ic.tassonomia()[0])
        self.assertTrue(any("JSON non valido" in k for k in errori))

    def test_uscita_diversa_da_zero_sui_file_guasti(self):
        """L'ispettore vale da cancello solo se il codice d'uscita lo dice."""
        r = buona()
        r["entities"][0]["label"] = "NONESISTE"
        argv, out = sys.argv, sys.stdout
        try:
            sys.stdout = io.StringIO()          # il rapporto non serve nell'output dei test
            sys.argv = ["inspect_contribution", scrivi([r])]
            guasto = ic.main()
            sys.argv = ["inspect_contribution", scrivi([buona()])]
            valido = ic.main()
        finally:
            sys.argv, sys.stdout = argv, out
        self.assertEqual(guasto, 1)
        self.assertEqual(valido, 0)


class MisuraDellaVarieta(unittest.TestCase):

    def test_struttura_quando_c_e_template_id(self):
        m, _ = ispeziona([buona("Mario", 1), buona("Giulia", 1), buona("Anna", 2)])
        self.assertEqual(m["grana"], "struttura")
        self.assertEqual(m["gruppi_distinti"], 2)      # due template_id
        self.assertEqual(m["righe_per_gruppo_max"], 2)

    def test_scheletro_come_ripiego_senza_template_id(self):
        righe = [buona("Mario"), buona("Giulia")]
        for r in righe:
            del r["template_id"]
        m, _ = ispeziona(righe)
        self.assertEqual(m["grana"], "scheletro")
        self.assertEqual(m["righe_senza_template_id"], 2)

    def test_lo_scheletro_sottostima_i_template_che_etichettano_tutto(self):
        """Il caso misurato che motiva la scelta della struttura.

        Nel template "bolletta" OGNI valore iniettato e' etichettato: righe diverse hanno lo
        STESSO scheletro, e contando gli scheletri sembra un template povero. Nel template
        "atto" il nome del tribunale NON e' etichettato: ogni valore diverso crea uno scheletro
        nuovo, e l'atto sembra piu' ricco. Le strutture dicono la verita': una ciascuno.
        """
        bollette, atti = [], []
        for citta in ("Roma", "Milano", "Napoli"):
            t = f"Intestatario Mario Rossi, fornitura in {citta}, totale 100,00 EUR."
            bollette.append(riga(t, [ent(t, "Mario", "GIVENNAME"), ent(t, "Rossi", "SURNAME"),
                                     ent(t, citta, "CITY"), ent(t, "100,00 EUR", "AMOUNT")], 1))
        for tribunale in ("Roma", "Milano", "Napoli"):
            t = f"Tribunale di {tribunale}: il ricorrente Mario Rossi chiede 100,00 EUR."
            atti.append(riga(t, [ent(t, "Mario", "GIVENNAME"), ent(t, "Rossi", "SURNAME"),
                                 ent(t, "100,00 EUR", "AMOUNT")], 2))

        b, _ = ispeziona(bollette)
        a, _ = ispeziona(atti)
        self.assertEqual(b["scheletri_distinti"], 1, "la bolletta etichetta tutto: 1 scheletro")
        self.assertEqual(a["scheletri_distinti"], 3, "l'atto lascia il tribunale in chiaro: 3")
        # la grana usata non si fa ingannare: una struttura per ciascuno
        self.assertEqual(b["gruppi_distinti"], 1)
        self.assertEqual(a["gruppi_distinti"], 1)

    def test_sovrapposizione_fra_due_file(self):
        a = scrivi([buona("Mario"), buona("Giulia")])
        b = scrivi([buona("Giulia"), buona("Anna")])
        _, _, mie = ic.ispeziona(a, ic.tassonomia()[0])
        self.assertEqual(len(mie & ic.impronte_di(b)), 1)

    def test_la_sintesi_contiene_i_numeri_da_incollare(self):
        m, _ = ispeziona([buona("Mario"), buona("Anna")])
        s = ic.sintesi(m)
        self.assertIn("2 righe", s)
        self.assertIn("strutture distinte", s)
        self.assertIn("0 errori", s)


class FileVuoto(unittest.TestCase):

    def test_non_esplode_e_non_divide_per_zero(self):
        m, errori = ispeziona([])
        self.assertEqual(m["righe"], 0)
        self.assertEqual(m["gruppi_distinti"], 0)
        self.assertEqual(m["errori_duri"], 0)
        self.assertIn("0 righe", ic.sintesi(m))


if __name__ == "__main__":
    unittest.main()
