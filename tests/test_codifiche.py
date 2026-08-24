# -*- coding: utf-8 -*-
"""`_text_from_bytes()`: da quali byte esce un testo su cui i rilevatori funzionano.

Il caso peggiore qui non è il testo illeggibile — quello si vede — ma il testo che resta
**leggibile a schermo** e sparisce ai rilevatori. Un `.txt` in utf-16 senza BOM finisce
decodificato a 8 bit e diventa `M\\0a\\0r\\0i\\0o`: in HTML i byte nulli sono invisibili,
quindi l'utente legge il documento intero, non vede nessun segnaposto e conclude che non ci
fosse niente da anonimizzare. Il documento esce in chiaro e sembra pulito.

Per questo i test misurano la **trovabilità della PII nel testo restituito**, non solo il
round-trip: sono due cose diverse ed è la prima che protegge l'utente.

`app.py` non è importabile senza torch/flask/fitz, quindi la funzione vera si ritaglia dal
sorgente con `ast`: si testa il codice che viene spedito, non una copia.

Tutti i valori sono SINTETICI.
"""

import ast
import os
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "src" / "app" / "app.py"

CF = "RSSMRA80A01H501U"
IBAN = "IT60X0542811101000000123456"
ATTO = ("TRIBUNALE DI MILANO - Verbale del 15/03/2026.\n"
        "Ricorrente: Mario Rossi, C.F. %s, IBAN %s.\n" % (CF, IBAN))
ACCENTI = "Perizia: la proprietà è accertata, perché così risulta.\n" + ATTO


def _carica():
    """`_text_from_bytes` e le sue dipendenze, senza importare tutto app.py."""
    albero = ast.parse(APP.read_text(encoding="utf-8"))
    ns = {"os": os, "re": re}
    for nodo in albero.body:
        if isinstance(nodo, ast.Assign) and any(
                getattr(t, "id", "") in ("TEXT_EXTS", "PDF_MAGIC") for t in nodo.targets):
            exec(compile(ast.Module([nodo], []), "app.py", "exec"), ns)
        elif isinstance(nodo, ast.FunctionDef) and nodo.name in ("_is_pdf", "_text_from_bytes"):
            exec(compile(ast.Module([nodo], []), "app.py", "exec"), ns)
    return ns["_text_from_bytes"]


leggi = _carica()


def trovabile(dati):
    """La PII e' ancora trovabile dai rilevatori nel testo restituito?"""
    testo = leggi("atto.txt", dati)
    return CF in testo and IBAN in testo


class Codifiche(unittest.TestCase):
    def test_round_trip_su_tutte_le_codifiche(self):
        for testo in (ATTO * 4, ACCENTI * 3):
            for enc in ("utf-8", "utf-8-sig", "cp1252", "latin-1",
                        "utf-16", "utf-16-le", "utf-16-be"):
                try:
                    dati = testo.encode(enc)
                except UnicodeEncodeError:
                    continue                      # non rappresentabile: non e' un caso
                self.assertEqual(leggi("atto.txt", dati), testo, "non torna dopo %s" % enc)

    def test_utf16_senza_bom_la_pii_resta_trovabile(self):
        """Il difetto per cui questa modifica esiste."""
        for enc in ("utf-16-le", "utf-16-be"):
            for testo in (ATTO, ATTO * 20, ACCENTI * 3):
                self.assertTrue(trovabile(testo.encode(enc)), "%s: PII persa" % enc)

    def test_utf16_con_caratteri_fuori_dal_latino(self):
        """Un allegato bilingue, o una tabella disegnata a cornice: i caratteri sopra
        U+00FF non hanno il byte alto nullo, ma la parte italiana sì."""
        for contorno in ("中华人民共和国上海市人民法院 " * 40,     # cinese
                         "สัญญาระหว่างคู่สัญญา " * 40,           # thai
                         "─" * 62 + "\n" + "─" * 62):            # cornice
            for enc in ("utf-16-le", "utf-16-be"):
                dati = (contorno + "\n" + ATTO * 3).encode(enc)
                self.assertTrue(trovabile(dati), "%s con %r: PII persa" % (enc, contorno[:8]))

    def test_i_file_a_8_bit_non_vengono_toccati(self):
        for enc in ("utf-8", "utf-8-sig", "cp1252", "latin-1"):
            self.assertTrue(trovabile((ATTO * 8).encode(enc)), enc)
            self.assertTrue(trovabile((ACCENTI * 4).encode(enc)), enc + " accenti")
        # l'euro e le virgolette tipografiche distinguono cp1252 da latin-1
        self.assertTrue(trovabile((ATTO + "Importo 1.250,00 €.\n").encode("cp1252")))

    def test_nul_vaganti_e_teste_azzerate(self):
        """File troncato da un crash, blob binario incollato in cima, export a record
        fissi NUL-terminati: nessuno di questi deve far perdere il documento."""
        base = bytearray((ATTO * 8).encode("utf-8"))
        for quanti in (1, 10, 60):
            dati = bytearray(base)
            for i in range(1, 1 + 2 * quanti, 2):
                dati[i] = 0
            self.assertTrue(trovabile(bytes(dati)), "%d NUL vaganti" % quanti)
        self.assertTrue(trovabile(b"\x00" * 700 + (ATTO * 40).encode("utf-8")))

    def test_casi_degeneri_non_sollevano(self):
        for dati in (b"", b"A", b"AB", b"ABC", b"a\x00", b"\x00a", b"\xef\xbb\xbf",
                     b"\xff\xfe", b"\xfe\xff", b"\x00" * 64, b"\x89PNG\r\n\x1a\n"):
            self.assertIsInstance(leggi("atto.txt", dati), str, repr(dati))


if __name__ == "__main__":
    unittest.main()
