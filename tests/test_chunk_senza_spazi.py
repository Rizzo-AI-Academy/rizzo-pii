# -*- coding: utf-8 -*-
"""Chunk limitati anche in caratteri, non solo in parole.

Il chunk si misurava in parole separate da spazi: un JSON compatto (o un export
senza spazi) era una parola sola, quindi un chunk solo, che la pipeline tronca in
silenzio a 8.192 token. Oltre il taglio il modello non legge nulla: su 1.000 record
anagrafici in JSON compatto restavano in chiaro i nomi di 753.

Il test non carica il modello: controlla la proprieta' da cui dipende tutto, cioe'
che nessun chunk superi il tetto e che ogni carattere del testo stia in un chunk.
`src/app/chunking.py` non importa torch.
"""

import json
import random
import re
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "app"))

import chunking  # noqa: E402


def _record(i):
    return {"id": i, "nome": "Giulia", "cognome": "Bianchi", "citta": "Genova",
            "email": f"u{i}@example.com"}


def json_compatto(n):
    return json.dumps([_record(i) for i in range(n)], separators=(",", ":"))


def prosa(n_parole, seed=0):
    rnd = random.Random(seed)
    parole = ["il", "ricorrente", "Mario", "Rossi", "nato", "a", "Genova", "chiede",
              "che", "la", "Corte", "d'Appello", "voglia", "accogliere", "ricorso."]
    return " ".join(rnd.choice(parole) for _ in range(n_parole))


def copertura(text, chunks):
    """Caratteri non-spazio che non stanno in nessun chunk."""
    dentro = [False] * len(text)
    for c, off in chunks:
        assert text[off:off + len(c)] == c, "offset sbagliato"
        for k in range(off, off + len(c)):
            dentro[k] = True
    return [k for k, ch in enumerate(text) if not ch.isspace() and not dentro[k]]


class TestoSenzaSpazi(unittest.TestCase):
    def test_json_compatto_non_diventa_un_chunk_solo(self):
        text = json_compatto(1000)                       # ~90.000 caratteri, zero spazi
        chunks = chunking.chunk_text(text)
        self.assertGreater(len(chunks), 1)
        self.assertLessEqual(max(len(c) for c, _ in chunks), chunking.MAX_CHARS)
        self.assertEqual(copertura(text, chunks), [])

    def test_i_campi_non_si_spezzano(self):
        """Ogni valore compare intero almeno in un chunk."""
        text = json_compatto(300)
        chunks = chunking.chunk_text(text)
        for m in re.finditer(r'"cognome":"Bianchi"', text):
            self.assertTrue(any(off <= m.start() and m.end() <= off + len(c)
                                for c, off in chunks), f"campo spezzato a {m.start()}")

    def test_i_chunk_si_aprono_e_chiudono_fra_due_campi(self):
        """Il taglio cade dopo una virgola: il modello non vede mai un nome troncato al
        bordo del chunk ("...,"nome":"Giu"), che etichetterebbe a meta'."""
        text = json_compatto(300)
        chunks = chunking.chunk_text(text)
        for c, off in chunks[:-1]:
            self.assertEqual(c[-1], ",", f"chunk chiuso a meta' campo: ...{c[-20:]!r}")
        for c, off in chunks[1:]:
            self.assertEqual(text[off - 1], ",", f"chunk aperto a meta' campo: {c[:20]!r}...")

    def test_blob_senza_separatori(self):
        text = "".join(random.Random(1).choice("ABCDEFabcdef0123456789+/") for _ in range(50000))
        chunks = chunking.chunk_text(text)
        self.assertLessEqual(max(len(c) for c, _ in chunks), chunking.MAX_CHARS)
        self.assertEqual(copertura(text, chunks), [])
        self.assertLess(len(chunks), 2 * len(text) / chunking.MAX_CHARS + 2,
                        "la sovrapposizione non deve far avanzare di un pezzo alla volta")

    def test_record_a_cavallo_di_due_chunk_c_e_intero_in_uno(self):
        text = json_compatto(300)
        chunks = chunking.chunk_text(text)
        for (a, oa), (b, ob) in zip(chunks, chunks[1:]):
            self.assertLess(ob, oa + len(a), "chunk consecutivi senza sovrapposizione")


class ProsaInvariata(unittest.TestCase):
    def _vecchio(self, text, max_words=120, overlap=20):
        """Il chunking di prima, per confronto."""
        words = list(re.finditer(r"\S+", text))
        chunks, i = [], 0
        while i < len(words):
            block = words[i:i + max_words]
            chunks.append((text[block[0].start():block[-1].end()], block[0].start()))
            if i + max_words >= len(words):
                break
            i += max_words - overlap
        return chunks

    def test_prosa_ordinaria_stessi_chunk_di_prima(self):
        for n in (1, 50, 120, 121, 700, 5000):
            with self.subTest(parole=n):
                text = prosa(n, seed=n)
                self.assertEqual(chunking.chunk_text(text), self._vecchio(text))

    def test_prosa_con_un_url_lungo(self):
        """Un URL di qualche centinaio di caratteri e' ancora una parola: non si spezza,
        e i chunk del documento restano quelli di prima."""
        url = "https://www.example.it/atti/" + "a1b2c3d4-" * 40
        text = prosa(300, seed=3) + f" consultabile su {url} " + prosa(300, seed=4)
        self.assertEqual(chunking.chunk_text(text), self._vecchio(text))

    def test_testo_vuoto(self):
        self.assertEqual(chunking.chunk_text(""), [])
        self.assertEqual(chunking.chunk_text("   \n "), [])


if __name__ == "__main__":
    unittest.main()
