# -*- coding: utf-8 -*-
"""L'inferenza gira sempre sullo stesso thread, qualunque richiesta la chieda (issue #139).

torch crea un pool OpenMP per ogni thread che lo chiama e su Windows non lo libera
quando il thread finisce. `app.run(threaded=True)` usa un thread nuovo per ogni
richiesta, quindi ogni `/analyze` lasciava vivo un pool intero, senza limite. Qui si
controlla da quale thread viene chiamato il modello, che e' cio' che decide quanti
pool nascono.

Tutti i valori sono SINTETICI.
"""

import sys
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))

from test_ora import APP  # noqa: E402  (stesso caricamento di app coi pesi finti)

TESTO = "Mario Rossi abita in via Roma 87 a Milano"


@unittest.skipIf(APP is None, "app.py non importabile (torch/flask/fitz assenti)")
class ThreadInferenza(unittest.TestCase):
    def test_un_solo_thread_per_tutte_le_richieste(self):
        visti = []
        codici = []

        def finto_nlp(chunks):
            visti.append(threading.current_thread())
            return [[] for _ in chunks]

        def richiesta():
            r = APP.app.test_client().post("/analyze", json={"text": TESTO})
            codici.append(r.status_code)

        with patch.object(APP, "nlp", finto_nlp):
            richieste = [threading.Thread(target=richiesta) for _ in range(5)]
            for t in richieste:
                t.start()
                t.join()

        self.assertEqual(codici, [200] * 5)
        # i Thread restano vivi in `visti`: il confronto e' per identita', e un ident
        # riusato da un thread finito non fa sembrare cinque richieste un thread solo
        self.assertEqual(len(set(visti)), 1)
        self.assertNotIn(visti[0], richieste)


if __name__ == "__main__":
    unittest.main()
