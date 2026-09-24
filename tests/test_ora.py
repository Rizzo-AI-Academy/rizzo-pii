# -*- coding: utf-8 -*-
"""`TIME`: la regex dell'ora completa le span del modello, e non gira da sola.

Il modello taglia i minuti (`18:28` -> `18`) e la metà che resta finisce in chiaro
accanto al segnaposto: `ore 18[TIME_1]28`. La regex dell'ora (`complete_time`) parte
solo dove il modello ha già visto un'ora e ne allarga i confini all'ora intera.

Da sola non girerebbe bene: due numeri separati da due punti sono anche una scala
catastale (`Scala 1:25`), un versetto (`Giovanni 3:16`), una coordinata. Ancorata al
modello non maschera nessuno di questi, e può accettare anche il punto (`ore 18.30`).

Tutti i valori sono SINTETICI.
"""

import os
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "app"))

import detectors  # noqa: E402


def ent(testo, pezzo, label="TIME", da=0):
    """Entita' del modello sulla prima occorrenza di `pezzo` a partire da `da`."""
    i = testo.index(pezzo, da)
    return {"label": label, "start": i, "end": i + len(pezzo), "score": 0.9,
            "validated": False, "source": "modello"}


def completa(testo, *ents):
    detectors.complete_time(list(ents), testo)
    return [testo[e["start"]:e["end"]] for e in ents]


def _carica_app():
    """Importa app coi pesi finti: qui il modello vero non serve.

    Tre accorgimenti, tutti necessari:
    - il finto transformers va messo in sys.modules, non con patch("transformers.
      pipeline"): un `from transformers import pipeline` su un _LazyModule NON passa
      dall'attributo patchato e prende la funzione vera;
    - la voce si sostituisce e ripristina A MANO: patch.dict(sys.modules) al
      ripristino fa clear()+update() dell'intero sys.modules, buttando via anche
      l'app appena importata (e ricaricando numpy);
    - PII_MODEL_DIR (l'override che app.py stesso documenta) puntato a una cartella
      esistente: senza, il controllo del modello mancante fa sys.exit(2) durante
      l'import - un SystemExit che l'except qui sotto non cattura.
    """
    if "app" in sys.modules:
        return sys.modules["app"]
    finto = MagicMock()
    finto.pipeline.return_value = MagicMock(return_value=[])
    vero = sys.modules.get("transformers")
    sys.modules["transformers"] = finto
    prima = os.environ.get("PII_MODEL_DIR")
    os.environ["PII_MODEL_DIR"] = str(ROOT / "tests")
    try:
        import app as pii_app
    finally:
        if vero is not None:
            sys.modules["transformers"] = vero
        else:
            del sys.modules["transformers"]
        if prima is not None:
            os.environ["PII_MODEL_DIR"] = prima
        else:
            del os.environ["PII_MODEL_DIR"]
    return pii_app


try:                                    # torch/transformers/flask/fitz presenti?
    APP = _carica_app()
except Exception:                       # noqa: BLE001 - qualunque import mancante
    APP = None


class CompletaOra(unittest.TestCase):
    def test_minuti_tagliati(self):
        t = "Ingresso ore 18:28, uscita ore 19:30."
        self.assertEqual(completa(t, ent(t, "18"), ent(t, "19:")), ["18:28", "19:30"])

    def test_meta_destra_e_secondi(self):
        t = "Accesso registrato alle 09:50:12."
        self.assertEqual(completa(t, ent(t, "50:12")), ["09:50:12"])

    def test_col_punto_se_il_modello_ha_visto_un_ora(self):
        t = "La riunione e' fissata per le ore 18.30."
        self.assertEqual(completa(t, ent(t, "18")), ["18.30"])

    def test_intervallo_col_trattino_attaccato(self):
        t = "Ricevimento dalle 9:00-12:30."
        self.assertEqual(completa(t, ent(t, "9"), ent(t, "12")), ["9:00", "12:30"])

    def test_span_gia_completa_o_piu_larga_non_si_restringe(self):
        t = "Udienza alle ore 10:30 in aula."
        self.assertEqual(completa(t, ent(t, "ore 10:30")), ["ore 10:30"])

    def test_nessuna_ora_senza_il_modello(self):
        """Senza un TIME del modello la regex non produce nulla: niente falsi positivi."""
        for t in ("Scala 1:25 della planimetria.", "Giovanni 3:16.",
                  "versione 1.30 del programma", "euro 10.30 di spesa"):
            ents = [ent(t, t.split()[0], label="ORG")]
            self.assertEqual(completa(t, *ents), [t.split()[0]], t)

    def test_non_mangia_pezzi_di_data_o_numeri_lunghi(self):
        t = "Nota del 15.03.2026, codice 1.2.30.4."
        self.assertEqual(completa(t, ent(t, "15"), ent(t, "30")), ["15", "30"])

    def test_ore_e_minuti_fuori_scala(self):
        for t in ("codice 24:00", "codice 25:30", "codice 10:60"):
            pezzo = t.split()[1][:2]
            self.assertEqual(completa(t, ent(t, pezzo)), [pezzo], t)

    def test_la_date_del_modello_non_si_tocca(self):
        """Il timestamp ISO la data la trova il modello: nessun TIME, nessuna modifica."""
        t = "Deposito telematico: 2026-03-15T10:30:00 (ricevuta PEC)."
        self.assertEqual(completa(t, ent(t, "2026-03-15T10:30:00", label="DATE")),
                         ["2026-03-15T10:30:00"])

    def test_time_non_e_nella_rete_regex(self):
        self.assertNotIn("TIME", {d[0] for d in detectors.DETECTORS})
        self.assertEqual(
            [e for e in detectors.detect_regex("ore 18:28") if e["label"] == "TIME"], [])


@unittest.skipIf(APP is None, "app.py non importabile (torch/flask/fitz assenti)")
class OraInAnalyze(unittest.TestCase):
    """La pipeline completa: il modello (finto) taglia i minuti, l'output no."""

    def _modello(self, testo, pezzi):
        def nlp(chunks):
            out = []
            for c in chunks:
                res = []
                for p in pezzi:
                    i = c.index(p)
                    res.append({"entity_group": "TIME", "start": i, "end": i + len(p),
                                "score": 0.9, "word": p})
                out.append(res)
            return out
        return patch.object(APP, "nlp", nlp)

    def test_nessun_minuto_in_chiaro(self):
        t = "Ingresso ore 18:28, uscita ore 19:30."
        with self._modello(t, ["18", "19:"]):
            out = APP.analyze(t)
        self.assertEqual(out["anonymized_text"],
                         "Ingresso ore [TIME_1], uscita ore [TIME_2].")
        self.assertEqual(out["mapping"], {"[TIME_1]": "18:28", "[TIME_2]": "19:30"})


if __name__ == "__main__":
    unittest.main()
