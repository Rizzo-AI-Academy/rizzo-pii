"""Coerenza dei segnaposti per le menzioni complete e parziali."""

import io
import re
import unittest
from unittest.mock import patch

from test_ora import APP, ent


TEXT = (
    "Paolo Rossi lavora in informatica. Paolo vive a Rocca Esempio. "
    "Elena Bianchi ed Elena Verdi lavorano insieme. Elena cura gli ordini. "
    "Alfa Servizi gestisce le consegne. Alfa opera in tutta la regione."
)
MENTIONS = (
    ("Paolo Rossi", "FULLNAME"), ("Paolo", "FULLNAME"), ("Rocca Esempio", "CITY"),
    ("Elena Bianchi", "FULLNAME"), ("Elena Verdi", "FULLNAME"), ("Elena", "FULLNAME"),
    ("Alfa Servizi", "ORG"), ("Alfa", "ORG"),
)
EXPECTED = (
    "[PERSONA_1] [PERSONA_2] lavora in informatica. [PERSONA_1] vive a [CITY_1]. "
    "[PERSONA_3] [PERSONA_4] ed [PERSONA_3] [PERSONA_5] lavorano insieme. "
    "[PERSONA_3] cura gli ordini. [ORG_1] [ORG_2] gestisce le consegne. "
    "[ORG_1] opera in tutta la regione."
)


def detected(text):
    mentions = []
    start = 0
    for value, label in MENTIONS:
        match = ent(text, value, label, da=start)
        mentions.append(match)
        start = match["end"]
    return mentions, 1


@unittest.skipIf(APP is None, "dipendenze app assenti")
class EntityPlaceholders(unittest.TestCase):
    def setUp(self):
        self.client = APP.app.test_client()
        self.enterContext(patch.object(APP, "detect_model", side_effect=detected))

    def test_riuso_e_ripristino(self):
        response = self.client.post("/analyze", json={"text": TEXT, "include_mapping": True})
        self.assertEqual(response.status_code, 200)
        result = response.get_json()
        self.assertEqual(result["anonymized_text"], EXPECTED)
        restored = re.sub(r"\[[A-Z]+_\d+\]", lambda match: result["mapping"][match[0]], EXPECTED)
        self.assertEqual(restored, TEXT)

    def test_nessun_valore_senza_dizionario(self):
        response = self.client.post("/analyze", json={"text": TEXT, "include_mapping": False})
        self.assertEqual(response.status_code, 200)
        result = response.get_json()
        self.assertEqual(result["anonymized_text"], EXPECTED)
        self.assertEqual(result["mapping"], {})
        self.assertNotIn("source_text", result)
        self.assertTrue(all("t" not in seg for seg in result["segments"] if "ph" in seg))

    def test_pdf_con_menzioni_complete_e_parziali(self):
        with APP.fitz.open() as pdf:
            page = pdf.new_page(width=1300, height=200)
            page.insert_text((30, 40), TEXT, fontsize=10)
            original = pdf.tobytes()

        response = self.client.post("/pdf", data={"file": (io.BytesIO(original), "esempio.pdf")})
        self.assertEqual(response.status_code, 200)
        with APP.fitz.open(stream=response.data, filetype="pdf") as pdf:
            output = pdf[0].get_text()
        for value, _ in MENTIONS:
            self.assertNotIn(value, output)
        self.assertEqual(output.count("[PERSONA_1]"), 2)
        self.assertEqual(output.count("[PERSONA_3]"), 3)
        self.assertEqual(output.count("[ORG_1]"), 2)
