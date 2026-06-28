# -*- coding: utf-8 -*-
from pathlib import Path
import fitz

OUT = str(Path(__file__).resolve().parents[2] / "experiments" / "test_doc.pdf")
doc = fitz.open()
page = doc.new_page()
testo = ("Il sottoscritto Mario Rossi, C.F. RSSMRA85M01H501Z, IBAN "
         "IT60X0542811101000000123456, email mario@studio.it, "
         "residente in Via Roma 12, Milano. Tel 3331234567.")
page.insert_text((72, 100), testo, fontsize=11)
doc.save(OUT)
doc.close()
print(f"creato {OUT}")
