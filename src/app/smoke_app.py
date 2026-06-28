# -*- coding: utf-8 -*-
import io, sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
import app  # carica il modello

txt = ("Il sottoscritto Mario Rossi, C.F. RSSMRA85M01H501Z, residente in Via Roma 10, "
       "Milano, chiede il pagamento sull'IBAN IT60X0542811101000000123456. "
       "Contatti: mario.rossi@example.it, +39 333 1234567. " * 20)  # forza piu' chunk
r = app.analyze(txt)
print("chunk:", r["n_chunks"], "| entita':", r["n_entities"], "| char:", r["n_chars"])
print("by_label:", r["by_label"])
print("\nESTRATTO CENSURATO:")
print(r["censored_text"][:400])
