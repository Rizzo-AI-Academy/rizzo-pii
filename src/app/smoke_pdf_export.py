# -*- coding: utf-8 -*-
"""Smoke test di pdf_export.py — NON carica il modello (solo PyMuPDF), gira in
pochi secondi ovunque:  python src/app/smoke_pdf_export.py

Copre i casi che rendono un PDF "anonimizzato" ancora leggibile: sillabazione a
fine riga, sottostringhe da non toccare, e le PII fuori dal content stream
(annotazioni, campi modulo, segnalibri, allegati, metadati). Exit code 1 se un
controllo fallisce.
"""

import io
import os
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import fitz  # PyMuPDF

import pdf_export as px

OK = []
CF_ONLY = "RSSMRA85H12H501V"
LINK_EMAIL = "mario.rossi@example.com"


def check(name, cond, extra=""):
    OK.append(bool(cond))
    print(("PASS " if cond else "FAIL ") + name + ((" | " + str(extra)) if extra else ""))


def build_pdf():
    """PDF di prova con le PII messe in tutti i posti in cui sanno nascondersi."""
    doc = fitz.open()
    p = doc.new_page()
    p.insert_text((50, 80), "Il sig. Mario Rossi, nato a Milano.", fontsize=11)
    p.insert_text((50, 100), "Il perito Fran-", fontsize=11)        # sillabazione
    p.insert_text((50, 116), "cesco Cordella firma.", fontsize=11)  # ...a capo
    p.insert_text((50, 140), "CORDELLA e DENSITA' non sono PII.", fontsize=11)
    p.insert_text((50, 160), "IBAN IT60X0542811101000000123456 - eta 45 anni.", fontsize=11)
    p.insert_text((50, 180), "Contatto: m.rossi@studio.it", fontsize=11)

    a = p.add_text_annot((300, 300), "Chiamare Mario Rossi")
    a.set_info(title="Mario Rossi", content="Chiamare Mario Rossi")
    a.update()

    w = fitz.Widget()
    w.rect = fitz.Rect(50, 400, 250, 420)
    w.field_name = "nome"
    w.field_type = fitz.PDF_WIDGET_TYPE_TEXT
    w.field_value = "Mario Rossi"
    p.add_widget(w)

    p2 = doc.new_page()
    p2.insert_text((50, 80), "Ancora Mario Rossi, pagina due.", fontsize=11)
    doc.set_toc([[1, "Fascicolo Mario Rossi", 1]])
    doc.embfile_add("nota.txt", b"Mario Rossi vive a Milano")
    doc.set_metadata({"author": "Mario Rossi", "title": "Referto Rossi"})
    out = doc.tobytes()
    doc.close()
    return out


MAPPING = {
    "[FULLNAME_1]": "Mario Rossi",
    "[FULLNAME_2]": "Francesco Cordella",
    "[CITY_1]": "Milano",
    "[IBAN_1]": "IT60X0542811101000000123456",
    "[EMAIL_1]": "m.rossi@studio.it",
    "[AGE_1]": "45",                       # 2 sole cifre -> deve finire in "skipped"
}

out, rep = px.redact_pdf(build_pdf(), MAPPING)
print("report:", {k: v for k, v in rep.items() if k != "by_placeholder"})
print("occorrenze:", rep["by_placeholder"], "\n")

with fitz.open(stream=out, filetype="pdf") as d:
    txt = px._readable_text(d)
    meta, toc, embs = d.metadata, d.get_toc(simple=True), d.embfile_names()
low = txt.lower()

check("nome semplice redatto", "mario rossi" not in low)
check("nome sillabato a fine riga redatto", "cesco cordella" not in low)
check("sottostringa CORDELLA non toccata", "cordella e densita" in low)
check("IBAN redatto", "it60x054281110100" not in low.replace(" ", ""))
check("email redatta", "m.rossi@studio.it" not in low)
check("citta redatta", "milano" not in low)
check("placeholder scritti al posto delle PII", "[FULLNAME_1]" in txt and "[IBAN_1]" in txt)
check("annotazioni ripulite", rep["annots"] > 0)
check("campi modulo ripuliti", rep["widgets"] > 0)
check("segnalibri ripuliti", rep["toc"] > 0 and "Mario Rossi" not in " ".join(e[1] for e in toc), toc)
check("allegati rimossi", rep["embedded"] == 1 and not embs, embs)
check("metadati azzerati", not meta.get("author") and not meta.get("title"))
check("valore troppo corto SALTATO e dichiarato", rep["skipped"] == ["[AGE_1]"], rep["skipped"])
check("nessun residuo", rep["residual"] == [], rep["residual"])
check("nessun valore introvabile", rep["not_found"] == [], rep["not_found"])

for name, fn in (("dizionario vuoto", lambda: px.redact_pdf(b"%PDF-", {})),
                 ("bytes non-PDF", lambda: px.redact_pdf(b"non un pdf", MAPPING)),
                 ("testo vuoto", lambda: px.text_to_pdf("   "))):
    try:
        fn()
        check(name + " -> PdfError", False)
    except px.PdfError:
        check(name + " -> PdfError", True)

# --- ancore: il frammento tagliato a meta' dal modello va trovato lo stesso,
#     ma un frammento corto NON puo' perdere i confini di parola ---------------
for val, txt, want in (
    ("-\ncesco Cordella", "Il perito Fran-\ncesco Cordella firma.", ["cesco Cordella"]),
    ("Rossi", "Rossini non e Rossi.", ["Rossi"]),          # niente match in "Rossini"
    (".it", "Il diritto e la vita in Italia.", []),        # troppo corto: resta ancorato
    ("Mario Rossi", "Il sig. Mario Rossi qui.", ["Mario Rossi"]),
):
    pat = px._value_pattern(val)
    got = [m.group(0) for m in pat.finditer(txt)] if pat else []
    check("pattern %r" % val, got == want, got)

pdf = px.text_to_pdf("Il sig. [FULLNAME_1] con IBAN [IBAN_1].\nRiga due — trattino, € 100.")
with fitz.open(stream=pdf, filetype="pdf") as d:
    t = d[0].get_text()
check("text_to_pdf impagina i placeholder", "[FULLNAME_1]" in t and "[IBAN_1]" in t)

# --- regressioni: PII scoperta solo fuori dal testo di pagina -----------------
doc = fitz.open()
p = doc.new_page()
p.insert_text((50, 80), "Documento di prova senza identificativi personali.", fontsize=11)
a = p.add_text_annot((200, 200), "Codice fiscale " + CF_ONLY)
a.set_info(content="Codice fiscale " + CF_ONLY)
a.update()
aux_only_pdf = doc.tobytes()
doc.close()
with fitz.open(stream=aux_only_pdf, filetype="pdf") as d:
    page_only_text = d[0].get_text()
check("CF solo in annotazione assente dal page text", CF_ONLY not in page_only_text)
aux_surfaces = px.extract_detection_surfaces(aux_only_pdf)
check("CF solo in annotazione scoperto dal percorso auxiliary",
      any(CF_ONLY in surface for surface in aux_surfaces))
check("residuo noto in annotazione rilevato prima della sanitizzazione",
      px._verify_residuals(aux_only_pdf, [("[CF_1]", CF_ONLY)]) == ["[CF_1]"])

aux_out, aux_rep = px.redact_pdf(aux_only_pdf, {"[CF_1]": CF_ONLY})
with fitz.open(stream=aux_out, filetype="pdf") as d:
    aux_after = px._readable_text(d)
check("CF solo in annotazione sanificato", CF_ONLY not in aux_after)
check("sanitizzazione auxiliary conteggiata come lavoro reale",
      aux_rep.get("sanitizations", 0) > 0, aux_rep)

# --- regressioni: target mailto persistente e non sovrapposto al testo --------
doc = fitz.open()
p = doc.new_page()
p.insert_text((50, 80), LINK_EMAIL, fontsize=11)
p.insert_text((50, 130), "Invia un messaggio", fontsize=11)
p.insert_link({"kind": fitz.LINK_URI, "from": fitz.Rect(45, 115, 180, 135),
               "uri": "mailto:" + LINK_EMAIL})
link_pdf = doc.tobytes()
doc.close()

doc = fitz.open()
p = doc.new_page()
p.insert_text((50, 80), "Invia un messaggio", fontsize=11)
p.insert_link({"kind": fitz.LINK_URI, "from": fitz.Rect(45, 65, 180, 85),
               "uri": "mailto:" + LINK_EMAIL})
link_only_pdf = doc.tobytes()
doc.close()
check("target link incluso nella verifica residui prima della sanitizzazione",
      px._verify_residuals(link_only_pdf, [("[EMAIL_1]", LINK_EMAIL)]) == ["[EMAIL_1]"])

link_out, link_rep = px.redact_pdf(link_pdf, {"[EMAIL_1]": LINK_EMAIL})
with fitz.open(stream=link_out, filetype="pdf") as d:
    link_text = d[0].get_text()
    links = d[0].get_links()
check("testo email visibile sanificato", LINK_EMAIL not in link_text)
check("link preservato", len(links) == 1, links)
check("target mailto sanificato",
      links and links[0].get("uri") == "mailto:%5BEMAIL_1%5D", links)
check("nessun residuo solo dopo testo e link sanificati", link_rep["residual"] == [], link_rep)

print("\n%d/%d PASS" % (sum(OK), len(OK)))
sys.exit(0 if all(OK) else 1)
