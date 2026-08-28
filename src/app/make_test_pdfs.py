# -*- coding: utf-8 -*-
"""Genera i PDF di prova per la verifica manuale delle 3 novita' del percorso
di input (vedi CHANGELOG 2026-08-17): OCR su scansioni, crop di intestazioni/
piè di pagina e gruppo multi-file con mappa condivisa.

Uso (dalla root del repo):
    py src/app/make_test_pdfs.py                 # -> test_output/ (creata qui)
    py src/app/make_test_pdfs.py -o tmp          # cartella di output diversa

Serve solo PyMuPDF. I nomi e i dati sono SINTETICI.

Genera:
  test_pdf_scan.pdf          misto: pag.1 nativa con PII, pag.2 "scansionata"
                             (immagine di testo: nessun layer testuale -> OCR)
  test_pdf_header_footer.pdf nativo con intestazione/piè di pagina contenenti
                             PII: prova il crop (8%/6%) dall'UI o da API
  test_pdf_group_a.pdf       e test_pdf_group_b.pdf: due file con la STESSA
                             persona (Mario Rossi) e lo stesso IBAN -> nel
                             gruppo devono ricevere gli stessi placeholder
"""

import argparse
from pathlib import Path

import fitz  # PyMuPDF

TESTI = {
    "header": "STUDIO LEGALE ROSSI & PARTNERS",
    "footer": "Via Roma 1, 20121 Milano MI - tel. +39 333 1234567",
    "corpo": (
        "Contratto di locazione tra il sig. Mario Rossi, codice fiscale "
        "RSSMRA85H12F205Z, residente in Via Garibaldi 24, e la societa' "
        "Edilnord S.r.l., partita IVA 12345678901. "
        "Le coordinate bancarie del locatore sono IBAN IT60X0542811101000000123456."
    ),
}


def _pagina_testo(doc, corpo):
    page = doc.new_page()
    page.insert_text((40, 30), TESTI["header"], fontsize=12)
    page.insert_textbox(fitz.Rect(40, 300, 555, 760), corpo, fontsize=11)
    page.insert_text((40, 790), TESTI["footer"], fontsize=10)
    return page


def _pagina_scansionata(doc):
    """Pagina senza layer testuale: testo renderizzato e re-impaginato come
    immagine, come esce da uno scanner."""
    tmp = fitz.open()
    t = tmp.new_page()
    t.insert_textbox(fitz.Rect(40, 40, 550, 780),
                     TESTI["header"] + "\n\n" + TESTI["corpo"],
                     fontsize=11)
    pix = t.get_pixmap(dpi=200, alpha=False)
    tmp.close()
    page = doc.new_page()
    page.insert_image(page.rect, pixmap=pix)
    return page


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--out", default="test_output",
                    help="cartella di output (default: test_output/)")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # 1) misto: pagina nativa + pagina scansionata
    doc = fitz.open()
    _pagina_testo(doc, TESTI["corpo"])
    _pagina_scansionata(doc)
    (out / "test_pdf_scan.pdf").write_bytes(doc.tobytes())
    doc.close()
    print("test_pdf_scan.pdf          (pag.1 nativa + pag.2 scansione -> OCR)")

    # 2) intestazione/piè di pagina con PII (per il crop)
    doc = fitz.open()
    _pagina_testo(doc, TESTI["corpo"])
    (out / "test_pdf_header_footer.pdf").write_bytes(doc.tobytes())
    doc.close()
    print("test_pdf_header_footer.pdf (intestazione e piè di pagina con PII)")

    # 3) gruppo: due file con la stessa persona
    for nome, pre in [("test_pdf_group_a.pdf", "Atto A - "),
                      ("test_pdf_group_b.pdf", "Atto B - ")]:
        doc = fitz.open()
        _pagina_testo(doc, pre + TESTI["corpo"])
        (out / nome).write_bytes(doc.tobytes())
        doc.close()
        print(nome, "                      (stesso Mario Rossi/IBAN: "
                    "nel gruppo -> stessi placeholder)")

    print("\nPoi, nell'app: carica piu' file insieme, oppure prova crop e OCR.")


if __name__ == "__main__":
    main()