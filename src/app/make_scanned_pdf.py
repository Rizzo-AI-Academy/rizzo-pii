# -*- coding: utf-8 -*-
"""Crea un PDF "scansionato" di prova (solo immagine, niente layer di testo) per
provare l'OCR: testo legale con PII sintetiche, renderizzato a 200 dpi, ruotato di
qualche decimo di grado e sporcato con un po' di rumore, come esce da uno scanner.

Uso:  python src/app/make_scanned_pdf.py [out.pdf] [--png out.png]
"""

import sys

import fitz
import numpy as np

TEXT = """TRIBUNALE ORDINARIO DI MILANO - Sezione Civile
R.G. n. 4521/2024

ATTO DI CITAZIONE

Il sig. Mario Rossi, nato a Napoli il 12/06/1985, codice fiscale
RSSMRA85H12F839X, residente in Via Garibaldi 24, 20121 Milano (MI),
rappresentato e difeso dall'avv. Giulia Bianchi del Foro di Milano
(PEC giulia.bianchi@pec.studiobianchi.it, tel. +39 333 1234567)

CITA

la societa' Edilnord S.r.l., P.IVA 12345678903, in persona del legale
rappresentante pro tempore, con sede in Corso Buenos Aires 15, Milano,
a comparire all'udienza del 15/03/2025 per sentir condannare la convenuta
al pagamento della somma di euro 12.500,00 da versare sul conto
IBAN IT60X0542811101000000123456 intestato all'attore.

Milano, 20 gennaio 2025                         Avv. Giulia Bianchi
"""


def make(angle=0.6, dpi=200, seed=7):
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    page.insert_textbox(fitz.Rect(60, 70, 540, 800), TEXT, fontname="tiro",
                        fontsize=11.5, lineheight=1.45)
    pix = page.get_pixmap(dpi=dpi, alpha=False)
    img = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, 3)
    import cv2
    h, w = img.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    img = cv2.warpAffine(img, m, (w, h), borderValue=(255, 255, 255))
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, 9, img.shape)
    img = np.clip(img.astype(np.float32) * 0.97 + 6 + noise, 0, 255).astype(np.uint8)
    ok, png = cv2.imencode(".png", img[:, :, ::-1])
    out = fitz.open()
    p2 = out.new_page(width=595, height=842)
    p2.insert_image(p2.rect, stream=png.tobytes())
    return out.tobytes(deflate=True), png.tobytes()


if __name__ == "__main__":
    dest = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("--") else "scansione_prova.pdf"
    pdf, png = make()
    with open(dest, "wb") as f:
        f.write(pdf)
    print("scritto", dest)
    if "--png" in sys.argv:
        p = sys.argv[sys.argv.index("--png") + 1]
        with open(p, "wb") as f:
            f.write(png)
        print("scritto", p)
