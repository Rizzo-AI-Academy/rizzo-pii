"""docmd: pipeline OCR "detect-and-route" su ONNX Runtime che converte documenti in Markdown.

Layout (PP-DocLayout) -> testo (PP-OCRv6 det+rec) / tabelle (SLANet_plus) / formule (PP-FormulaNet_plus).

Copia della pipeline docmd (OCR su CPU con i modelli ufficiali PaddleOCR esportati in ONNX)
adattata a rizzo-pii: box per carattere dalle posizioni CTC, testo di timbri e loghi, formule
opzionali. La usa src/app/ocr.py; i modelli si procurano con scripts/fetch_ocr_models.py.
"""

from .pipeline import DocPipeline

__all__ = ["DocPipeline"]
