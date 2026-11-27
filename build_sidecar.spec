# -*- mode: python ; coding: utf-8 -*-
# PyInstaller spec per il BACKEND SIDECAR dell'app Tauri (build CPU, windowed).
# Entry: src/app/serve.py (solo server Flask, nessun browser).
# Build:  build_env\Scripts\pyinstaller.exe build_sidecar.spec --noconfirm \
#           --distpath tauri/src-tauri/backend
# Output: tauri/src-tauri/backend/pii-backend/pii-backend.exe (+ _internal, modello, assets)
import os as _os
# Disable ORT telemetry before build hooks import native libraries.
_os.environ["ORT_DISABLE_TELEMETRY"] = "1"
_os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
from PyInstaller.utils.hooks import collect_all

datas, binaries, hiddenimports = [], [], []
# Inferenza con ONNX Runtime (src/app/onnx_ner.py): NIENTE torch/transformers nel pacchetto.
for pkg in ("onnxruntime", "tokenizers"):
    d, b, h = collect_all(pkg)
    datas += d
    binaries += b
    hiddenimports += h

# modello + asset grafici impacchettati (app.py li risolve via _resource_path / _MEIPASS)
# Del modello si impacchetta SOLO cio' che serve a onnxruntime: il model.onnx (fp32, gli
# stessi pesi di PyTorch, prodotto da src/training/export_onnx.py) + tokenizer e config.
# Il model.safetensors (1,2 GB) resta fuori: con lui dentro il modello sarebbe doppio.
#
# PII_BUNDLE_MODEL=0 -> sidecar SENZA modello (installer leggero): l'app Tauri lo scarica
# da Hugging Face al primo avvio (tauri/src-tauri/src/model.rs) e lo passa con
# PII_MODEL_DIR. Default 1 = modello dentro (installer offline / build locali).
# Il modello impacchettato si puo' anche aggiungere DOPO, copiando i 6 file in
# _internal/pii_model/ (e' quello che fa la CI per l'installer offline).
MODEL_DIR = "models/rizzo-pii-0.3B-v1.5.0"
import os as _os
BUNDLE_MODEL = _os.environ.get("PII_BUNDLE_MODEL", "1") != "0"
if BUNDLE_MODEL:
    if not _os.path.isfile(f"{MODEL_DIR}/onnx/model.onnx_data"):
        raise SystemExit(f"manca {MODEL_DIR}/onnx/model.onnx(_data): "
                         f"python src/training/export_onnx.py --model-dir {MODEL_DIR}")
    # model.onnx = grafo, model.onnx_data = pesi (mappati in memoria): sempre insieme.
    datas += [(f"{MODEL_DIR}/onnx/model.onnx", "pii_model"),
              (f"{MODEL_DIR}/onnx/model.onnx_data", "pii_model")]
    datas += [(f"{MODEL_DIR}/{f}", "pii_model")
              for f in ("config.json", "tokenizer.json", "tokenizer_config.json",
                        "special_tokens_map.json")]
datas += [("src/app/assets", "assets")]
hiddenimports += ["fitz", "flask", "onnx_ner", "notarial_app", "notarial_server_config",
                  "notarial_ocr", "contextual", "review", "span_repairs",
                  "notarial_rules", "notarial_places", "notarial_evidence", "company_context"]

# --- OCR delle scansioni (src/app/ocr.py -> src/app/docmd/): modelli ONNX in "ocr_models" ---
# Si procurano con: python scripts/fetch_ocr_models.py  (-> models/ocr/). Si impacchettano
# solo inference.onnx + inference.yml dei modelli presenti; il modello delle formule
# (PP-FormulaNet, 321 MB) resta fuori: per atti e contratti non serve.
# PII_BUNDLE_OCR=0 -> build senza OCR (le scansioni restituiscono "modelli OCR non installati").
OCR_DIR = "models/ocr"
if _os.environ.get("PII_BUNDLE_OCR", "1") != "0":
    _ocr = [m for m in sorted(_os.listdir(OCR_DIR)) if _os.path.isfile(f"{OCR_DIR}/{m}/inference.onnx")
            and m != "PP-FormulaNet_plus-S"] if _os.path.isdir(OCR_DIR) else []
    if not any(m.startswith("PP-DocLayout") for m in _ocr) or "PP-OCRv6_small_rec" not in _ocr:
        raise SystemExit("mancano i modelli OCR in models/ocr: python scripts/fetch_ocr_models.py "
                         "(oppure PII_BUNDLE_OCR=0 per una build senza OCR)")
    for _m in _ocr:
        datas += [(f"{OCR_DIR}/{_m}/{_f}", f"ocr_models/{_m}") for _f in ("inference.onnx", "inference.yml")]
datas += [("src/app/ui", "ui")]           # la UI (index.html) servita da app.py
hiddenimports += ["ocr", "docmd", "docmd.pipeline", "docmd.layout", "docmd.text", "docmd.table",
                  "docmd.reading_order", "docmd.common", "cv2", "pyclipper", "yaml"]

excludes = [
    "tensorflow", "tensorflow_intel", "tf_keras", "keras", "jax", "jaxlib", "flax",
    "vllm", "FlagEmbedding", "flagembedding", "torchvision", "torchaudio",
    "torch", "transformers", "safetensors", "accelerate", "sklearn", "sympy",
    "matplotlib", "pandas", "scipy", "IPython", "notebook", "PyQt5", "PySide2",
    "docmd.formula", "PIL",     # ramo formule di docmd: escluso insieme al suo modello
]

a = Analysis(
    ["src/app/serve.py"],
    pathex=["src/app"],          # cosi' PyInstaller trova il modulo 'app' importato da serve.py
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=excludes,
    noarchive=False,
)
# opencv-python-headless porta con se' ffmpeg (~25 MB) per i video: all'OCR non serve
a.binaries = [b for b in a.binaries if "opencv_videoio_ffmpeg" not in b[0]]
# e i classificatori Haar (riconoscimento volti, ~10 MB di XML): all'OCR non servono
a.datas = [d for d in a.datas if not d[0].replace("\\", "/").startswith("cv2/data/")]
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="pii-backend",
    console=False,              # windowed: nessuna finestra console (i log vanno su file)
    disable_windowed_traceback=False,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="pii-backend",
)
