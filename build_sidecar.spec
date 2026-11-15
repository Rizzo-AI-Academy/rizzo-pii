# -*- mode: python ; coding: utf-8 -*-
# PyInstaller spec per il BACKEND SIDECAR dell'app Tauri (build CPU, windowed).
# Entry: src/app/serve.py (solo server Flask, nessun browser).
# Build:  build_env\Scripts\pyinstaller.exe build_sidecar.spec --noconfirm \
#           --distpath tauri/src-tauri/backend
# Output: tauri/src-tauri/backend/pii-backend/pii-backend.exe (+ _internal, modello, assets)
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
hiddenimports += ["fitz", "flask", "onnx_ner"]

excludes = [
    "tensorflow", "tensorflow_intel", "tf_keras", "keras", "jax", "jaxlib", "flax",
    "vllm", "FlagEmbedding", "flagembedding", "torchvision", "torchaudio",
    "torch", "transformers", "safetensors", "accelerate", "sklearn", "sympy",
    "matplotlib", "pandas", "scipy", "IPython", "notebook", "PyQt5", "PySide2",
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
