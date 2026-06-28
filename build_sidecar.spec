# -*- mode: python ; coding: utf-8 -*-
# PyInstaller spec per il BACKEND SIDECAR dell'app Tauri (build CPU, windowed).
# Entry: src/app/serve.py (solo server Flask, nessun browser).
# Build:  build_env\Scripts\pyinstaller.exe build_sidecar.spec --noconfirm \
#           --distpath tauri/src-tauri/backend
# Output: tauri/src-tauri/backend/pii-backend/pii-backend.exe (+ _internal, modello, assets)
from PyInstaller.utils.hooks import collect_all

datas, binaries, hiddenimports = [], [], []
for pkg in ("transformers", "tokenizers", "safetensors", "huggingface_hub", "regex"):
    d, b, h = collect_all(pkg)
    datas += d
    binaries += b
    hiddenimports += h

# modello + asset grafici impacchettati (app.py li risolve via _resource_path / _MEIPASS)
datas += [("models/rizzo-pii-0.3B-v1.2.0", "pii_model")]
datas += [("src/app/assets", "assets")]
hiddenimports += ["fitz", "flask", "sklearn.utils._typedefs"]

excludes = [
    "tensorflow", "tensorflow_intel", "tf_keras", "keras", "jax", "jaxlib", "flax",
    "vllm", "FlagEmbedding", "flagembedding", "torchvision", "torchaudio",
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
