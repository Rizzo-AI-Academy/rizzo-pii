# -*- mode: python ; coding: utf-8 -*-
# PyInstaller spec per l'app desktop CPU. Build: pyinstaller build.spec --noconfirm
from PyInstaller.utils.hooks import collect_all

datas, binaries, hiddenimports = [], [], []
# raccoglie codice + dati delle librerie con import dinamici
for pkg in ("transformers", "tokenizers", "safetensors", "huggingface_hub", "regex"):
    d, b, h = collect_all(pkg)
    datas += d
    binaries += b
    hiddenimports += h

# il modello addestrato va incluso nel pacchetto (sorgente: models/rizzo-pii-0.3B,
# destinazione dentro l'exe: "pii_model" -> app.py lo risolve via _resource_path).
# Finche' rizzo-pii-0.3B non e' stato addestrato si puo' usare models/pii_model_legacy.
datas += [("models/rizzo-pii-0.3B-v1.2.0", "pii_model")]
datas += [("src/app/assets", "assets")]   # mascotte/icone -> app.py le serve da _resource_path("assets")
hiddenimports += ["fitz", "flask", "sklearn.utils._typedefs"]

# escludi tutto cio' che non serve (riduce dimensione e rumore)
excludes = [
    "tensorflow", "tensorflow_intel", "tf_keras", "keras", "jax", "jaxlib", "flax",
    "vllm", "FlagEmbedding", "flagembedding", "torchvision", "torchaudio",
    "matplotlib", "pandas", "scipy", "IPython", "notebook", "PyQt5", "PySide2",
]

a = Analysis(
    ["src/app/desktop_app.py"],
    pathex=["src/app"],          # cosi' PyInstaller trova il modulo 'app' importato da desktop_app
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
    name="AnonimizzatorePII",
    console=True,            # True per vedere i log; metti False per nasconderli
    disable_windowed_traceback=False,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="AnonimizzatorePII",
)
