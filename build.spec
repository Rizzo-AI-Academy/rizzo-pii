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
datas += [("models/rizzo-pii-0.3B-v1.5.0", "pii_model")]
datas += [("src/app/assets", "assets")]   # mascotte/icone -> app.py le serve da _resource_path("assets")
# pytesseract/Pillow: pdf_text li importa SOLO quando serve l'OCR -> vanno dichiarati
# qui o dentro l'exe l'OCR non parte ("Installa le dipendenze di requirements.txt").
hiddenimports += ["fitz", "flask", "sklearn.utils._typedefs", "pytesseract", "PIL"]

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

# I metadati *.dist-info\licenses\... di torch annidano percorsi oltre i 260 caratteri di
# Windows: la copia fallisce su cartelle profonde e l'installer (ISCC) non li raggiunge.
# Sono solo testi di licenza, mai letti a runtime: si scartano qui, alla fonte.
import re as _re
_RE_LICENSES = _re.compile(r"\.dist-info[/\\]licenses([/\\]|$)", _re.IGNORECASE)
a.datas = [t for t in a.datas if not _RE_LICENSES.search(t[1])]

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

# Cintura di sicurezza: qualunque cartella licenses di metadati sfuggita al filtro
# sopra viene rimossa dal disco dopo l'assemblaggio (robocopy/ISCC non la reggono).
import shutil as _shutil
from pathlib import Path as _P
_internal = _P(DISTPATH) / "AnonimizzatorePII" / "_internal"
if _internal.is_dir():
    for _lic in _internal.glob("*.dist-info/licenses"):
        _shutil.rmtree(_lic, ignore_errors=True)
