# -*- mode: python ; coding: utf-8 -*-
# PyInstaller spec per l'app desktop CPU. Build: pyinstaller build.spec --noconfirm
from PyInstaller.utils.hooks import collect_all

datas, binaries, hiddenimports = [], [], []
# raccoglie codice + dati delle librerie con import dinamici
# Inferenza con ONNX Runtime (src/app/onnx_ner.py): NIENTE torch/transformers nel pacchetto.
for pkg in ("onnxruntime", "tokenizers"):
    d, b, h = collect_all(pkg)
    datas += d
    binaries += b
    hiddenimports += h

# il modello addestrato va incluso nel pacchetto (sorgente: models/rizzo-pii-0.3B,
# destinazione dentro l'exe: "pii_model" -> app.py lo risolve via _resource_path).
# Finche' rizzo-pii-0.3B non e' stato addestrato si puo' usare models/pii_model_legacy.
# Del modello si impacchetta SOLO cio' che serve a onnxruntime: il model.onnx (fp32, gli
# stessi pesi di PyTorch, prodotto da src/training/export_onnx.py) + tokenizer e config.
# Il model.safetensors (1,2 GB) resta fuori: con lui dentro il modello sarebbe doppio.
MODEL_DIR = "models/rizzo-pii-0.3B-v1.5.0"
import os as _os
if not _os.path.isfile(f"{MODEL_DIR}/onnx/model.onnx_data"):
    raise SystemExit(f"manca {MODEL_DIR}/onnx/model.onnx(_data): "
                     f"python src/training/export_onnx.py --model-dir {MODEL_DIR}")
# model.onnx = grafo, model.onnx_data = pesi (mappati in memoria): sempre insieme.
datas += [(f"{MODEL_DIR}/onnx/model.onnx", "pii_model"),
          (f"{MODEL_DIR}/onnx/model.onnx_data", "pii_model")]
datas += [(f"{MODEL_DIR}/{f}", "pii_model")
          for f in ("config.json", "tokenizer.json", "tokenizer_config.json",
                    "special_tokens_map.json")]
datas += [("src/app/assets", "assets")]   # mascotte/icone -> app.py le serve da _resource_path("assets")
hiddenimports += ["fitz", "flask", "onnx_ner"]

# escludi tutto cio' che non serve (riduce dimensione e rumore)
excludes = [
    "tensorflow", "tensorflow_intel", "tf_keras", "keras", "jax", "jaxlib", "flax",
    "vllm", "FlagEmbedding", "flagembedding", "torchvision", "torchaudio",
    "torch", "transformers", "safetensors", "accelerate", "sklearn", "sympy",
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
