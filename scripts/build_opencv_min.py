# -*- coding: utf-8 -*-
"""
Wheel di OpenCV MINIMA per l'OCR dell'app (src/app/docmd/ usa solo core + imgproc).

La wheel ufficiale opencv-python-headless porta con se' video (ffmpeg, aom, vpx),
OpenBLAS/LAPACK, OpenCL e una ventina di moduli: su Linux sono ~143 MB scompattati
e il modulo cv2 li carica TUTTI all'avvio (DT_NEEDED), quindi non si possono togliere
dopo. L'unico modo di alleggerire e' compilarla con i soli moduli che servono:

    core, imgproc + binding Python   -> resize, cvtColor, warpAffine/Perspective,
                                        findContours, minAreaRect, fillPoly, mean...

Il sorgente e' l'sdist ufficiale di opencv-python-headless su PyPI (hash sha256
verificato). Il risultato e' una wheel opencv_python_headless-<ver>-...whl che si
installa al posto di quella ufficiale prima di PyInstaller (vedi docs/BUILD.md e i
workflow di build). imgcodecs non c'e': cv2.imencode/imdecode NON esistono (l'app
legge e scrive le immagini con PyMuPDF).

Uso:
    python scripts/build_opencv_min.py                     # -> build/opencv_min/*.whl
    python scripts/build_opencv_min.py --out dist_cv --version 4.12.0.88

Serve un compilatore C++ (MSVC su Windows, gcc/clang su Linux/macOS) e CMake.
Circa 5-10 minuti; la CI tiene la wheel in cache.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VERSION = "4.12.0.88"

# Moduli e funzionalita'. Tutto il resto e' spento: niente video, niente BLAS/LAPACK,
# niente OpenCL, niente codec di immagini, niente G-API/DNN/Java.
CMAKE_FLAGS = [
    "-DBUILD_LIST=core,imgproc,python3",
    "-DPYTHON3_LIMITED_API=ON",          # deve comparire letterale: setup.py lo cerca
    "-DWITH_FFMPEG=OFF", "-DWITH_GSTREAMER=OFF", "-DWITH_V4L=OFF", "-DWITH_MSMF=OFF",
    "-DWITH_DSHOW=OFF", "-DWITH_AVFOUNDATION=OFF", "-DWITH_OBSENSOR=OFF",
    "-DWITH_LAPACK=OFF", "-DWITH_EIGEN=OFF", "-DWITH_OPENCL=OFF", "-DWITH_VA=OFF",
    "-DWITH_VA_INTEL=OFF", "-DWITH_CUDA=OFF", "-DWITH_ITT=OFF", "-DWITH_PROTOBUF=OFF",
    "-DWITH_ADE=OFF", "-DWITH_OPENEXR=OFF", "-DBUILD_OPENEXR=OFF", "-DWITH_JPEG=OFF",
    "-DWITH_PNG=OFF", "-DWITH_TIFF=OFF", "-DWITH_WEBP=OFF", "-DWITH_OPENJPEG=OFF",
    "-DWITH_JASPER=OFF", "-DWITH_AVIF=OFF", "-DWITH_IMGCODEC_HDR=OFF",
    "-DWITH_IMGCODEC_SUNRASTER=OFF", "-DWITH_IMGCODEC_PXM=OFF", "-DWITH_IMGCODEC_PFM=OFF",
    "-DWITH_GTK=OFF", "-DWITH_QT=OFF", "-DWITH_WIN32UI=OFF", "-DWITH_VTK=OFF",
    "-DBUILD_opencv_java=OFF", "-DBUILD_JAVA=OFF", "-DBUILD_TESTS=OFF",
    "-DBUILD_PERF_TESTS=OFF", "-DBUILD_EXAMPLES=OFF", "-DBUILD_DOCS=OFF",
    "-DENABLE_PRECOMPILED_HEADERS=OFF", "-DCV_TRACE=OFF",
    # IPP (Intel, statica): +27 MB nel modulo e nessun guadagno misurabile sull'OCR
    # (1,45 s/pagina con e senza, testo identico byte per byte): spenta.
    "-DWITH_IPP=OFF",
]


def sdist(version: str, dest: Path) -> Path:
    meta = json.load(urllib.request.urlopen(
        f"https://pypi.org/pypi/opencv-python-headless/{version}/json", timeout=60))
    src = next(u for u in meta["urls"] if u["packagetype"] == "sdist")
    path = dest / src["filename"]
    if not path.exists() or hashlib.sha256(path.read_bytes()).hexdigest() != src["digests"]["sha256"]:
        print(f"scarico {src['url']} ({src['size'] / 1e6:.0f} MB)")
        tmp = path.with_suffix(".part")
        with urllib.request.urlopen(src["url"], timeout=600) as r, open(tmp, "wb") as f:
            shutil.copyfileobj(r, f)
        tmp.replace(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != src["digests"]["sha256"]:
        sys.exit(f"sha256 non corrisponde per {path.name}")
    return path


def main() -> int:
    ap = argparse.ArgumentParser(description="Compila una wheel OpenCV minima (core + imgproc).")
    ap.add_argument("--version", default=DEFAULT_VERSION)
    ap.add_argument("--out", type=Path, default=ROOT / "build" / "opencv_min")
    ap.add_argument("--work", type=Path, default=ROOT / "build" / "opencv_min_src")
    ap.add_argument("--jobs", type=int, default=os.cpu_count() or 2)
    ap.add_argument("--cmake-arg", action="append", default=[],
                    help="flag CMake in piu' (ripetibile), es. -DWITH_IPP=OFF")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    args.work.mkdir(parents=True, exist_ok=True)

    tgz = sdist(args.version, args.work)
    src = args.work / f"opencv-python-headless-{args.version}"
    if not src.exists():
        print(f"estraggo {tgz.name}")
        with tarfile.open(tgz) as t:
            t.extractall(args.work, filter="data") if hasattr(tarfile, "data_filter") \
                else t.extractall(args.work)

    # setup.py copia nella wheel file che una build minima non produce, e fallisce a
    # compilazione finita se ne manca uno: la DLL di ffmpeg (Windows) e i file di tipi
    # (py.typed, cv2/typing: il generatore degli stub vuole imgcodecs). Via le tre regole.
    setup_py = src / "setup.py"
    text = setup_py.read_text(encoding="utf-8")
    for old, new in (
        ('[r"bin/opencv_videoio_ffmpeg\\d{4}%s\\.dll" % ("_64" if is64 else "")]\n'
         '            if os.name == "nt"', "[]\n            if False"),
        ('[ r"python/cv2/py.typed" ] if sys.version_info', "[] if sys.version_info"),
        ('rearrange_cmake_output_data["cv2.typing"] = ["python/cv2" + r"/typing/.*\\.py"]',
         "pass  # build minima: niente cv2/typing"),
    ):
        if old in text:
            text = text.replace(old, new)
        elif new not in text:
            sys.exit(f"setup.py di opencv-python cambiato, regola non trovata: {old[:50]}")
    setup_py.write_text(text, encoding="utf-8")

    # build sempre da zero: la cache CMake di un giro precedente punta agli header di
    # numpy dell'ambiente di build temporaneo di pip, che nel frattempo non esiste piu'
    shutil.rmtree(src / "_skbuild", ignore_errors=True)

    env = dict(os.environ)
    env.pop("CI", None)              # setup.py in "modalita' CI" riaccende LAPACK e V4L
    env["ENABLE_HEADLESS"] = "1"
    env["CMAKE_ARGS"] = " ".join(CMAKE_FLAGS + args.cmake_arg)
    env["CMAKE_BUILD_PARALLEL_LEVEL"] = str(args.jobs)
    # Windows: il generatore lo sceglie scikit-build (>= 0.19 prova VS 2026, poi 2022 con
    # i set di strumenti v144/v143): non forzarlo, i runner windows-latest hanno solo VS
    # 2026 e una macchina di sviluppo puo' avere solo il 2022. MSBuild invece non regge i
    # percorsi lunghi: tieni --work corto (default: build/ nella repo).
    for old in args.out.glob("opencv_python_headless-*.whl"):
        old.unlink()
    cmd = [sys.executable, "-m", "pip", "wheel", str(src), "--no-deps", "-w", str(args.out), "-v"]
    print("+", " ".join(cmd))
    subprocess.run(cmd, check=True, env=env)

    wheels = list(args.out.glob("opencv_python_headless-*.whl"))
    if not wheels:
        sys.exit("nessuna wheel prodotta")
    for w in wheels:
        print(f"OK {w}  ({w.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
