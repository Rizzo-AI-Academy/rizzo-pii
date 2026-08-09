# Third-party licenses

This file lists everything that ships **inside the released binaries**, and under which licence.
It is a list of verifiable facts, not legal advice.

The short version:

| What you get | Licence |
| --- | --- |
| **This repository (source code)** | **MIT** — see [LICENSE](LICENSE) |
| **Released binaries** (`.exe`, `.dmg`, `.deb`, `.AppImage`) | **AGPL-3.0**, because they bundle PyMuPDF |
| **Docker image** | you build it yourself (`docker build`), so nothing combined is distributed by us — but the image you build contains PyMuPDF and is therefore AGPL-3.0 if you pass it on |

Nothing in this project is *relicensed* by the above. The source code is MIT everywhere, always.
The binaries are AGPL-3.0 because an executable that embeds an AGPL library is a combined work,
and the licence of the whole follows the strictest component. Take `src/` on its own and it is MIT.

## Why the binaries are AGPL-3.0

[PyMuPDF](https://pypi.org/project/PyMuPDF/) is dual-licensed: **GNU AGPL-3.0 or a commercial
licence from Artifex**. We use it for the part of the app that matters most — `page.apply_redactions()`,
which removes the glyphs from the PDF content stream instead of drawing a black rectangle over them —
plus page rendering for the preview and character-level text extraction for precise matching.

It is compiled into the shipped executables: [`build_sidecar.spec:19`](build_sidecar.spec#L19)
lists `fitz` in `hiddenimports`, and every platform build script
([`build_linux.sh`](build_linux.sh), [`build_macos.sh`](build_macos.sh), [`build_mac.sh`](build_mac.sh))
runs `pyinstaller build_sidecar.spec`.

So: the artefacts published under **Releases** are conveyed under the terms of the AGPL-3.0.
The corresponding source required by §6 is this repository, which is public.

### What the AGPL does *not* do

- **It does not touch your documents.** Copyleft applies to the software, not to the output.
  A law firm anonymising a contract with this app owes nobody anything and publishes nothing.
- **It does not forbid commercial use.** You may use the binaries at work, and you may sell them.
  What you may not do is distribute them *closed* — recipients must get the corresponding source.
- **It does not affect you if you install from source.** Clone the repo, `pip install -r requirements.txt`,
  and you assemble the combination on your own machine without distributing anything.

### What it does require

- **If you redistribute a binary** (yours or ours, modified or not), ship the corresponding source
  under the AGPL-3.0, including your modifications.
- **§13, the network clause:** if you run the app as a service reachable by other people
  (`--host 0.0.0.0`, or Docker on an office server), you must offer the corresponding source to
  those remote users. Running an unmodified build satisfies this by pointing at this repository.
  The default bind is `127.0.0.1`, so ordinary desktop use never triggers it.
- **If you want to ship a closed product** built on this, either replace PyMuPDF or buy the
  [Artifex commercial licence](https://artifex.com/licensing/) — that is precisely what the dual
  licensing is for.

## Python dependencies bundled in the binaries

Collected by [`build_sidecar.spec`](build_sidecar.spec) and installed by the platform build scripts.
Licences as declared on PyPI.

| Package | Licence | Note |
| --- | --- | --- |
| PyMuPDF (`fitz`) | **AGPL-3.0 or Artifex Commercial** | the reason this file exists |
| torch | Apache-2.0 (with BSD-2-Clause / LLVM-exception parts) | CPU build |
| transformers | Apache-2.0 | |
| tokenizers | Apache-2.0 | |
| safetensors | Apache-2.0 | |
| huggingface_hub | Apache-2.0 | |
| regex | Apache-2.0 AND CNRI-Python | |
| requests | Apache-2.0 | |
| packaging | Apache-2.0 OR BSD-2-Clause | |
| numpy | BSD-3-Clause (with 0BSD / MIT / Zlib / CC0 parts) | |
| Flask | BSD-3-Clause | |
| Werkzeug, Jinja2, Click, ItsDangerous, MarkupSafe | BSD-3-Clause | Flask's dependencies |
| gunicorn | MIT | Docker image only |
| filelock | MIT | |
| PyYAML | MIT | |
| tqdm | MPL-2.0 AND MIT | |
| certifi | MPL-2.0 | |

The Apache-2.0 packages require their `NOTICE` files to be reproduced on redistribution (§4);
they are preserved inside the bundle by PyInstaller's `collect_all`.

## Desktop shell

| Component | Licence |
| --- | --- |
| [Tauri](https://tauri.app) v2 and its Rust crates | MIT OR Apache-2.0 |
| serde, serde_json | MIT OR Apache-2.0 |
| WebView2 (Windows) / WKWebView (macOS) / WebKitGTK (Linux) | system components, not redistributed by us |

## Model weights and training data

The binaries also embed the trained model: [`build_sidecar.spec:17`](build_sidecar.spec#L17) bundles
`models/rizzo-pii-0.3B-v1.5.0` into the executable. So the terms below travel with the binaries too,
not only with the weights published on Hugging Face.

| Artefact | Licence as declared |
| --- | --- |
| [rizzoaiacademy/rizzo-pii-0.3B](https://huggingface.co/rizzoaiacademy/rizzo-pii-0.3B) (our weights) | MIT |
| [jhu-clsp/mmBERT-base](https://huggingface.co/jhu-clsp/mmBERT-base) (backbone) | MIT |
| [ai4privacy/open-pii-masking-500k-ai4privacy](https://huggingface.co/datasets/ai4privacy/open-pii-masking-500k-ai4privacy) | card declares `license: other` with `license_name: cc-by-4.0` — attribution required |
| [DeepMount00/pii-masking-ita](https://huggingface.co/datasets/DeepMount00/pii-masking-ita) | **no licence declared on the dataset card** |

Two honest caveats:

- **DeepMount00/pii-masking-ita states no licence.** Absent a declaration, redistribution terms are
  undefined. We do not redistribute the dataset — only weights trained on it, alongside three other
  sources — but anyone planning to redistribute that data should ask the author first.
- The Ai4Privacy card is tagged `other` while naming CC-BY-4.0. Under CC-BY-4.0 commercial use is
  fine and attribution is mandatory; this file and the README are that attribution.

## Reproducing this list

```bash
pip download -r requirements.txt --no-deps -d /tmp/deps   # or inspect the build venv
pip-licenses --format=markdown                            # if you have pip-licenses installed
```

Licences were last verified against PyPI and the Hugging Face API on **2026-08-09**.
If a dependency changes licence, this file is the thing to update — and if PyMuPDF is ever removed
from the bundle, note that the binaries *already published* remain AGPL-3.0: that cannot be undone
retroactively.
