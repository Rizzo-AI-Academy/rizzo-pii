<div align="center">

<img src="report/images/mascot_shield.png" alt="rizzo-pii mascot — a purple hedgehog guarding a document with a shield" width="180" />

# rizzo-pii

### Local, reversible PII anonymization for Italian legal text

**_Use frontier models without giving up your data._**

<p>
<img src="https://img.shields.io/badge/100%25-LOCAL-7c3aed?style=for-the-badge" alt="100% local" />
<img src="https://img.shields.io/badge/GDPR-BY%20DESIGN-7c3aed?style=for-the-badge" alt="GDPR by design" />
<img src="https://img.shields.io/badge/EU%20AI%20ACT-ALIGNED-7c3aed?style=for-the-badge" alt="EU AI Act aligned" />
</p>

<p>
<img src="https://img.shields.io/badge/params-%E2%89%880.3B-blue" alt="0.3B parameters" />
<img src="https://img.shields.io/badge/RAM-~0.5%20GB%20·%20CPU-blue" alt="0.5 GB RAM CPU" />
<img src="https://img.shields.io/badge/PII%20categories-22-blue" alt="22 PII categories" />
<img src="https://img.shields.io/badge/micro--F1-0.989-brightgreen" alt="0.989 micro-F1" />
<img src="https://img.shields.io/badge/offline-no%20API%20key-brightgreen" alt="offline, no API key" />
</p>

**📄 [Read the full technical report (PDF)](report/rizzo-pii-report.pdf)** — model, dataset, method and experiments in detail

<p>
<a href="https://github.com/Rizzo-AI-Academy/rizzo-pii/releases/latest"><img src="https://img.shields.io/badge/⬇%20Download%20for-Windows-0078D6?style=for-the-badge&logo=windows&logoColor=white" alt="Download for Windows" /></a>
<a href="https://github.com/Rizzo-AI-Academy/rizzo-pii/releases/latest"><img src="https://img.shields.io/badge/⬇%20Download%20for-macOS-000000?style=for-the-badge&logo=apple&logoColor=white" alt="Download for macOS" /></a>
<a href="https://github.com/Rizzo-AI-Academy/rizzo-pii/releases/latest"><img src="https://img.shields.io/badge/⬇%20Download%20for-Linux-FCC624?style=for-the-badge&logo=linux&logoColor=black" alt="Download for Linux" /></a>
</p>

<sub>🪟 **Windows installer** · 🍎 **macOS** (Apple Silicon) · 🐧 **Linux AppImage** — all available now</sub>

</div>

**`rizzo-pii:0.3B`** is a lightweight, CPU-friendly, **Italian-first** token-classification
model (≈0.3B parameters, [mmBERT](https://huggingface.co/jhu-clsp/mmBERT-base) / ModernBERT
backbone) that detects **22 categories of personal data** — including the Italian-legal
identifiers (**codice fiscale**, **partita IVA**, **dati catastali**) that no other open model
covers — and drives a fully **reversible** anonymization workflow:

<div align="center">

🔒 **anonymize locally** → 🏷️ **placeholder + reversible local dictionary** → ☁️ **frontier LLM** → 🔓 **restore locally**

</div>

It is built for law firms, accountants, notaries and anyone bound by the **GDPR** who wants to
keep using ChatGPT / Claude / Gemini on sensitive documents **without ever sending the real data
out**.

<table>
<tr>
<td width="120" align="center" valign="middle">
<img src="report/images/mascot_doc.png" alt="rizzo-pii redacting a document" width="110" />
</td>
<td valign="middle">

| ≈0.3B | ~0.5 GB | 22 | 0.989 |
|:---:|:---:|:---:|:---:|
| parameters (mmBERT-base) | RAM footprint, CPU | PII categories | micro-F1 (real IT validation) |

The hedgehog mascot does one job: it grabs your document, blacks out every identifier, and
**stays inside the EU** while doing it.

</td>
</tr>
</table>

---

## The problem: convenience is leaking your data

People summarize contracts, draft replies and ask legal questions simply by **pasting the
document in**. It is fast and useful — and it quietly moves enormous amounts of personal and
confidential data off the user's device. Names, addresses, tax codes, IBANs, health details,
unsigned-contract clauses: all of it crosses the network to servers the user does not control,
where it may be logged, cached, retained or exposed in a breach. For a law firm or a hospital this
is not hypothetical; under the GDPR it can be a direct compliance failure.

The intuitive fix is to **stop sending data out** and run an open model locally — but a
frontier-grade open model is large and expensive to serve (€9,000–€10,000 of hardware), and the
small models that fit a normal laptop are **not** in the same league on the hard tasks (legal
reasoning, dense contracts, long official documents) — exactly where Italian professionals need
the most help.

**The trade-off we actually want:** keep the frontier model and *remove the data from the
equation*. Anonymize the document **locally** on a CPU, send only placeholders to the cloud, and
restore the real values **locally** from the answer. The sensitive content never leaves the
machine.

---

## rizzo-pii in one picture

The workflow has three local steps and one remote step. Locally, rizzo-pii tags every span of
personal data and replaces each one with a stable, type-aware placeholder
(`[FULLNAME_1]`, `[IBAN_1]`, `[CF_1]`), recording the mapping `placeholder → real value` in a
dictionary that **stays on disk**. Identical values share the same placeholder, so the frontier
model still sees a coherent text and can reason about it. The anonymized text is sent to
ChatGPT / Claude / Gemini; when the answer comes back, a local pass swaps the placeholders for the
true values. **The cloud provider never receives a single real name, code or number.**

<div align="center">
<img src="report/images/schema_explainable.png" alt="The rizzo-pii workflow: everything runs locally on CPU; only placeholder text crosses to the cloud, and the answer is re-identified locally" width="780" />
</div>

Everything except the frontier query happens on the user's CPU; only placeholder text crosses the
boundary, and the answer is re-identified locally.

---

## Why this is different: privacy that is actually private

This is not "yet another PII detector". It is an **architecture for using powerful models without
surrendering data**, built so the privacy guarantee is structural rather than a promise:

- **The data never leaves the device.** Detection and re-identification run locally on a CPU.
  No API key, no telemetry, no upload. What the cloud receives is already stripped of identifiers.
- **GDPR by design.** The workflow implements **data minimization** (Art. 5) almost literally:
  the third-party processor only ever sees pseudonymized text, so the most common reason a cloud
  LLM call is unlawful (transferring identifiable data to a third party without a basis) is removed
  at the source.
- **Aligned with the EU AI Act.** Keeping personal data under local control and out of
  third-party model pipelines supports the Act's emphasis on data governance.
- **Accessible to everyone.** The model is ≈0.3B parameters and runs on a CPU in well under 1 GB
  of RAM — the privacy layer costs nothing extra in hardware. A normal laptop is enough.
- **Reversible, not destructive.** Classic redaction throws information away. rizzo-pii
  **pseudonymizes**: the answer from the frontier model is reconstructed with the real values, so
  the tool is useful in real work, not just compliance theater.

### How it compares

| Property | **rizzo-pii:0.3B** | OpenAI Privacy Filter | MS Presidio |
|---|---|---|---|
| Type | Dense encoder (mmBERT / ModernBERT) | Sparse MoE encoder | NER + rules pipeline |
| Parameters | ≈0.3B dense (all active) | 1.5B total / ≈50M active | spaCy model + rules |
| Memory to load | 0.5–1.2 GB | 1.5B params resident | Varies (spaCy) |
| Runs on | CPU, under 1 GB RAM | On-device | CPU |
| Categories | **22 (incl. IT-legal)** | 8 generic | Configurable; EN defaults |
| Italian CF / PIVA / catasto | **Yes** | No | Not by default |
| Primary language | Italian (+7 more) | English | English |
| Checksum validation | **Yes** (IBAN/CF/PIVA/card) | No | Some recognizers |
| Reversible mapping | **Yes** (local dict) | Masking | Anonymization |

The differentiators are **Italian-legal coverage**, a **smaller memory footprint**, and a
**checksum-backed safety net** (mod-97 for IBAN, Luhn for cards, the official CF/PIVA algorithms)
that the larger generic models do not provide.

> **Concrete example.** Take *"Il Sig. Mario Rossi, C.F. RSSMRA85H12F205Z, P.IVA 12345678901, è
> titolare dell'immobile al Foglio 12, particella 345, sub. 6."* rizzo-pii tags `FULLNAME`, `CF`,
> `PIVA` and `CATASTO` and rewrites it as *"Il Sig. [FULLNAME_1], C.F. [CF_1], P.IVA [PIVA_1], è
> titolare dell'immobile al [CATASTO_1]."* A generic English-first model has no label for the
> fiscal code, the VAT number or the cadastral reference — the three most sensitive identifiers in
> the sentence — and would leave them in the clear.

---

## The taxonomy: 22 tags, and why

rizzo-pii predicts 22 entity types in **BIO** format (a `B-`/`I-` label per tag, plus `O`). The
raw datasets are left untouched; the mapping to these 22 tags is applied **at load time** through a
single `TAG_MAP` in `train_pii.py`, so the taxonomy can be changed in one place without
re-annotating anything. Details in **[docs/TASSONOMIA_TAG.md](docs/TASSONOMIA_TAG.md)**.

| Tag | Meaning | Example | Source |
|---|---|---|---|
| `FULLNAME` | Person name (incl. legal roles: judge, lawyer, parties, witness) | Mario Rossi | real+synth |
| `AGE` | Age | 45 anni | real |
| `GENDER` | Sex / gender | Femmina | real |
| `DATE` | Calendar date | 12/06/1985 | real+synth |
| `TIME` | Time of day | ore 15:30 | real |
| `STREET` | Street / square | Via Garibaldi | real+synth |
| `BUILDINGNUM` | Street number | 24 | real+synth |
| `ZIPCODE` | Postal code (CAP) | 00185 | real+synth |
| `CITY` | City | Milano | real+synth |
| `PROVINCE` | Province abbreviation | MI | synth |
| `EMAIL` | E-mail (incl. PEC) | m.rossi@studio.it | real+synth |
| `TELEPHONENUM` | Phone number | +39 333 1234567 | real+synth |
| `CF` | Codice fiscale (personal tax code) | RSSMRA85H12F205Z | synth |
| `PIVA` | Partita IVA (VAT number) | 12345678901 | real+synth |
| `ID_DOC` | ID / passport / licence / social number | CA12345AB | real+synth |
| `IBAN` | IBAN / bank account | IT60X05428… | synth |
| `CREDITCARDNUMBER` | Credit-card number | 4111 1111 1111 1111 | real |
| `AMOUNT` | Money amount | € 12.500,00 | synth |
| `TARGA` | Vehicle plate | AB 123 CD | synth |
| `ORG` | Private company / firm / bank | Edilnord S.r.l. | synth |
| `DOCID` | Act identifier (RG, protocol, repertory, ruling) | 1234/2024 | synth |
| `CATASTO` | Cadastral data (sheet, parcel, sub.) | Foglio 12, part. 345 | synth |

**Two design decisions stand out.** First, **legal roles collapse into `FULLNAME`** — whether
"Mario Rossi" is the judge, the lawyer or a witness is not a property of the string; the role, if
needed, is recovered downstream as metadata. Second, **raw types that mean the same thing are
merged**: names + surnames → `FULLNAME`; `PEC` → `EMAIL`; `TAXNUM` → `PIVA`; ID card / passport /
licence / social number → `ID_DOC`; account number → `IBAN`. Honorifics (`Dott.`, `Avv.`) and the
name of the **court** itself are dropped to `O` because they are not identifiers to mask.

The five Italian-legal tags (`CF`, `PIVA`, `CATASTO`, `DOCID`, `PROVINCE`) are the reason
rizzo-pii exists: they do not appear as labeled data in any public corpus, so they are created
through synthesis with mathematically valid checksums.

---

## Dataset & training

The model is fine-tuned on a **multilingual** pool of ≈**745k** labeled rows (Italian reinforced to
~45%) assembled from four sources — real (Ai4Privacy, DeepMount) and synthetic — all remapped to the
22 tags at load time. The synthetic part follows the **"LLM author, code labeler"** principle: an
LLM writes only Italian legal prose with placeholders (`{SLOT}`) and our code injects the real
values (CF/PIVA/IBAN with valid checksums), so BIO labels are exact, identifiers are valid by
construction, and **no real personal data is ever produced by the LLM**. The backbone is
**mmBERT-base** (ModernBERT architecture, native 8192-token context) and training runs in a single
epoch on one 16 GB consumer GPU.

> 📄 **The full dataset composition, the synthesis method, the training recipe and the
> experiments are described in the [technical report](report/rizzo-pii-report.pdf).** See also
> [docs/DATASET.md](docs/DATASET.md) and [CLAUDE.md](CLAUDE.md) for the operational details.

---

## Results

Training was a single epoch (~26.6k steps over 744,912 rows). The loss fell from ≈6.2 to below 0.1
within the first few hundred steps, then settled into a clean, monotone, low regime; the validation
loss decreased monotonically and was **still falling** when the epoch ended. Final training loss
≈**0.003** and validation loss ≈**0.006** are both very low and very close, so the model is **not
over-fitting** — a second epoch would very likely push it lower still.

<div align="center">
<img src="report/images/v12_training_loss_zoom.png" alt="Training loss (smoothed zoom): drops sharply then stays low and stable" width="46%" />
<img src="report/images/v12_valid_loss.png" alt="Validation loss: decreases monotonically across the epoch, still falling at the end" width="46%" />
</div>
<p align="center"><sub><b>Left:</b> training loss (smoothed zoom) — fast drop, then low and stable. <b>Right:</b> validation loss — monotone, still decreasing when training stopped.</sub></p>

On the 7,000-row held-out **real Italian** benchmark (`validation_real.jsonl`):

| 0.987 | 0.990 | 0.989 | 0.998 |
|:---:|:---:|:---:|:---:|
| micro precision | micro recall | micro F1 | token accuracy |

The unweighted per-tag mean (macro-F1) across all 22 tags is **0.987**, and **every one of the five
Italian-legal identifiers scores a perfect 1.000**.

### Per-tag precision / recall / F1 (v1.2.0)

| Tag | Sup. | P | R | F1 | | Tag | Sup. | P | R | F1 |
|---|---:|---:|---:|---:|---|---|---:|---:|---:|---:|
| FULLNAME | 4390 | .989 | .990 | .990 | | GENDER | 472 | 1.00 | 1.00 | 1.00 |
| CATASTO | 1200 | 1.00 | 1.00 | 1.00 | | PROVINCE | 400 | 1.00 | 1.00 | 1.00 |
| CITY | 953 | .961 | .963 | .962 | | DOCID | 400 | 1.00 | 1.00 | 1.00 |
| DATE | 922 | 1.00 | 1.00 | 1.00 | | CF | 400 | 1.00 | 1.00 | 1.00 |
| TELEPHONENUM | 874 | 1.00 | 1.00 | 1.00 | | AGE | 385 | .979 | .977 | .978 |
| ID_DOC | 800 | 1.00 | 1.00 | 1.00 | | ZIPCODE | 299 | .938 | .967 | .952 |
| EMAIL | 748 | .999 | .999 | .999 | | IBAN | 278 | .996 | .996 | .996 |
| TIME | 637 | .991 | .992 | .991 | | CREDITCARD | 257 | .919 | .973 | .945 |
| STREET | 617 | .951 | .969 | .960 | | AMOUNT | 146 | 1.00 | .993 | .997 |
| BUILDINGNUM | 594 | .969 | .958 | .964 | | ORG | 145 | .967 | 1.00 | .983 |
| PIVA | 514 | 1.00 | 1.00 | 1.00 | | TARGA | 43 | 1.00 | 1.00 | 1.00 |

All five Italian-legal identifiers (`CF`, `PIVA`, `CATASTO`, `DOCID`, `PROVINCE`) score a perfect
1.000, as do `ID_DOC`, `DATE`, `TELEPHONENUM`, `GENDER` and `TARGA`. The remaining soft spots are
the open, high-variability classes (`ZIPCODE`, `CREDITCARDNUMBER`, `STREET`, `CITY`) and `ORG`,
which is exactly where a larger, better-balanced dataset would help.

---

## Deployment: it runs on a normal computer

The released checkpoint is ~1.2 GB on disk in fp32 and runs comfortably on a **CPU**: quantized, its
memory footprint is on the order of **0.5 GB**, with **no GPU required**. That is the whole point —
the privacy layer is cheap enough to run on the laptop the user already owns.

In production the neural model is **never used alone**. It is paired with a deterministic **regex +
checksum** network for the structured identifiers (EMAIL, phone, IBAN, CF, PIVA, credit card,
amount, plate), where IBAN/PIVA/card must pass their **checksum** (mod-97 / Luhn) to be accepted,
and a valid checksum **overrides** the model. This eliminates the classic failure mode of a neural
tagger fragmenting a long code, and gives mathematically certain detection for exactly the
identifiers whose leakage is most damaging. The app adds the reversible layer (stable placeholders,
downloadable local dictionary, a "restore" tab tolerant to markdown/format drift), chunking with
overlap for long PDFs, and a colored per-tag UI.

| To use the model (inference) | To retrain the model |
|---|---|
| Any 64-bit CPU (no GPU) | A single 16 GB GPU is enough |
| 0.5–1.2 GB RAM for the model | Reference run: RTX 5060 Ti, ~2 h |
| Windows (installer) / Linux / macOS | PyTorch `cu128` for Blackwell |
| Fully offline; no API key | ~745k rows, regenerable from scripts |

The desktop app **Rizzo PII** (Tauri) launches the Python/Flask backend as a bundled CPU
"sidecar"; a CPU-only PyTorch build keeps it fully **offline** on Windows (WebView2), macOS and Linux.
Packaging instructions in **[docs/BUILD.md](docs/BUILD.md)**.

> **⬇️ Download.** Grab the ready-to-use build from the
> **[Releases page](https://github.com/Rizzo-AI-Academy/rizzo-pii/releases/latest)** — no Python or
> setup required: a **Windows installer** (double-click), a **macOS `.dmg`** (Apple Silicon /
> arm64 — **signed & notarized** by Apple, just open it), and a **Linux AppImage** (`chmod +x` then
> run) are all available now.

---

## Quickstart

> Prerequisites and critical environment constraints (Blackwell GPU, torch **cu128**, etc.) are in
> **[CLAUDE.md](CLAUDE.md)**. The `dataset/raw/` sources are downloaded from Hugging Face (see the
> `hf download` commands in CLAUDE.md). All scripts force UTF-8 and resolve their paths from
> `__file__`, so they run from any working directory.

> 💡 Just want to **use** the app? You don't need any of this — download the ready-to-use build
> (Windows / macOS / Linux) from the
> [Releases page](https://github.com/Rizzo-AI-Academy/rizzo-pii/releases/latest). The steps
> below are for developers who want to regenerate the data and retrain the model.

### 0) Install

```powershell
git clone https://github.com/Rizzo-AI-Academy/rizzo-pii
cd rizzo-pii
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt          # NVIDIA Blackwell? install torch cu128 first — see requirements.txt
copy .env.example .env                    # optional: add W&B / Gemini keys
```

### 1) Generate the data

```powershell
python src/data_pipeline/llm_template_bank.py --per-type 5 --append          # (opt.) legal templates via Gemini
python src/data_pipeline/generate_synthetic_pii.py -n 200000 --out dataset/synthetic/synthetic_pii_it_200k.jsonl
python src/data_pipeline/augment_real_pii.py     -n 40000  --out dataset/synthetic/synthetic_pii_it_realaug.jsonl
python src/data_pipeline/prepare_deepmount.py                                 # requires HF login
python src/data_pipeline/build_validation.py                                 # real validation (7k, it)
python src/data_pipeline/build_subset.py                                     # 10k/5k subsets for smoke tests
```

### 2) Train

```powershell
# fast smoke test / tuning on the subset (~3 min)  -> experiments/subset_smoke/
python src/training/train_pii.py --type subset

# full run on the whole dataset  -> models/rizzo-pii-0.3B-v{VERSION}/ + experiments/full_run_v{VERSION}/
python src/training/train_pii.py --type full
python src/training/train_pii.py --type full --version 1.2.0   # or an explicit version
```

### 3) Use the model

```powershell
python src/training/test_pii.py "Mi chiamo Mario Rossi, IBAN IT60X0542811101000000123456"
python src/app/app.py            # http://127.0.0.1:5005  (paste text or upload a PDF)
```

The web app assigns every PII a **reversible ID** (`[FULLNAME_1]`, `[IBAN_1]`…) plus a local
dictionary, pairing the model with the regex/checksum net. You copy the anonymized text into an LLM
and **restore** the real values from the response.

---

## Repository structure

```
rizzo_pii/
├─ README.md                 this file
├─ LICENSE                   MIT
├─ CONTRIBUTING.md           how to contribute (code, docs, data)
├─ requirements.txt          Python dependencies (see the cu128 note for Blackwell GPUs)
├─ .env.example              template for the optional W&B / Gemini keys
├─ CLAUDE.md                 operating instructions + environment constraints (GPU, CUDA…)
├─ report/                   the technical report (PDF + Typst source)
├─ docs/
│   ├─ DATASET.md            full composition of train/validation
│   ├─ TASSONOMIA_TAG.md     the 22 final tags and the merge decisions
│   ├─ BUILD.md              desktop app build (Tauri recommended + PyInstaller legacy)
│   └─ CHANGELOG.md          change log, with rationale
├─ src/
│   ├─ data_pipeline/        data generation & preparation
│   │   ├─ llm_template_bank.py       Gemini writes legal templates → legal_templates.json
│   │   ├─ generate_synthetic_pii.py  injects checksum-valid values into the templates
│   │   ├─ augment_real_pii.py        injects synthetic entities into real Ai4Privacy sentences
│   │   ├─ prepare_deepmount.py       remaps DeepMount (56 types) onto our 22 tags
│   │   ├─ build_validation.py        builds the single real validation set (it)
│   │   └─ build_subset.py            stratified subsets for smoke tests / tuning
│   ├─ training/
│   │   ├─ train_pii.py               train, evaluate, save the model + metrics
│   │   ├─ evaluate_pii.py            per-tag (P/R/F1) evaluation on validation_real
│   │   └─ test_pii.py                CLI inference on the saved model
│   ├─ inspect/                       read-only utilities (counts, lengths, checksums)
│   └─ app/                           local anonymization app
│       ├─ app.py                     Flask server: reversible anonymization + regex/checksum net
│       ├─ serve.py                   headless entry (backend of the Tauri app, no browser)
│       ├─ desktop_app.py             legacy PyInstaller entry (opens the browser)
│       └─ assets/                    mascot (the hedgehog) and icons
├─ tauri/                    native desktop app (Tauri) + Windows installer — see docs/BUILD.md
├─ dataset/                  (gitignored — regenerable from the scripts)
├─ models/                   (gitignored) trained models, one folder per version
└─ experiments/             (gitignored) run artifacts (logs, plots, metrics, checkpoints)
```

---

## Limitations

Stated plainly:

- **Validation is Italian-only.** Training is multilingual, but the 7,000-row benchmark measures
  Italian only, by design. The other seven languages are trained but not certified.
- **The IT-legal tags are validated against injected entities.** `CF`, `PIVA`, `CATASTO`, `DOCID`,
  `PROVINCE` have no real public data, so even in validation they are generated entities placed into
  real sentences — a good proxy, not a fully blind test.
- **Class imbalance and under-represented categories.** The corpus is heavily skewed (`FULLNAME`
  outnumbers `CREDITCARDNUMBER` ~97×), so the rarer tags are noisier. The clearest case is
  organizations (`ORG`): an open, highly variable class that today comes largely from synthetic
  templates and off-domain Faker data — exactly where a larger, balanced dataset would help most.
- **Off-domain synthetic values.** DeepMount supplies US-style names/addresses: useful for
  form/context, not as Italian values.
- **Evaluation is sentence-level, not document-level.** The benchmark is built from short
  sentences, whereas the real use case is whole documents. Measuring true end-to-end behaviour
  (long context, PDF chunking with overlap, real act structure) needs a dedicated test set of
  large, real Italian documents, which still has to be assembled.

The mitigation that matters in practice: **always pair the model with the regex/checksum safety
net** (`src/inspect/validate_checksums.py` is the blueprint). The two together are stronger than
either alone.

---

## Next step: a community-owned Italian PII dataset

<img src="report/images/mascot_eu_hat.png" alt="rizzo-pii mascot wearing an EU cap, thumbs up" width="150" align="right" />

rizzo-pii proves the thesis: you **can** keep using frontier models and still keep your data
private, on ordinary hardware, with Italian-legal coverage no other open model offers. The single
biggest lever on quality from here is **data** — a large, real, lawfully collected Italian corpus:
both for the legal identifiers that are scarce today and, above all, to **balance the classes** and
add genuine coverage where the model is weakest (organizations), plus a **test set of large, real
documents** so the model can be measured end-to-end on the documents it is actually meant to
anonymize.

So the project is **open source**, and this is an open invitation. The community dataset lives on
Hugging Face at
**[`rizzoaiacademy/anonimizzazione-testi-italiano`](https://huggingface.co/datasets/rizzoaiacademy/anonimizzazione-testi-italiano)**
— a public, collaborative corpus we are starting to fill. If you work with Italian documents —
lawyers, accountants, notaries, developers, researchers — help build it: contribute templates,
annotation, edge cases, and review. **A privacy tool for Italy is something Italy should build
together, for the good of everyone's privacy.**

---

## Documentation

| Document | Contents |
|---|---|
| [CLAUDE.md](CLAUDE.md) | Environment constraints, repo map, architectural decisions, commands |
| [docs/DATASET.md](docs/DATASET.md) | Full composition of train (~745k) and validation (7k) |
| [docs/TASSONOMIA_TAG.md](docs/TASSONOMIA_TAG.md) | The 22 final tags and the merges (`TAG_MAP`) |
| [docs/BUILD.md](docs/BUILD.md) | Desktop executable build (CPU, Windows) |
| [docs/CHANGELOG.md](docs/CHANGELOG.md) | Change log for the pipeline, with rationale |
| [CONTRIBUTING.md](CONTRIBUTING.md) | How to contribute code, docs and (above all) data |

## License

Released under the **[MIT License](LICENSE)** © 2026 Simone Rizzo — Rizzo AI Academy.

Note on third-party data: the training corpus draws on
[Ai4Privacy](https://huggingface.co/datasets/ai4privacy/open-pii-masking-500k-ai4privacy)
(CC-BY-4.0) and [DeepMount00](https://huggingface.co/datasets/DeepMount00/pii-masking-ita); the
backbone is [mmBERT-base](https://huggingface.co/jhu-clsp/mmBERT-base). Please respect their
respective licenses when redistributing data or weights.

---

<div align="center">

<img src="report/images/mascot_idle.png" alt="rizzo-pii mascot waving" width="130" />

### Contribute to the project

⭐ Star it · 🐛 open an issue · 🔀 send a pull request · 📎 or just share a hard example

**Author** — Simone Rizzo · **Sponsor** — [Rizzo AI Academy](https://rizzoaiacademy.com)

_The mascot, a hedgehog, guards the document and **stays inside the EU**. Built and trained in Italy._ 🇮🇹🇪🇺

</div>

---

### References

1. mmBERT: a multilingual ModernBERT encoder. JHU-CLSP, `jhu-clsp/mmBERT-base`, Hugging Face.
2. Warner, B. et al. *Smarter, Better, Faster, Longer: A Modern Bidirectional Encoder* (ModernBERT), 2024.
3. Ai4Privacy. `open-pii-masking-500k`, Hugging Face (CC-BY-4.0).
4. DeepMount00. `pii-masking-ita`, Hugging Face.
5. OpenAI. *Introducing OpenAI Privacy Filter*, 2026; model `openai/privacy-filter`, Hugging Face (Apache-2.0).
6. Microsoft. *Presidio: Data Protection and De-identification SDK*. Open source (MIT).
7. Regulation (EU) 2016/679: General Data Protection Regulation (GDPR).
8. Regulation (EU) 2024/1689: Artificial Intelligence Act (EU AI Act).
