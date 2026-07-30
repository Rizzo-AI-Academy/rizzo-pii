# Contributing to rizzo-pii

Thanks for helping build a privacy tool for Italy. 🦔🇮🇹

The single biggest lever on quality is **data** — a large, real, lawfully-collected Italian
corpus, especially for **organizations (`ORG`)** and the legal identifiers that are scarce today.
Contributions of code, documentation and (above all) data are all welcome.

## Ways to contribute

- **🐛 Report a bug or a missed entity.** Open an issue with a short, **anonymized** example
  (never paste real personal data) showing what was tagged wrong.
- **🔀 Code / docs.** Fork, branch, and open a pull request (see workflow below).
- **📚 Data.** The community dataset lives on Hugging Face:
  **[`rizzoaiacademy/anonimizzazione-testi-italiano`](https://huggingface.co/datasets/rizzoaiacademy/anonimizzazione-testi-italiano)**.
  The quickest way to help is the contribution script, which uses **your own Gemini key** to
  write **brand-new** legal prose every run, injects valid values (CF/PIVA/IBAN checksums) with
  exact BIO labels, and opens a **Pull Request** for review:

  ```bash
  pip install huggingface_hub                                     # the data script needs no torch
  export GEMINI_API_KEY=...      # https://aistudio.google.com/apikey   (PowerShell: $env:GEMINI_API_KEY=...)
  # …or write the templates with a LOCAL model instead — no API key, nothing leaves the machine:
  export LLM_BASE_URL=http://127.0.0.1:8080/v1   # llama.cpp / Ollama / vLLM / LM Studio
  export LLM_MODEL=your-model-name
  hf auth login                  # or: export HF_TOKEN=hf_xxx
  # dry run (no upload), then the real batch boosting the weak tags:
  python src/data_pipeline/contribute_dataset.py --n 300  --handle yourname --no-upload
  python src/data_pipeline/contribute_dataset.py --n 5000 --handle yourname --per-type 3 \
      --boost ORG=6 IBAN=4 CF=4 CATASTO=3 DOCID=3
  ```

  The **exact row format** (JSONL, BIO scheme, offsets, required checksums) is documented in
  [docs/FORMATO_DATI.md](docs/FORMATO_DATI.md) — read it if you want to produce examples by
  hand. The README also has a ready-to-paste prompt for a coding agent.

> ⚠️ **Never contribute real personal data.** This project exists to protect PII. Examples in
> issues, PRs and the dataset must be synthetic or fully anonymized. The whole synthetic pipeline
> is built on the *"LLM author, code labeler"* principle precisely so that no real PII is ever
> handled — please keep it that way.

## Development setup

```bash
git clone https://github.com/Rizzo-AI-Academy/rizzo-pii
cd rizzo-pii
python -m venv .venv && . .venv/Scripts/activate   # Windows; use bin/activate on Linux/macOS
pip install -r requirements.txt                    # see the cu128 note for NVIDIA Blackwell GPUs
cp .env.example .env                               # optional: add W&B / Gemini keys
```

Regenerate the data and run a quick smoke-test training (see [README](README.md#quickstart) and
[CLAUDE.md](CLAUDE.md) for the full pipeline and environment constraints):

```bash
python src/data_pipeline/build_subset.py
python src/training/train_pii.py --type subset     # ~3 min
```

## Pull request workflow

1. Create a topic branch: `git checkout -b fix/short-description`.
2. Keep changes focused; match the style of the surrounding code.
3. If you change the taxonomy, edit **only** `TAG_MAP` / `DROP_TYPES` in `src/training/train_pii.py`
   (the raw datasets are remapped at load time) and update
   [docs/TASSONOMIA_TAG.md](docs/TASSONOMIA_TAG.md).
4. Note user-facing changes in [docs/CHANGELOG.md](docs/CHANGELOG.md), with the rationale.
5. Open the PR against `main` and describe **what** changed and **why**.

## License

By contributing, you agree that your contributions are licensed under the
[MIT License](LICENSE) that covers this project.
