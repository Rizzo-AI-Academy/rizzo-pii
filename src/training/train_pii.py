# -*- coding: utf-8 -*-
"""
Training token classification PII con mmBERT su TUTTO il dataset Ai4Privacy.

- Carica l'intero train + validation (tutte le lingue; filtrabile su una sola).
- 1 epoca su GPU (bf16, Blackwell).
- Logga train loss a OGNI step; calcola validation loss e metriche ALLA FINE.
- Unifica i tag persona (nomi + ruoli legali) su FULLNAME.
- Salva il modello, il plot train-vs-val loss e stampa le metriche sul val subset.
"""

import io
import json
import os
import random
import sys
import time

# Riduce la frammentazione VRAM (blocchi espandibili): evita il thrashing
# dell'allocatore quando le lunghezze di sequenza variano vicino al tetto dei 16 GB.
# Va impostato PRIMA di inizializzare CUDA (quindi prima di importare torch).
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
from dotenv import load_dotenv
from torch.utils.data import Dataset
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)
from transformers.trainer_pt_utils import LengthGroupedSampler


class LengthGroupedTrainer(Trainer):
    """Come Trainer ma per group_by_length usa lunghezze PRECALCOLATE (conteggio
    parole) invece di ri-tokenizzare tutto il dataset lazy. Raggruppa sequenze di
    lunghezza simile -> meno padding sprecato e niente thrashing VRAM sul run grande."""

    def __init__(self, *args, train_lengths=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._train_lengths = train_lengths

    def _get_train_sampler(self, train_dataset=None):
        if self.args.group_by_length and self._train_lengths is not None:
            ds = train_dataset if train_dataset is not None else self.train_dataset
            bs = self.args.train_batch_size * self.args.gradient_accumulation_steps
            return LengthGroupedSampler(bs, dataset=ds, lengths=self._train_lengths)
        return super()._get_train_sampler(train_dataset)

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

# Weights & Biases: carica WANDB_API_KEY / WANDB_PROJECT dal .env (se presente).
load_dotenv()
USE_WANDB = bool(os.environ.get("WANDB_API_KEY"))
print(f"W&B: {'attivo (' + os.environ.get('WANDB_PROJECT', 'default') + ')' if USE_WANDB else 'disattivo (nessun WANDB_API_KEY)'}")

# --------------------------------------------------------------------------- #
# PARAMETRI
# --------------------------------------------------------------------------- #
# Radice del repo (questo file e' in src/training/) -> path assoluti: lo script gira
# da qualsiasi cartella di lavoro.
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "dataset"

TRAIN_PATH = str(DATA / "raw" / "ai4privacy_500k" / "data" / "train" / "train.jsonl")
VAL_PATH = str(DATA / "raw" / "ai4privacy_500k" / "data" / "validation" / "test.jsonl")
MODEL_NAME = "jhu-clsp/mmBERT-base"
LANG = None            # None = tutte le lingue (multilingue); "it" = solo italiano.
                       # Multilingue + i dati IT (synth/augment/DeepMount) come rinforzo italiano.
MAX_LEN = 768          # copre i sintetici (max 771 subword) e DeepMount (660); tronca
                       # solo poche centinaia di Ai4 estremi. Padding dinamico: i batch
                       # corti (media ~112 subword) non rallentano.
EPOCHS = 1
BATCH = 14             # run grande a MAX_LEN 768 su 16 GB CONDIVISI col desktop: 16 tiene il picco
                       # di attivazioni a ~8-9 GB (24 arrivava a ridosso del tetto -> thrashing sui
                       # batch lunghi). Con GRAD_ACCUM=2 il batch EFFETTIVO resta 32 (vedi sotto).
GRAD_ACCUM = 2         # accumulo gradiente: batch effettivo = BATCH * GRAD_ACCUM = 32, a costo VRAM di 16.
EVAL_BATCH = 32        # eval forward a MAX_LEN 768: 32 evita OOM durante l'eval periodica
                       # (optimizer residente in VRAM). L'eval e' infrequente: costo trascurabile.
LR = 5e-5
TRAIN_EVAL_N = 5000    # esempi di train per le metriche finali (None = tutti, ma e' un pass lungo)
SYNTH_PATHS = [
    str(DATA / "synthetic" / "synthetic_pii_it_200k.jsonl"),    # template (CF/IBAN/ORG/CATASTO/...)
    str(DATA / "synthetic" / "synthetic_pii_it_realaug.jsonl"), # entita' sintetiche in testo reale
]
DEEPMOUNT_TRAIN = str(DATA / "processed" / "deepmount_pii_it_train.jsonl")   # -> pool di train
# UNICA validation reale: Ai4Privacy val + DeepMount test + tag mancanti iniettati in
# frasi reali held-out (build_validation.py). Il DeepMount TEST e' consumato lì, non qui.
VALIDATION_PATH = str(DATA / "validation" / "validation_real.jsonl")
# Famiglia del modello: "0.3B" e' la DIMENSIONE (mmBERT-base ~0.3B param), non la versione.
MODEL_FAMILY = "rizzo-pii-0.3B"
# VERSIONE del modello: ogni run grande salva in models/rizzo-pii-0.3B-v{VERSION}/ (storico
# completo, niente sovrascrittura) e logga su W&B come rizzo-pii:0.3B-v{VERSION}. Bumpala qui a
# ogni cambiamento significativo di dati/training, oppure passala con --version sulla CLI.
# Storia: v1.0.0 = baseline (vecchio models/rizzo-pii-0.3B). v1.1.0 = case-augmentation +
# ORG vari + template-elenco (nomi/societa' consecutivi).
MODEL_VERSION = "1.1.0"

# Modalita' del run, scelta da riga di comando:
#   --type full    (default) -> run grande -> models/rizzo-pii-0.3B-v{VERSION} + experiments/full_run_v{VERSION}
#   --type subset            -> smoke test / tuning sui subset 10k/5k -> experiments/subset_smoke
# (compatibilita': anche env PII_SUBSET=1 forza la modalita' subset.)
import argparse
_ap = argparse.ArgumentParser(description="Training token-classification PII (rizzo-pii:0.3B).")
_ap.add_argument("--type", choices=["full", "subset"], default="full",
                 help="full = run grande su tutto il dataset; subset = smoke test 10k/5k")
_ap.add_argument("--version", default=None,
                 help=f"versione del modello (default {MODEL_VERSION}); il run grande salva in models/{MODEL_FAMILY}-v<versione>")
_args, _ = _ap.parse_known_args()
SUBSET = (_args.type == "subset") or (os.environ.get("PII_SUBSET") == "1")
MODEL_VERSION = _args.version or MODEL_VERSION
SUBSET_TRAIN_PATH = str(DATA / "subsets" / "train_subset_10k.jsonl")
SUBSET_VAL_PATH = str(DATA / "subsets" / "val_subset_5k.jsonl")
# Nome del run su W&B: versionato per il run grande, generico per lo smoke test.
WANDB_RUN_NAME = f"rizzo-pii:0.3B-v{MODEL_VERSION}" if not SUBSET else "rizzo-pii:0.3B-subset"
if SUBSET:
    # Smoke test: artefatti in experiments/subset_smoke, modello buttato lì (non versionato).
    RUN_DIR = str(ROOT / "experiments" / "subset_smoke")
    SAVE_DIR = os.path.join(RUN_DIR, "pii_model_subset")
    PLOT_PATH = os.path.join(RUN_DIR, "training_loss_subset.png")
    # MAX_LEN ridotto -> ~3-4x meno VRAM, niente thrashing, ~0,27 s/step.
    MAX_LEN = 256
    BATCH = 32         # a MAX_LEN 256 c'e' VRAM in abbondanza
else:
    # Run grande: modello VERSIONATO in models/ (storico) + artefatti per-versione in experiments/.
    SAVE_DIR = str(ROOT / "models" / f"{MODEL_FAMILY}-v{MODEL_VERSION}")
    RUN_DIR = str(ROOT / "experiments" / f"full_run_v{MODEL_VERSION}")
    PLOT_PATH = os.path.join(RUN_DIR, "training_loss.png")
os.makedirs(RUN_DIR, exist_ok=True)
print(f"Versione modello: v{MODEL_VERSION} | save: {SAVE_DIR}")
if SUBSET:
    print(f"MODALITA' SUBSET attiva: train 10k / val 5k, MAX_LEN={MAX_LEN} -> {RUN_DIR}/")

assert torch.cuda.is_available(), "CUDA non disponibile: serve la build GPU di torch."
print(f"GPU: {torch.cuda.get_device_name(0)} | torch {torch.__version__}")

# --------------------------------------------------------------------------- #
# Unificazione tassonomia (applicata al caricamento, i file grezzi restano intatti).
#
#  - Nomi e RUOLI legali sono TUTTI nomi di persona da mascherare: GIVENNAME/SURNAME
#    e GIUDICE/AVVOCATO/CONVENUTO/ATTORE/TESTIMONE -> FULLNAME. Il ruolo NON e' un tag
#    a se' (sarebbe keyword-matching dalla posizione nel template, e lo stesso nome puo'
#    avere ruoli diversi): se serve, si deriva a valle come metadato sopra FULLNAME.
#  - Fusioni di tag che sono lo stesso concetto:
#       SEX + GENDER -> GENDER      (stesso contenuto: M/F/Maschio/Femmina/...)
#       TAXNUM + PIVA -> PIVA       (TAXNUM nei dati = partita IVA; copre l'asimmetria)
#       PEC + EMAIL  -> EMAIL       (la PEC e' un'email)
#       RG + DOCID   -> DOCID       (il n. di Ruolo Generale e' un codice di documento;
#                                    stesso formato NNNN/AAAA, distinzione solo contestuale)
#       IDCARDNUM + PASSPORTNUM + DRIVERLICENSENUM + SOCIALNUM -> ID_DOC
#                                   (tutti "numero di documento d'identita' personale";
#                                    distinzione solo contestuale -> un unico tag robusto)
#       CONTO + IBAN -> IBAN        (il n. di conto e' un sottoinsieme dell'IBAN)
#  - TITLE (Dott./Avv./Sig.) -> O: appellativo, non identificatore.
# --------------------------------------------------------------------------- #
TAG_MAP = {
    "GIVENNAME": "FULLNAME", "SURNAME": "FULLNAME",
    "GIUDICE": "FULLNAME", "AVVOCATO": "FULLNAME", "CONVENUTO": "FULLNAME",
    "ATTORE": "FULLNAME", "TESTIMONE": "FULLNAME",
    "SEX": "GENDER",
    "TAXNUM": "PIVA",
    "PEC": "EMAIL",
    "RG": "DOCID",
    "IDCARDNUM": "ID_DOC", "PASSPORTNUM": "ID_DOC",
    "DRIVERLICENSENUM": "ID_DOC", "SOCIALNUM": "ID_DOC",
    "CONTO": "IBAN",
}
DROP_TYPES = {"TITLE", "TRIBUNAL"}    # tipi rimappati a O (non vanno anonimizzati):
#   TITLE    = appellativo (Dott./Avv./Sig.)
#   TRIBUNAL = ufficio giudiziario, ente pubblico: non e' PII da mascherare


def normalize_labels(labels):
    """Rimappa i tipi (TAG_MAP), scarta i DROP_TYPES (-> O) e fonde token adiacenti
    dello stesso tipo in un'unica entita': es. 'Mario'(GIVENNAME)+'Rossi'(SURNAME)
    -> B-FULLNAME I-FULLNAME."""
    out, prev = [], None
    for l in labels:
        typ = TAG_MAP.get(l[2:], l[2:]) if l != "O" else None
        if typ is None or typ in DROP_TYPES:
            out.append("O"); prev = None
            continue
        out.append(("I-" if typ == prev else "B-") + typ)
        prev = typ
    return out


# --------------------------------------------------------------------------- #
# Case augmentation: il modello e' CASED e tutti i nomi nel training sono
# capitalizzati -> impara "maiuscola iniziale" come feature dominante e non
# riconosce "mario rossi". Qui, SOLO sul train, riscriviamo il casing di una
# frazione degli esempi (labels invariate: il casing non sposta i confini token).
# La validation NON viene toccata -> la metrica resta confrontabile coi run vecchi.
#
# Profilo AGGRESSIVO (le percentuali sono per-esempio):
#   35% frase tutta minuscola | 10% frase tutta MAIUSCOLA | 8% random per-token
#   22% ENTITY-LEVEL: ricasa SOLO i token delle entita' nominali (nomi/vie/citta'/
#       province/societa') a minuscolo / MAIUSCOLO / Iniziale, contesto invariato
#       -> copre "il sottoscritto mario rossi" e "comparso MARIO ROSSI" negli atti.
#   25% invariato.
# (i valori strutturati CF/IBAN/PIVA/... vengono minuscolizzati solo nel 35% "frase
#  intera"; la rete regex+checksum li copre comunque in produzione.)
RECASE_NAME_TAGS = {"FULLNAME", "STREET", "CITY", "PROVINCE", "ORG"}


def _recase_entities(tokens, labels, mode):
    out = []
    for t, l in zip(tokens, labels):
        if l != "O" and l[2:] in RECASE_NAME_TAGS:
            out.append(t.lower() if mode == "lower" else t.upper() if mode == "upper" else t.capitalize())
        else:
            out.append(t)
    return out


def recase_tokens(tokens, labels):
    r = random.random()
    if r < 0.35:
        return [t.lower() for t in tokens]
    if r < 0.45:
        return [t.upper() for t in tokens]
    if r < 0.53:
        return [(t.upper() if random.random() < 0.5 else t.lower()) for t in tokens]
    if r < 0.75:
        return _recase_entities(tokens, labels, random.choice(["lower", "upper", "title"]))
    return tokens


# --------------------------------------------------------------------------- #
# 1) Caricamento dati (tutto il file; eventuale filtro lingua)
# --------------------------------------------------------------------------- #
def load_records(path, lang=None, limit=None):
    recs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if lang and r.get("language") != lang:
                continue
            toks, labs = r.get("mbert_tokens"), r.get("mbert_token_classes")
            if toks and labs and len(toks) == len(labs):
                recs.append((toks, normalize_labels(labs)))
            if limit and len(recs) >= limit:
                break
    return recs


def load_synth(path):
    """Sintetici: schema con 'tokens' + 'bio_labels' (vedi generate_synthetic_pii.py)."""
    recs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            toks, labs = r.get("tokens"), r.get("bio_labels")
            if toks and labs and len(toks) == len(labs):
                recs.append((toks, normalize_labels(labs)))
    return recs


t_load = time.time()
random.seed(42)

if SUBSET:
    # --- subset rappresentativi gia' fusi e normalizzati (schema tokens/bio_labels) ---
    train_recs = load_synth(SUBSET_TRAIN_PATH)
    val_recs = load_synth(SUBSET_VAL_PATH) if os.path.exists(SUBSET_VAL_PATH) else []
    random.shuffle(train_recs)
    print(f"SUBSET: train {len(train_recs)} ({SUBSET_TRAIN_PATH}) | "
          f"val {len(val_recs)} ({SUBSET_VAL_PATH}) | caricati in {time.time()-t_load:.0f}s")
else:
    # --- TRAIN pool: Ai4Privacy + sintetici (template + augment) + DeepMount train ---
    train_recs = load_records(TRAIN_PATH, LANG)
    n_ai4_train = len(train_recs)

    synth_n = 0
    for sp in SYNTH_PATHS:
        if os.path.exists(sp):
            loaded = load_synth(sp)
            train_recs += loaded
            synth_n += len(loaded)
            print(f"Sintetici da {sp}: {len(loaded)}")
        else:
            print(f"ATTENZIONE: sintetico mancante (saltato): {sp}")

    n_dm_train = 0
    if os.path.exists(DEEPMOUNT_TRAIN):
        dm_train = load_synth(DEEPMOUNT_TRAIN)
        train_recs += dm_train
        n_dm_train = len(dm_train)
        print(f"DeepMount train: {n_dm_train}")
    else:
        print(f"ATTENZIONE: DeepMount train mancante (saltato): {DEEPMOUNT_TRAIN}")

    random.shuffle(train_recs)        # interleave tutte le fonti (non tutte in coda)

    # --- UNICA validation reale unificata (build_validation.py) ---
    if os.path.exists(VALIDATION_PATH):
        val_recs = load_synth(VALIDATION_PATH)
    else:
        val_recs = []
        print(f"ATTENZIONE: validation mancante: {VALIDATION_PATH} (esegui build_validation.py)")

    print(f"Train: {len(train_recs)} (Ai4Privacy {n_ai4_train} + sintetici {synth_n} + "
          f"DeepMount {n_dm_train}) | lingua: {LANG or 'tutte'} + it-synth | "
          f"Validation reale: {len(val_recs)} | caricati in {time.time()-t_load:.0f}s")

# Case augmentation SOLO sul train (la validation resta a casing reale).
train_recs = [(recase_tokens(toks, labs), labs) for toks, labs in train_recs]

# tassonomia dall'unione di train + validation
label_set = set()
for recs in (train_recs, val_recs):
    for _, labs in recs:
        label_set.update(labs)
label_list = sorted(label_set)
label2id = {l: i for i, l in enumerate(label_list)}
id2label = {i: l for l, i in label2id.items()}
O_ID = label2id.get("O", 0)
print(f"Numero di label (BIO): {len(label_list)}")

# --------------------------------------------------------------------------- #
# 2) Tokenizer + dataset LAZY (tokenizza on-the-fly: niente RAM esplosa su 464k)
# --------------------------------------------------------------------------- #
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


def encode(tokens, labels):
    enc = tokenizer(tokens, is_split_into_words=True, truncation=True, max_length=MAX_LEN)
    word_ids = enc.word_ids()
    aligned, prev = [], None
    for wid in word_ids:
        if wid is None:
            aligned.append(-100)
        elif wid != prev:
            aligned.append(label2id.get(labels[wid], O_ID))
        else:
            aligned.append(-100)
        prev = wid
    enc["labels"] = aligned
    return enc


class PiiDataset(Dataset):
    def __init__(self, recs):
        self.recs = recs

    def __len__(self):
        return len(self.recs)

    def __getitem__(self, i):
        return encode(*self.recs[i])


train_ds = PiiDataset(train_recs)
eval_ds = PiiDataset(val_recs)        # unica validation reale

# --------------------------------------------------------------------------- #
# 3) Modello
# --------------------------------------------------------------------------- #
model = AutoModelForTokenClassification.from_pretrained(
    MODEL_NAME, num_labels=len(label_list), id2label=id2label, label2id=label2id,
)
print(f"Modello: {MODEL_NAME} | parametri: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")

collator = DataCollatorForTokenClassification(tokenizer)

microbatches = len(train_ds) // BATCH + (len(train_ds) % BATCH > 0)
steps_per_epoch = max(1, microbatches // GRAD_ACCUM)   # step OTTIMIZZATORE (con accumulo gradiente)
# Eval intermedia LEGGERA: ~4 valutazioni (solo eval_loss) durante l'epoca, per vedere su
# W&B la curva train-vs-val e cogliere overfit/anomalie prima della fine del run lungo.
# Le metriche P/R/F1 entity-level restano calcolate ALLA FINE (sezione 6).
EVAL_EVERY = max(1, steps_per_epoch // 4)
print(f"Step ottimizzatore/epoca: ~{steps_per_epoch} (microbatch {microbatches}, accum {GRAD_ACCUM}) | "
      f"eval_loss ogni {EVAL_EVERY} step | P/R/F1 ALLA FINE")

args = TrainingArguments(
    output_dir=os.path.join(RUN_DIR, "out"),
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH,
    gradient_accumulation_steps=GRAD_ACCUM,   # batch effettivo = BATCH * GRAD_ACCUM
    per_device_eval_batch_size=EVAL_BATCH,
    learning_rate=LR,
    weight_decay=0.01,               # regolarizzazione AdamW standard per il fine-tuning
    warmup_ratio=0.05,               # ~5% warmup: stabilizza i primi step (testa nuova a LR pieno)
    bf16=True,
    logging_steps=1,                 # train loss a OGNI step (per il plot e per W&B)
    eval_strategy="steps",           # eval_loss periodica durante il training
    eval_steps=EVAL_EVERY,
    report_to=(["wandb"] if USE_WANDB else []),
    run_name=WANDB_RUN_NAME,         # versionato sul run grande (rizzo-pii:0.3B-v{VERSION})
    save_strategy="no",
    dataloader_num_workers=0,        # Windows
    group_by_length=(not SUBSET),    # run grande: raggruppa per lunghezza (meno padding/VRAM)
)

# Lunghezze (conteggio parole) per group_by_length, gia' disponibili senza tokenizzare.
train_lengths = [len(t) for t, _ in train_recs] if not SUBSET else None

trainer = LengthGroupedTrainer(model=model, args=args, train_dataset=train_ds,
                               eval_dataset=eval_ds, data_collator=collator,
                               tokenizer=tokenizer, train_lengths=train_lengths)

# --------------------------------------------------------------------------- #
# 4) Training
# --------------------------------------------------------------------------- #
print(f"\nTrainer device: {trainer.args.device} | pesi su {next(model.parameters()).device}")
t0 = time.time()
trainer.train()
torch.cuda.synchronize()
print(f"\nTraining completato in {time.time()-t0:.0f}s | VRAM picco "
      f"{torch.cuda.max_memory_allocated()/1e6:.0f} MB")

# salvataggio
trainer.save_model(SAVE_DIR)
tokenizer.save_pretrained(SAVE_DIR)
print(f"Modello salvato in: {SAVE_DIR}")

# --------------------------------------------------------------------------- #
# 5) Plot della training loss (un punto per step)
# --------------------------------------------------------------------------- #
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

train_pts = [(h["step"], h["loss"]) for h in trainer.state.log_history if "loss" in h]
if train_pts:
    xs, ys = zip(*train_pts)
    plt.figure(figsize=(10, 5))
    plt.plot(xs, ys, lw=0.8, color="#c1121f", label="train loss (ogni step)")
    plt.xlabel("step")
    plt.ylabel("training loss")
    plt.title(f"Training loss - mmBERT-base | {len(train_ds)} train | {EPOCHS} epoca")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=130)
    print(f"Plot salvato in: {PLOT_PATH} ({len(train_pts)} punti)")

# --------------------------------------------------------------------------- #
# 6) Valutazione finale: loss + metriche su TRAIN e VALIDATION
# --------------------------------------------------------------------------- #
def spans(tags):
    out, cur = [], None
    for i, t in enumerate(tags + ["O"]):
        if t == "O" or t.startswith("B-") or (cur and t[2:] != cur[0]):
            if cur:
                out.append((cur[0], cur[1], i)); cur = None
        if t.startswith("B-") or (t.startswith("I-") and cur is None):
            cur = (t[2:], i)
    return set(out)


def evaluate_metrics(ds):
    pred = trainer.predict(ds)
    preds = np.argmax(pred.predictions, axis=-1)
    gold = pred.label_ids
    tp = fp = fn = tok_ok = tok_tot = 0
    for p_row, g_row in zip(preds, gold):
        p_tags, g_tags = [], []
        for p, g in zip(p_row, g_row):
            if g == -100:
                continue
            p_tags.append(id2label[int(p)]); g_tags.append(id2label[int(g)])
            tok_tot += 1; tok_ok += int(p == g)
        gs, ps = spans(g_tags), spans(p_tags)
        tp += len(gs & ps); fp += len(ps - gs); fn += len(gs - ps)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"loss": pred.metrics.get("test_loss"),
            "token_acc": tok_ok / tok_tot if tok_tot else 0.0,
            "precision": prec, "recall": rec, "f1": f1}


# un pass sull'intero train sarebbe lungo: si usa un sottoinsieme (TRAIN_EVAL_N)
train_eval_ds = PiiDataset(train_recs if TRAIN_EVAL_N is None else train_recs[:TRAIN_EVAL_N])

print("\nCalcolo metriche finali su train e validation reale...")
m_tr = evaluate_metrics(train_eval_ds)
m_va = evaluate_metrics(eval_ds) if len(val_recs) else None

cols = [("TRAIN", train_eval_ds, m_tr), ("VALIDATION reale", eval_ds, m_va)]
print("\n" + "=" * 60)
print(f"{'':<16}" + "".join(f"{name + ' (' + str(len(ds)) + ')':>22}"
                            for name, ds, _ in cols))
print("-" * 60)
for k, name in [("loss", "Loss"), ("token_acc", "Token accuracy"),
                ("precision", "Precision"), ("recall", "Recall"), ("f1", "F1-score")]:
    row = ""
    for _, _, m in cols:
        row += f"{(f'{m[k]:.4f}' if m and m[k] is not None else '-'):>22}"
    print(f"{name:<16}{row}")
print("=" * 60)

# Salva train/val loss e metriche su file dentro RUN_DIR (sia full che subset).
final_train_loss = train_pts[-1][1] if train_pts else None
mean_train_loss = (sum(y for _, y in train_pts) / len(train_pts)) if train_pts else None
summary = {
    "run": "subset_smoke_test" if SUBSET else "full_run",
    "train_rows": len(train_ds), "val_rows": len(eval_ds),
    "num_labels": len(label_list),
    "final_step_train_loss": final_train_loss,
    "mean_train_loss": mean_train_loss,
    "train_eval": m_tr, "validation": m_va,
}
with open(os.path.join(RUN_DIR, "metrics.json"), "w", encoding="utf-8") as fj:
    json.dump(summary, fj, ensure_ascii=False, indent=2)
with open(os.path.join(RUN_DIR, "metrics.txt"), "w", encoding="utf-8") as ft:
    ft.write(f"{'SUBSET smoke test' if SUBSET else 'FULL run'} - metriche finali\n")
    ft.write(f"train rows: {len(train_ds)} | val rows: {len(eval_ds)} | "
             f"label BIO: {len(label_list)}\n")
    ft.write(f"training loss (ultimo step): {final_train_loss}\n")
    ft.write(f"training loss (media step):  {mean_train_loss}\n\n")
    for nm, m in (("TRAIN (eval subset)", m_tr), ("VALIDATION", m_va)):
        if not m:
            continue
        ft.write(f"[{nm}]\n")
        for k in ("loss", "token_acc", "precision", "recall", "f1"):
            v = m.get(k)
            ft.write(f"  {k:<12}: {v:.4f}\n" if v is not None else f"  {k:<12}: -\n")
        ft.write("\n")
print(f"Metriche salvate in {RUN_DIR}/metrics.json e {RUN_DIR}/metrics.txt")
# Suggerimento: per il dettaglio PER-TAG usa src/training/evaluate_pii.py

# --------------------------------------------------------------------------- #
# Registry dei modelli: storico di TUTTE le versioni del run grande in
# models/registry.json (append). Ogni voce: versione, cartella, data, dimensioni,
# F1 validation -> tabella unica per confrontare i modelli nel tempo.
# --------------------------------------------------------------------------- #
if not SUBSET:
    from datetime import datetime
    registry_path = ROOT / "models" / "registry.json"
    entry = {
        "version": MODEL_VERSION,
        "model_dir": os.path.relpath(SAVE_DIR, ROOT).replace("\\", "/"),
        "base_model": MODEL_NAME,
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "lang": LANG or "multi",
        "max_len": MAX_LEN,
        "effective_batch": BATCH * GRAD_ACCUM,
        "train_rows": len(train_ds),
        "val_rows": len(eval_ds),
        "num_labels": len(label_list),
        "final_step_train_loss": final_train_loss,
        "val_f1": (m_va or {}).get("f1"),
        "val_precision": (m_va or {}).get("precision"),
        "val_recall": (m_va or {}).get("recall"),
        "wandb_run": WANDB_RUN_NAME,
        "experiments_dir": os.path.relpath(RUN_DIR, ROOT).replace("\\", "/"),
    }
    registry = []
    if registry_path.exists():
        try:
            registry = json.loads(registry_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            registry = []
    # sostituisce una voce con la stessa versione (re-run) invece di duplicarla
    registry = [e for e in registry if e.get("version") != MODEL_VERSION]
    registry.append(entry)
    registry.sort(key=lambda e: e.get("trained_at", ""))
    registry_path.write_text(json.dumps(registry, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Registry aggiornato: {registry_path} (v{MODEL_VERSION}, val F1={entry['val_f1']})")

# Metriche finali su W&B (oltre alla curva di train loss loggata a ogni step)
if USE_WANDB:
    import wandb
    log = {f"final/train_{k}": v for k, v in m_tr.items() if v is not None}
    if m_va:
        log.update({f"final/val_{k}": v for k, v in m_va.items() if v is not None})
    wandb.log(log)
    wandb.finish()
