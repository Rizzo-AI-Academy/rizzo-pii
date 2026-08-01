# -*- coding: utf-8 -*-
"""
Politica di anonimizzazione: per ogni tag decide se MASCHERARE l'entita' con un
placeholder reversibile (comportamento storico, default) o LASCIARLA IN CHIARO.

Perche' serve: alcuni compiti chiesti al modello di frontiera dipendono proprio dai
valori che l'anonimizzazione rimuove — il confronto fra due importi dello stesso
contratto, l'eta' e il sesso in uno studio clinico. Con la policy l'utente sceglie
per tag cosa esce dal documento; senza configurazione si maschera tutto, come prima.

Risoluzione con precedenza, la stessa catena di server_config:

    CLI  >  env (PII_PROFILE / PII_KEEP_TAGS)  >  policy.json  >  default

Il profilo da' un insieme di tag di partenza; i tag indicati esplicitamente si
AGGIUNGONO a quelli del profilo (unione). Per mascherare tutto: profilo "full",
nessun tag.

policy.json sta nella stessa directory di config.json (server_config.config_dir())
ma e' un file SEPARATO: config.json e' scritto anche dall'app Tauri dal lato Rust,
che lo riscrive come {"host", "port"} e cancellerebbe una chiave estranea.

    Formato:  {"profile": "clinical", "keep_tags": ["AMOUNT"]}

Il modulo e' puro (nessun import di torch/flask/transformers): si puo' testare e
usare senza caricare il modello.
"""

import json
import os
import sys
from pathlib import Path

import server_config

# Azioni possibili su un'entita' rilevata.
ACTION_MASK = "mask"    # -> [TAG_n] + voce nel dizionario reversibile
ACTION_KEEP = "keep"    # -> resta in chiaro nel testo anonimizzato
ACTIONS = (ACTION_MASK, ACTION_KEEP)

# Motivo riportato nell'output per un'entita' rilevata ma non mascherata.
REASON_CONFIG = "excluded_by_config"

DEFAULT_PROFILE = "full"

# Profili preconfezionati: nome -> tag lasciati in chiaro.
PROFILES = {
    "full": (),                                        # maschera tutto (storico)
    "clinical": ("AGE", "GENDER", "DATE", "TIME"),     # cartelle/studi clinici
    "compare-amounts": ("AMOUNT",),                    # confronti fra importi
}

# Identificatori diretti: lasciarli in chiaro e' una scelta legittima ma pesante,
# quindi la si segnala una volta al caricamento. Non e' un divieto.
HIGH_RISK_TAGS = frozenset({
    "FULLNAME", "CF", "PIVA", "IBAN", "CREDITCARDNUMBER", "ID_DOC",
    "EMAIL", "TELEPHONENUM",
})

POLICY_FILENAME = "policy.json"


def _warn(message: str) -> None:
    print(f"[policy] {message}", file=sys.stderr)


def policy_path() -> Path:
    """Percorso di policy.json (stessa directory di config.json)."""
    return server_config.config_dir() / POLICY_FILENAME


def load_file() -> dict:
    """Legge policy.json; ritorna {} se manca o e' corrotto (come server_config)."""
    p = policy_path()
    if p.exists():
        try:
            data = json.loads(p.read_text("utf-8"))
            if isinstance(data, dict):
                return data
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def save_file(profile: str, keep_tags) -> None:
    """Scrive policy.json (crea la directory se necessario)."""
    d = server_config.config_dir()
    d.mkdir(parents=True, exist_ok=True)
    payload = {"profile": profile, "keep_tags": list(parse_tags(keep_tags))}
    (d / POLICY_FILENAME).write_text(json.dumps(payload, indent=2), "utf-8")


def parse_tags(raw) -> tuple:
    """Normalizza una lista di tag scritta come stringa o come lista.

    Accetta "age, gender" oppure ["AGE", "GENDER"]; ritorna una tupla di tag in
    MAIUSCOLO, senza vuoti e senza duplicati, nell'ordine di apparizione.
    """
    if raw is None:
        return ()
    items = raw.replace(";", ",").split(",") if isinstance(raw, str) else list(raw)
    out = []
    for item in items:
        tag = str(item).strip().upper()
        if tag and tag not in out:
            out.append(tag)
    return tuple(out)


class Policy:
    """Cosa fare di ogni entita' rilevata, tag per tag.

    keep_tags: tag lasciati in chiaro; tutto il resto e' mascherato.
    """

    def __init__(self, keep_tags=(), profile: str = DEFAULT_PROFILE):
        self.profile = profile
        self.keep_tags = frozenset(parse_tags(keep_tags))

    def action(self, label: str) -> str:
        """ACTION_KEEP se il tag va lasciato in chiaro, altrimenti ACTION_MASK."""
        return ACTION_KEEP if str(label).upper() in self.keep_tags else ACTION_MASK

    def keeps(self, label: str) -> bool:
        return self.action(label) == ACTION_KEEP

    def as_dict(self) -> dict:
        """Rappresentazione serializzabile (risposta API / UI)."""
        return {"profile": self.profile, "keep_tags": sorted(self.keep_tags)}

    def __repr__(self):
        return f"Policy(profile={self.profile!r}, keep_tags={sorted(self.keep_tags)})"


def _profile_tags(name: str, warn) -> tuple:
    """Tag del profilo; profilo sconosciuto -> avviso e default."""
    if name in PROFILES:
        return PROFILES[name]
    warn(f"profilo sconosciuto '{name}': uso '{DEFAULT_PROFILE}'. "
         f"Disponibili: {', '.join(sorted(PROFILES))}")
    return PROFILES[DEFAULT_PROFILE]


def load_policy(cli_keep_tags=None, cli_profile=None, known_tags=None, warn=_warn) -> Policy:
    """Risolve la policy con la catena CLI > env > policy.json > default.

    cli_keep_tags / cli_profile: valori da riga di comando (None = non specificati).
    known_tags: tassonomia valida (label del modello + della rete regex). I tag non
        riconosciuti vengono segnalati e ignorati, non fanno fallire il caricamento.
    warn: funzione di avviso, iniettabile nei test.

    Il primo livello che fornisce dei tag vince (non si sommano fra loro); i tag del
    profilo si aggiungono sempre.
    """
    cfg = load_file()

    profile = (cli_profile
               or os.environ.get("PII_PROFILE")
               or cfg.get("profile")
               or DEFAULT_PROFILE)
    profile = str(profile).strip().lower()

    explicit = ()
    for source in (cli_keep_tags, os.environ.get("PII_KEEP_TAGS"), cfg.get("keep_tags")):
        explicit = parse_tags(source)
        if explicit:
            break

    tags = list(_profile_tags(profile, warn))
    for tag in explicit:
        if tag not in tags:
            tags.append(tag)

    if known_tags:
        known = {str(t).upper() for t in known_tags}
        unknown = [t for t in tags if t not in known]
        if unknown:
            warn(f"tag non presenti nella tassonomia, ignorati: {', '.join(unknown)}")
        tags = [t for t in tags if t in known]

    risky = sorted(t for t in tags if t in HIGH_RISK_TAGS)
    if risky:
        warn(f"ATTENZIONE: identificatori diretti lasciati IN CHIARO: {', '.join(risky)}")

    return Policy(keep_tags=tags, profile=profile)
