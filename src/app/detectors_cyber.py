# -*- coding: utf-8 -*-
"""
Detector deterministici per documenti di SICUREZZA (pacchetto opt-in "cyber").

Chi scrive report di assessment, timeline forensi, ticket di incidente o estratti di
log ha lo stesso problema degli studi legali — usare un LLM di frontiera senza
spedirgli i dati di un terzo — ma i suoi identificatori non sono nella tassonomia dei
22 tag: indirizzi IP, domini, URL, MAC, hash, wallet, identificativi cloud, percorsi
che contengono lo username, account di dominio, ASN. Oggi restano tutti in chiaro.

Sono tutti dati a FORMA specifica: si riconoscono con regex + validatore, senza
toccare il modello e senza riaddestrare nulla. Il pacchetto si attiva esplicitamente
(env PII_DETECTORS=cyber, oppure --detectors cyber): con il pacchetto spento il
comportamento sui documenti legali resta identico, cosi' un numero di repertorio non
rischia di diventare un IP.

Due scelte da spiegare:

1. **Le forme "defanged" si matchano direttamente** (203[.]0[.]113[.]42, evil(.)com,
   hxxps://). Normalizzare il testo prima dell'analisi sfaserebbe gli offset, e un IP
   defangato non mascherato e' comunque un leak: e' la forma in cui gli indirizzi
   compaiono piu' spesso nei documenti di sicurezza.

2. **La keeplist (KEEP_PATTERNS) e' volutamente stretta.** Protegge i riferimenti
   pubblici (CVE, CWE, CAPEC, RFC, tecniche ATT&CK) dall'essere mascherati: senza,
   "CVE-2024-3094" sparirebbe e la frase diventerebbe incomprensibile per l'LLM. Ma
   una keeplist e' l'inverso di un detector: un falso positivo qui non produce una
   parola illeggibile, produce un dato sensibile lasciato in chiaro. Per questo si
   protegge solo cio' che ha un prefisso inequivocabile, e non forme generiche come
   "S1234"/"G0016" degli id ATT&CK di software e gruppi.

Non e' incluso il tag SECRET (password, API key, JWT, chiavi PEM, seed phrase): lo
copre gia' la PR #37, che aggiunge anche il supporto ai gruppi in detect_regex per
mascherare il solo valore lasciando leggibile l'etichetta. Duplicarlo qui creerebbe
solo un conflitto.

Ogni voce di DETECTORS ha la stessa forma del core in app.py:
    (label, regex compilata, validatore o None, strict)
"""

import hashlib
import ipaddress
import re

# --------------------------------------------------------------------------- #
# Defang: separatori alternativi usati per rendere non cliccabili gli indicatori
# --------------------------------------------------------------------------- #
DOT = r"(?:\.|\[\.\]|\(\.\)|\[dot\])"
COLON = r"(?::|\[:\])"

# Coda di uno span "lungo" (URL, path, risorsa cloud): prende tutto tranne gli spazi,
# ma non puo' FINIRE con punteggiatura, altrimenti il punto di fine frase o la virgola
# di un elenco entrano nel placeholder (e quindi anche nel dizionario reversibile).
TAIL = r"[^\s<>\"']*[^\s<>\"'.,;:!?)\]]"


def _refang(value):
    """Riporta un indicatore defangato alla forma canonica, per poterlo validare."""
    value = re.sub(r"\[\.\]|\(\.\)|\[dot\]", ".", value, flags=re.IGNORECASE)
    return value.replace("[:]", ":")


# --------------------------------------------------------------------------- #
# Validatori
# --------------------------------------------------------------------------- #
def ip_ok(value):
    """IPv4/IPv6, singolo o come rete CIDR. Usa ipaddress della stdlib: niente
    'regex che sembra giusta' (999.1.1.1, /48 su IPv4, 6 gruppi esadecimali...)."""
    value = _refang(value).strip()
    try:
        if "/" in value:
            ipaddress.ip_network(value, strict=False)
        else:
            ipaddress.ip_address(value)
    except ValueError:
        return False
    return True


def url_ok(value):
    """Forma minima di un URL: schema + host non vuoto. Il validatore serve anche a
    dare all'URL la priorita' sul dominio che contiene (vedi _merge in app.py: a pari
    fonte vince lo span validato, poi il piu' lungo)."""
    value = _refang(value)
    m = re.match(r"[A-Za-z][A-Za-z0-9+.\-]*://([^/\s]+)", value)
    return bool(m and m.group(1))


def hash_ok(value):
    """Hash esadecimale di lunghezza nota. Deve contenere almeno una lettera a-f:
    una sequenza di sole cifre e' quasi sempre un altro identificativo."""
    return bool(re.search(r"[a-fA-F]", value))


def asn_ok(value):
    number = int(re.sub(r"\D", "", value))
    return 0 < number <= 4294967295


_B58 = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"


def btc_ok(value):
    """Base58Check di un indirizzo Bitcoin legacy/P2SH: versione + doppio SHA-256.
    E' l'equivalente del mod-97 dell'IBAN: elimina i falsi positivi su parole lunghe."""
    number = 0
    for char in value:
        index = _B58.find(char)
        if index < 0:
            return False
        number = number * 58 + index
    if number >= 256 ** 25:
        return False
    raw = number.to_bytes(25, "big")
    if raw[0] not in (0x00, 0x05):
        return False
    return hashlib.sha256(hashlib.sha256(raw[:21]).digest()).digest()[:4] == raw[21:]


# --------------------------------------------------------------------------- #
# Domini: TLD piu' comuni + suffissi delle reti interne.
# Esclusi di proposito i TLD che sono anche estensioni di file (zip, mov, app):
# "allegato.zip" non e' un dominio.
# --------------------------------------------------------------------------- #
TLDS = """
com net org edu gov mil int info biz name pro io co me tv cc ly sh gg
it eu de fr uk es nl be ch at pt ie dk se no fi is pl cz sk hu ro bg gr hr si lt lv ee
ru ua by tr il ae sa in cn jp kr hk sg my th vn id ph au nz za eg ma ng ke
us ca mx br ar cl pe uy ve
dev cloud tech online site website space store shop blog news live link click
ai app agency digital email network solutions systems security expert
local internal lan corp intranet intra home arpa onion test invalid example localdomain
""".split()
_TLD_ALT = "|".join(sorted(TLDS, key=len, reverse=True))

_LABEL = r"[A-Za-z0-9](?:[A-Za-z0-9\-]{0,61}[A-Za-z0-9])?"

# --------------------------------------------------------------------------- #
# Detector (stessa forma delle voci di DETECTORS in app.py)
# --------------------------------------------------------------------------- #
DETECTORS = [
    # URL prima del dominio: e' lo span piu' lungo e validato, quindi vince il merge.
    ("URL",
     re.compile(rf"(?<![\w@])(?:h[xt]{{2}}ps?|ftps?|wss?){COLON}//{TAIL}"),
     url_ok, True),

    # IPv4 (anche in forma CIDR e defangata). Il lookbehind esclude le versioni
    # software scritte come v1.2.3.4; "versione 1.2.3.4" resta un falso positivo noto.
    # La coda e' (?!\.?\d) e non (?![\w.]): con quest'ultima un indirizzo a FINE FRASE
    # ("il C2 e' 203.0.113.42.") non veniva rilevato — falso negativo, cioe' un leak.
    ("IP",
     re.compile(rf"(?<![\w.])\d{{1,3}}(?:{DOT}\d{{1,3}}){{3}}(?:/\d{{1,2}})?(?!\.?\d)(?![\w])"),
     ip_ok, True),

    # IPv6: la regex e' larga di proposito, la selezione la fa ipaddress
    # (cosi' un MAC 00:1A:2B:3C:4D:5E o un orario 15:30:45 vengono scartati).
    ("IP",
     re.compile(r"(?<![\w:.])(?:[0-9A-Fa-f]{0,4}:){2,7}[0-9A-Fa-f]{0,4}"
                r"(?:/\d{1,3})?(?![\w:])"),
     ip_ok, True),

    ("MAC",
     re.compile(r"(?<![\w:.\-])(?:[0-9A-Fa-f]{2}[:\-]){5}[0-9A-Fa-f]{2}(?![\w:\-])"
                r"|(?<![\w.\-])(?:[0-9A-Fa-f]{4}\.){2}[0-9A-Fa-f]{4}(?![\w.\-])"),
     None, True),

    ("HASH",
     re.compile(r"(?<![0-9A-Za-z])(?:[0-9a-fA-F]{128}|[0-9a-fA-F]{64}"
                r"|[0-9a-fA-F]{40}|[0-9a-fA-F]{32})(?![0-9A-Za-z])"),
     hash_ok, True),

    # Wallet. Ethereum: solo forma (il checksum EIP-55 richiede keccak256, che NON e'
    # in hashlib — sha3_256 e' un altro algoritmo, confonderli e' un errore classico).
    ("WALLET",
     re.compile(r"(?<![\w])0x[0-9a-fA-F]{40}(?![\w])"
                r"|(?<![\w])bc1[023456789acdefghjklmnpqrstuvwxyz]{11,71}(?![\w])"),
     None, True),
    ("WALLET",
     re.compile(r"(?<![\w])[13][a-km-zA-HJ-NP-Z1-9]{25,34}(?![\w])"),
     btc_ok, True),

    ("CLOUDID",
     re.compile(rf"arn:[a-z0-9\-]*:[a-z0-9\-]+:[a-z0-9\-]*:\d{{0,12}}:{TAIL}"
                rf"|(?:s3|gs|az|abfss?|wasbs?)://{TAIL}"
                r"|(?<![\w\-])(?:i|vol|ami|snap|sg|vpc|subnet|eni|acl|rtb|igw|nat)"
                r"-[0-9a-f]{8,17}(?![\w\-])"
                r"|(?<![\w\-])[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}"
                r"-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}(?![\w\-])", re.IGNORECASE),
     None, True),

    # Solo i percorsi che incorporano un'identita' (utente o host): /etc/passwd non
    # identifica nessuno, C:\Users\m.rossi e \\FS01\condivisa si'.
    ("PATH",
     re.compile(rf"[A-Za-z]:\\Users\\{TAIL}"
                rf"|\\\\[A-Za-z0-9._\-]+\\{TAIL}"
                rf"|/(?:home|Users|root)/{TAIL}", re.IGNORECASE),
     None, True),

    # Account di dominio (DOMINIO\utente) e utenze di servizio.
    ("USER",
     re.compile(r"(?<![\\\w])[A-Za-z][A-Za-z0-9._\-]{1,30}\\[A-Za-z][A-Za-z0-9._\-]{1,30}"
                r"(?![\\\w])"
                r"|(?<![\w\-])(?:svc|srv|sa)[._\-][A-Za-z0-9._\-]{2,30}(?![\w\-])"),
     None, True),

    ("ASN",
     re.compile(r"(?<![\w])AS\d{1,10}(?![\w])"),
     asn_ok, True),

    # Il dominio per ultimo: se e' dentro un URL, vince l'URL (piu' lungo e validato).
    ("DOMAIN",
     re.compile(rf"(?<![\w@.\-])(?:{_LABEL}{DOT})+(?:{_TLD_ALT})(?![\w\-])",
                re.IGNORECASE),
     None, True),
]

# --------------------------------------------------------------------------- #
# Keeplist: riferimenti PUBBLICI che nessun detector deve mascherare.
# Volutamente ristretta ai prefissi inequivocabili (vedi il docstring in testa).
# --------------------------------------------------------------------------- #
KEEP_PATTERNS = [
    re.compile(r"\bCVE-\d{4}-\d{4,7}\b", re.IGNORECASE),
    re.compile(r"\bCWE-\d{1,4}\b", re.IGNORECASE),
    re.compile(r"\bCAPEC-\d{1,4}\b", re.IGNORECASE),
    re.compile(r"\bRFC\s?\d{1,5}\b", re.IGNORECASE),
    re.compile(r"\bTA?\d{4}(?:\.\d{3})?\b"),          # tecniche/tattiche ATT&CK
    re.compile(r"\b(?:attack\.mitre\.org|cve\.mitre\.org|nvd\.nist\.gov|cwe\.mitre\.org)"
               r"(?:/[^\s\"'<>]*)?", re.IGNORECASE),
]
