#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generatore di dataset sintetico PII per il dominio legale italiano.

Livello 1: entita' strutturate con CHECKSUM VALIDI (Codice Fiscale, Partita IVA, IBAN).
Livello 2: slot-filling su template di atti italiani -> testo + label BIO esatte (zero annotazione manuale).

Output: JSONL, una riga per esempio:
  {
    "source_text": "...",
    "language": "it",
    "template_id": 3,
    "entities": [{"value": "...", "label": "CF", "start": 10, "end": 26}, ...],
    "tokens": ["Il", "sottoscritto", ...],
    "bio_labels": ["O", "O", "B-GIVENNAME", ...]
  }

Nessuna dipendenza esterna. Esecuzione: python generate_synthetic_pii.py
"""

import json
import random
import re
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SYNTH_DIR = ROOT / "dataset" / "synthetic"

random.seed(42)  # riproducibile

# --------------------------------------------------------------------------- #
# Tabelle e dati di base                                                       #
# --------------------------------------------------------------------------- #

MALE_NAMES = ["Mario", "Luca", "Giuseppe", "Francesco", "Alessandro", "Antonio",
              "Marco", "Andrea", "Stefano", "Roberto", "Giovanni", "Matteo",
              "Davide", "Simone", "Lorenzo", "Federico", "Riccardo", "Daniele",
              "Gabriele", "Emanuele", "Vincenzo", "Salvatore", "Domenico", "Paolo",
              "Michele", "Nicola", "Filippo", "Tommaso", "Pietro", "Carlo",
              "Enrico", "Fabio", "Claudio", "Sergio", "Massimo", "Giorgio",
              "Alberto", "Edoardo", "Cristian", "Manuel",
              "Angelo", "Raffaele", "Luigi", "Franco", "Bruno", "Aldo", "Renato",
              "Dario", "Mauro", "Walter", "Gianluca", "Gianni", "Maurizio", "Valerio",
              "Alessio", "Diego", "Samuele", "Giacomo", "Leonardo", "Mattia", "Nicolo'",
              "Elia", "Christian", "Ettore", "Umberto", "Vittorio", "Cesare", "Guido",
              "Marcello", "Fabrizio", "Gaetano", "Rocco", "Cosimo", "Donato", "Pasquale",
              "Saverio", "Silvio", "Ugo", "Vito", "Adriano", "Amedeo", "Arturo", "Augusto",
              "Corrado", "Dino", "Elio", "Ennio", "Ezio", "Ivano", "Luciano", "Oreste",
              "Osvaldo", "Remo", "Tiziano", "Tullio", "Valentino", "Gennaro", "Ignazio",
              "Pierluigi", "Giancarlo", "Gianfranco", "Gianmarco", "Flavio", "Achille"]
FEMALE_NAMES = ["Giulia", "Anna", "Francesca", "Maria", "Sara", "Laura",
                "Elena", "Chiara", "Valentina", "Martina", "Federica", "Paola",
                "Alessia", "Silvia", "Roberta", "Cristina", "Giorgia", "Beatrice",
                "Alice", "Aurora", "Greta", "Ilaria", "Marta", "Noemi",
                "Caterina", "Eleonora", "Veronica", "Camilla", "Arianna", "Sofia",
                "Lucia", "Teresa", "Angela", "Rosa", "Carla", "Monica",
                "Daniela", "Simona", "Barbara", "Antonella",
                "Giovanna", "Rita", "Franca", "Luisa", "Patrizia", "Stefania", "Sabrina",
                "Manuela", "Claudia", "Emanuela", "Loredana", "Gabriella", "Raffaella",
                "Tiziana", "Cinzia", "Sonia", "Nadia", "Michela", "Serena", "Erica",
                "Debora", "Vanessa", "Jessica", "Melissa", "Rebecca", "Asia", "Emma",
                "Ginevra", "Vittoria", "Bianca", "Matilde", "Nicole", "Gaia", "Viola",
                "Diletta", "Letizia", "Adele", "Agnese", "Benedetta", "Carlotta", "Cecilia",
                "Costanza", "Elisa", "Elisabetta", "Flavia", "Gloria", "Irene", "Lara",
                "Lavinia", "Margherita", "Marina", "Miriam", "Ornella", "Rosanna", "Susanna",
                "Valeria", "Concetta", "Carmela", "Assunta", "Filomena", "Pierina", "Annalisa"]
SURNAMES = ["Rossi", "Bianchi", "Ferrari", "Russo", "Esposito", "Romano",
            "Colombo", "Ricci", "Marino", "Greco", "De Luca", "Conti",
            "Gallo", "Costa", "Fontana", "Bruno", "Rizzo", "Moretti", "Barbieri",
            "Mancini", "Lombardi", "Giordano", "Rinaldi", "Caruso", "Ferrara",
            "Galli", "Martini", "Leone", "Longo", "Gentile", "Martinelli",
            "Vitale", "Lombardo", "Serra", "Coppola", "De Santis", "D'Angelo",
            "Marchetti", "Parisi", "Villa", "Conte", "Ferretti", "Bianco",
            "Fabbri", "Marini", "Grassi", "Santoro", "Pellegrini", "Carbone",
            "Sala", "De Angelis", "Gatti", "Testa", "Montanari", "Guerra",
            "Palumbo", "Sanna", "Farina", "Rizzi", "Monti",
            "Caputo", "Ferraro", "Ferri", "Fiore", "De Rosa", "Battaglia", "Sartori",
            "Neri", "Riva", "Benedetti", "Mariani", "Amato", "Silvestri", "Vitali",
            "Pagano", "Negri", "Basile", "Donati", "Mazza", "Cattaneo", "Valentini",
            "Orlando", "De Simone", "Cocco", "Pellegrino", "Bernardi", "Castelli",
            "Antonelli", "Santini", "Bellini", "Brambilla", "Palmieri", "Pace",
            "Sorrentino", "Morelli", "Ruggiero", "Bartolini", "Piras", "Melis", "Pinna",
            "Pisani", "Aiello", "Costantini", "Catalano", "De Marco", "Giuliani", "Manca",
            "Mele", "Nardi", "Tedesco", "Zanetti", "Bevilacqua", "Ceccarelli",
            "Di Stefano", "Di Marco", "Di Maio", "La Rosa", "Lo Russo", "Cervone",
            "Damiani", "Guerrini", "Marchesi", "Carli", "Franchi", "Belli", "Rossetti"]

# citta' con relativo codice catastale (Belfiore) e sigla provincia (codici reali:
# il CF resta con checksum valido; ampliabile aggiungendo altre coppie qui)
CITIES = {
    "Roma": ("H501", "RM"), "Milano": ("F205", "MI"), "Napoli": ("F839", "NA"),
    "Torino": ("L219", "TO"), "Bologna": ("A944", "BO"), "Firenze": ("D612", "FI"),
    "Genova": ("D969", "GE"), "Palermo": ("G273", "PA"), "Bari": ("A662", "BA"),
    "Catania": ("C351", "CT"), "Venezia": ("L736", "VE"), "Verona": ("L781", "VR"),
    "Padova": ("G224", "PD"), "Trieste": ("L424", "TS"), "Brescia": ("B157", "BS"),
    "Parma": ("G337", "PR"), "Modena": ("F257", "MO"), "Bergamo": ("A794", "BG"),
    "Bolzano": ("A952", "BZ"), "Trento": ("L378", "TN"), "Perugia": ("G478", "PG"),
    "Ancona": ("A271", "AN"), "Cagliari": ("B354", "CA"), "Salerno": ("H703", "SA"),
    "Lecce": ("E506", "LE"), "Rimini": ("H294", "RN"), "Ferrara": ("D548", "FE"),
    "Livorno": ("E625", "LI"), "Pisa": ("G702", "PI"), "Lucca": ("E715", "LU"),
    "Latina": ("E472", "LT"), "Cosenza": ("D086", "CS"), "Messina": ("F158", "ME"),
    "Como": ("C933", "CO"), "Monza": ("F704", "MB"), "Novara": ("F952", "NO"),
    "Udine": ("L483", "UD"), "Treviso": ("L407", "TV"), "Cremona": ("D150", "CR"),
    "Pavia": ("G388", "PV"),
    # --- ampliamento: capoluoghi + citta' maggiori (codici Belfiore/provincia reali) ---
    "Reggio Calabria": ("H224", "RC"), "Reggio Emilia": ("H223", "RE"),
    "Vicenza": ("L840", "VI"), "Foggia": ("D643", "FG"), "Taranto": ("L049", "TA"),
    "Sassari": ("I452", "SS"), "Pescara": ("G482", "PE"), "Ravenna": ("H199", "RA"),
    "Forli'": ("D704", "FC"), "Piacenza": ("G535", "PC"), "Catanzaro": ("C352", "CZ"),
    "Terni": ("L117", "TR"), "Trapani": ("L331", "TP"), "Siracusa": ("I754", "SR"),
    "Brindisi": ("B180", "BR"), "Pistoia": ("G713", "PT"), "Arezzo": ("A390", "AR"),
    "Caserta": ("B963", "CE"), "Varese": ("L682", "VA"), "Asti": ("A479", "AT"),
    "Alessandria": ("A182", "AL"), "La Spezia": ("E463", "SP"), "Grosseto": ("E202", "GR"),
    "Avellino": ("A509", "AV"), "Benevento": ("A783", "BN"), "Potenza": ("G942", "PZ"),
    "Matera": ("F052", "MT"), "Crotone": ("D122", "KR"), "Vibo Valentia": ("F537", "VV"),
    "Aosta": ("A326", "AO"), "Savona": ("I480", "SV"), "Imperia": ("E290", "IM"),
    "Cuneo": ("D205", "CN"), "Biella": ("A859", "BI"), "Vercelli": ("L750", "VC"),
    "Lodi": ("E648", "LO"), "Lecco": ("E507", "LC"), "Sondrio": ("I829", "SO"),
    "Mantova": ("E897", "MN"), "Rovigo": ("H620", "RO"), "Belluno": ("A757", "BL"),
    "Pordenone": ("G888", "PN"), "Gorizia": ("E098", "GO"), "Massa": ("F023", "MS"),
    "Prato": ("G999", "PO"), "Siena": ("I726", "SI"), "Viterbo": ("M082", "VT"),
    "Rieti": ("H282", "RI"), "Frosinone": ("D810", "FR"), "Teramo": ("L103", "TE"),
    "Chieti": ("C632", "CH"), "L'Aquila": ("A345", "AQ"), "Campobasso": ("B519", "CB"),
    "Isernia": ("E335", "IS"), "Caltanissetta": ("B429", "CL"), "Agrigento": ("A089", "AG"),
    "Enna": ("C342", "EN"), "Ragusa": ("H163", "RG"), "Nuoro": ("F979", "NU"),
    "Oristano": ("G113", "OR"), "Macerata": ("E783", "MC"), "Ascoli Piceno": ("A462", "AP"),
    "Fermo": ("D542", "FM"), "Pesaro": ("G479", "PU"), "Andria": ("A285", "BT"),
    "Barletta": ("A669", "BT"), "Cesena": ("C573", "FC"), "Carpi": ("B819", "MO"),
    "Pozzuoli": ("G964", "NA"), "Giugliano in Campania": ("E054", "NA"),
    "Torre del Greco": ("L259", "NA"), "Marsala": ("E974", "TP"), "Gela": ("D960", "CL"),
    "Aprilia": ("A341", "LT"), "Velletri": ("L719", "RM"), "Civitavecchia": ("C773", "RM"),
    "Tivoli": ("L182", "RM"), "Pomezia": ("G811", "RM"), "Anzio": ("A323", "RM"),
    "Aversa": ("A512", "CE"), "Battipaglia": ("A717", "SA"), "Cava de' Tirreni": ("C361", "SA"),
    "Scafati": ("I483", "SA"), "Nocera Inferiore": ("F912", "SA"), "Eboli": ("D390", "SA"),
    "Acireale": ("A028", "CT"), "Bagheria": ("A546", "PA"), "Modica": ("F258", "RG"),
    "Vittoria": ("M088", "RG"), "Faenza": ("D458", "RA"), "Imola": ("E289", "BO"),
    "Carrara": ("B832", "MS"), "Bassano del Grappa": ("A703", "VI"), "Sassuolo": ("I462", "MO"),
    "Formia": ("D708", "LT"),
}
CITY_NAMES = list(CITIES.keys())

STREETS = ["Via Roma", "Via Garibaldi", "Via Mazzini", "Corso Italia", "Via Dante",
           "Viale Europa", "Via Verdi", "Piazza Marconi", "Via Manzoni", "Via Cavour",
           "Via Nazionale", "Via Veneto", "Corso Vittorio Emanuele", "Via XX Settembre",
           "Viale della Repubblica", "Via San Martino", "Piazza Garibaldi", "Via Carducci",
           "Via Leopardi", "Via Foscolo", "Largo Augusto", "Via Battisti", "Via Marconi",
           "Viale dei Tigli", "Via delle Rose", "Corso Umberto I", "Via Petrarca",
           "Via Galilei", "Via Volta", "Via Diaz", "Piazza Dante", "Via Mameli",
           "Via Bixio", "Viale Kennedy", "Via Bellini", "Via Rossini",
           # toponimi geografici
           "Via Trieste", "Via Trento", "Via Bologna", "Via Milano", "Via Torino",
           "Via Firenze", "Via Napoli", "Via Venezia", "Via Sicilia", "Via Sardegna",
           "Via Calabria", "Via Toscana", "Via Lombardia", "Via Piave", "Via Isonzo",
           "Via Po", "Via Adige", "Via Arno", "Via Tevere", "Via Ticino",
           # figure storiche/politiche
           "Via Gramsci", "Via Matteotti", "Via Don Minzoni", "Via Aldo Moro",
           "Via De Gasperi", "Via Einaudi", "Via Turati", "Via Cairoli", "Via Mentana",
           # letterati / artisti / scienziati
           "Via Pascoli", "Via Ungaretti", "Via Montale", "Via Pirandello", "Via Alfieri",
           "Via Goldoni", "Via Machiavelli", "Via Galvani", "Via Fermi", "Via Meucci",
           "Via Michelangelo", "Via Raffaello", "Via Donatello", "Via Tiziano",
           "Via Caravaggio", "Via Giotto", "Via Bernini", "Via Canova",
           # altri tipi di odonimo (Viale/Corso/Piazza/Largo/Vicolo/Strada/Borgo/Lungo...)
           "Corso Garibaldi", "Corso Mazzini", "Corso Buenos Aires", "Corso Francia",
           "Corso Sempione", "Corso Magenta", "Corso Porta Nuova",
           "Viale Trastevere", "Viale Libertà", "Viale Italia", "Viale dei Mille",
           "Viale delle Acacie", "Viale dei Platani", "Viale Trieste",
           "Piazza San Marco", "Piazza Duomo", "Piazza della Repubblica", "Piazza Cavour",
           "Piazza Vittorio", "Piazza della Libertà", "Piazza del Popolo",
           "Largo Argentina", "Largo Cairoli", "Largo Europa",
           "Vicolo Stretto", "Vicolo del Pozzo", "Vicolo dei Soldati",
           "Strada Provinciale", "Strada Statale 16", "Strada Maggiore",
           "Borgo Pinti", "Borgo San Frediano", "Salita Castello",
           "Lungomare Caracciolo", "Lungarno Vespucci", "Lungotevere dei Mellini",
           "Via dei Mille", "Via delle Acacie", "Via dei Platani", "Via dei Glicini",
           "Via delle Magnolie", "Via dei Gelsomini", "Via delle Camelie",
           "Traversa Marina", "Rotonda Diaz"]

# i tribunali si generano dalle citta' (vedi tribunal_piece): varieta' gratuita
TRIBUNAL_TYPES = ["Tribunale di", "Corte d'Appello di", "Tribunale Ordinario di",
                  "Procura della Repubblica di", "Giudice di Pace di"]

# tabella dispari (ODD) per check char del Codice Fiscale / CIN IBAN
ODD = {
    "0": 1, "1": 0, "2": 5, "3": 7, "4": 9, "5": 13, "6": 15, "7": 17, "8": 19,
    "9": 21, "A": 1, "B": 0, "C": 5, "D": 7, "E": 9, "F": 13, "G": 15, "H": 17,
    "I": 19, "J": 21, "K": 2, "L": 4, "M": 18, "N": 20, "O": 11, "P": 3, "Q": 6,
    "R": 8, "S": 12, "T": 14, "U": 16, "V": 10, "W": 22, "X": 25, "Y": 24, "Z": 23,
}
def _even_val(c):
    return int(c) if c.isdigit() else ord(c) - ord("A")

MONTH_LETTERS = "ABCDEHLMPRST"
PLATE_LETTERS = "ABCDEFGHJKLMNPRSTVWXYZ"  # targhe IT: escluse I, O, Q, U

# --------------------------------------------------------------------------- #
# Generatori di entita' strutturate con checksum valido                       #
# --------------------------------------------------------------------------- #

def _strip(s):
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    return re.sub(r"[^A-Za-z]", "", s).upper()

def _cons(s):
    return [c for c in s if c in "BCDFGHJKLMNPQRSTVWXYZ"]

def _vow(s):
    return [c for c in s if c in "AEIOU"]

def _surname_code(s):
    s = _strip(s)
    code = (_cons(s) + _vow(s) + list("XXX"))[:3]
    return "".join(code)

def _name_code(s):
    s = _strip(s)
    c = _cons(s)
    if len(c) >= 4:
        code = [c[0], c[2], c[3]]
    else:
        code = (c + _vow(s) + list("XXX"))[:3]
    return "".join(code)

def codice_fiscale(given, surname, sex, year, month, day, city):
    belfiore = CITIES[city][0]
    dd = day + 40 if sex == "F" else day
    body = (_surname_code(surname) + _name_code(given) +
            f"{year % 100:02d}" + MONTH_LETTERS[month - 1] + f"{dd:02d}" + belfiore)
    tot = sum((ODD[ch] if i % 2 == 0 else _even_val(ch)) for i, ch in enumerate(body))
    return body + chr(ord("A") + tot % 26)

def partita_iva():
    d = [random.randint(0, 9) for _ in range(10)]
    tot = 0
    for i, n in enumerate(d):
        if i % 2 == 0:          # posizioni dispari (1-indexed) = indici pari
            tot += n
        else:
            x = n * 2
            tot += x - 9 if x > 9 else x
    check = (10 - tot % 10) % 10
    return "".join(map(str, d)) + str(check)

def _cin(bban22):
    tot = sum((ODD[ch] if i % 2 == 0 else _even_val(ch)) for i, ch in enumerate(bban22))
    return chr(ord("A") + tot % 26)

def iban_it():
    abi = f"{random.randint(0, 99999):05d}"
    cab = f"{random.randint(0, 99999):05d}"
    conto = f"{random.randint(0, 10**12 - 1):012d}"
    bban22 = abi + cab + conto
    bban = _cin(bban22) + bban22                      # CIN + ABI + CAB + conto
    rearr = bban + "IT00"
    num = int("".join(str(ord(c) - 55) if c.isalpha() else c for c in rearr))
    check = 98 - num % 97
    return f"IT{check:02d}{bban}"

# --------------------------------------------------------------------------- #
# Helper di alto livello (ritornano liste di pezzi (testo, label|None))       #
# --------------------------------------------------------------------------- #

def _person():
    if random.random() < 0.5:
        return random.choice(MALE_NAMES), random.choice(SURNAMES), "M"
    return random.choice(FEMALE_NAMES), random.choice(SURNAMES), "F"

def _date():
    d, m, y = random.randint(1, 28), random.randint(1, 12), random.randint(1955, 2005)
    return f"{d:02d}/{m:02d}/{y}", (d, m, y)

def full_name():
    g, s, _ = _person()
    return [(g, "GIVENNAME"), (" ", None), (s, "SURNAME")]

def role(label):
    g, s, _ = _person()
    return [(f"{g} {s}", label)]   # nome intero taggato col ruolo legale

def cf_piece():
    g, s, sex = _person()
    _, (d, m, y) = _date()
    return [(codice_fiscale(g, s, sex, y, m, d, random.choice(CITY_NAMES)), "CF")]

def address():
    city = random.choice(CITY_NAMES)
    prov = CITIES[city][1]
    cap = f"{random.randint(10, 98):02d}{random.randint(0, 999):03d}"
    return [(random.choice(STREETS), "STREET"), (" ", None),
            (str(random.randint(1, 250)), "BUILDINGNUM"), (", ", None),
            (cap, "ZIPCODE"), (" ", None),
            (city, "CITY"), (" (", None), (prov, "PROVINCE"), (")", None)]

def email_piece(domain="example.it"):
    g, s, _ = _person()
    user = f"{_strip(g).lower()}.{_strip(s).lower()}"
    return [(f"{user}@{domain}", "PEC" if domain == "pec.it" else "EMAIL")]

def phone_piece():
    return [(f"+39 3{random.randint(0,99):02d} {random.randint(1000000,9999999)}", "TELEPHONENUM")]

def amount_piece():
    euros = random.randint(500, 250000)
    return [(f"€ {euros:,}".replace(",", ".") + ",00", "AMOUNT")]

def rg_piece():
    return [(f"{random.randint(100, 9999)}/{random.randint(2018, 2025)}", "RG")]

def tribunal_piece():
    # il tribunale (ente pubblico) NON e' PII: il nome resta nel testo come contesto O
    return [(f"{random.choice(TRIBUNAL_TYPES)} {random.choice(CITY_NAMES)}", None)]

def targa_piece():
    L = lambda n: "".join(random.choice(PLATE_LETTERS) for _ in range(n))
    return [(f"{L(2)} {random.randint(100,999)} {L(2)}", "TARGA")]

def idcard_piece():
    L = lambda n: "".join(random.choice(PLATE_LETTERS) for _ in range(n))
    return [(f"CA{random.randint(0,99999):05d}{L(2)}", "IDCARDNUM")]

def driving_piece():
    L = lambda n: "".join(random.choice(PLATE_LETTERS) for _ in range(n))
    return [(f"{L(2)}{random.randint(1000000,9999999)}{L(1)}", "DRIVERLICENSENUM")]

def city_piece():
    return [(random.choice(CITY_NAMES), "CITY")]

def province_piece():
    return [(CITIES[random.choice(CITY_NAMES)][1], "PROVINCE")]

def date_piece():
    return [(_date()[0], "DATE")]

def piva_piece():
    return [(partita_iva(), "PIVA")]

def iban_piece():
    return [(iban_it(), "IBAN")]

# stem per ragioni sociali (nomi di fantasia, nessun riferimento reale)
COMPANY_STEMS = ["Adriatica", "Meridionale", "Lombarda", "Tirrenica", "Alfa", "Beta",
                 "Costruzioni Moderne", "Logistica Veloce", "Tecnoimpianti",
                 "Immobiliare Centro", "Edilnord", "Sviluppo Sud", "Gamma", "Delta",
                 "Nuova Edilizia", "Servizi Integrati", "Consulenza Globale",
                 "Trasporti Riuniti", "Italtecnica", "Mediterranea", "Progetti e Forniture",
                 "Industriale Padana", "Commerciale Veneta", "Energia Futura",
                 "Sistemi Avanzati", "Officine Riunite", "Agricola del Sole",
                 "Grafiche Moderne", "Impianti Sicuri", "Distribuzione Italia",
                 "Padana", "Subalpina", "Triveneta", "Romagnola", "Siciliana", "Sarda",
                 "Verde Ambiente", "Tecno Service", "Global Trade", "Smart Solutions",
                 "Costa Azzurra", "Val di Sole", "Pianura Servizi", "Alpe Adria",
                 "Manifatture Italiane", "Cantieri Navali", "Acque Potabili",
                 "Idraulica del Nord", "Termotecnica", "Elettroforniture", "Carpenterie Unite",
                 "Editoriale Nuova", "Farmaceutica Centrale", "Chimica Industriale"]

# brand a parola/forma "moderna" (spesso senza suffisso legale: caso piu' difficile)
BRAND_WORDS = ["Tecnova", "Inforge", "Logitalia", "Edilmax", "Aurea", "Novamed",
               "Italgest", "Sintex", "Prometeo", "Quadrifoglio", "Ergonet", "Sferanet",
               "Biotecna", "Geocart", "Mediaplan", "Tecnosistemi", "Idealcasa",
               "Ecofuturo", "Solaria", "Faber", "Vianova", "Plurima", "Univerde",
               "Demetra", "Kairos", "Helios", "Sinergia", "Zenit", "Klimex", "Optima",
               "Nordest Servizi", "Cantieri del Sud", "Gruppo Vesuvio"]

LEGAL_FORMS = ["S.r.l.", "S.p.A.", "S.n.c.", "S.a.s.", "S.r.l.s.", "Soc. Coop.",
               "S.c.a.r.l.", "& C. S.a.s."]
FOREIGN_FORMS = ["Ltd", "GmbH", "S.A.", "B.V.", "Group", "International", "Italia S.p.A."]
ORG_KINDS = ["Cooperativa", "Societa' Cooperativa", "Consorzio", "Fondazione",
             "Associazione", "Onlus", "Ente"]

def _acronym():
    return "".join(random.choice("ABCDEFGHILMNOPRSTUVZ") for _ in range(random.randint(2, 4)))

def org_piece():
    """Nome di societa'/studio/banca/ente: tutta la stringa e' una sola entita' ORG.
    Tante forme diverse (suffissi legali, brand nudi, acronimi, gruppi, cooperative,
    sigle estere, ditte familiari) per non far imparare al modello solo 'X S.r.l.'."""
    r = random.random()
    if r < 0.20:
        name = f"{random.choice(COMPANY_STEMS)} {random.choice(LEGAL_FORMS)}"
    elif r < 0.32:
        name = f"Studio Legale {random.choice(SURNAMES)}" + random.choice(["", " & Associati", " e Associati"])
    elif r < 0.40:
        name = f"Banca {random.choice(COMPANY_STEMS)} S.p.A."
    elif r < 0.52:
        name = random.choice(BRAND_WORDS) + random.choice(["", "", " Italia", " Group", f" {random.choice(LEGAL_FORMS)}"])
    elif r < 0.60:
        name = f"Gruppo {random.choice(COMPANY_STEMS + BRAND_WORDS)}"
    elif r < 0.68:
        name = f"{random.choice(ORG_KINDS)} {random.choice(COMPANY_STEMS)}"
    elif r < 0.76:
        name = _acronym() + random.choice(["", "", f" {random.choice(LEGAL_FORMS)}", " S.p.A."])
    elif r < 0.85:
        s = random.choice(SURNAMES)
        name = random.choice([f"F.lli {s}", f"{s} & Figli", f"{s} & C.",
                              f"{s} {random.choice(SURNAMES)} {random.choice(LEGAL_FORMS)}"])
    elif r < 0.93:
        name = f"{random.choice(COMPANY_STEMS)} {random.choice(FOREIGN_FORMS)}"
    else:
        name = random.choice(BRAND_WORDS)   # brand nudo, nessun suffisso legale
    return [(name, "ORG")]

def docid_piece():
    """Codice di un atto (protocollo/repertorio/sentenza): solo il codice, il prefisso
    (Prot. n., sentenza n., Rep. n.) sta nel template — come RG."""
    forms = [f"{random.randint(1000, 99999)}/{random.randint(2018, 2025)}",
             f"{random.randint(1, 999)}/{random.randint(2018, 2025)}",
             f"{random.randint(10000, 999999)}"]
    return [(random.choice(forms), "DOCID")]

def catasto_piece():
    """Dati catastali: i numeri (foglio/particella/sub) sono CATASTO, le parole-chiave O."""
    return [("Foglio ", None), (str(random.randint(1, 200)), "CATASTO"),
            (", particella ", None), (str(random.randint(1, 999)), "CATASTO"),
            (", sub. ", None), (str(random.randint(1, 30)), "CATASTO")]

def conto_piece():
    """Numero di conto corrente (12 cifre), distinto dall'IBAN."""
    return [(f"{random.randint(0, 10**12 - 1):012d}", "CONTO")]

# separatori d'elenco: OGNUNO deve produrre un TOKEN (virgola/';'/'-'/parola 'e').
# Quel token e' O -> normalize_labels() resetta prev e tiene le entita' DISTINTE
# (due nomi consecutivi NON si fondono in un'unica B-/I-). NIENTE '\n' puro: lo
# whitespace non viene tokenizzato -> i nomi si fonderebbero in un'unica entita'.
LIST_SEPS = [", ", ", ", "; ", "\n- ", " - ", " e "]

def name_list():
    """Elenco di 2-7 nomi persona separati (caso 'nomi uno dopo l'altro')."""
    sep = random.choice(LIST_SEPS)
    out = []
    for i in range(random.randint(2, 7)):
        if i:
            out.append((sep, None))
        out.extend(full_name())
    return out

def org_list():
    """Elenco di 2-6 societa'/enti separati."""
    sep = random.choice(LIST_SEPS)
    out = []
    for i in range(random.randint(2, 6)):
        if i:
            out.append((sep, None))
        out.extend(org_piece())
    return out

def mixed_list():
    """Elenco misto persone+societa' (es. parti, presenti, soggetti coinvolti)."""
    sep = random.choice(LIST_SEPS)
    out = []
    for i in range(random.randint(3, 7)):
        if i:
            out.append((sep, None))
        out.extend(full_name() if random.random() < 0.5 else org_piece())
    return out

# mappa slot -> generatore
SLOTS = {
    "FULLNAME": full_name,
    "JUDGE": lambda: role("GIUDICE"),
    "LAWYER": lambda: role("AVVOCATO"),
    "PLAINTIFF": lambda: role("ATTORE"),
    "DEFENDANT": lambda: role("CONVENUTO"),
    "WITNESS": lambda: role("TESTIMONE"),
    "CF": cf_piece, "PIVA": piva_piece, "IBAN": iban_piece,
    "ADDRESS": address, "EMAIL": email_piece, "PEC": lambda: email_piece("pec.it"),
    "PHONE": phone_piece, "AMOUNT": amount_piece, "RG": rg_piece,
    "TRIBUNAL": tribunal_piece, "TARGA": targa_piece, "IDCARD": idcard_piece,
    "DRIVING": driving_piece, "CITY": city_piece, "DATE": date_piece,
    "ORG": org_piece, "DOCID": docid_piece, "CATASTO": catasto_piece,
    "CONTO": conto_piece, "PROVINCE": province_piece,
    "NAMELIST": name_list, "ORGLIST": org_list, "MIXEDLIST": mixed_list,
}

# --------------------------------------------------------------------------- #
# Template di atti italiani                                                    #
# --------------------------------------------------------------------------- #

TEMPLATES = [
    "Con atto di citazione l'avvocato {LAWYER}, nell'interesse dell'attore {PLAINTIFF}, "
    "conviene in giudizio {DEFENDANT} dinanzi al {TRIBUNAL}, R.G. n. {RG}.",

    "Il {TRIBUNAL}, nella persona del giudice {JUDGE}, ha pronunciato la seguente "
    "sentenza nella causa iscritta al R.G. n. {RG}.",

    "Il sottoscritto {FULLNAME}, nato a {CITY} il {DATE}, C.F. {CF}, "
    "residente in {ADDRESS}, dichiara quanto segue.",

    "Le somme dovute, pari a {AMOUNT}, dovranno essere versate sull'IBAN {IBAN} "
    "intestato a {FULLNAME}.",

    "Per ogni comunicazione si prega di contattare il numero {PHONE} o l'indirizzo "
    "email {EMAIL} (PEC: {PEC}).",

    "La societa', P.IVA {PIVA}, con sede in {ADDRESS}, in persona del legale "
    "rappresentante {FULLNAME}, conferisce mandato.",

    "Si ingiunge a {DEFENDANT}, C.F. {CF}, di pagare la somma di {AMOUNT} oltre "
    "interessi, come da decreto del giudice {JUDGE}.",

    "Documento d'identita' n. {IDCARD}, patente n. {DRIVING}, veicolo targato {TARGA}, "
    "intestati a {FULLNAME}.",

    "Delego l'avv. {LAWYER}, C.F. {CF}, a rappresentarmi con domicilio eletto in "
    "{ADDRESS}, tel. {PHONE}.",

    "Sentito il teste {WITNESS}, residente in {CITY}, nato il {DATE}, "
    "il giudice {JUDGE} dispone il rinvio.",

    "La {ORG}, P.IVA {PIVA}, con sede in {ADDRESS}, conto corrente n. {CONTO}, "
    "in persona del legale rappresentante {FULLNAME}.",

    "Visto il bene immobile sito in {ADDRESS}, censito al Catasto Fabbricati al {CATASTO}, "
    "di proprieta' di {FULLNAME}.",

    "Vista la sentenza n. {DOCID} pronunciata dal {TRIBUNAL}, lo {ORG}, in persona "
    "dell'avv. {LAWYER}, propone appello (prot. n. {DOCID}).",

    # --- elenchi: entita' consecutive separate da virgola/a-capo (caso 'nomi/societa'
    #     uno dopo l'altro', non coperto dai template in prosa) ---
    "Sono comparsi personalmente i signori: {NAMELIST}.",
    "Risultano presenti all'assemblea i soci: {NAMELIST}.",
    "Le parti del presente atto sono: {NAMELIST}.",
    "Si citano quali testimoni: {NAMELIST}.",
    "Eredi legittimi: {NAMELIST}.",
    "Hanno partecipato alla gara le seguenti societa': {ORGLIST}.",
    "Fornitori autorizzati: {ORGLIST}.",
    "Creditori ammessi al passivo: {ORGLIST}.",
    "Si elencano i soggetti coinvolti nel procedimento: {MIXEDLIST}.",
    "Allegato A - anagrafica:\n{MIXEDLIST}",
    "{NAMELIST}",
    "{ORGLIST}",
]

SLOT_RE = re.compile(r"\{(\w+)\}")
TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)

def load_external_templates(path=str(SYNTH_DIR / "legal_templates.json")):
    """Carica i template generati dall'LLM, scartando quelli con slot non gestiti."""
    import os
    if not os.path.exists(path):
        return []
    out = []
    for t in json.load(open(path, encoding="utf-8")):
        slots = set(SLOT_RE.findall(t["text"]))
        if slots and not (slots - set(SLOTS)):
            out.append(t["text"])
    return out

def build_example(template_id, templates):
    template = templates[template_id]
    text = ""
    entities = []
    pos = 0
    for part in re.split(r"(\{\w+\})", template):
        if not part:
            continue
        m = SLOT_RE.fullmatch(part)
        if m:
            for piece_text, label in SLOTS[m.group(1)]():
                start = len(text)
                text += piece_text
                if label is not None:
                    entities.append({"value": piece_text, "label": label,
                                     "start": start, "end": len(text)})
        else:
            text += part
    return text, entities

def to_bio(text, entities):
    tokens, spans = [], []
    for m in TOKEN_RE.finditer(text):
        tokens.append(m.group()); spans.append((m.start(), m.end()))
    labels = []
    for ts, te in spans:
        tag = "O"
        for ent in entities:
            if ts >= ent["start"] and te <= ent["end"]:
                tag = ("B-" if ts == ent["start"] else "I-") + ent["label"]
                break
        labels.append(tag)
    return tokens, labels

# --------------------------------------------------------------------------- #
# Main                                                                         #
# --------------------------------------------------------------------------- #

def main(n=100, out_path=str(SYNTH_DIR / "synthetic_pii_it.jsonl")):
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", type=int, default=n)
    ap.add_argument("--out", default=out_path)
    args = ap.parse_args()
    n, out_path = args.n, args.out

    templates = TEMPLATES + load_external_templates()   # built-in + documenti LLM
    print(f"Template disponibili: {len(templates)} "
          f"({len(TEMPLATES)} built-in + {len(templates) - len(TEMPLATES)} da LLM)")

    label_counts = {}
    with open(out_path, "w", encoding="utf-8") as f:
        for i in range(n):
            tid = random.randrange(len(templates))
            text, entities = build_example(tid, templates)
            tokens, bio = to_bio(text, entities)
            for e in entities:
                label_counts[e["label"]] = label_counts.get(e["label"], 0) + 1
            f.write(json.dumps({
                "source_text": text, "language": "it", "template_id": tid,
                "entities": entities, "tokens": tokens, "bio_labels": bio,
            }, ensure_ascii=False) + "\n")
    print(f"Generati {n} esempi -> {out_path}\n")
    print("Conteggio entita' per label:")
    for label, c in sorted(label_counts.items(), key=lambda x: -x[1]):
        print(f"  {label:18s} {c}")
    print("\nEsempio:")
    text, entities = build_example(2, templates)
    tokens, bio = to_bio(text, entities)
    print(" ", text)
    for tok, tag in zip(tokens, bio):
        if tag != "O":
            print(f"    {tok:20s} {tag}")

if __name__ == "__main__":
    main()
