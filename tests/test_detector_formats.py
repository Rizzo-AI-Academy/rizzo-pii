# -*- coding: utf-8 -*-
"""Formati reali per la rete regex+checksum di src/app/detectors.py.

Ogni caso e' un identificativo scritto come lo si trova STAMPATO in un
documento italiano: l'IBAN raggruppato a quattro di una fattura, un codice
fiscale omocodico, una carta separata da punti, un telefono con i trattini.
Tutti i valori sono SINTETICI e hanno checksum calcolati apposta.
"""

import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "app"))

import detectors  # noqa: E402


def spans(text, label):
    """Sottostringhe rilevate da detect_regex per quel tag."""
    return [text[e["start"]:e["end"]]
            for e in detectors.detect_regex(text) if e["label"] == label]


def validated(text, label):
    """Sottostringhe rilevate E con checksum verificato."""
    return [text[e["start"]:e["end"]]
            for e in detectors.detect_regex(text)
            if e["label"] == label and e["validated"]]


# (nome del caso, testo, tag, valore che deve essere rilevato)
POSITIVI = [
    # IBAN: mod-97 obbligatorio (strict), quindi la regex deve arrivare al
    # validatore anche quando il numero e' stampato con i separatori.
    ("iban_compatto", "Bonifico su IT60X0542811101000000123456 entro il 30/09.",
     "IBAN", "IT60X0542811101000000123456"),
    ("iban_gruppi_di_4", "IBAN IT60 X054 2811 1010 0000 0123 456 presso la filiale.",
     "IBAN", "IT60 X054 2811 1010 0000 0123 456"),
    ("iban_con_trattini", "IBAN: IT60-X054-2811-1010-0000-0123-456.",
     "IBAN", "IT60-X054-2811-1010-0000-0123-456"),
    ("iban_minuscolo", "iban it60x0542811101000000123456 intestato allo studio.",
     "IBAN", "it60x0542811101000000123456"),
    ("iban_estero_gruppi", "Accredito su ES91 2100 0418 4502 0005 1332.",
     "IBAN", "ES91 2100 0418 4502 0005 1332"),
    # Il testo estratto da un PDF non usa sempre lo spazio semplice.
    ("iban_spazi_non_breaking",
     "IBAN IT60\xa0X054\xa02811\xa01010\xa00000\xa00123\xa0456 della filiale.",
     "IBAN", "IT60\xa0X054\xa02811\xa01010\xa00000\xa00123\xa0456"),
    ("iban_a_capo_nel_pdf",
     "Coordinate: IT60 X054 2811\n1010 0000 0123 456 presso la filiale.",
     "IBAN", "IT60 X054 2811\n1010 0000 0123 456"),
    # Codice fiscale, forma canonica e forma omocodica (le cifre diventano
    # 0->L 1->M 2->N 3->P 4->Q 5->R 6->S 7->T 8->U 9->V).
    ("cf_canonico", "Il sig. Rossi, C.F. RSSMRA85H12F205Y, dichiara.",
     "CF", "RSSMRA85H12F205Y"),
    ("cf_omocodia_2_cifre", "Il sig. Rossi, C.F. RSSMRA85H12F2LRE, dichiara.",
     "CF", "RSSMRA85H12F2LRE"),
    ("cf_omocodia_1_cifra", "Codice fiscale BNCLCU90A41H50MO del richiedente.",
     "CF", "BNCLCU90A41H50MO"),
    ("cf_minuscolo", "codice fiscale rssmra85h12f205y del ricorrente.",
     "CF", "rssmra85h12f205y"),
    # Carta di credito: Luhn obbligatorio (strict).
    ("carta_spazi", "Pagamento con carta 4111 1111 1111 1111.",
     "CREDITCARDNUMBER", "4111 1111 1111 1111"),
    ("carta_trattini", "Pagamento con carta 4111-1111-1111-1111.",
     "CREDITCARDNUMBER", "4111-1111-1111-1111"),
    ("carta_punti", "Pagamento con carta 4111.1111.1111.1111.",
     "CREDITCARDNUMBER", "4111.1111.1111.1111"),
    ("carta_compatta", "Pagamento con carta 4111111111111111.",
     "CREDITCARDNUMBER", "4111111111111111"),
    # Telefono.
    ("cellulare_spazi", "Cell. 333 123 4567 per contatti.",
     "TELEPHONENUM", "333 123 4567"),
    ("cellulare_trattini", "Cell. 333-123-4567 per contatti.",
     "TELEPHONENUM", "333-123-4567"),
    ("cellulare_prefisso_trattini", "Cell. +39-333-123-4567 per contatti.",
     "TELEPHONENUM", "+39-333-123-4567"),
    ("cellulare_prefisso_spazio", "Cell. +39 333 123 4567 per contatti.",
     "TELEPHONENUM", "+39 333 123 4567"),
    ("fisso_punti", "Tel. 010.2471234 per contatti.",
     "TELEPHONENUM", "010.2471234"),
    ("fisso_trattini", "Tel. 010-2471234 per contatti.",
     "TELEPHONENUM", "010-2471234"),
    # Partita IVA.
    ("piva_nuda", "P.IVA 00743110157 iscritta al registro imprese.",
     "PIVA", "00743110157"),
    ("piva_con_prefisso_it", "Partita IVA IT00743110157, sede in Milano.",
     "PIVA", "00743110157"),
    # Indirizzo IP: forma semplice, subnet CIDR, range.
    ("ip_semplice", "Il server risponde su 192.168.1.10 sulla rete interna.",
     "IPADDR", "192.168.1.10"),
    ("ip_pubblico", "Connessione in ingresso da 8.8.8.8 registrata nel log.",
     "IPADDR", "8.8.8.8"),
    ("ip_cidr", "La subnet 192.168.1.0/24 e' riservata al reparto IT.",
     "IPADDR", "192.168.1.0/24"),
    ("ip_range", "Range assegnato 192.168.1.1-192.168.1.20 per i client DHCP.",
     "IPADDR", "192.168.1.1-192.168.1.20"),
    ("ip_range_con_spazi", "Range assegnato 192.168.1.1 - 192.168.1.20 per i client.",
     "IPADDR", "192.168.1.1 - 192.168.1.20"),
]

# Valori con la forma giusta ma il checksum sbagliato: dove il detector e'
# strict devono sparire, e' la ragione per cui strict esiste.
CHECKSUM_ERRATO = [
    ("iban_checksum_errato", "IBAN IT61 X054 2811 1010 0000 0123 456.", "IBAN"),
    ("iban_compatto_checksum_errato", "IBAN IT61X0542811101000000123456.", "IBAN"),
    ("carta_luhn_errato", "Carta 4111 1111 1111 1112.", "CREDITCARDNUMBER"),
    ("carta_punti_luhn_errato", "Carta 4111.1111.1111.1112.", "CREDITCARDNUMBER"),
    ("piva_checksum_errato", "P.IVA 00743110158 del fornitore.", "PIVA"),
    ("cf_omocodia_checksum_errato", "C.F. RSSMRA85H12F2LRA del ricorrente.", "CF"),
]

# Testo di un atto che NON contiene PII strutturata: nessuno di questi span
# deve essere rilevato dai tag validati con checksum.
NEGATIVI = [
    ("data_estesa", "Provvedimento emesso il 12.03.2024 dal collegio."),
    ("date_in_sequenza", "Udienze del 01.02.2024 e 03.04.2025, rinviate."),
    ("protocollo", "Protocollo n. 2024-000123-456 del registro generale."),
    ("catasto", "Immobile al Foglio 12, particella 345, sub. 6."),
    ("piva_come_parola", "La p.iva verra' comunicata in separata sede."),
    ("numeri_di_riga", "Cfr. pagg. 12-15, punti 3.4.5 e 6.7.8 della memoria."),
    ("importo_e_articoli", "Ai sensi dell'art. 2043 c.c. per euro 1.250,00."),
    ("sequenza_lunga_non_luhn", "Codice pratica 1234567890123456789 del 2024."),
    ("protocollo_con_punti", "Protocollo 2024.0001234.5678 del registro generale."),
    ("mandato_con_punti", "Mandato n. 12.345.678.901.234 del tesoriere."),
]

# Versioni software con un numero fuori range 0-255: non devono diventare IPADDR.
# E' il caso esplicito segnalato nell'issue #83 (es. "2.3.55.987").
IP_NEGATIVI = [
    ("versione_software", "Aggiornata alla versione 2.3.55.987 del pacchetto."),
    ("versione_a_5_cifre", "Changelog della build 10.20.30.40500 di oggi."),
]

TAG_CON_CHECKSUM = ("IBAN", "CF", "PIVA", "CREDITCARDNUMBER")


class DetectorFormatTests(unittest.TestCase):
    """I formati reali devono arrivare al validatore, non fermarsi alla regex."""

    def test_formati_reali_rilevati(self):
        for nome, testo, label, atteso in POSITIVI:
            with self.subTest(nome):
                self.assertIn(atteso, spans(testo, label))

    def test_checksum_verificato_sui_tag_validati(self):
        for nome, testo, label, atteso in POSITIVI:
            if label not in TAG_CON_CHECKSUM:
                continue
            with self.subTest(nome):
                self.assertIn(atteso, validated(testo, label))

    def test_checksum_errato_scartato_dove_strict(self):
        for nome, testo, label in CHECKSUM_ERRATO:
            with self.subTest(nome):
                if label == "CF":
                    # il CF non e' strict: si redige sulla sola forma, ma senza
                    # il ✓ del checksum.
                    self.assertEqual([], validated(testo, label))
                else:
                    self.assertEqual([], spans(testo, label))

    def test_nessun_falso_positivo_sul_testo_di_un_atto(self):
        for nome, testo in NEGATIVI:
            with self.subTest(nome):
                trovati = [(e["label"], testo[e["start"]:e["end"]])
                           for e in detectors.detect_regex(testo)
                           if e["label"] in TAG_CON_CHECKSUM]
                self.assertEqual([], trovati)

    def test_nessun_falso_positivo_telefono_su_date_e_richiami(self):
        # Il trattino tra i gruppi e' l'aggiunta piu' esposta ai falsi positivi:
        # una data o un richiamo a pagine non deve diventare un numero.
        for testo in ("Provvedimento emesso il 12.03.2024 dal collegio.",
                      "Udienze del 01.02.2024 e 03.04.2025, rinviate.",
                      "Cfr. pagg. 12-15 e 20-24 della memoria."):
            with self.subTest(testo):
                self.assertEqual([], spans(testo, "TELEPHONENUM"))

    def test_versione_software_non_diventa_ip(self):
        for nome, testo in IP_NEGATIVI:
            with self.subTest(nome):
                self.assertEqual([], spans(testo, "IPADDR"))

    def test_iban_non_ingoia_la_parola_successiva(self):
        # La regex raggruppata e' golosa: se assorbe il token dopo l'IBAN il
        # checksum fallisce e con strict=True il valore si perde del tutto.
        testo = "Accredito su ES91 2100 0418 4502 0005 1332 come da mandato."
        self.assertEqual(["ES91 2100 0418 4502 0005 1332"], spans(testo, "IBAN"))

    def test_cf_span_unico_quando_le_due_voci_coincidono(self):
        # La voce omocodica accetta anche la forma canonica: i candidati sono
        # due ma sullo STESSO span, cosi' _merge ne tiene uno solo.
        testo = "C.F. RSSMRA85H12F205Y."
        trovati = {(e["start"], e["end"])
                   for e in detectors.detect_regex(testo) if e["label"] == "CF"}
        self.assertEqual(1, len(trovati))

    def test_modulo_senza_dipendenze_pesanti(self):
        # E' la premessa della CI: i test dei detector girano senza il modello.
        # L'import va fatto in un processo pulito, altrimenti l'esito dipende da
        # cosa hanno gia' importato gli altri file di test.
        codice = (f"import sys; sys.path.insert(0, {str(ROOT / 'src' / 'app')!r});"
                  " import detectors;"
                  " print([m for m in ('torch', 'transformers', 'fitz', 'flask')"
                  " if m in sys.modules])")
        out = subprocess.run([sys.executable, "-c", codice],
                             capture_output=True, text=True, check=True)
        self.assertEqual("[]", out.stdout.strip())


if __name__ == "__main__":
    unittest.main()
