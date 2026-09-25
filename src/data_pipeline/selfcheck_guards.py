#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Autoverifica delle guardie che tengono la PII fuori dai template.

Le guardie di llm_template_bank decidono quali template dell'LLM entrano nella banca:
se sono troppo larghe entra PII non taggata (il modello impara a NON taggarla), se sono
troppo strette si perdono template buoni -- e si perdono soprattutto nei domini nuovi,
dove il lessico non e' quello dei tribunali. Un caso per ogni modo di sbagliare, cosi'
la prossima modifica alle guardie si accorge subito di quale dei due lati ha rotto.

Uso:  python selfcheck_guards.py [--templates FILE.json]
Esce con codice 1 se un caso non si comporta come atteso.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import llm_template_bank as tb  # noqa: E402

# nomi di persona scritti nel testo invece che come segnaposto: DEVONO essere presi
NOMI_DA_PRENDERE = [
    "Il Sig. Mario Rossi, nato a {CITY}, dichiara quanto segue.",
    "Comparso il sottoscritto Sferrazza, difeso dal {LAWYER}.",
    "L'intestatario Ludovico Marinetti chiede il rimborso di {AMOUNT}.",
    "Il paziente Bianchi si e' presentato alle ore {TIME}.",
    "La lettura e' stata comunicata da Giulia al numero {PHONE}.",
    "Il teste Aldo Verdi ha riferito i fatti.",
]

# lessico amministrativo e istituzionale: NON sono nomi di persona. Sono i casi su cui la
# guardia basata su un vocabolario giuridico scritto a mano scartava template buoni.
NON_SONO_NOMI = [
    "L'Ufficio Protocollo ha registrato l'istanza al n. {DOCID}.",
    "La pratica e' assegnata alle Risorse Umane di {ORG}.",
    "Il Servizio Clienti di {ORG} risponde entro 30 giorni.",
    "Visti gli artt. 2043 e 1218 del Codice Civile e il D.Lgs. 196/2003.",
    "L'Azienda Sanitaria Locale ha trasmesso il referto al {FULLNAME}.",
    "Istituto Comprensivo Statale, classe frequentata da {FULLNAME}.",
    "Il Tribunale di {CITY}, Sezione Prima Civile, ha pronunciato la {DOCID}.",
    "Al Giudice di Pace di {CITY}, ricorso in opposizione.",
    "La Corte di Giustizia Tributaria di Secondo Grado respinge il ricorso.",
    "Il presente atto e' rogato dal sottoscritto Notaio in {CITY}.",
    "Presente all'udienza, assistito dal Cancelliere. La causa prosegue.",
    "Il {FULLNAME}, di seguito denominato Locatore, e il {FULLNAME}, Conduttore.",
    "Somma corrisposta dal Sig. {FULLNAME} al Venditore tramite bonifico.",
    "Consumo rilevato pari a 3.450 kWh nel periodo di riferimento.",
    "Valutazione finale: Ottimo. Media dei voti 8,5 su 10.",
    "L'offerta sottoscritta Mobile Plan Pro Canone prevede minuti illimitati.",
    "Gentile Servizio Clienti, chiedo il rimborso del titolo di viaggio.",
    "Gentile Cliente, la informiamo che la fattura e' disponibile.",
    "Il presente verbale e' redatto dal sottoscritto Agente accertatore.",
    "Il Venditore consegna il bene entro il {DATE}.",
    "La pratica passa al Responsabile del Procedimento entro 10 giorni.",
]

# codici e date scritti inline: entita' presenti nel testo ma assenti dalle label
CODICI_DA_PRENDERE = [
    "Codice fiscale RSSMRA85M01H501Z del richiedente.",
    "Accredito su IT60X0542811101000000123456 entro il termine.",
    "Recapito telefonico 3331234567 per comunicazioni.",
    "Nato il 12/03/1985 a {CITY}.",
]

# numeri legittimi della prosa: norme, importi con segnaposto, unita' di misura
NON_SONO_CODICI = [
    "Ai sensi del D.Lgs. 196/2003 e del Regolamento UE 679/2016.",
    "Importo di {AMOUNT} da versare entro il {DATE}.",
    "Consumo di 3.450 kWh, potenza impegnata 4,5 kW.",
    "Visto l'art. 2043 c.c. e la legge n. 241/1990.",
]


def _riga(atteso_ok, testo, esito):
    print(f"  [{'ok  ' if atteso_ok else 'ERRORE'}] {testo[:58]:<58} -> {esito}")


def controlla():
    errori = 0
    print("nomi inline che devono essere presi:")
    for t in NOMI_DA_PRENDERE:
        got = sorted(set(tb.find_stray_names(t)))
        errori += not got
        _riga(bool(got), t, got[:2])

    print("\nlessico amministrativo che non deve essere preso:")
    for t in NON_SONO_NOMI:
        got = sorted(set(tb.find_stray_names(t)))
        errori += bool(got)
        _riga(not got, t, got[:2])

    print("\ncodici e date inline che devono essere presi:")
    for t in CODICI_DA_PRENDERE:
        got = tb.inline_code(t)
        errori += not got
        _riga(bool(got), t, repr(got))

    print("\nnumeri legittimi che non devono essere presi:")
    for t in NON_SONO_CODICI:
        got = tb.inline_code(t)
        errori += bool(got)
        _riga(not got, t, repr(got))
    return errori


def sulla_banca(percorso):
    """Ripassa le guardie sui template GIA' accettati: quelli segnalati contengono PII
    entrata quando le guardie erano piu' larghe, e va tolta dalla banca."""
    if not os.path.exists(percorso):
        print(f"\n(banca non trovata in {percorso}: salto il controllo sui template)")
        return 0
    banca = json.load(open(percorso, encoding="utf-8"))
    print(f"\nle guardie sui {len(banca)} template della banca:")
    segnalati = 0
    for t in banca:
        nomi = sorted(set(tb.find_stray_names(t["text"])))
        cod = tb.inline_code(t["text"])
        if nomi or cod:
            segnalati += 1
            print(f"  #{t.get('id', '?'):<4} {t.get('doc_type', '')[:32]:<32} "
                  f"{'nomi ' + str(nomi[:2]) if nomi else ''} "
                  f"{'codice ' + repr(cod) if cod else ''}")
    print(f"  -> {segnalati}/{len(banca)} da rivedere")
    return 0                              # informativo: non fa fallire l'autoverifica


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--templates",
                    default=str(tb.ROOT / "dataset" / "synthetic" / "legal_templates.json"),
                    help="banca di template su cui ripassare le guardie")
    args = ap.parse_args()
    errori = controlla()
    sulla_banca(args.templates)
    print(f"\n{'tutti i casi corretti' if not errori else str(errori) + ' CASI SBAGLIATI'}")
    return 1 if errori else 0


if __name__ == "__main__":
    sys.exit(main())
