#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Banca di template legali scritti a mano, senza chiamare un LLM.

Stessa idea di `llm_template_bank.py` - l'autore scrive SOLO la prosa con i
segnaposto `{SLOT}`, il codice inietta i dati - ma l'autore qui e' una persona
(o un agente di coding) invece di Gemini: serve a chi non ha una GEMINI_API_KEY,
e i 60 template che ne escono sono deterministici, quindi il dataset e'
riproducibile.

I template puntano sui tag che il resto del sintetico copre poco: CATASTO,
DOCID, TARGA, CF, ID_DOC, AMOUNT, IBAN, PIVA, ORG, PROVINCE.

Come per i template generati, NESSUN dato reale compare qui: solo `{SLOT}`. La
verifica non e' sulla fiducia ma sulla stessa guardia del repo
(`llm_template_bank.clean_and_validate`), che scarta i template con segnaposto
non iniettabili, con nomi propri scritti inline o con caratteri non latini.

    python src/data_pipeline/author_templates.py            # accoda al file
    python src/data_pipeline/author_templates.py --dry-run  # valida e basta
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "data_pipeline"))

import generate_synthetic_pii as gen  # noqa: E402
import llm_template_bank as tb  # noqa: E402

OUT = ROOT / "dataset" / "synthetic" / "legal_templates.json"

# (tipo di documento, testo con i soli segnaposto)
DRAFTS = [
    ("contratto di compravendita immobiliare",
     "Con la presente scrittura privata il venditore {FULLNAME}, C.F. {CF}, cede all'acquirente "
     "{FULLNAME}, C.F. {CF}, la piena proprieta' dell'immobile sito in {ADDRESS}, censito al Catasto "
     "Fabbricati al {CATASTO}, per il prezzo di {AMOUNT}, versato mediante bonifico sull'IBAN {IBAN}."),

    ("decreto ingiuntivo",
     "Il {TRIBUNAL}, letto il ricorso iscritto al R.G. n. {RG}, ingiunge alla {ORG}, P.IVA {PIVA}, "
     "in persona del legale rappresentante {FULLNAME}, di pagare la somma di {AMOUNT} in favore del "
     "ricorrente, oltre interessi, come da fattura prot. n. {DOCID}."),

    ("atto di citazione",
     "L'avvocato {LAWYER}, C.F. {CF}, giusta procura in atti, nell'interesse dell'attore {PLAINTIFF}, "
     "residente in {ADDRESS}, cita in giudizio il convenuto {DEFENDANT} dinanzi al {TRIBUNAL}, con "
     "udienza fissata per il {DATE}, R.G. n. {RG}."),

    ("verbale di udienza",
     "All'udienza del {DATE} dinanzi al giudice {JUDGE}, comparso il teste {WITNESS}, identificato a "
     "mezzo carta d'identita' n. {IDCARD}, si da' atto delle dichiarazioni rese nella causa R.G. n. {RG}."),

    ("procura alle liti",
     "Il sottoscritto {FULLNAME}, C.F. {CF}, nato a {CITY} il {DATE}, delega a rappresentarlo e difenderlo "
     "l'avv. {LAWYER}, eleggendo domicilio in {ADDRESS}, PEC {PEC}, tel. {PHONE}."),

    ("contratto di locazione",
     "Tra il locatore {FULLNAME}, C.F. {CF}, e il conduttore {FULLNAME}, C.F. {CF}, si conviene la "
     "locazione dell'immobile in {ADDRESS}, identificato al {CATASTO}, per un canone mensile di {AMOUNT}, "
     "da versare sul conto corrente n. {CONTO}."),

    ("atto di diffida",
     "La {ORG}, P.IVA {PIVA}, con sede in {ADDRESS}, in persona dell'avv. {LAWYER}, diffida formalmente "
     "il Sig. {FULLNAME}, C.F. {CF}, ad adempiere entro il {DATE}, pena l'azione esecutiva; ogni "
     "comunicazione all'indirizzo email {EMAIL} o PEC {PEC}."),

    ("ricorso per decreto ingiuntivo",
     "Il ricorrente {PLAINTIFF}, rappresentato dall'avv. {LAWYER}, chiede al {TRIBUNAL} l'emissione di "
     "decreto ingiuntivo nei confronti della {ORG}, P.IVA {PIVA}, per il pagamento di {AMOUNT}, come da "
     "contratto rep. n. {DOCID} e da estratto del conto corrente n. {CONTO}."),

    ("comparsa di costituzione e risposta",
     "Si costituisce in giudizio il convenuto {DEFENDANT}, C.F. {CF}, rappresentato e difeso dall'avv. "
     "{LAWYER}, con domicilio eletto in {ADDRESS}, il quale contesta la domanda attorea nella causa "
     "R.G. n. {RG} dinanzi al {TRIBUNAL}."),

    ("sentenza civile",
     "Il {TRIBUNAL}, nella persona del giudice {JUDGE}, definitivamente pronunciando nella causa R.G. n. "
     "{RG}, condanna il convenuto {DEFENDANT} al pagamento di {AMOUNT} in favore dell'attore {PLAINTIFF}, "
     "come da sentenza n. {DOCID}."),

    ("contratto di compravendita immobiliare",
     "Il promittente venditore {FULLNAME}, C.F. {CF}, promette di vendere al promissario acquirente "
     "{FULLNAME} l'immobile in {ADDRESS}, censito al {CATASTO}, al prezzo di {AMOUNT}; a titolo di "
     "acconto si versa la somma sull'IBAN {IBAN}."),

    ("verbale di udienza",
     "Nel procedimento R.G. n. {RG}, il giudice {JUDGE} da' atto della presenza dell'avv. {LAWYER} per "
     "l'attore {PLAINTIFF} e dell'avv. {LAWYER} per il convenuto {DEFENDANT}, rinviando l'udienza al {DATE}."),

    ("atto di diffida",
     "Con la presente si diffida il Sig. {FULLNAME}, proprietario del veicolo targato {TARGA}, patente n. "
     "{DRIVING}, a provvedere al risarcimento di {AMOUNT} entro il {DATE}, mediante versamento sull'IBAN {IBAN}."),

    ("procura alle liti",
     "La {ORG}, P.IVA {PIVA}, in persona del legale rappresentante {FULLNAME}, C.F. {CF}, conferisce "
     "mandato all'avv. {LAWYER} per la controversia dinanzi al {TRIBUNAL}, R.G. n. {RG}, con ogni "
     "comunicazione a mezzo PEC {PEC}."),

    ("decreto ingiuntivo",
     "Visto il ricorso, il {TRIBUNAL} ingiunge al debitore {FULLNAME}, C.F. {CF}, residente in {ADDRESS}, "
     "di pagare {AMOUNT} in favore della {ORG}, con addebito sul conto corrente n. {CONTO}, giusta "
     "documentazione prot. n. {DOCID}."),

    ("contratto di locazione",
     "Il conduttore {FULLNAME}, C.F. {CF}, si obbliga a versare al locatore {FULLNAME} il canone di "
     "{AMOUNT} mensili sull'IBAN {IBAN}, per l'immobile in {ADDRESS} identificato al {CATASTO}; recapiti: "
     "tel. {PHONE}, email {EMAIL}."),

    ("sentenza civile",
     "Definitivamente pronunciando, il {TRIBUNAL}, giudice {JUDGE}, nella causa R.G. n. {RG} tra l'attore "
     "{PLAINTIFF} e il convenuto {DEFENDANT}, dichiara risolto il contratto rep. n. {DOCID} e condanna "
     "al pagamento di {AMOUNT}."),

    ("comparsa di costituzione e risposta",
     "Si costituisce la {ORG}, P.IVA {PIVA}, in persona del legale rappresentante {FULLNAME}, difesa "
     "dall'avv. {LAWYER}, C.F. {CF}, la quale eccepisce l'inadempimento nella causa R.G. n. {RG}, "
     "chiedendo il rigetto della domanda."),

    ("ricorso per decreto ingiuntivo",
     "Il creditore {FULLNAME}, C.F. {CF}, chiede al {TRIBUNAL} decreto ingiuntivo nei confronti del "
     "debitore {FULLNAME}, per la somma di {AMOUNT} portata dalla fattura n. {DOCID}, da accreditare "
     "sull'IBAN {IBAN}."),

    ("atto di citazione",
     "L'avv. {LAWYER}, per conto della {ORG}, P.IVA {PIVA}, conviene in giudizio dinanzi al {TRIBUNAL} "
     "il Sig. {FULLNAME}, C.F. {CF}, proprietario del veicolo targato {TARGA}, per il risarcimento di "
     "{AMOUNT}, R.G. n. {RG}."),

    ("atto di precetto",
     "Si intima e fa precetto al debitore {FULLNAME}, C.F. {CF}, residente in {ADDRESS}, di pagare "
     "entro dieci giorni la somma di {AMOUNT} di cui alla sentenza n. {DOCID} del {TRIBUNAL}, "
     "mediante accredito sull'IBAN {IBAN}, con avvertenza dell'esecuzione forzata."),
    ("pignoramento presso terzi",
     "Ad istanza del creditore {FULLNAME}, si pignorano le somme dovute dal terzo {ORG}, P.IVA {PIVA}, "
     "al debitore {FULLNAME}, C.F. {CF}, fino alla concorrenza di {AMOUNT}, giusta titolo esecutivo "
     "R.G. n. {RG} del {TRIBUNAL}."),
    ("verbale di assemblea",
     "L'assemblea della {ORG}, P.IVA {PIVA}, con sede in {ADDRESS}, tenutasi il {DATE}, vede la "
     "partecipazione dei soci: {NAMELIST}. Presiede il legale rappresentante {FULLNAME}."),
    ("denuncia di successione",
     "Alla successione di {FULLNAME}, C.F. {CF}, deceduto il {DATE}, concorrono gli eredi: {NAMELIST}. "
     "Cade in successione l'immobile in {ADDRESS}, censito al Catasto Fabbricati al {CATASTO}."),
    ("contratto preliminare",
     "Il promittente venditore {FULLNAME}, C.F. {CF}, e il promissario acquirente {FULLNAME} convengono "
     "il preliminare relativo all'immobile in {ADDRESS} ({PROVINCE}), identificato al {CATASTO}, prezzo "
     "{AMOUNT}, con caparra versata sul conto corrente n. {CONTO}."),
    ("quietanza",
     "Il sottoscritto {FULLNAME}, C.F. {CF}, dichiara di aver ricevuto dalla {ORG}, P.IVA {PIVA}, la "
     "somma di {AMOUNT} a saldo della fattura n. {DOCID}, rilasciando ampia quietanza; accredito "
     "sull'IBAN {IBAN}."),
    ("atto di cessione del credito",
     "La {ORG}, P.IVA {PIVA}, cede a {FULLNAME}, C.F. {CF}, il credito di {AMOUNT} vantato verso il "
     "debitore ceduto {FULLNAME}, di cui alla scrittura rep. n. {DOCID}, con pagamento sul conto "
     "corrente n. {CONTO}."),
    ("ricorso per ingiunzione",
     "Il ricorrente {PLAINTIFF}, difeso dall'avv. {LAWYER}, PEC {PEC}, chiede al {TRIBUNAL} ingiunzione "
     "verso i condebitori: {NAMELIST}, per la somma complessiva di {AMOUNT}, R.G. n. {RG}."),
    ("verbale di conciliazione",
     "All'udienza del {DATE}, dinanzi al giudice {JUDGE}, le parti {PLAINTIFF} e {DEFENDANT} raggiungono "
     "conciliazione nella causa R.G. n. {RG}: il convenuto versera' {AMOUNT} sull'IBAN {IBAN} entro il {DATE}."),
    ("contratto di locazione commerciale",
     "Il locatore {FULLNAME}, C.F. {CF}, concede in locazione alla {ORG}, P.IVA {PIVA}, l'immobile "
     "commerciale in {ADDRESS}, censito al {CATASTO}, per un canone annuo di {AMOUNT} da versare sul "
     "conto corrente n. {CONTO}; recapiti PEC {PEC}, tel. {PHONE}."),
    ("atto di diffida e messa in mora",
     "Con la presente si costituisce in mora il Sig. {FULLNAME}, C.F. {CF}, proprietario del veicolo "
     "targato {TARGA}, patente n. {DRIVING}, affinche' corrisponda {AMOUNT} entro il {DATE}, PEC {PEC}."),
    ("comparsa conclusionale",
     "Nell'interesse dell'attore {PLAINTIFF}, l'avv. {LAWYER}, C.F. {CF}, insiste per la condanna del "
     "convenuto {DEFENDANT} al pagamento di {AMOUNT}, richiamando la documentazione prot. n. {DOCID}, "
     "nella causa R.G. n. {RG}."),
    ("procura speciale",
     "Il sottoscritto {FULLNAME}, C.F. {CF}, nato a {CITY} ({PROVINCE}) il {DATE}, nomina procuratore "
     "speciale l'avv. {LAWYER}, con facolta' di transigere la lite R.G. n. {RG} dinanzi al {TRIBUNAL}."),
    ("fideiussione",
     "Il fideiussore {FULLNAME}, C.F. {CF}, garantisce le obbligazioni della {ORG}, P.IVA {PIVA}, verso "
     "il creditore fino a {AMOUNT}, con addebito sul conto corrente n. {CONTO} in caso di escussione."),
    ("verbale di udienza",
     "Nel procedimento R.G. n. {RG} dinanzi al giudice {JUDGE} compaiono i testimoni: {NAMELIST}, "
     "identificati e ammoniti; si rinvia al {DATE} per l'assunzione delle ulteriori prove."),
    ("atto di citazione",
     "L'avv. {LAWYER}, per l'attore {PLAINTIFF}, residente in {ADDRESS} ({PROVINCE}), cita il convenuto "
     "{DEFENDANT}, C.F. {CF}, dinanzi al {TRIBUNAL}, chiedendo il risarcimento di {AMOUNT}, R.G. n. {RG}."),
    ("elenco creditori",
     "Nella procedura R.G. n. {RG} risultano ammessi al passivo i creditori: {ORGLIST}, per un importo "
     "complessivo di {AMOUNT}, come da stato passivo prot. n. {DOCID}."),
    ("decreto ingiuntivo",
     "Il {TRIBUNAL}, giudice {JUDGE}, ingiunge a {FULLNAME}, C.F. {CF}, il pagamento di {AMOUNT} in "
     "favore della {ORG}, P.IVA {PIVA}, oltre spese, come da fattura n. {DOCID}; accredito IBAN {IBAN}."),
    ("visura catastale",
     "Si attesta che l'immobile sito in {ADDRESS} ({PROVINCE}), intestato a {FULLNAME}, C.F. {CF}, "
     "risulta censito al Catasto Fabbricati al {CATASTO}, con annotazione prot. n. {DOCID}."),
    ("contratto di compravendita di veicolo",
     "Il venditore {FULLNAME}, C.F. {CF}, cede all'acquirente {FULLNAME} il veicolo targato {TARGA}, per "
     "il prezzo di {AMOUNT} versato sull'IBAN {IBAN}; consegna della carta di circolazione n. {IDCARD}."),
    ("atto di precetto",
     "Ad istanza di {FULLNAME}, si intima ai coobbligati: {NAMELIST}, il pagamento di {AMOUNT} portato "
     "dal titolo esecutivo n. {DOCID} del {TRIBUNAL}, con accredito sul conto corrente n. {CONTO}."),
    ("comparsa di costituzione",
     "Si costituisce il convenuto {DEFENDANT}, C.F. {CF}, difeso dall'avv. {LAWYER}, PEC {PEC}, che "
     "eccepisce il difetto di legittimazione nella causa R.G. n. {RG} dinanzi al {TRIBUNAL}."),
    ("ricorso per decreto ingiuntivo",
     "La {ORG}, P.IVA {PIVA}, in persona di {FULLNAME}, chiede ingiunzione verso {FULLNAME}, C.F. {CF}, "
     "per {AMOUNT} di cui alle fatture: {ORGLIST} non pagate, R.G. n. {RG}."),
    ("sentenza civile",
     "Il {TRIBUNAL}, giudice {JUDGE}, nella causa R.G. n. {RG}, condanna in solido i convenuti: "
     "{NAMELIST}, al pagamento di {AMOUNT} in favore dell'attore {PLAINTIFF}, sentenza n. {DOCID}."),
    ("contratto di mutuo",
     "La {ORG}, P.IVA {PIVA}, concede a {FULLNAME}, C.F. {CF}, un mutuo di {AMOUNT}, garantito da ipoteca "
     "sull'immobile in {ADDRESS} censito al {CATASTO}; rimborso mediante addebito sul conto n. {CONTO}."),
    ("atto di diffida",
     "Lo studio legale {ORG}, PEC {PEC}, per conto di {FULLNAME}, diffida i condomini morosi: {NAMELIST}, "
     "al pagamento degli oneri pari a {AMOUNT} entro il {DATE}."),
    ("verbale di assemblea condominiale",
     "L'assemblea del condominio, tenutasi il {DATE} in {ADDRESS} ({PROVINCE}), con la presenza dei "
     "condomini {NAMELIST}, delibera lavori per {AMOUNT}, delega l'amministratore {FULLNAME}, C.F. {CF}."),
    ("procura alle liti",
     "Delego l'avv. {LAWYER} e l'avv. {LAWYER} a rappresentarmi nella causa dinanzi al {TRIBUNAL}, R.G. "
     "n. {RG}, eleggendo domicilio in {ADDRESS}, PEC {PEC}, tel. {PHONE}."),
    ("atto di appello",
     "Avverso la sentenza n. {DOCID} del {TRIBUNAL}, l'avv. {LAWYER}, per {FULLNAME}, C.F. {CF}, propone "
     "appello chiedendo la riforma della condanna al pagamento di {AMOUNT}, R.G. n. {RG}."),
    ("contratto di appalto",
     "La committente {ORG}, P.IVA {PIVA}, affida all'appaltatore {FULLNAME}, C.F. {CF}, i lavori "
     "sull'immobile in {ADDRESS} censito al {CATASTO}, per il corrispettivo di {AMOUNT}, IBAN {IBAN}."),
    ("dichiarazione sostitutiva",
     "Il sottoscritto {FULLNAME}, C.F. {CF}, nato a {CITY} ({PROVINCE}) il {DATE}, documento d'identita' "
     "n. {IDCARD}, dichiara sotto la propria responsabilita' i dati anagrafici sopra riportati."),
    ("elenco fornitori",
     "Si autorizzano al pagamento i seguenti fornitori: {ORGLIST}, per gli importi indicati in fattura, "
     "con addebito sul conto corrente n. {CONTO} della {ORG}, P.IVA {PIVA}."),
    ("atto di transazione",
     "Le parti {FULLNAME}, C.F. {CF}, e {FULLNAME}, C.F. {CF}, transigono ogni pretesa relativa alla "
     "causa R.G. n. {RG}: si conviene il versamento di {AMOUNT} sull'IBAN {IBAN} entro il {DATE}."),
    ("ricorso",
     "Il ricorrente {PLAINTIFF}, difeso dall'avv. {LAWYER}, C.F. {CF}, adisce il {TRIBUNAL} avverso il "
     "provvedimento prot. n. {DOCID}, chiedendone l'annullamento, R.G. n. {RG}."),
    ("verbale di pignoramento",
     "L'ufficiale giudiziario, ad istanza di {FULLNAME}, pignora presso il debitore {FULLNAME}, C.F. "
     "{CF}, il veicolo targato {TARGA}, stimato in {AMOUNT}, di cui al titolo n. {DOCID}."),
    ("contratto di comodato",
     "Il comodante {FULLNAME}, C.F. {CF}, concede in comodato a {FULLNAME} l'immobile in {ADDRESS} "
     "({PROVINCE}) censito al {CATASTO}, a titolo gratuito, con oneri accessori regolati a parte."),
    ("anagrafica soggetti",
     "Allegato - anagrafica dei soggetti del procedimento R.G. n. {RG}:\n{MIXEDLIST}"),
    ("atto di intervento",
     "Interviene volontariamente nel giudizio R.G. n. {RG} la {ORG}, P.IVA {PIVA}, difesa dall'avv. "
     "{LAWYER}, spiegando domanda di condanna al pagamento di {AMOUNT} verso {FULLNAME}, C.F. {CF}."),
    ("verbale di conciliazione",
     "Dinanzi al giudice {JUDGE}, R.G. n. {RG}, le parti convengono che {FULLNAME} corrispondera' ad "
     "{FULLNAME} la somma di {AMOUNT}, in rate mensili accreditate sull'IBAN {IBAN}."),
    ("decreto ingiuntivo",
     "Ingiunzione verso i debitori solidali: {NAMELIST}, per {AMOUNT} portati dalla scrittura rep. n. "
     "{DOCID}, emessa dal {TRIBUNAL}, giudice {JUDGE}, R.G. n. {RG}."),
]


def valida(text):
    """Il template ripulito, oppure None con il motivo dello scarto.

    Si usa la guardia del repo invece di riscriverne una: e' quella che decide
    per i template generati da Gemini, e due guardie che divergono sono un modo
    silenzioso di far entrare nel dataset cio' che l'altra scarterebbe.
    """
    clean = tb.clean_and_validate(text)
    if not clean:
        return None, "guardia clean_and_validate"
    slots = set(gen.SLOT_RE.findall(clean))
    non_iniettabili = slots - set(gen.SLOTS)
    if non_iniettabili:
        return None, "slot non iniettabili %s" % sorted(non_iniettabili)
    return clean, None


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dry-run", action="store_true",
                    help="valida i template e stampa l'esito, senza scrivere nulla")
    args = ap.parse_args()

    esistenti = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else []
    tid = len(esistenti)
    nuovi, scartati = [], 0
    for doc_type, text in DRAFTS:
        clean, errore = valida(text)
        if not clean:
            scartati += 1
            print("SCARTATO (%s): %s" % (doc_type, errore))
            continue
        nuovi.append({"id": tid, "doc_type": doc_type, "text": clean})
        tid += 1

    print("\nTemplate validi: %d/%d | scartati: %d" % (len(nuovi), len(DRAFTS), scartati))
    if args.dry_run:
        print("--dry-run: nessuna scrittura.")
        return
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(esistenti + nuovi, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print("Totale nel file: %d -> %s" % (len(esistenti) + len(nuovi), OUT))


if __name__ == "__main__":
    main()
