#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Batch 2 template legali IT (autore = Claude). Include liste e {PROVINCE} per
varieta' strutturale. Validazione: slot in gen.SLOTS + guardia anti nomi inline
(tb.find_stray_names). Accoda a dataset/synthetic/legal_templates.json. Solo {SLOT}."""
import json, re, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[0]
sys.path.insert(0, str(ROOT / "src" / "data_pipeline"))
import llm_template_bank as tb
import generate_synthetic_pii as gen

DRAFTS = [
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


def validate(text):
    text = re.sub(r"^```.*?\n|```$", "", text.strip(), flags=re.MULTILINE).strip()
    slots = set(gen.SLOT_RE.findall(text))
    if not slots:
        return None, "nessun segnaposto"
    bad = slots - set(gen.SLOTS)
    if bad:
        return None, f"slot non iniettabili {bad}"
    stray = tb.find_stray_names(text)
    if stray:
        return None, f"nomi inline {sorted(set(stray))[:6]}"
    return text, None


def main():
    out = ROOT / "dataset" / "synthetic" / "legal_templates.json"
    existing = json.load(open(out, encoding="utf-8")) if out.exists() else []
    tid = len(existing)
    new = 0
    for doc_type, text in DRAFTS:
        clean, err = validate(text)
        if not clean:
            print(f"SCARTATO ({doc_type}): {err}"); continue
        existing.append({"id": tid, "doc_type": doc_type, "text": clean}); tid += 1; new += 1
    json.dump(existing, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\nBatch 2: nuovi validi {new}/{len(DRAFTS)} | totale template nel file: {len(existing)}")


if __name__ == "__main__":
    main()
