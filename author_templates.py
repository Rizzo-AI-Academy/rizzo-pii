#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Autore = Claude (al posto di Gemini): template legali IT con SOLO segnaposto.
Valida con la stessa guardia del repo (clean_and_validate + slot in gen.SLOTS) e
accoda a dataset/synthetic/legal_templates.json. Nessuna PII reale: solo {SLOT}."""
import json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[0]
sys.path.insert(0, str(ROOT / "src" / "data_pipeline"))
import llm_template_bank as tb
import generate_synthetic_pii as gen

# (doc_type, testo). Ricchi di tag rari: CATASTO, DOCID, TARGA, CF, IDCARD/DRIVING, AMOUNT, CONTO, PIVA, ORG.
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
]


def main():
    out = ROOT / "dataset" / "synthetic" / "legal_templates.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    existing = json.load(open(out, encoding="utf-8")) if out.exists() else []
    tid = len(existing)
    kept, skipped = existing, 0
    valid_slots = set(gen.SLOTS)
    for doc_type, text in DRAFTS:
        clean = tb.clean_and_validate(text)                    # guardia repo: slot ammessi + no nomi inline
        if not clean:
            skipped += 1; print(f"SCARTATO ({doc_type}): guardia clean_and_validate"); continue
        slots = set(gen.SLOT_RE.findall(clean))
        if slots - valid_slots:                                # slot iniettabili?
            skipped += 1; print(f"SCARTATO ({doc_type}): slot non iniettabili {slots - valid_slots}"); continue
        kept.append({"id": tid, "doc_type": doc_type, "text": clean}); tid += 1
    json.dump(kept, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    new = len(kept) - len(existing)
    print(f"\nTemplate nuovi validi: {new}/{len(DRAFTS)} | scartati: {skipped} | totale nel file: {len(kept)}")
    print(f"Salvato -> {out}")


if __name__ == "__main__":
    main()
