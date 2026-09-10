#!/usr/bin/env python3
"""
Test di regressione per reverse() (ripristino dei placeholder nel PDF anonimizzato).

reverse() e' JS incorporato in `PAGE` (src/app/app.py): questa suite non e' un
unittest.TestCase apposta, cosi' come test_confini_parola.py -- `unittest
discover` non ci trova nessuna TestCase e non installa Node, restando fedele
al commento nel workflow ("nessun pip install, solo libreria standard"). E'
un test manuale, allo stesso modo di test_confini_parola.py, ma eseguibile
offline (nessun backend da avviare): estrae il blocco reverse()/dictFile
letterale da app.py e lo esegue con `node`, stubando solo il DOM.

Copre due difetti gia' chiusi da questa funzione e cinque emersi in code
review sulla stessa riga, per evitare che una futura modifica li riapra senza
che nessun test se ne accorga:
  - spazi/tab/a-capo intorno a un placeholder senza parentesi (non vanno persi)
  - un indice inventato dal modello ([CF_12] con CF_1 in mappa) non deve
    scrivere un valore SBAGLIATO: o matcha tutto o resta visibile
  - lettera accentata subito dopo un placeholder senza parentesi (\b di JS e'
    ASCII-only, "CF_1e'" non vedeva il confine ed incollava il valore --
    simmetrico al caso ASCII "ilCF_1", che restava correttamente intatto)
  - asterischi di grassetto markdown condivisi fra due placeholder adiacenti
    (l'esito non deve dipendere dall'ordine con cui le chiavi sono elaborate)
  - grassetto NON correlato al placeholder non va assorbito
  - il valore sostituito per una chiave non deve poter contenere, per
    coincidenza, il nome di un'altra chiave e venire ri-sostituito
  - un file dizionario con una chiave vuota o un valore non testuale va
    rifiutato, non silenziosamente accettato (altrimenti corrompe l'intero
    documento al ripristino)

Richiede `node` in PATH (gia' necessario per buildare il frontend Tauri).

Uso:
    python tests/test_ripristino_placeholder.py
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

APP_PY = Path(__file__).resolve().parents[1] / "src" / "app" / "app.py"
START_MARK = "/* ---- reverse ---- */"
END_MARK = "r.readAsText(f);};"

# Stub minimale di DOM/app: solo cio' che reverse() e dictFile.onchange usano.
HARNESS = r"""
let MAP = {};
const elements = {};
function $(id) {
  if (!elements[id]) elements[id] = { value: '', textContent: '', innerHTML: '', _raw: '', onclick: null, onchange: null };
  return elements[id];
}
function toast() {}
function tt(k) { return k; }
const T = { it: { dict_loaded_n: (n) => 'loaded ' + n } };
const L = 'it';
global.FileReader = class {
  set onload(fn) { this._onload = fn; }
  get onload() { return this._onload; }
  readAsText(_f) { this.result = global.__pendingJsonText; this._onload(); }
};

%(reverse_block)s

function setDict(obj) { MAP = obj; }
function loadDictFromJsonText(jsonText) {
  global.__pendingJsonText = jsonText;
  $('dictFile').onchange({ target: { files: [{}] } });
}
function callReverse(inputText) { $('rin').value = inputText; reverse(); return $('rout')._raw; }

let pass = 0, fail = 0;
function eq(name, got, want) {
  const ok = got === want;
  console.log((ok ? 'OK  ' : 'FAIL') + ' | ' + name + (ok ? '' : ('  got=' + JSON.stringify(got) + ' want=' + JSON.stringify(want))));
  ok ? pass++ : fail++;
}

setDict({ '[FULLNAME_1]': 'Mario Rossi' });
eq('spazi intorno alla forma nuda preservati', callReverse('Il FULLNAME_1 ha firmato.'), 'Il Mario Rossi ha firmato.');
eq('colonna TAB preservata', callReverse('col1\tFULLNAME_1\tcol3'), 'col1\tMario Rossi\tcol3');
eq('riga a-capo preservata', callReverse('riga\nFULLNAME_1\nriga'), 'riga\nMario Rossi\nriga');

setDict({ '[CF_1]': 'RSSMRA78S03L750K' });
eq('indice inventato [CF_12] resta non risolto', callReverse('vedi [CF_12] qui'), 'vedi [CF_12] qui');
eq('indice inventato nudo CF_12 resta non risolto', callReverse('vedi CF_12 qui'), 'vedi CF_12 qui');
eq('incollato ASCII (ilCF_1) resta intatto', callReverse('ilCF_1 resta intatto'), 'ilCF_1 resta intatto');
eq('incollato a lettera accentata resta intatto (simmetrico al caso ASCII)', callReverse('il codice CF_1è già stato controllato'), 'il codice CF_1è già stato controllato');
eq('stesso placeholder con uno spazio vero si risolve', callReverse('il codice CF_1 è già stato controllato'), 'il codice RSSMRA78S03L750K è già stato controllato');

setDict({ '[CF_1]': 'AAAA111', '[CF_2]': 'BBBB222' });
eq('asterischi condivisi fra due placeholder adiacenti, esito deterministico', callReverse('**CF_1****CF_2**'), 'AAAA111BBBB222');

setDict({ '[CF_1]': 'RSSMRA80A01H501U' });
eq('grassetto non correlato al placeholder non viene assorbito', callReverse('Vedi CF_1**parola importante** qui'), 'Vedi RSSMRA80A01H501U**parola importante** qui');

setDict({ '[ORG_1]': 'Studio Legale CF_2 & Partners', '[CF_2]': 'RSSMRA80A01H501U' });
eq('un valore non viene ri-sostituito se contiene il nome di un\'altra chiave', callReverse('Rivolgersi a ORG_1 per informazioni.'), 'Rivolgersi a Studio Legale CF_2 & Partners per informazioni.');

console.log('--- caricamento dizionario da file ---');
loadDictFromJsonText(JSON.stringify({ '[CF_1]': 'RSSMRA80A01H501U' }));
eq('dizionario valido viene caricato', JSON.stringify(MAP), JSON.stringify({ '[CF_1]': 'RSSMRA80A01H501U' }));
const beforeMap = JSON.stringify(MAP);
loadDictFromJsonText(JSON.stringify({ '': 'XXXX' }));
eq('chiave vuota viene rifiutata, MAP invariata', JSON.stringify(MAP), beforeMap);
loadDictFromJsonText(JSON.stringify({ '[CF_1]': 123 }));
eq('valore non testuale viene rifiutato, MAP invariata', JSON.stringify(MAP), beforeMap);

console.log('\n' + pass + ' passati, ' + fail + ' falliti');
process.exit(fail ? 1 : 0);
"""


def main() -> int:
    if shutil.which("node") is None:
        print("node non trovato in PATH: necessario per questo test "
              "(e' gia' un prerequisito per buildare il frontend Tauri).",
              file=sys.stderr)
        return 2

    src = APP_PY.read_text(encoding="utf-8")
    start = src.find(START_MARK)
    end = src.find(END_MARK, start)
    if start == -1 or end == -1:
        print(f"Blocco reverse()/dictFile non trovato in {APP_PY} "
              "(marcatori spostati o rinominati?).", file=sys.stderr)
        return 2
    reverse_block = src[start:end + len(END_MARK)]

    script = HARNESS % {"reverse_block": reverse_block}
    with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False, encoding="utf-8") as f:
        f.write(script)
        script_path = f.name

    try:
        result = subprocess.run(["node", script_path], capture_output=True, text=True, timeout=30)
    finally:
        Path(script_path).unlink(missing_ok=True)

    print(result.stdout, end="")
    if result.stderr:
        print(result.stderr, file=sys.stderr, end="")
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
